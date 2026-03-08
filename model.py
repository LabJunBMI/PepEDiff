import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.t5.modeling_t5 import T5Stack, T5Attention, T5Config

from typing import Any, List, Dict
from egnn_clean import EGNN
from copy import deepcopy
from dataset import *


class SELayer(nn.Module):
    # according to the paper: https://arxiv.org/pdf/2401.13858
    def __init__(self, t5_config: T5Config, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(t5_config.d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(t5_config.d_model, elementwise_affine=False)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(t5_config.d_model, t5_config.d_model, bias=True),
            nn.SiLU(),
            nn.Linear(t5_config.d_model, 6 * t5_config.d_model, bias=True),
        )

        self.attn = T5Attention(t5_config, **block_kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(t5_config.d_model, int(t5_config.d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(t5_config.dropout_rate),
            nn.Linear(int(t5_config.d_model * mlp_ratio), t5_config.d_model),
            nn.Dropout(t5_config.dropout_rate),
        )

        nn.init.zeros_(self.adaLN_modulation[0].weight)
        nn.init.zeros_(self.adaLN_modulation[0].bias)

    def forward(self, x, c, mask):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self._modulate(
            self.norm1(self.attn(x, mask)[0]), shift_msa, scale_msa
        )
        x = x + gate_mlp * self._modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp)
        return x

    def _modulate(self, x, shift, scale):
        return x * (1 + scale) + shift


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    Built primarily for score-based models.

    Source:
    https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, embed_dim: int = 384, scale: float = 2 * torch.pi):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        w = torch.randn(embed_dim // 2) * scale
        assert w.requires_grad == False
        self.register_buffer("W", w)

    def forward(self, x: torch.Tensor):
        """
        takes as input the time vector and returns the time encoding
        time (x): (batch_size, )
        output  : (batch_size, embed_dim)
        """
        if x.ndim > 1:
            x = x.squeeze()
        elif x.ndim < 1:
            x = x.unsqueeze(0)
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embed


class PepEDiff(nn.Module):
    def __init__(
        self,
        t5_config,
        node_input_dim,
        edge_input_dim,
        num_layers,
        augment_eps,
    ):
        super(PepEDiff, self).__init__()
        self.augment_eps = augment_eps
        self.hidden_dim = t5_config.d_model

        # Receptor Layers
        self.EGNN_encoder = EGNN(
            in_node_nf=node_input_dim,
            in_edge_nf=edge_input_dim,
            out_node_nf=self.hidden_dim,
            hidden_nf=self.hidden_dim,
            n_layers=num_layers,
            attention=True,
            normalize=True,
            tanh=True,
        )
        self.seq_feat_proj = nn.Linear(1024, self.hidden_dim)  # 1024 from prott5
        self.receptor_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.receptor_encoder = T5Stack(t5_config)
        # Peptide Layers
        self.timestep_projector = GaussianFourierProjection(self.hidden_dim)
        # self.timestep_emb = SELayer(t5_config)
        self.peptide_proj = nn.Sequential(
            nn.Linear(1024, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1024),
        )
        decoder_config = deepcopy(t5_config)
        decoder_config.is_decoder = True
        self.timestep_emb = SELayer(t5_config)
        self.peptide_encoder = T5Stack(decoder_config)
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        # receptors
        coords,
        edge_index,
        structure_feat,
        seq_feat,
        pocket_mask,
        # peptides
        noised_emb,
        emb_mask,
        timesteps,
        batch_id,
    ):
        # Extend Mask
        # emb_mask = self._exetend_attention_mask(emb_mask)
        # pocket_mask = self._exetend_attention_mask(pocket_mask)
        # Data augmentation
        if self.training and self.augment_eps > 0:
            coords = (  # [#node, 5 ("N", "Ca", "C", "O", "R"), 3(x,y,z)
                coords + torch.randn_like(coords) * self.augment_eps
            )
        # Cross Res structure Feat.
        h_V_geo, h_E = get_geo_feat(coords, edge_index)  # _, [#edge, 450]
        structure_feat = torch.cat([structure_feat, h_V_geo], dim=-1)  # [#node, 20+184]
        structure_feat, _ = self.EGNN_encoder(
            structure_feat, coords[:, 1, :], edge_index, h_E
        )
        # Sequence Feat.
        seq_feat = self.seq_feat_proj(seq_feat)
        # Receptor Feat.
        feature_embedding = torch.concat(
            [structure_feat, seq_feat], dim=1
        )  # [#node,hid*2]
        receptor_feat, receptor_mask = to_dense_batch(
            feature_embedding, batch_id
        )  # [B,max(L),hid*2]
        receptor_feat = self.receptor_proj(receptor_feat)  # [B,max(L),hid]
        receptor_feat = self.receptor_encoder(
            inputs_embeds=receptor_feat, 
            attention_mask=receptor_mask
        ).last_hidden_state
        # Peptide feat.
        timestep_proj = self.timestep_projector(timesteps.squeeze(dim=-1)).unsqueeze(1)
        peptide_emb = self.peptide_encoder(  # Aim to learn to true emb.
            inputs_embeds=noised_emb,
            attention_mask=emb_mask,
            encoder_hidden_states=receptor_feat,
            encoder_attention_mask=pocket_mask,
        ).last_hidden_state
        emb_mask = self.peptide_encoder.get_extended_attention_mask(emb_mask, None)
        peptide_emb = self.timestep_emb(  # Scale true emb based on timestep
            peptide_emb, timestep_proj, emb_mask
        )
        output = (  # = (noised_emb-peptide_emb) in NN, calculate the noise
            peptide_emb + noised_emb
        )
        return output


class PepEDiffTrainer(PepEDiff, pl.LightningModule):
    def __init__(
        self,
        attn_config: T5Config,
        node_input_dim,
        edge_input_dim,
        num_layers,
        augment_eps,
        max_len=1024,
        epochs: int = 1,
        lr_scheduler=None,
        l2_lambda: float = 0.0,
        steps_per_epoch: int = 250,
        learning_rate: float = 5e-5,
        **kwargs,
    ):
        PepEDiff.__init__(
            self,
            attn_config,
            node_input_dim,
            edge_input_dim,
            num_layers,
            augment_eps,
        )
        # Store information about leraning rates and loss
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.train_epoch_losses = []
        self.valid_epoch_losses = []
        self.epochs = epochs
        self.max_len = max_len
        self.l2_lambda = l2_lambda
        self.train_epoch_counter = 0
        self.attn_config = attn_config
        self.loss_mse = nn.MSELoss()
        self.loss_cos = nn.CosineEmbeddingLoss()

    def _get_loss_terms(self, data) -> List[torch.Tensor]:
        """
        Returns the loss terms for the model. Length of the returned list
        is equivalent to the number of features we are fitting to.
        """
        batch_size = len(data.batch.unique())
        pocket_mask = to_dense_batch(data.pocket_mask, data.batch)[0]
        noised_emb = data.noised_emb.reshape(batch_size, -1, self.max_len)
        emb_mask = data.peptide_mask.reshape(batch_size, -1, self.max_len)
        known_noise = data.known_noise.reshape(batch_size, -1, self.max_len)
        predicted_noise = self.forward(
            # Receptor Structure Features
            coords=data.X,
            edge_index=data.edge_index,
            structure_feat=data.receptor_structure,
            # Receptor Sequence Features
            seq_feat=data.receptor_seq_emb,
            pocket_mask=pocket_mask,
            # Peptide Features
            noised_emb=noised_emb,
            emb_mask=emb_mask,
            # Others
            timesteps=data.timestep,
            batch_id=data.batch,
        )
        peptide_mask = data.peptide_mask.reshape(batch_size, self.max_len).bool()
        predicted_noise = predicted_noise[peptide_mask]
        known_noise = known_noise[peptide_mask]
        assert (
            known_noise.shape == predicted_noise.shape
        ), f"{known_noise.shape} != {predicted_noise.shape}"
        mse_loss = self.loss_mse(predicted_noise, known_noise)
        cos_loss = self.loss_cos(
            predicted_noise,
            known_noise,
            torch.ones((known_noise.shape[0]), device=self.device),
        )
        loss = mse_loss + cos_loss
        return loss, mse_loss, cos_loss

    def training_step(self, batch, batch_idx):
        """
        Training step, runs once per batch
        """
        total_loss, mse_loss, cos_loss = self._get_loss_terms(batch)
        self.log_dict(
            {"train_loss": total_loss, "mse": mse_loss, "cos": cos_loss}, sync_dist=True
        )  # Don't seem to need rank zero or sync dist
        return total_loss

    def on_train_batch_end(self, outputs, batch_idx, dataloader_idx=0) -> None:
        """Log the average training loss over the epoch"""
        # pl.utilities.rank_zero_info(outputs)
        self.train_epoch_losses.append(float(outputs["loss"]))

    def on_train_epoch_end(self) -> None:
        pl.utilities.rank_zero_info(
            f"Traning Loss:{sum(self.train_epoch_losses)/len(self.train_epoch_losses)}"
        )
        self.train_epoch_losses = []
        self.train_epoch_counter += 1

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Validation step
        """
        with torch.no_grad():
            total_loss, mse_loss, cos_loss = self._get_loss_terms(batch)
        # with rank zero it seems that we don't need to use sync_dist
        self.log_dict(
            {"val_loss": total_loss, "mse": mse_loss, "cos": cos_loss},
            rank_zero_only=True,
            sync_dist=True,
        )
        return total_loss

    def on_validation_batch_end(self, outputs, batch_idx, dataloader_idx=0) -> None:
        """Log the average validation loss over the epoch"""
        self.valid_epoch_losses.append(float(outputs))

    def on_validation_epoch_end(self) -> None:
        pl.utilities.rank_zero_info(
            f"Validation Loss:{sum(self.valid_epoch_losses)/len(self.valid_epoch_losses)}"
        )
        self.valid_epoch_losses = []

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Return optimizer. Limited support for some optimizers
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optim}
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optim,
                        max_lr=1e-2,
                        epochs=self.epochs,
                        steps_per_epoch=self.steps_per_epoch,
                    ),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            elif self.lr_scheduler == "LinearWarmup":
                # https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
                # Transformers typically do well with linear warmup
                warmup_steps = int(self.epochs * 0.1)
                pl.utilities.rank_zero_info(
                    f"Using linear warmup with {warmup_steps}/{self.epochs} warmup steps"
                )
                retval["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optim,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.epochs,
                    ),
                    "frequency": 1,
                    "interval": "epoch",  # Call after 1 epoch
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval
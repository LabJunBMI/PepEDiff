import torch
from torch import nn
from transformers import T5Config
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from lightning_fabric.utilities.seed import seed_everything

from dataset import NoisedReceptorPeptideDataset
from model import PepEDiff
from diff_utils import *
from dataset import *

from typing import Tuple
from tqdm import tqdm
import pickle
import gc
import os

gc.enable()

MODEL_PATH = "./model_diff.pt"
SPLIT_FILE = "./data/data_split.pt"
OUTPUT_FOLDER = "./data/sample_res/sample_test_{random_seed}"

CONFIG = dict(
    # Computation Resource
    gpu_id=[5],
    num_thread=16,
    loader_num_workers=16,
    loader_prefetch_factor=4,
    # For Training
    lr=5e-5,
    l2_lambda=0.1,
    batch_size=32,
    random_seed=0,
    min_epochs=1,
    max_epochs=500,
    gradient_clip=1.0,
    lr_scheduler="LinearWarmup",
    data_noise_ration=0.1,
    augment_eps=0.15,
    max_seq_len=1024,
    diff_timestep=1000,
    diff_step_size=1,
    # For Model
    node_input_dim=20 + 184,  # precalculated + generated
    edge_input_dim=450,
    d_model=1024,
    d_ff=2048,
    num_layers=2,
    num_heads=4,
    dropout_rate=0.1,
    feed_forward_proj="gated-gelu",
)


def get_dataloader() -> Tuple[DataLoader, DataLoader]:
    data_split = torch.load(SPLIT_FILE)
    ds_train = NoisedReceptorPeptideDataset(dataset=data_split["train"])
    dl_train = DataLoader(
        ds_train,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=CONFIG["num_thread"],
        prefetch_factor=CONFIG["num_thread"],
    )
    ds_val = NoisedReceptorPeptideDataset(dataset=data_split["val"])
    dl_val = DataLoader(
        ds_val,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=CONFIG["num_thread"],
        prefetch_factor=CONFIG["num_thread"],
    )
    return dl_train, dl_val


def load_model() -> PepEDiff:
    attn_config = T5Config(
        d_ff=CONFIG["d_ff"],
        d_model=CONFIG["d_model"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout_rate=CONFIG["dropout_rate"],
        feed_forward_proj=CONFIG["feed_forward_proj"],
        is_decoder=False,
        use_cache=False,
        is_encoder_decoder=False,
    )
    model = PepEDiff(
        attn_config,
        CONFIG["node_input_dim"],
        CONFIG["edge_input_dim"],
        CONFIG["num_layers"],
        CONFIG["augment_eps"],
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(f"cuda:{CONFIG['gpu_id'][0]}").eval()
    return model


@torch.no_grad()
def p_sample(
    model: PepEDiff,
    noised_emb: torch.Tensor,
    batch,
    timestep: int,
    betas: torch.Tensor,
) -> torch.Tensor:
    device = next(model.parameters()).device
    if timestep <= 1:  # skip timestep=0
        return noised_emb, noised_emb
    alpha_beta_values = compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])
    sqrt_recip_alphas_t = sqrt_recip_alphas[timestep]
    betas_t = betas[timestep]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][timestep]
    batch_size = len(batch.batch.unique())
    pocket_mask = to_dense_batch(batch.pocket_mask, batch.batch)[0]
    noised_emb = noised_emb.reshape(batch_size, CONFIG["max_seq_len"], -1)
    emb_mask = batch.peptide_mask.reshape(batch_size, CONFIG["max_seq_len"], -1)
    timesteps = torch.full(
        (noised_emb.shape[0],), timestep, device=device, dtype=torch.long
    )
    predicted_noise = model(
        # Receptor Structure Features
        coords=batch.X,
        edge_index=batch.edge_index,
        structure_feat=batch.receptor_structure,
        # Receptor Sequence Features
        seq_feat=batch.receptor_seq_emb,
        pocket_mask=pocket_mask,
        # Peptide Features
        noised_emb=noised_emb,
        emb_mask=emb_mask,
        # Others
        timesteps=timesteps,
        batch_id=batch.batch,
    )
    model_mean = sqrt_recip_alphas_t * (
        noised_emb - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = alpha_beta_values["posterior_variance"][timestep]
    noise = torch.randn_like(noised_emb)
    return model_mean + torch.sqrt(posterior_variance_t) * noise, predicted_noise


@torch.no_grad()
def p_sample_loop(
    model: PepEDiff,
    batch,
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """
    device = next(model.parameters()).device
    batch = batch.to(device)
    ligand_emb_noise = torch.randn_like(batch.known_noise, device=device)
    print("init noise:", ligand_emb_noise.max(), ligand_emb_noise.min())
    b = ligand_emb_noise.shape[0]
    noises = []
    tqdm_bar = tqdm(
        reversed(range(0, CONFIG["diff_timestep"], CONFIG["diff_step_size"])),
        desc="sampling loop time step",
        total=int(CONFIG["diff_timestep"] / CONFIG["diff_step_size"]),
        dynamic_ncols=True,
    )
    for i in tqdm_bar:
        # Shape is (batch, seq_len, 1024)
        ligand_emb_noise, pred_noise = p_sample(
            model=model,
            noised_emb=ligand_emb_noise,
            batch=batch,
            timestep=i,
            betas=betas,
        )
        tqdm_bar.set_postfix(
            emb_min=float(ligand_emb_noise.min()),
            emb_max=float(ligand_emb_noise.max()),
            pred_min=float(pred_noise.min()),
            pred_max=float(pred_noise.max()),
        )
        noises.append(ligand_emb_noise.cpu())
    del b, ligand_emb_noise
    return torch.stack(noises)


def sample_batch(model: PepEDiff, dataset: NoisedReceptorPeptideDataset):
    test_dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"], prefetch_factor=2, num_workers=16
    )
    for batch_idx, batch in enumerate(test_dataloader):
        print(f"Generating Batch {batch_idx}/{len(test_dataloader)}")
        # Sample noise and sample the lengths
        sampled = p_sample_loop(
            model=model,
            batch=batch,
            betas=dataset.alpha_beta_terms["betas"],
        )
        # Gets to size (timesteps, seq_len, n_ft)
        batch_size = len(batch.batch.unique())
        peptide_mask = batch.peptide_mask.reshape(batch_size, -1, CONFIG["max_seq_len"])
        ligand_length = [m.sum().int() for m in peptide_mask]
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy() for i, l in enumerate(ligand_length)
        ]
        trimmed_sampled = [s[-1] for s in trimmed_sampled]  # extract last time step
        print(trimmed_sampled[0].shape)
        for pdb_id, sampled in zip(batch["pdb_id"], trimmed_sampled):
            output_folder = OUTPUT_FOLDER.format(random_seed=CONFIG["random_seed"])
            os.makedirs(output_folder, exist_ok=True)
            torch.save(sampled, os.path.join(output_folder, f"{pdb_id}.pkl"))
        del sampled, ligand_length, trimmed_sampled
    return None


def get_test_dataset():
    data_split = torch.load(SPLIT_FILE)
    test_ds = NoisedReceptorPeptideDataset(data_split["test"])
    return test_ds


def get_train_val_dataset():
    data_split = torch.load(SPLIT_FILE)
    ds = NoisedReceptorPeptideDataset(data_split["train"] + data_split["val"])
    return ds


if __name__ == "__main__":
    random_seeds = [1,2,3,4,5,6,7,8,9]
    print("Sampling seeds:", random_seeds)
    torch.set_float32_matmul_precision("medium")
    torch.set_num_threads(CONFIG["num_thread"])
    ds = get_test_dataset()
    # ds = get_train_val_dataset()
    model = load_model()
    for i in random_seeds:
        CONFIG["random_seed"] = i
        seed_everything(CONFIG["random_seed"])
        sample_batch(model, ds)

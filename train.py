import torch
import pytorch_lightning as pl
from transformers import T5Config
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_fabric.utilities.seed import seed_everything

from dataset import NoisedReceptorPeptideDataset
from model import PepEDiffTrainer

MODEL_PATH = "./model_diff.pt"
SPLIT_FILE = "./data/data_split.pt"

CONFIG = dict(
    # Computation Resource
    gpu_id=[4,5,6,7],
    num_thread=16,
    loader_num_workers=16,
    loader_prefetch_factor=4,
    # For Training
    lr=5e-5,
    l2_lambda=0.1,
    batch_size=8,
    random_seed=0,
    min_epochs=1,
    max_epochs=500,
    gradient_clip=1.0,
    lr_scheduler="LinearWarmup",
    data_noise_ration=0.1,
    augment_eps=0.15,
    max_seq_len=1024,
    diff_timestep=1000,
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


def get_dataloader():
    data_split = torch.load(SPLIT_FILE)
    ds_train = NoisedReceptorPeptideDataset(
        dataset=data_split["train"],
        timesteps=CONFIG["diff_timestep"]
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=CONFIG["num_thread"],
        prefetch_factor=CONFIG["num_thread"],
    )
    ds_val = NoisedReceptorPeptideDataset(
        dataset=data_split["val"],
        timesteps=CONFIG["diff_timestep"]
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=CONFIG["num_thread"],
        prefetch_factor=CONFIG["num_thread"],
    )
    return dl_train, dl_val


def train_model(
    attn_config: T5Config,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    checkpoint_callback = ModelCheckpoint(
        dirpath="./",
        monitor="val_loss",  # Monitor the validation loss
        filename="best_val_model",  # Filename template
        save_top_k=1,  # Only save the best model
        mode="max",  # Save the model with the highest validation loss
    )
    model = PepEDiffTrainer(
        attn_config=attn_config,
        node_input_dim=CONFIG["node_input_dim"],
        edge_input_dim=CONFIG["edge_input_dim"],
        num_layers=CONFIG["num_layers"],
        augment_eps=CONFIG["augment_eps"],
        max_len=CONFIG["max_seq_len"],
        epochs=CONFIG["max_epochs"],
        lr_scheduler=CONFIG["lr_scheduler"],
        l2_lambda=CONFIG["l2_lambda"],
        steps_per_epoch=len(train_dataloader),
        learning_rate=CONFIG["lr"],
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters")
    trainer = pl.Trainer(
        default_root_dir="./",
        gradient_clip_val=CONFIG["gradient_clip"],
        callbacks=[checkpoint_callback],
        min_epochs=CONFIG["min_epochs"],
        max_epochs=CONFIG["max_epochs"],
        check_val_every_n_epoch=1,
        log_every_n_steps=30,
        accelerator="gpu",
        devices=CONFIG["gpu_id"],
        strategy='ddp_find_unused_parameters_true',
        # move_metrics_to_cpu=False,  # Saves memory
    )
    print("Start training")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return trainer, model


if __name__ == "__main__":
    seed_everything(CONFIG["random_seed"])
    torch.set_float32_matmul_precision("medium")
    torch.set_num_threads(CONFIG["num_thread"])
    train_dataloader, val_dataloader = get_dataloader()
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
    trainer, model = train_model(attn_config, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), MODEL_PATH)

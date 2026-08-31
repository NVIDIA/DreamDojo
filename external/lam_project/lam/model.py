from os import makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import piq
import torch
from PIL import Image
from einops import rearrange
# from kmeans_pytorch import kmeans
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

OptimizerCallable = Callable[[Iterable], Optimizer]

from lam.modules import LatentActionModel


class LAM(LightningModule):
    def __init__(
        self,
        image_channels: int = 3,
        # Latent action autoencoder
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        beta: float = 0.01,
        log_interval: int = 1000,
        log_path: str = "log_imgs",
        optimizer: OptimizerCallable = AdamW,
        init_checkpoint_path: str = None,
        gradient_loss_scale: float = 1.0,
    ) -> None:
        super(LAM, self).__init__()
        if not np.isfinite(gradient_loss_scale) or gradient_loss_scale <= 0:
            raise ValueError("gradient_loss_scale must be finite and positive")
        self.lam = LatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout
        )
        self.beta = beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.gradient_loss_scale = gradient_loss_scale

        if init_checkpoint_path:
            checkpoint = torch.load(
                init_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.load_state_dict(state_dict, strict=True)
            del checkpoint, state_dict
            print(f"Initialized LAM weights from {init_checkpoint_path}")

        self.save_hyperparameters()

    def shared_step(self, batch: Dict) -> Tuple:
        outputs = self.lam(batch)
        gt_future_frames = batch["videos"][:, 1:]

        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        kl_loss = -0.5 * torch.sum(1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(), dim=1).mean()
        loss = mse_loss + self.beta * kl_loss

        # Compute monitoring measurements
        gt = gt_future_frames.clamp(0, 1).reshape(-1, *gt_future_frames.shape[2:]).permute(0, 3, 1, 2)
        recon = outputs["recon"].clamp(0, 1).reshape(-1, *outputs["recon"].shape[2:]).permute(0, 3, 1, 2)
        psnr = piq.psnr(gt, recon).mean()
        ssim = piq.ssim(gt, recon).mean()
        return outputs, loss, (
            ("mse_loss", mse_loss),
            ("kl_loss", kl_loss),
            ("psnr", psnr),
            ("ssim", ssim)
        )

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False
        )

        if batch_idx % self.log_interval == 0:  # Start of the epoch
            self.log_images(batch, outputs, "train")
        # DDP averages gradients uniformly across ranks.  Scaling the local
        # mean loss lets ranks use different local batch sizes while still
        # producing the exact sample-weighted global-batch gradient.
        return loss * self.gradient_loss_scale

    # @torch.no_grad()
    # def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
    #     # Compute the validation loss
    #     outputs, loss, aux_losses = self.shared_step(batch)
    #
    #     # Log the validation loss
    #     self.log_dict(
    #         {**{"val_loss": loss}, **{f"val/{k}": v for k, v in aux_losses}},
    #         prog_bar=True,
    #         logger=True,
    #         on_step=True,
    #         on_epoch=True,
    #         sync_dist=True
    #     )
    #
    #     if batch_idx % self.log_interval == 0:  # Start of the epoch
    #         self.log_images(batch, outputs, "val")
    #     return loss

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log_images(batch, outputs, "test")
        return loss

    def log_images(self, batch: Dict, outputs: Dict, split: str) -> None:
        gt_seq = batch["videos"][0].clamp(0, 1).cpu()
        recon_seq = outputs["recon"][0].clamp(0, 1).cpu()
        recon_seq = torch.cat([gt_seq[:1], recon_seq], dim=0)
        compare_seq = torch.cat([gt_seq, recon_seq], dim=1)
        compare_seq = rearrange(compare_seq * 255, "t h w c -> h (t w) c")
        compare_seq = compare_seq.detach().numpy().astype(np.uint8)
        img_path = path.join(self.log_path, f"{split}_step{self.global_step:06}.png")
        makedirs(path.dirname(img_path), exist_ok=True)
        img = Image.fromarray(compare_seq)
        try:
            img.save(img_path)
        except:
            pass

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim

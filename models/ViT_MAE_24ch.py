import lightning as L
from transformers import ViTMAEForPreTraining
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
def unpatchify(patches, patch_size, num_channels, img_h, img_w):
    """
    Reconstructs the image from ViT-MAE patches.
    Args:
        patches (torch.Tensor): (N, L, patch_dim) - the output patches from ViT-MAE.
        patch_size (int): Size of each patch.
        num_channels (int): Number of image channels.
        img_h (int): Original height of the spectrogram.
        img_w (int): Original width of the spectrogram.
    Returns:
        torch.Tensor: Reconstructed full image of shape (N, num_channels, H, W).
    """
    N, L, patch_dim = patches.shape
    h = w = int(L ** 0.5)
    assert h * w == L, f"Mismatch: L={L}, but computed h*w={h*w}"
    patches = patches.view(N, h, w, patch_size, patch_size, num_channels)
    img = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
    img = img.view(N, num_channels, h * patch_size, w * patch_size)
    return img
class Model(L.LightningModule):
    """
    PyTorch Lightning model for pre-training a ViT-MAE on 24-channel spectrograms.
    """
    def __init__(self, lr=0.001, min_lr=1e-7, model_name="facebook/vit-mae-base", mask_ratio=0.75, **kwargs):
        """
        Initializes the model.
        Args:
            lr (float): Learning rate.
            min_lr (float): Minimum learning rate.
            model_name (str): Pre-trained model name.
            mask_ratio (float): Masking ratio for pretraining.
            **kwargs: Additional keyword arguments for flexibility.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.min_lr = min_lr
        self.mask_ratio = mask_ratio
        self.model = ViTMAEForPreTraining.from_pretrained(model_name)
        self.model.config.num_channels = 24
        self._modify_patch_embedding()
        self.model.decoder.decoder_pred = nn.Linear(
            in_features=self.model.config.decoder_hidden_size,
            out_features=24 * self.model.config.patch_size ** 2
        )
        self.patch_size = self.model.config.patch_size
        self.num_channels = 24
        self.img_h = self.img_w = self.model.config.image_size
    def _modify_patch_embedding(self):
        """
        Modifies the patch embedding layer to accept 24 channels.
        """
        old_patch_embed = self.model.vit.embeddings.patch_embeddings.projection
        embed_dim = old_patch_embed.out_channels
        patch_size = old_patch_embed.kernel_size
        new_patch_embed = nn.Conv2d(
            in_channels=24,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=old_patch_embed.bias is not None
        )
        with torch.no_grad():
            pretrained_weights = old_patch_embed.weight
            expansion_factor = 24 // pretrained_weights.shape[1]
            expanded_weights = pretrained_weights.repeat(1, expansion_factor, 1, 1)
            new_patch_embed.weight[:, :pretrained_weights.shape[1], :, :] = pretrained_weights
            if old_patch_embed.bias is not None:
                new_patch_embed.bias[:] = old_patch_embed.bias
        self.model.vit.embeddings.patch_embeddings.projection = new_patch_embed
        self.model.vit.embeddings.num_channels = 24
        self.model.vit.embeddings.patch_embeddings.num_channels = 24
    def forward(self, pixel_values):
        """
        Forward pass for masked image reconstruction.
        Args:
            pixel_values (torch.Tensor): Input spectrograms.
        Returns:
            vitmae_loss (torch.Tensor): Loss value of the ViTMAE (reconstruction error of the masked patches).
            spec_loss (torch.Tensor): Mean squared error loss between the original spectrogram image and its reconstruction.
            reconstructed_img (torch.Tensor): The reconstructed image.
            reconstructed_img_plus_visible (torch.Tensor): The image of the reconstructed masked patches + visible patches.
        """
        outputs = self.model(pixel_values)
        vitmae_loss = outputs.loss
        reconstructed_patches = outputs.logits
        mask = outputs.mask
        reconstructed_img = unpatchify(
            reconstructed_patches, self.patch_size, self.num_channels, self.img_h, self.img_w
        )
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size * self.patch_size * self.num_channels)
        mask = unpatchify(mask, self.patch_size, self.num_channels, self.img_h, self.img_w)
        mask = mask.to(pixel_values.dtype)
        reconstructed_img_plus_visible = pixel_values * (1 - mask) + reconstructed_img * mask # [b_size, 24, 224, 224]
        # spec_loss = F.mse_loss(reconstructed_img, pixel_values)
        spec_loss = F.mse_loss(reconstructed_img, pixel_values, reduction='none')
       
        return vitmae_loss, spec_loss, reconstructed_img_plus_visible, reconstructed_img
    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        vitmae_loss, spec_loss, _, _ = self(batch)
        self.log("train_vitmae_loss", vitmae_loss, prog_bar=True, logger=True)
        self.log("train_spec_loss", spec_loss, prog_bar=True, logger=True)
        return spec_loss
    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        vitmae_loss, spec_loss, _, _ = self(batch)
        self.log("val_vitmae_loss", vitmae_loss, prog_bar=True, logger=True)
        self.log("val_spec_loss", spec_loss, prog_bar=True, logger=True)
        return spec_loss
    def test_step(self, batch, batch_idx):
        """Test step for one batch."""
        vitmae_loss, spec_loss, _, _ = self(batch)
        self.log("test_vitmae_loss", vitmae_loss, prog_bar=True, logger=True)
        self.log("test_spec_loss", spec_loss, prog_bar=True, logger=True)
        return spec_loss
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        Returns:
            dict: Optimizer and learning rate scheduler.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, min_lr=self.min_lr)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_spec_loss"}}
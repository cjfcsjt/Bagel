import torch
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

class SSLDecoder(nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):
        super().__init__()
        self.patch_size = patch_size
        self.decoder_norm = nn.LayerNorm(dec_embed_dim)
        self.decoder_pred = nn.Linear(dec_embed_dim, 2*(output_dim)*self.patch_size**2, bias=True) # decoder to patch

    def forward(self, hidden):
        x = self.decoder_norm(hidden)
        x = self.decoder_pred(x)
        pred, conf = torch.chunk(x, 2, dim=-1)
        return pred, conf

class ConfLoss(nn.Module):
    def __init__(
        self,
        patch_size,
    ):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, imgs, pred, conf, mask, repeat_masks=None):
        """
        imgs: [B, V, 3, H, W]
        pred: [B, V, L, p*p*3]
        conf: [B, V, L, p*p*3]
        mask: [B, V, L], 0=keep, 1=remove
        repeat_masks: [B, V] bool tensor, True=repeated view (excluded from loss). Optional.
        """
        target = F.pixel_unshuffle(imgs, downscale_factor=self.patch_size) # 把每个 14×14 的 patch 展平为一个 588 维向量，作为 SSL 解码器需要预测的目标像素值
        target = target.permute(0,1,3,4,2).flatten(2, 3)

        mse = (pred - target) ** 2 # [B, V, L, p²×3]
        loss = ((conf+0.1) * mse).mean(dim=-1)  # [B, V, L], mean loss per patch

        # Build effective mask: exclude repeated views from loss
        effective_mask = mask.clone()
        if repeat_masks is not None:
            # Zero out mask for repeated views so they contribute zero loss
            # repeat_masks: (B, V), expand to (B, V, L)
            effective_mask = effective_mask * (~repeat_masks).unsqueeze(-1).float()

        # 只在 masked patch (non-repeated views) 上计算损失
        mask_sum = effective_mask.sum()
        if mask_sum == 0:
            # Edge case: all views are repeated or no masked patches
            final_loss = loss.new_zeros(1).squeeze()
            details = {
                'ssl_mse': loss.new_zeros(1).squeeze(),
                'ssl_per_instance_loss': loss.new_zeros(1).squeeze(),
                'ssl_conf_reg': loss.new_zeros(1).squeeze(),
            }
            return final_loss, details

        per_instance_loss = (loss * effective_mask).sum(dim=(1,2)) / effective_mask.sum(dim=(1,2)).clamp(min=1)
        loss_val = (loss * effective_mask).sum() / mask_sum  # mean loss on removed patches (non-repeated)

        # 置信度正则化 (only on non-repeated views)
        conf_reg = -torch.log(conf + 1e-6).mean(dim=-1)
        conf_reg_val = (conf_reg * effective_mask).sum() / mask_sum

        final_loss = loss_val + 0.1 * conf_reg_val
        details = {
            'ssl_mse': (mse.detach().mean(dim=-1) * effective_mask).sum() / mask_sum,
            'ssl_per_instance_loss': per_instance_loss.detach().mean(),
            'ssl_conf_reg': conf_reg_val.detach(),
        }
        return final_loss, details
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

    def forward(self, imgs, pred, conf, mask):
        """
        imgs: [B, V, 3, H, W]
        pred: [B, V, L, p*p*3]
        conf: [B, V, L, p*p*3]
        mask: [B, V, L], 0=keep, 1=remove
        """
        target = F.pixel_unshuffle(imgs, downscale_factor=self.patch_size) # 把每个 14×14 的 patch 展平为一个 588 维向量，作为 SSL 解码器需要预测的目标像素值
        target = target.permute(0,1,3,4,2).flatten(2, 3)

        mse = (pred - target) ** 2 # [B, V, L, p²×3]
        loss = ((conf+0.1) * mse).mean(dim=-1)  # [N, L], mean loss per patch, 模型同时预测"像素值"和"我对这个预测有多大把握"。置信度高的 patch 损失被放大（模型被迫对自信的预测更准确），置信度低的 patch 损失被缩小（允许模型对困难区域"承认不确定"）
        # 只在 masked patch 上计算损失
        per_instance_loss = (loss * mask).sum(dim=(1,2)) / mask.sum(dim=(1,2))

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # 置信度正则化, 防止置信度坍缩的正则项。 
        # 如果模型把所有 conf → 0，那第2步的损失就趋近于 0（作弊了）-log(conf) 对低置信度施加惩罚：conf 越小，-log(conf) 越大
        # 效果：鼓励模型给出高置信度，而不是简单地降低 conf 来"逃避"损失
        conf_reg = -torch.log(conf + 1e-6).mean(dim=-1)
        conf_reg = (conf_reg * mask).mean()
        final_loss = loss + 0.1 * conf_reg
        details = {
            'ssl_mse': (mse.detach().mean(dim=-1) * mask).sum() / mask.sum(),
            'ssl_per_instance_loss': per_instance_loss.detach().mean(),
            'ssl_conf_reg': conf_reg.detach(),
        }
        return final_loss, details
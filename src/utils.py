import torch
import torch.nn.functional as F_func
from typing import Tuple

def ssim_map(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute per-image SSIM and contrast-structure (CS) maps.
    Returns:
      ssim_map: Tensor of shape (N,), the mean SSIM per sample.
      cs_map:   Tensor of shape (N,), the mean CS per sample.
    """
    # local means
    mu1 = F.conv2d(img1, window, padding=window.size(-1)//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window.size(-1)//2, groups=img2.size(1))

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # variances and covariance
    sigma1_sq = F.conv2d(img1*img1, window, padding=window.size(-1)//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window.size(-1)//2, groups=img2.size(1)) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding=window.size(-1)//2, groups=img1.size(1)) - mu1_mu2

    # SSIM numerator and denominator
    ssim_num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    ssim_den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map_tensor = ssim_num / (ssim_den + eps)

    # contrast–structure term
    cs_num = 2 * sigma12 + C2
    cs_den = sigma1_sq + sigma2_sq + C2
    cs_map_tensor = cs_num / (cs_den + eps)

    # mean over channels & spatial dims
    ssim_per_sample = ssim_map_tensor.mean(dim=[1,2,3])
    cs_per_sample   = cs_map_tensor.mean(dim=[1,2,3])
    return ssim_per_sample, cs_per_sample


def ms_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    weights: Tuple[float,...] = (0.3, 0.3, 0.4),
    levels: int = 3
) -> torch.Tensor:
    """
    Multi-scale SSIM.
    Args:
      img1, img2:   Float tensors in [N, C, H, W], assumed normalized to [−1,1] or [0,1].
      window:       Gaussian filter of shape [C,1,ks,ks] (create once via your gaussian_window).
      weights:      Importance of each scale (sum == 1).
      levels:       Number of scales (len(weights) should be >= levels).
    Returns:
      scalar MS-SSIM averaged over the batch.
    """
    mssim_vals = []
    mcs_vals   = []

    x1, x2 = img1, img2
    for l in range(levels):
        ssim_val, cs_val = ssim_map(x1, x2, window)
        mssim_vals.append(ssim_val)
        mcs_vals.append(cs_val)

        # downsample for next scale
        x1 = F.avg_pool2d(x1, kernel_size=2)
        x2 = F.avg_pool2d(x2, kernel_size=2)

    # combine: ∏_{i=1..L−1} (CS_i ^ weight_i) × (SSIM_L ^ weight_L)
    # in log-domain for stability
    log_mcs = torch.stack([w * torch.log(mcs_vals[i] + 1e-6)
                           for i, w in enumerate(weights[:-1])], dim=0).sum(dim=0)
    log_mssim = weights[-1] * torch.log(mssim_vals[-1] + 1e-6)
    ms_ssim_per_sample = torch.exp(log_mcs + log_mssim)

    return ms_ssim_per_sample.mean()

# SAM Loss
class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()
        self.eps = 1e-8

    def forward(self, pred, target):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        cos_sim = nn.functional.cosine_similarity(pred_flat, target_flat, dim=1, eps=self.eps)
        sam_loss = torch.acos(torch.clamp(cos_sim, -1 + self.eps, 1 - self.eps)).mean()
        return sam_loss

def histogram_loss(fake, real, bins=64):
    loss = 0.0
    for c in range(real.shape[1]):
        real_hist = torch.histc(real[:, c].flatten(), bins=bins, min=0.0, max=1.0)
        fake_hist = torch.histc(fake[:, c].flatten(), bins=bins, min=0.0, max=1.0)
        real_hist = real_hist / (real_hist.sum() + 1e-8)
        fake_hist = fake_hist / (fake_hist.sum() + 1e-8)
        loss += torch.sum((real_hist - fake_hist) ** 2)  # MSE yerine KL de olur
    return loss / real.shape[1]

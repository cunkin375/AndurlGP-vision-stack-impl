import torch
import torch.nn.functional as F
import numpy as np

# Cached coordinate grids — built once per (H, W) per worker process
_GRID_CACHE: dict = {}


def _grid(h: int, w: int):
    if (h, w) not in _GRID_CACHE:
        ys = torch.arange(h, dtype=torch.float32)
        xs = torch.arange(w, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        _GRID_CACHE[(h, w)] = (gx, gy)
    return _GRID_CACHE[(h, w)]


def build_single_heatmap(corners: np.ndarray, h: int, w: int, sigma: float = 4.0) -> torch.Tensor:
    target = torch.zeros(4, h, w)
    if len(corners) == 0:
        return target
    gx, gy = _grid(h, w)
    for row in corners:
        c_type = int(row[0])
        px = float(row[1]) * w
        py = float(row[2]) * h
        gauss = torch.exp(-((gx - px) ** 2 + (gy - py) ** 2) / (2 * sigma ** 2))
        target[c_type] = torch.max(target[c_type], gauss)
    return target


def build_heatmap_targets(corners_list, heatmap_h: int, heatmap_w: int, sigma: float = 4.0) -> torch.Tensor:
    B = len(corners_list)
    targets = torch.zeros(B, 4, heatmap_h, heatmap_w)

    ys = torch.arange(heatmap_h, dtype=torch.float32)
    xs = torch.arange(heatmap_w, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

    for b, corners in enumerate(corners_list):
        if corners is None or len(corners) == 0:
            continue
        corners = torch.as_tensor(corners, dtype=torch.float32)
        for row in corners:
            c_type = int(row[0].item())
            px = row[1].item() * heatmap_w
            py = row[2].item() * heatmap_h
            gaussian = torch.exp(
                -((grid_x - px) ** 2 + (grid_y - py) ** 2) / (2 * sigma ** 2)
            )
            # Multiple corners of the same type (e.g. two gates) take the max
            targets[b, c_type] = torch.max(targets[b, c_type], gaussian)

    return targets


def heatmap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def decode_keypoints_multi(heatmaps: torch.Tensor, peak_thresh: float = 0.5, nms_radius: int = 8) -> list:
    B, C, H, W = heatmaps.shape
    kernel = 2 * nms_radius + 1
    pooled  = F.max_pool2d(heatmaps, kernel_size=kernel, stride=1, padding=nms_radius)
    is_peak = (heatmaps == pooled) & (heatmaps >= peak_thresh)

    results = []
    for b in range(B):
        detections = {}
        for c in range(C):
            mask = is_peak[b, c]
            if not mask.any():
                continue
            ys, xs = mask.nonzero(as_tuple=True)
            scores  = heatmaps[b, c][ys, xs]
            order   = scores.argsort(descending=True)
            detections[c] = [
                (xs[i].item() / W, ys[i].item() / H, scores[i].item())
                for i in order
            ]
        results.append(detections)
    return results


def decode_keypoints(heatmaps: torch.Tensor, peak_thresh: float = 0.5) -> list:
    B, C, H, W = heatmaps.shape
    results = []
    for b in range(B):
        detections = {}
        for c in range(C):
            score, flat_idx = heatmaps[b, c].view(-1).max(0)
            score = score.item()
            if score >= peak_thresh:
                py = (flat_idx // W).item()
                px = (flat_idx % W).item()
                detections[c] = (px / W, py / H, score)
        results.append(detections)
    return results

"""
Test-Time Augmentation (TTA) for improved inference
"""

import torch
import torch.nn.functional as F
from torchvision import transforms


class TestTimeAugmentation:
    """Apply multiple augmentations at test time and average predictions"""
    def __init__(self, model, num_augments=8):
        self.model = model
        self.num_augments = num_augments

    def _get_augmentations(self):
        """Generate different augmentation transforms"""
        augs = [
            lambda x: x,  # Original
            lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
            lambda x: torch.flip(x, dims=[2]),  # Vertical flip
            lambda x: torch.flip(torch.flip(x, dims=[2]), dims=[3]),  # Both flips
            lambda x: self._rotate(x, 5),  # Rotate +5°
            lambda x: self._rotate(x, -5),  # Rotate -5°
            lambda x: self._zoom(x, 1.05),  # Zoom in 5%
            lambda x: self._zoom(x, 0.95),  # Zoom out 5%
        ]
        return augs[:self.num_augments]

    def _rotate(self, x, angle):
        """Rotate image by angle degrees"""
        rad = angle * 3.14159 / 180
        theta = torch.tensor([
            [torch.cos(rad), -torch.sin(rad), 0],
            [torch.sin(rad), torch.cos(rad), 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def _zoom(self, x, scale):
        """Zoom image by scale factor"""
        theta = torch.tensor([
            [scale, 0, 0],
            [0, scale, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    @torch.no_grad()
    def predict(self, x):
        """Run TTA and return averaged predictions"""
        self.model.eval()
        augmentations = self._get_augmentations()

        all_preds = []
        for aug_fn in augmentations:
            aug_x = aug_fn(x)
            pred = self.model(aug_x)
            all_preds.append(F.softmax(pred, dim=1))

        # Average all predictions
        avg_pred = torch.stack(all_preds).mean(dim=0)
        return avg_pred

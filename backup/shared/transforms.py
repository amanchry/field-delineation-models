import random
import torch
import torch.nn.functional as F


class UNetTransform:
    """Random horizontal/vertical flips and 90° rotations for {"image", "mask"} dicts."""

    def __call__(self, sample: dict) -> dict:
        img, mask = sample["image"], sample["mask"]
        if random.random() > 0.5:
            img  = torch.flip(img,  dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if random.random() > 0.5:
            img  = torch.flip(img,  dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        k = random.randint(0, 3)
        if k:
            img  = torch.rot90(img,  k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        return {"image": img, "mask": mask}


class Sam1Transform:
    """
    Prepares SAM1 training samples from {"image", "mask"}.

    SAM1 is single-frame. Pads image to max_image_size square, samples nsel
    random field instances, and returns point prompts + binary GT masks.

    Returns:
        image    : [C, max_image_size, max_image_size]  float32
        gt_masks : [nsel, H_orig, W_orig]               float32 binary
        points   : [nsel, 1, 2]                         int64  (x, y)
        labels   : [nsel, 1]                            int64  (all 1)
    """

    def __init__(self, nsel: int = 3, max_image_size: int = 1024):
        self.nsel     = nsel
        self.max_size = max_image_size

    def _pad_to_square(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape
        size = max(H, W)
        scale = self.max_size / size
        new_h, new_w = int(H * scale), int(W * scale)
        img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        img = F.pad(img, (0, self.max_size - new_w, 0, self.max_size - new_h))
        return img

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        mask  = sample["mask"]

        instance_ids = mask.unique()
        instance_ids = instance_ids[instance_ids > 0].tolist() or [0]
        chosen = [random.choice(instance_ids) for _ in range(self.nsel)]

        gt_masks, points = [], []
        for iid in chosen:
            bin_mask = (mask == iid).float()
            gt_masks.append(bin_mask)
            ys, xs = torch.where(bin_mask > 0)
            if len(ys) == 0:
                cx, cy = mask.shape[1] // 2, mask.shape[0] // 2
            else:
                cx = int(xs.float().mean().item())
                cy = int(ys.float().mean().item())
            points.append([[cx, cy]])

        return {
            "image":    self._pad_to_square(image),
            "gt_masks": torch.stack(gt_masks),
            "points":   torch.tensor(points, dtype=torch.int64),
            "labels":   torch.ones(self.nsel, 1, dtype=torch.int64),
        }


class Sam2Transform:
    """
    Prepares SAM2 training samples from {"window_a", "window_b", "mask"}.

    Resizes both windows to max_image_size (padded square), samples nsel
    random field instances, and returns point prompts + binary GT masks.

    Returns:
        img_a    : [C, max_image_size, max_image_size]  float32
        img_b    : [C, max_image_size, max_image_size]  float32
        gt_masks : [nsel, H_orig, W_orig]               float32 binary
        points   : [nsel, 1, 2]                         int64  (x, y)
        labels   : [nsel, 1]                            int64  (all 1)
    """

    def __init__(self, nsel: int = 3, max_image_size: int = 1024):
        self.nsel     = nsel
        self.max_size = max_image_size

    def _pad_to_square(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape
        size = max(H, W)
        scale = self.max_size / size
        new_h, new_w = int(H * scale), int(W * scale)
        img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        img = F.pad(img, (0, self.max_size - new_w, 0, self.max_size - new_h))
        return img

    def __call__(self, sample: dict) -> dict:
        img_a = sample["window_a"]
        img_b = sample["window_b"]
        mask  = sample["mask"]

        instance_ids = mask.unique()
        instance_ids = instance_ids[instance_ids > 0].tolist() or [0]
        chosen = [random.choice(instance_ids) for _ in range(self.nsel)]

        gt_masks, points = [], []
        for iid in chosen:
            bin_mask = (mask == iid).float()
            gt_masks.append(bin_mask)
            ys, xs = torch.where(bin_mask > 0)
            if len(ys) == 0:
                cx, cy = mask.shape[1] // 2, mask.shape[0] // 2
            else:
                cx = int(xs.float().mean().item())
                cy = int(ys.float().mean().item())
            points.append([[cx, cy]])

        return {
            "img_a":    self._pad_to_square(img_a),
            "img_b":    self._pad_to_square(img_b),
            "gt_masks": torch.stack(gt_masks),
            "points":   torch.tensor(points, dtype=torch.int64),
            "labels":   torch.ones(self.nsel, 1, dtype=torch.int64),
        }

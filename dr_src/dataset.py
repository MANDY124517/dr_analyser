# dr_src/dataset.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class AptosDataset(Dataset):
    """
    Generic APTOS dataset reader.
    - CSV must contain an image identifier column: id_code or image_id (falls back to first column).
    - Optional 'diagnosis' column; if missing, labels are set to 0 for inference.
    - Image files may be .png/.jpg/.jpeg (case-insensitive). The dataset will try common extensions.
    - Augmentations: pass an Albumentations transform via aug.
    """
    def __init__(self, df: pd.DataFrame, img_dir, aug=None, id_column=None, label_column="diagnosis"):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = Path(img_dir)
        self.aug = aug

        # Resolve id column
        if id_column is not None:
            self.id_col = id_column
        elif "id_code" in self.df.columns:
            self.id_col = "id_code"
        elif "image_id" in self.df.columns:
            self.id_col = "image_id"
        else:
            self.id_col = self.df.columns[0]

        # Ensure label column exists for training compatibility
        if label_column not in self.df.columns:
            self.df[label_column] = 0
        self.label_col = label_column

        # Normalize id column to string
        self.df[self.id_col] = self.df[self.id_col].astype(str)

        # Precompute lowercase filename candidates to speed up __getitem__
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, rid: str) -> Path:
        # Direct match if id already includes an extension
        p = self.img_dir / rid
        if p.suffix:
            return p if p.exists() else None

        # Try common extensions
        exts = [".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"]
        for ext in exts:
            q = self.img_dir / f"{rid}{ext}"
            if q.exists():
                return q
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rid = str(row[self.id_col])
        label = int(row[self.label_col]) if self.label_col in self.df.columns else 0

        # Cached path lookup
        p = self._cache.get(rid)
        if p is None:
            p = self._find_image_path(rid)
            if p is None:
                raise FileNotFoundError(f"Image not found for id='{rid}' under {self.img_dir}")
            self._cache[rid] = p

        im_bgr = cv2.imread(str(p))
        if im_bgr is None:
            raise IOError(f"Failed to read image file: {p}")
        im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        if self.aug is not None:
            out = self.aug(image=im)
            im = out["image"]

        # Convert HWC uint8 or float to CHW float32 tensor in [0,1] if not already
        if isinstance(im, np.ndarray):
            if im.dtype != np.float32:
                im = im.astype(np.float32) / 255.0
            im = np.transpose(im, (0, 1, 2)) if im.ndim == 3 and im.shape[0] in (1,3) and im.shape[-1] not in (1,3) else im
            if im.ndim == 3 and im.shape[-1] in (1,3):  # HWC -> CHW
                im = np.transpose(im, (2, 0, 1))
            tensor = torch.from_numpy(im)
        else:
            tensor = im  # assume transform already produced a tensor

        return tensor, label

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_aug(size):
    return A.Compose([
        A.Resize(size,size),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8,8), p=0.7),
        A.GaussianBlur(blur_limit=(3,5), p=0.3),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2,0.2,p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def get_val_aug(size):
    return A.Compose([
        A.Resize(size,size),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8,8), p=1.0),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

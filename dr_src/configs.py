from pathlib import Path

class CFG:
    # Set this to your exact root shown in structure.txt
    PROJECT_ROOT = Path(r"D:\lenovo\MY PROJECT")
    DATA_ROOT = PROJECT_ROOT / "data"
    APTOS_DIR = DATA_ROOT / "aptos2019-blindness-detection"
    APTOS_CSV = APTOS_DIR / "train.csv"
    APTOS_IMGS = APTOS_DIR / "train_images"

    NUM_CLASSES = 5
    IMG_SIZE = 384

    EPOCHS = 20
    BATCH_SIZE = 2           # safe for 4 GB
    VAL_BATCH_SIZE = 2
    ACCUM_STEPS = 4          # effective global batch 8
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 0          # Jupyter/Windows stability
    PIN_MEMORY = True
    AMP = True
    GRAD_CLIP = 1.0
    ONECYCLE = True
    SEED = 42

    CKPT_DIR = PROJECT_ROOT / "ckpt"
    RUNS_DIR = PROJECT_ROOT / "runs"

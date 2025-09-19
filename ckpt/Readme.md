
Put the pretrained model weights for inference in this folder using the structure below.

ckpt/
├─ rsg_res34/
│  └─ best.pt
├─ effnet_b3/
│  └─ best.pt
├─ vit_small/
│  └─ best.pt
└─ fusion_meta_lr_calib2.npz

Files required
- rsg_res34/best.pt          # ResNet-34 based model checkpoint
- effnet_b3/best.pt          # EfficientNet-B3 model checkpoint
- vit_small/best.pt          # ViT-Small model checkpoint
- fusion_meta_lr_calib2.npz  # Calibrated logistic-fusion weights (coefficients, intercepts, temps)

Notes
- Do NOT commit these large files to Git. Keep them local, or distribute via a release link or shared drive.
- If a different fusion meta is preferred, update the path in dr_src/configs.py (CFG.FUSION_META) or via the Streamlit sidebar.
- File sizes: each .pt is typically hundreds of MB; ensure enough disk space.
- If running on CPU-only machines, weights remain the same; inference will be slower but functional.

Verification checklist
1) After placing files, your tree should look like:
   - ckpt/rsg_res34/best.pt
   - ckpt/effnet_b3/best.pt
   - ckpt/vit_small/best.pt
   - ckpt/fusion_meta_lr_calib2.npz
2) Start the app:
   streamlit run app.py
3) The sidebar/system status should show "Models: Ready" and predictions should work on a sample image.

Troubleshooting
- FileNotFoundError: Ensure the exact paths and filenames match the structure above.
- RuntimeError: unexpected EOF while loading state_dict: The checkpoint file is incomplete; re-download the .pt file.
- ModuleNotFoundError (timm, albumentations, etc.): Install dependencies from requirements.txt.

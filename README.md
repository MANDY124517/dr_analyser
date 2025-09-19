# 👁️ DR AI Analyzer

Advanced AI-powered Diabetic Retinopathy detection and grading with a clean, responsive Streamlit interface. Upload a retinal fundus image to get an instant 0–4 grade prediction with calibrated confidence and clear clinical cues.[^1]

## Why this matters

Diabetic retinopathy is a leading cause of preventable blindness. Early detection and triage can significantly reduce vision loss by guiding timely referrals. This project delivers a practical, easy-to-use tool that brings strong computer-vision models to clinicians and researchers.

## Features

- Fast image upload and one-click inference via Streamlit UI.
- 0–4 grade prediction: No DR, Mild, Moderate, Severe, Proliferative.
- Clear numerical indicator and color-coded confidence bars.
- Ensemble of CNNs and ViT combined with calibrated logistic fusion for robust results.
- Works on CPU or GPU; identical predictions to the batch CLI.


## Demo

- Launch locally and open the browser UI to upload a PNG/JPG fundus image and inspect the predicted grade and probabilities.


## Project structure

- app.py — Streamlit app with responsive UI and inference logic.
- dr_src/ — Core library
    - configs.py — global CFG, relative paths, and runtime flags.
    - transforms.py — validation transforms and preprocessing.
    - dataset.py — robust image loader for PNG/JPG.
    - models/ — model definitions for ResNet-34, EfficientNet-B3, ViT-Small.
- ckpt/ — model weights directory (see below).
- requirements.txt — runtime dependencies.


## Installation

1) Create a Python environment and install dependencies:

- pip install -r requirements.txt

2) Place pretrained weights in ckpt/ as described below.
3) Run the app:

- streamlit run app.py


## Checkpoints

Create the following structure and place weights locally. Large files are not committed to Git.

- ckpt/
    - rsg_res34/best.pt
    - effnet_b3/best.pt
    - vit_small/best.pt
    - fusion_meta_lr_calib2.npz

Tip: If using a different fusion meta, update the path in dr_src/configs.py or via the app sidebar.

## How it works

- Three backbones predict class logits; optional flip TTA averages two views.
- Temperatures rescale each model’s logits for better probability calibration.
- A logistic-regression fusion combines all logits into final calibrated probabilities.
- The top class determines the grade; the UI displays the numeric label (0–4), text label, and confidence.


## Usage tips

- Use high-quality, centered fundus images for best results.
- Confidence < 0.7 indicates uncertainty; consider manual review.
- CPU inference works but is slower; a CUDA GPU accelerates predictions.


## Troubleshooting

- ModuleNotFoundError: Ensure dependencies are installed from requirements.txt.
- Checkpoint not found: Verify the ckpt/ paths and filenames exactly match the structure above.
- Streamlit/protobuf error: Pin protobuf to 3.20.x if needed.


## Responsible AI

- This tool is intended for research and clinical decision support, not as a sole diagnostic. Always verify with a qualified ophthalmologist.


## Acknowledgments

- Built with PyTorch, timm, Albumentations, and Streamlit. Model architectures include ResNet-34, EfficientNet-B3, and ViT-Small with a calibrated fusion head.


## License

- GNU GENERAL PUBLIC LICENSE

Contributions and feedback are welcome—issues and PRs help improve model robustness and usability.




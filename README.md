# 👁️ DR AI Analyzer

Advanced AI-powered Diabetic Retinopathy detection and grading with a clean, responsive Streamlit interface. Upload a retinal fundus image to get an instant 0–4 grade prediction with calibrated confidence and clear clinical cues.[^1]

## Why this matters

Diabetic retinopathy is a leading cause of preventable blindness. Early detection and triage can significantly reduce vision loss by guiding timely referrals. This project delivers a practical, easy-to-use tool that brings strong computer-vision models to clinicians and researchers.[^1]

## Features

- Fast image upload and one-click inference via Streamlit UI.[^1]
- 0–4 grade prediction: No DR, Mild, Moderate, Severe, Proliferative.[^1]
- Clear numerical indicator and color-coded confidence bars.[^1]
- Ensemble of CNNs and ViT combined with calibrated logistic fusion for robust results.[^1]
- Works on CPU or GPU; identical predictions to the batch CLI.[^1]


## Demo

- Launch locally and open the browser UI to upload a PNG/JPG fundus image and inspect the predicted grade and probabilities.[^1]


## Project structure

- app.py — Streamlit app with responsive UI and inference logic.[^1]
- dr_src/ — Core library
    - configs.py — global CFG, relative paths, and runtime flags.[^1]
    - transforms.py — validation transforms and preprocessing.[^1]
    - dataset.py — robust image loader for PNG/JPG.[^1]
    - models/ — model definitions for ResNet-34, EfficientNet-B3, ViT-Small.[^1]
- ckpt/ — model weights directory (see below).[^1]
- requirements.txt — runtime dependencies.[^1]


## Installation

1) Create a Python environment and install dependencies:

- pip install -r requirements.txt[^1]

2) Place pretrained weights in ckpt/ as described below.[^1]
3) Run the app:

- streamlit run app.py[^1]


## Checkpoints

Create the following structure and place weights locally. Large files are not committed to Git.[^1]

- ckpt/
    - rsg_res34/best.pt
    - effnet_b3/best.pt
    - vit_small/best.pt
    - fusion_meta_lr_calib2.npz[^1]

Tip: If using a different fusion meta, update the path in dr_src/configs.py or via the app sidebar.[^1]

## How it works

- Three backbones predict class logits; optional flip TTA averages two views.[^1]
- Temperatures rescale each model’s logits for better probability calibration.[^1]
- A logistic-regression fusion combines all logits into final calibrated probabilities.[^1]
- The top class determines the grade; the UI displays the numeric label (0–4), text label, and confidence.[^1]


## Usage tips

- Use high-quality, centered fundus images for best results.[^1]
- Confidence < 0.7 indicates uncertainty; consider manual review.[^1]
- CPU inference works but is slower; a CUDA GPU accelerates predictions.[^1]


## Troubleshooting

- ModuleNotFoundError: Ensure dependencies are installed from requirements.txt.[^1]
- Checkpoint not found: Verify the ckpt/ paths and filenames exactly match the structure above.[^1]
- Streamlit/protobuf error: Pin protobuf to 3.20.x if needed.[^1]


## Responsible AI

- This tool is intended for research and clinical decision support, not as a sole diagnostic. Always verify with a qualified ophthalmologist.[^1]


## Acknowledgments

- Built with PyTorch, timm, Albumentations, and Streamlit. Model architectures include ResNet-34, EfficientNet-B3, and ViT-Small with a calibrated fusion head.[^1]


## License

- Add a suitable OSI license (e.g., MIT) to enable clear reuse.[^1]

Contributions and feedback are welcome—issues and PRs help improve model robustness and usability.[^1]




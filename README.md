# COEN816 Digital Image Processing Term Project

**Name:** Mustapha Mohammed Yusuf  
**Reg No:** P23EGCP8034  
**Course:** COEN816 Digital Image Processing  

## Project Structure
COEN816_P23EGCP8034_MustaphaMohammedYusuf.zip
│
├── code/
│   ├── run_all.py
│   ├── dataset_generation.py
│   ├── analysis.py
│   ├── filters.py
│   ├── restoration.py
│   ├── segmentation.py
│   ├── integration.py
│   └── utils.py
│
├── dataset/
│   └── base/
│       ├── img1.jpg
│       ├── img2.jpg
│       ├── img3.jpg
│       ├── img4.jpg
│       ├── img5.jpg
│       └── img6.jpg
│
├── outputs/
│   ├── corrupted/
│   ├── intermediate/
│   └── final/
│
├── report/
│   └── project_report.pdf
│
├── hand_calculations/
│   └── handwritten_derivations.pdf
│
└── logs/
    └── experiments.csv

## Project Summary
This repository contains the full implementation of an integrated digital image processing pipeline:
- Dataset generation using seed-based corruption
- Exploratory analysis
- Histogram enhancement (manual equalization)
- Spatial filtering (Gaussian + Median)
- Frequency filtering (Notch)
- Restoration (Wiener + Blind)
- Segmentation and morphology
- Robustness experiments (+/-20% noise perturbation)

## How to Run
Install dependencies:
```bash
pip install numpy opencv-python matplotlib scipy
```
## Run full pipeline:
```bash
python run_all.py
```

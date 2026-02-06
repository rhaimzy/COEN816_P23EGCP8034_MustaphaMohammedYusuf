# COEN816 Digital Image Processing Term Project

**Name:** Mustapha Mohammed Yusuf  
**Reg No:** P23EGCP8034  
**Course:** COEN816 Digital Image Processing  


## Project Structure

```
COEN816_P23EGCP8034_MustaphaMohammedYusuf/
│
├── code/
│ ├── run_all.py
│ ├── dataset_generation.py
│ ├── analysis.py
│ ├── filters.py
│ ├── restoration.py
│ ├── segmentation.py
│ ├── integration.py
│ └── utils.py
│
├── dataset/
│ └── base/
│ ├── img1.jpg
│ ├── img2.jpg
│ ├── img3.jpg
│ ├── img4.jpg
│ ├── img5.jpg
│ └── img6.jpg
│
├── outputs/
│ ├── corrupted/
│ ├── intermediate/
│ └── final/
│
├── report/
│ └── project_report.pdf
│
├── hand_calculations/
│ └── handwritten_derivations.pdf
│
└── logs/
└── experiments.csv

```

## Project Summary

This repository contains the full implementation of an integrated digital image processing pipeline:

- Dataset generation using seed-based corruption
- Exploratory analysis (histogram, mean/variance, DFT magnitude)
- Histogram enhancement (manual histogram equalization + adaptive enhancement)
- Spatial filtering (Gaussian smoothing + median filtering)
- Frequency-domain filtering (adaptive notch filter)
- Restoration (Wiener filter + blind restoration)
- Segmentation and morphology
- Robustness experiments (+/-20% noise perturbation)

---

## Requirements

Install dependencies:

```bash
pip install numpy opencv-python matplotlib scipy

cd code
python run_all.py

"Machine Learning Pipeline Development" 
# Nuclear Scaling ML Pipeline

A modular microscopy analysis pipeline for **nuclear segmentation, ROI extraction, and quantitative analysis** using machine learning and image processing.

This project is designed for **large-scale microscopy datasets**, including multi-channel, multi-Z, and time-lapse imaging, with compatibility for HPC environments.

---

## 🧬 Project Overview

This pipeline performs:

1. **Image Handling**
   - ND2 → TIFF conversion
   - Hyperstack concatenation
   - Multi-dimensional image support (C, Z, T)

2. **Segmentation (U-Net)**
   - Nuclear classification
   - Probability map generation
   - Binary mask output

3. **ROI Extraction**
   - Identification of individual nuclei
   - Filtering based on size, circularity, and proximity

4. **Quantification**
   - Nuclear area (µm²)
   - N/C ratio calculations
   - Time-resolved measurements

---

## 🧠 Design Philosophy

- **Modular**: Each step is isolated (IO, segmentation, ROI, measurements)
- **Reproducible**: Config-driven workflows
- **Scalable**: Designed for HPC + large datasets
- **Debuggable**: Notebook-friendly but not notebook-dependent

---

## 📁 Repository Structure

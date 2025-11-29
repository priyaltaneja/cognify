---
title: SHIA - Brain MRI Segmentation
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SHIA - Structured Health Intelligence for Alzheimer's

Fast GPU-accelerated brain MRI segmentation API.

## Setup Instructions

### 1. Convert the model (run locally first)

```bash
# Install conversion dependencies
pip install tensorflowjs tensorflow

# Convert tfjs model to H5 format
cd huggingface
python convert_model.py ../public/models/model18cls ./model18cls
```

### 2. Upload to Hugging Face Space

The `model18cls/` folder should now contain `model.h5` and `saved_model/`.

## API Endpoints

### POST /segment/tensor (Recommended)
Upload pre-processed tensor data (256³ uint8, gzipped) for GPU inference.
This is the recommended endpoint as it ensures identical preprocessing to local inference.

```bash
# Frontend sends the conformed tensor from NiiVue directly
# See atrophy/main.js for implementation
```

### POST /segment
Upload a NIfTI file (.nii or .nii.gz) for segmentation.
Note: Server-side preprocessing may differ slightly from local NiiVue preprocessing.

```bash
curl -X POST -F "file=@brain.nii.gz" https://YOUR-SPACE.hf.space/segment
```

### POST /segment/compact
Same as /segment but returns base64-gzipped results.

### GET /health
Check API status and GPU availability.

## Credits

Based on [BrainChop](https://github.com/neuroneural/brainchop) by the Neuroneural Lab.

import os
import io
import time
import json
import numpy as np
import tensorflow as tf
import nibabel as nib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gzip

app = FastAPI(title="SHIA - Brain MRI Segmentation API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model = None
MODEL_PATH = "model18cls"

def load_model():
    """
    Load TensorFlow model on startup.
    Supports H5, SavedModel, or Keras formats.

    NOTE: Convert tfjs models first using convert_model.py
    """
    global model
    if model is None:
        print(f"Loading model from {MODEL_PATH}...")

        # Check for different model formats
        h5_path = os.path.join(MODEL_PATH, "model.h5")
        keras_path = os.path.join(MODEL_PATH, "model.keras")
        saved_model_dir = os.path.join(MODEL_PATH, "saved_model")

        if os.path.exists(h5_path):
            print("Loading H5 format...")
            model = tf.keras.models.load_model(h5_path)
        elif os.path.exists(keras_path):
            print("Loading Keras format...")
            model = tf.keras.models.load_model(keras_path)
        elif os.path.exists(saved_model_dir):
            print("Loading SavedModel format...")
            model = tf.keras.models.load_model(saved_model_dir)
        elif os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH):
            # Try loading directory as SavedModel
            print("Loading as SavedModel directory...")
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(
                f"No model found in {MODEL_PATH}. "
                "Please convert the tfjs model first using: "
                "python convert_model.py ../public/models/model18cls ./model18cls"
            )

        print("Model loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    return model

def parse_nifti(file_bytes: bytes, filename: str = "temp.nii"):
    """Parse NIfTI file from bytes"""
    import tempfile

    # Determine file extension for nibabel
    is_gzipped = file_bytes[:2] == b'\x1f\x8b' or filename.endswith('.gz')
    suffix = '.nii.gz' if is_gzipped else '.nii'

    # Write to temp file and load with nibabel
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        img = nib.load(tmp_path)
        data = img.get_fdata()
        header = img.header
    finally:
        # Clean up temp file
        import os
        os.unlink(tmp_path)

    return data, header

def min_max_normalize(data):
    """Normalize data to 0-1 range"""
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min == 0:
        return data
    return (data - data_min) / (data_max - data_min)

def preprocess_volume(data):
    """
    Preprocess MRI volume for model input.
    Returns preprocessed data and info needed to unpad output.
    """
    original_shape = data.shape

    # Normalize
    data = min_max_normalize(data)

    # Ensure float32
    data = data.astype(np.float32)

    # Transpose to match model expectations
    data = np.transpose(data, (2, 1, 0))
    transposed_shape = data.shape

    # Pad to 256x256x256 if needed (model requires fixed input size)
    target_shape = (256, 256, 256)
    current_shape = data.shape
    pad_info = None

    if current_shape != target_shape:
        print(f"Padding volume from {current_shape} to {target_shape}")
        padded = np.zeros(target_shape, dtype=np.float32)

        # Calculate padding offsets (center the volume)
        offsets = [(t - c) // 2 for t, c in zip(target_shape, current_shape)]

        # Handle cases where input is larger than target (crop instead)
        slices_src = []
        slices_dst = []
        for i in range(3):
            if current_shape[i] <= target_shape[i]:
                # Pad: source is full, destination is offset
                slices_src.append(slice(0, current_shape[i]))
                slices_dst.append(slice(offsets[i], offsets[i] + current_shape[i]))
            else:
                # Crop: source is cropped, destination is full
                start = (current_shape[i] - target_shape[i]) // 2
                slices_src.append(slice(start, start + target_shape[i]))
                slices_dst.append(slice(0, target_shape[i]))

        padded[slices_dst[0], slices_dst[1], slices_dst[2]] = data[slices_src[0], slices_src[1], slices_src[2]]
        data = padded
        pad_info = {
            'original_transposed_shape': transposed_shape,
            'slices_dst': slices_dst
        }

    # Add batch and channel dimensions
    data = np.expand_dims(data, axis=0)  # batch
    data = np.expand_dims(data, axis=-1)  # channel

    return data, pad_info, original_shape


def postprocess_segmentation(segmentation, pad_info, original_shape):
    """
    Remove padding from segmentation output and transpose back to original orientation.
    """
    # If we padded, extract the original region
    if pad_info is not None:
        slices = pad_info['slices_dst']
        segmentation = segmentation[slices[0], slices[1], slices[2]]

    # Transpose back to original orientation
    segmentation = np.transpose(segmentation, (2, 1, 0))

    return segmentation

def run_inference(data):
    """Run model inference on preprocessed data"""
    loaded_model = load_model()

    # Run prediction
    prediction = loaded_model.predict(data, verbose=0)

    # Get argmax for segmentation labels
    segmentation = np.argmax(prediction, axis=-1)

    # Remove batch dimension and transpose back
    segmentation = segmentation[0]
    segmentation = np.transpose(segmentation, (2, 1, 0))

    return segmentation

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "SHIA - Brain MRI Segmentation",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "gpu": tf.config.list_physical_devices('GPU')}

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Segment a brain MRI scan.

    Upload a NIfTI file (.nii or .nii.gz) and receive segmentation results.
    """
    try:
        start_time = time.time()

        # Validate file type
        if not file.filename.endswith(('.nii', '.nii.gz')):
            raise HTTPException(400, "File must be a NIfTI file (.nii or .nii.gz)")

        # Read file
        print(f"Processing: {file.filename}")
        file_bytes = await file.read()

        # Parse NIfTI
        parse_start = time.time()
        data, header = parse_nifti(file_bytes, file.filename)
        parse_time = time.time() - parse_start
        print(f"Volume shape: {data.shape}, Parse time: {parse_time:.2f}s")

        # Preprocess
        preprocess_start = time.time()
        processed, pad_info, original_shape = preprocess_volume(data)
        preprocess_time = time.time() - preprocess_start
        print(f"Preprocessed shape: {processed.shape}, Time: {preprocess_time:.2f}s")

        # Run inference
        inference_start = time.time()
        segmentation = run_inference(processed)
        segmentation = postprocess_segmentation(segmentation, pad_info, original_shape)
        inference_time = time.time() - inference_start
        print(f"Inference time: {inference_time:.2f}s")

        total_time = time.time() - start_time

        # Get unique labels found
        unique_labels = np.unique(segmentation).tolist()

        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "original_shape": list(data.shape),
            "segmentation_shape": list(segmentation.shape),
            "unique_labels": unique_labels,
            "num_labels": len(unique_labels),
            "timing": {
                "parse": round(parse_time, 3),
                "preprocess": round(preprocess_time, 3),
                "inference": round(inference_time, 3),
                "total": round(total_time, 3)
            },
            # Return segmentation as nested list (can be large!)
            "segmentation": segmentation.astype(np.uint8).tolist()
        })

    except Exception as e:
        import traceback
        print(f"ERROR in /segment: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, f"Segmentation failed: {str(e)}")

@app.post("/segment/compact")
async def segment_compact(file: UploadFile = File(...)):
    """
    Segment a brain MRI scan and return compressed results.

    Returns base64-encoded gzipped segmentation for efficiency.
    """
    import base64

    try:
        start_time = time.time()

        if not file.filename.endswith(('.nii', '.nii.gz')):
            raise HTTPException(400, "File must be a NIfTI file (.nii or .nii.gz)")

        file_bytes = await file.read()
        print(f"Processing: {file.filename}, size: {len(file_bytes)} bytes")

        data, header = parse_nifti(file_bytes, file.filename)
        print(f"Parsed volume shape: {data.shape}")

        processed, pad_info, original_shape = preprocess_volume(data)
        print(f"Preprocessed shape: {processed.shape}")

        segmentation = run_inference(processed)
        print(f"Raw segmentation shape: {segmentation.shape}")

        segmentation = postprocess_segmentation(segmentation, pad_info, original_shape)
        print(f"Final segmentation shape: {segmentation.shape}")

        total_time = time.time() - start_time

        # Compress segmentation
        seg_bytes = segmentation.astype(np.uint8).tobytes()
        compressed = gzip.compress(seg_bytes)
        encoded = base64.b64encode(compressed).decode('utf-8')

        return JSONResponse({
            "success": True,
            "shape": list(segmentation.shape),
            "dtype": "uint8",
            "encoding": "base64_gzip",
            "inference_time": round(total_time, 3),
            "data": encoded
        })

    except Exception as e:
        import traceback
        print(f"ERROR in /segment/compact: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, f"Segmentation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

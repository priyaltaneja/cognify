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
    """Parse NIfTI file from bytes and reorient to canonical (RAS+) orientation"""
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

        # Reorient to canonical RAS+ orientation (like NiiVue does)
        # This ensures consistent orientation regardless of how the file was saved
        img_canonical = nib.as_closest_canonical(img)
        data = img_canonical.get_fdata()
        header = img_canonical.header

        print(f"Original orientation: {nib.aff2axcodes(img.affine)}")
        print(f"Canonical orientation: {nib.aff2axcodes(img_canonical.affine)}")

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


def conform_volume(data, header, target_shape=(256, 256, 256), target_voxel_size=1.0):
    """
    Conform MRI volume to standard dimensions (like FreeSurfer's mri_convert --conform).
    Resamples to 1mm isotropic voxels and 256^3 dimensions.
    """
    from scipy.ndimage import zoom

    # Get current voxel sizes from header
    try:
        voxel_sizes = header.get_zooms()[:3]
    except:
        voxel_sizes = (1.0, 1.0, 1.0)

    print(f"Original voxel sizes: {voxel_sizes}")
    print(f"Original shape: {data.shape}")

    # Calculate zoom factors to get to target voxel size, then to target shape
    # Step 1: Resample to target voxel size (1mm isotropic)
    zoom_to_1mm = [vs / target_voxel_size for vs in voxel_sizes]

    # Resample to 1mm isotropic
    data_1mm = zoom(data, zoom_to_1mm, order=1)  # order=1 = linear interpolation
    print(f"After 1mm resample shape: {data_1mm.shape}")

    # Step 2: Pad or crop to target shape (256^3)
    current_shape = data_1mm.shape
    result = np.zeros(target_shape, dtype=data_1mm.dtype)

    # Calculate start indices for centering
    starts_src = [max(0, (cs - ts) // 2) for cs, ts in zip(current_shape, target_shape)]
    starts_dst = [max(0, (ts - cs) // 2) for cs, ts in zip(current_shape, target_shape)]

    # Calculate the size of the region to copy
    sizes = [min(cs, ts) for cs, ts in zip(current_shape, target_shape)]

    # Adjust for offset
    sizes = [min(s, ts - sd, cs - ss) for s, ts, sd, cs, ss in
             zip(sizes, target_shape, starts_dst, current_shape, starts_src)]

    # Copy data
    result[
        starts_dst[0]:starts_dst[0]+sizes[0],
        starts_dst[1]:starts_dst[1]+sizes[1],
        starts_dst[2]:starts_dst[2]+sizes[2]
    ] = data_1mm[
        starts_src[0]:starts_src[0]+sizes[0],
        starts_src[1]:starts_src[1]+sizes[1],
        starts_src[2]:starts_src[2]+sizes[2]
    ]

    print(f"Conformed shape: {result.shape}")
    return result

def preprocess_volume(data, header):
    """
    Preprocess MRI volume for model input.
    Conforms to 256^3 at 1mm isotropic, normalizes, and prepares for model.
    """
    # Conform to 256^3 at 1mm isotropic (like FreeSurfer)
    data = conform_volume(data, header)

    # Normalize
    data = min_max_normalize(data)

    # Ensure float32
    data = data.astype(np.float32)

    # The model expects input in a specific orientation
    # After canonical reorientation, data is in RAS+ (Right-Anterior-Superior)
    # The tfjs model was trained with transposed input, so we transpose here
    # This matches the local frontend's behavior
    data = np.transpose(data, (2, 1, 0))
    print(f"After transpose shape: {data.shape}")

    # Add batch and channel dimensions
    data = np.expand_dims(data, axis=0)  # batch
    data = np.expand_dims(data, axis=-1)  # channel

    return data


def postprocess_segmentation(segmentation):
    """
    Transpose segmentation back to standard RAS+ orientation.
    Output is 256^3 (conformed space).
    """
    # Transpose back to RAS+ orientation
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

        # Preprocess (conform to 256^3 + normalize)
        preprocess_start = time.time()
        processed = preprocess_volume(data, header)
        preprocess_time = time.time() - preprocess_start
        print(f"Preprocessed shape: {processed.shape}, Time: {preprocess_time:.2f}s")

        # Run inference
        inference_start = time.time()
        segmentation = run_inference(processed)
        segmentation = postprocess_segmentation(segmentation)
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

        processed = preprocess_volume(data, header)
        print(f"Preprocessed shape: {processed.shape}")

        segmentation = run_inference(processed)
        print(f"Raw segmentation shape: {segmentation.shape}")

        segmentation = postprocess_segmentation(segmentation)
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


@app.post("/segment/tensor")
async def segment_tensor(file: UploadFile = File(...)):
    """
    Segment using pre-processed tensor from frontend.

    Accepts gzipped raw tensor data (256x256x256 uint8) that has already been
    conformed by NiiVue. This ensures identical preprocessing to local inference.

    The frontend sends the conformed volume, server just runs inference.
    """
    import base64

    try:
        start_time = time.time()

        # Read gzipped tensor data
        compressed_bytes = await file.read()
        print(f"Received {len(compressed_bytes)} bytes of compressed tensor data")

        # Decompress
        try:
            raw_bytes = gzip.decompress(compressed_bytes)
        except:
            # Maybe not compressed
            raw_bytes = compressed_bytes

        expected_size = 256 * 256 * 256
        if len(raw_bytes) != expected_size:
            raise HTTPException(400, f"Expected {expected_size} bytes (256³), got {len(raw_bytes)}")

        # Convert to numpy array
        data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((256, 256, 256))
        print(f"Tensor shape: {data.shape}, dtype: {data.dtype}")

        # Normalize to [0, 1] - same as brainchop's minMaxNormalizeVolumeData
        data = data.astype(np.float32)
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min > 0:
            data = (data - data_min) / (data_max - data_min)
        print(f"Normalized range: [{data.min():.3f}, {data.max():.3f}]")

        # Transpose - same as brainchop with enableTranspose=true
        data = np.transpose(data, (2, 1, 0))
        print(f"After transpose: {data.shape}")

        # Add batch and channel dimensions
        data = np.expand_dims(data, axis=0)   # batch
        data = np.expand_dims(data, axis=-1)  # channel
        print(f"Model input shape: {data.shape}")

        # Run inference
        inference_start = time.time()
        loaded_model = load_model()
        prediction = loaded_model.predict(data, verbose=0)
        segmentation = np.argmax(prediction, axis=-1)[0]

        # Transpose back to match frontend expectations
        segmentation = np.transpose(segmentation, (2, 1, 0))
        inference_time = time.time() - inference_start
        print(f"Inference time: {inference_time:.2f}s, output shape: {segmentation.shape}")

        total_time = time.time() - start_time

        # Compress and encode result
        seg_bytes = segmentation.astype(np.uint8).tobytes()
        compressed = gzip.compress(seg_bytes)
        encoded = base64.b64encode(compressed).decode('utf-8')

        return JSONResponse({
            "success": True,
            "shape": list(segmentation.shape),
            "dtype": "uint8",
            "encoding": "base64_gzip",
            "inference_time": round(inference_time, 3),
            "total_time": round(total_time, 3),
            "data": encoded
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"ERROR in /segment/tensor: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, f"Tensor inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS # Optional: May be needed for local development
import time # For potential timing or frame processing delays
import tempfile # For handling uploaded video files
import requests

# --- Configuration ---
MODEL_PATH = "new_model_11.keras"
HF_URL = "https://huggingface.co/thkash18/crowd-count-model/resolve/main/new_model_11.keras"
def download_model_from_hf(url=HF_URL, save_path=MODEL_PATH):
    """Download the model from Hugging Face if not already present."""
    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}")
        return
    print(f"Downloading model from Hugging Face: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Model downloaded and saved to {save_path}")

MODEL_INPUT_SIZE = (512, 512)   # Must match model training
DENSITY_OUTPUT_SIZE = (64, 64)  # Must match model output shape
# ImageNet Mean/Std for normalization (since VGG base is likely used)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Default frame skip for video processing if not provided by frontend
DEFAULT_FRAME_SKIP = 5

# --- Define and Register Custom Layer ---
# This MUST match the definition used during saving/Streamlit exactly
@tf.keras.utils.register_keras_serializable()
class CBAMLayer(layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = int(input_shape[-1])
        hidden_units = max(channel // self.reduction_ratio, 1)
        self.shared_dense_one = layers.Dense(hidden_units, activation='relu', name=self.name + '_dense1')
        self.shared_dense_two = layers.Dense(channel, activation=None, name=self.name + '_dense2')
        self.conv_spatial = layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid', name=self.name + '_spatial')
        # Explicitly build internal layers - Adjust input shapes based on call logic
        dense_input_shape = (None, 1, 1, channel)
        self.shared_dense_one.build(dense_input_shape)
        self.shared_dense_two.build((None, 1, 1, hidden_units))
        conv_input_shape = list(input_shape)
        conv_input_shape[-1] = 2 # concatenated avg and max spatial pools
        self.conv_spatial.build(tuple(conv_input_shape))
        super(CBAMLayer, self).build(input_shape) # Call super build last

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))
        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        x = inputs * channel_attention
        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)
        spatial_attention = self.conv_spatial(concat)
        out = x * spatial_attention
        return out

    def get_config(self):
        config = super(CBAMLayer, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load Model Globally ---
model = None
try:
    download_model_from_hf(HF_URL, MODEL_PATH)
    model = load_model(MODEL_PATH, custom_objects={'CBAMLayer': CBAMLayer}, compile=False)
    print("✅ Keras model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading Keras model: {e}")
    model = None

# --- Helper Functions ---

def predict_density_for_patch(patch_rgb):
    """Preprocesses a single patch with Mean/Std Norm, predicts density, clamps negatives."""
    if model is None: raise ValueError("Model not loaded")
    img = cv2.resize(patch_rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)
    density = np.squeeze(pred)
    density = np.maximum(density, 0)
    return density

def aggregate_image_density(image_rgb):
    """Splits image, predicts per patch, aggregates, calculates final count."""
    h, w, _ = image_rgb.shape
    ph, pw = MODEL_INPUT_SIZE
    n_patches_h = (h + ph - 1) // ph
    n_patches_w = (w + pw - 1) // pw
    padded_h = n_patches_h * ph
    padded_w = n_patches_w * pw

    pad_bottom = padded_h - h
    pad_right = padded_w - w
    img_padded = cv2.copyMakeBorder(image_rgb, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)

    scale = MODEL_INPUT_SIZE[0] // DENSITY_OUTPUT_SIZE[0]
    per_pixel_h = padded_h // scale
    per_pixel_w = padded_w // scale
    aggregate_density_small = np.zeros((per_pixel_h, per_pixel_w), dtype=np.float32)

    total_predicted_count = 0.0

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y0 = i * ph
            x0 = j * pw
            patch = img_padded[y0:y0+ph, x0:x0+pw, :]
            density_small = predict_density_for_patch(patch)
            aggregate_density_small[i*DENSITY_OUTPUT_SIZE[0]:(i+1)*DENSITY_OUTPUT_SIZE[0],
                                    j*DENSITY_OUTPUT_SIZE[1]:(j+1)*DENSITY_OUTPUT_SIZE[1]] = density_small
            total_predicted_count += float(density_small.sum())

    agg_density_for_viz = cv2.resize(aggregate_density_small, (padded_w, padded_h), interpolation=cv2.INTER_LINEAR)
    agg_density_for_viz[agg_density_for_viz < 0] = 0
    agg_density_for_viz = agg_density_for_viz[:h, :w]

    return agg_density_for_viz, total_predicted_count

def create_heatmap_base64(image_pil_or_np_rgb, density_map, alpha=0.6):
    """Generates heatmap overlay and encodes as base64 data URL."""
    if isinstance(image_pil_or_np_rgb, Image.Image):
        img_np = np.array(image_pil_or_np_rgb).astype(np.uint8)
    elif isinstance(image_pil_or_np_rgb, np.ndarray):
        img_np = image_pil_or_np_rgb.astype(np.uint8)
    else:
        raise TypeError("Input image must be a PIL Image or NumPy array")

    h, w, _ = img_np.shape
    norm = density_map.copy()
    max_val = norm.max()
    if max_val > 1e-8: norm = norm / max_val
    else: norm = np.zeros_like(norm)

    heat = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET) # BGR heatmap
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Convert original RGB to BGR
    overlay_bgr = cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0)

    is_success, buffer = cv2.imencode(".png", overlay_bgr)
    if not is_success: raise ValueError("Could not encode heatmap overlay to PNG")

    base64_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{base64_string}"


# --- API Routes ---

@app.route('/')
def serve_index():
    """Serves the index.html file."""
    return send_from_directory('.', 'index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image_endpoint():
    """Handles image upload, prediction, and returns JSON."""
    if model is None: return jsonify({"error": "Model is not loaded."}), 500
    if 'image' not in request.files: return jsonify({"error": "No image file."}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({"error": "No selected file."}), 400

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None: return jsonify({"error": "Could not decode image."}), 400
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_rgb)
        density_map_viz, predicted_count = aggregate_image_density(img_rgb)
        heatmap_base64 = create_heatmap_base64(image_pil, density_map_viz, alpha=0.6)
        return jsonify({"count": predicted_count, "heatmap_base64": heatmap_base64})
    except Exception as e:
        print(f"ERROR in /predict_image: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500

@app.route('/predict_video', methods=['POST'])
def predict_video_endpoint():
    """Handles video upload, processes frames, returns count data."""
    if model is None: return jsonify({"error": "Model is not loaded."}), 500
    if 'video' not in request.files: return jsonify({"error": "No video file provided."}), 400

    video_file = request.files['video']
    if video_file.filename == '': return jsonify({"error": "No selected video file."}), 400

    # Get frame skip value from request form data (optional, default used if not provided)
    try:
        frame_skip = int(request.form.get('frame_skip', DEFAULT_FRAME_SKIP))
        if frame_skip < 1: frame_skip = 1 # Ensure at least 1
    except ValueError:
        frame_skip = DEFAULT_FRAME_SKIP

    tfile_path = None
    cap = None
    results = [] # To store [frame_number, count]

    try:
        # Save video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as tfile:
            video_file.save(tfile.name)
            tfile_path = tfile.name

        cap = cv2.VideoCapture(tfile_path)
        if not cap.isOpened(): return jsonify({"error": "Could not open video file."}), 400

        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {video_file.filename}, Total Frames: {frame_count_total}, Skipping: {frame_skip}")

        processed_frame_counter = 0
        while cap.isOpened():
            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret: break

            # Process only every Nth frame
            if current_frame_pos % frame_skip == 0:
                processed_frame_counter += 1
                print(f"  Processing frame {current_frame_pos}...")
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, predicted_count = aggregate_image_density(img_rgb) # Don't need heatmap for video count graph
                results.append([current_frame_pos, predicted_count])

        print(f"Finished processing video. Processed {processed_frame_counter} frames.")
        return jsonify({"frame_counts": results}) # Return list of [frame_num, count]

    except Exception as e:
        print(f"ERROR in /predict_video: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Error processing video: {e}"}), 500
    finally:
        if cap is not None and cap.isOpened(): cap.release()
        if tfile_path and os.path.exists(tfile_path):
            try: os.unlink(tfile_path) # Clean up temporary file
            except Exception as e_clean: print(f"Error cleaning up temp video file: {e_clean}")


@app.route('/predict_webcam', methods=['POST'])
def predict_webcam_endpoint():
    """Handles a single webcam frame (sent as base64), returns count and heatmap."""
    if model is None: return jsonify({"error": "Model is not loaded."}), 500

    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({"error": "No image_base64 data provided."}), 400

    try:
        # Decode base64 string
        # Remove header like "data:image/jpeg;base64," if present
        base64_data = data['image_base64'].split(',')[1] if ',' in data['image_base64'] else data['image_base64']
        img_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None: return jsonify({"error": "Could not decode base64 image."}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Use PIL image for heatmap generation consistency if needed, or directly use numpy array
        # image_pil = Image.fromarray(img_rgb)

        density_map_viz, predicted_count = aggregate_image_density(img_rgb)
        # Pass the numpy array directly to heatmap function
        heatmap_base64 = create_heatmap_base64(img_rgb, density_map_viz, alpha=0.6)

        return jsonify({"count": predicted_count, "heatmap_base64": heatmap_base64})

    except Exception as e:
        print(f"ERROR in /predict_webcam: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Error processing webcam frame: {e}"}), 500


# --- Run the App ---
if __name__ == '__main__':
    # Use 0.0.0.0 for broader accessibility, debug=True for development
    app.run(host='0.0.0.0', port=5000, debug=True)



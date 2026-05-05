from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import torch
import rasterio
import tempfile
import os

app = Flask(__name__)
CORS(app,  resources={
    r"/predict": {"origins": "https://urban-encroachment-detection.vercel.app"}
})

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()


# =========================
# MODEL (same as training)
# =========================
class UNet(torch.nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, 3, padding=1),
            torch.nn.ReLU()
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU()
        )
        self.pool = torch.nn.MaxPool2d(2)
        self.dec1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 2, stride=2),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.dec1(x2)
        return self.out(x3)


# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/encroachment_model.pt"

try:
    model = UNet(in_channels=16).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None


# =========================
# BUILD 16 BAND STACK
# =========================
def build_stack(past_path, present_path):
    with rasterio.open(past_path) as src:
        b2_b, b3_b, b4_b, b8_b, b11_b = src.read()

    with rasterio.open(present_path) as src:
        b2_a, b3_a, b4_a, b8_a, b11_a = src.read()

    # NDVI
    ndvi_b = (b8_b - b4_b) / (b8_b + b4_b + 1e-6)
    ndvi_a = (b8_a - b4_a) / (b8_a + b4_a + 1e-6)

    # NDBI
    ndbi_b = (b11_b - b8_b) / (b11_b + b8_b + 1e-6)
    ndbi_a = (b11_a - b8_a) / (b11_a + b8_a + 1e-6)

    stack = np.stack([
        b2_b, b3_b, b4_b, b8_b, b11_b,
        ndvi_b, ndbi_b,
        b2_a, b3_a, b4_a, b8_a, b11_a,
        ndvi_a, ndbi_a,
        ndvi_a - ndvi_b,
        ndbi_a - ndbi_b
    ], axis=0)

    return stack.astype(np.float32)


# =========================
# INFERENCE (same as training)
# =========================
def run_inference(stack):
    if model is None:
        raise Exception("Model not loaded")

    C, H, W = stack.shape

    # normalize (IMPORTANT)
    stack = (stack - np.percentile(stack, 2)) / (
        np.percentile(stack, 98) - np.percentile(stack, 2) + 1e-6
    )
    stack = np.clip(stack, 0, 1)

    PATCH = 256
    STRIDE = 128

    pred_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H - PATCH + 1, STRIDE):
            for x in range(0, W - PATCH + 1, STRIDE):

                patch = stack[:, y:y+PATCH, x:x+PATCH]

                patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                pred = torch.sigmoid(model(patch_tensor)).cpu().numpy()[0, 0]

                pred_map[y:y+PATCH, x:x+PATCH] += pred
                count_map[y:y+PATCH, x:x+PATCH] += 1

    pred_map /= (count_map + 1e-6)

    return pred_map


# =========================
# API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "past_image" not in request.files or "present_image" not in request.files:
            return jsonify({"error": "Upload both images"}), 400

        past_file = request.files["past_image"]
        present_file = request.files["present_image"]

        past_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(past_file.filename))
        present_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(present_file.filename))

        past_file.save(past_path)
        present_file.save(present_path)

        # =========================
        # BUILD STACK
        # =========================
        stack = build_stack(past_path, present_path)
        print("✅ Stack:", stack.shape)

        # =========================
        # INFERENCE
        # =========================
        pred_map = run_inference(stack)
        print("✅ Inference:", pred_map.shape)

        # =========================
        # 🔥 RGB IMAGE (IMPORTANT FIX)
        # =========================
        with rasterio.open(present_path) as src:
            b2_a, b3_a, b4_a, b8_a, b11_a = src.read()

        # Create RGB (same as training)
        rgb = np.stack([b4_a, b3_a, b2_a], axis=-1)

        rgb = (rgb - np.percentile(rgb, 2)) / (
            np.percentile(rgb, 98) - np.percentile(rgb, 2) + 1e-6
        )
        rgb = np.clip(rgb, 0, 1)

        print("✅ RGB built:", rgb.shape)

        # =========================
        # STATS
        # =========================
        stats = {
            "mean_risk": float(pred_map.mean()),
            "max_risk": float(pred_map.max()),
            "std_risk": float(pred_map.std()),
            "high_confidence_pixels": int((pred_map > 0.6).sum()),
            "high_confidence_percent": float((pred_map > 0.6).mean() * 100)
        }

        # =========================
        # RESPONSE
        # =========================
        return jsonify({
            "status": "success",
            "full_prediction_map": pred_map.tolist(),   # ✅ FIXED
            "rgb_image": rgb.tolist(),
            "shape": list(pred_map.shape),
            "statistics": stats
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": device
    })


if __name__ == "__main__":
    print("🚀 Server started")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
import os
import gdown
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# === Path to model ===
MODEL_PATH = "best_model.keras"

# === Google Drive model URL (converted to direct gdown link) ===
MODEL_URL = "https://drive.google.com/uc?id=17HfixT0TvMHdQNMFbGA39N5DX_k6xOhi"

# === Download model if missing ===
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
else:
    print("✅ Model already exists locally.")

# === Load model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Load category mappings ===
cat_df = pd.read_csv(
    "list_category_cloth.txt",
    sep=r"\s{2,}",
    engine="python",
    skiprows=1,
    names=["category_id", "category_name", "category_type"]
)

# Mapping: category_id → index (used during training)
cat_ids = sorted(cat_df["category_id"].unique())
id_to_index = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

# Mapping: index → category_name (used during prediction)
index_to_name = {
    id_to_index[row["category_id"]]: row["category_name"]
    for _, row in cat_df.iterrows()
}

# === Image preprocessing ===
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # Drop alpha if RGBA
        image = image[..., :3]
    return np.expand_dims(image, axis=0)

# === Predict ===
def predict(image: Image.Image) -> str:
    processed = preprocess_image(image)
    preds = model.predict(processed)
    predicted_index = np.argmax(preds, axis=1)[0]
    return index_to_name.get(predicted_index, "Unknown")

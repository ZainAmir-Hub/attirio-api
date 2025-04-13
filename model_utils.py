# model_utils.py

import os
import gdown
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

MODEL_PATH = "best_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=17HfixT0TvMHdQNMFbGA39N5DX_k6xOhi"

model = None  # Do not load on import

# === Download model if missing ===
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# === Load category mappings (these are small, fine to load now) ===
cat_df = pd.read_csv(
    "list_category_cloth.txt",
    sep=r"\s{2,}",
    engine="python",
    skiprows=1,
    names=["category_id", "category_name", "category_type"]
)

cat_ids = sorted(cat_df["category_id"].unique())
id_to_index = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
index_to_name = {
    id_to_index[row["category_id"]]: row["category_name"]
    for _, row in cat_df.iterrows()
}

def load_model():
    global model
    if model is None:
        print("🧠 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.expand_dims(image, axis=0)

def predict(image: Image.Image) -> str:
    model_instance = load_model()
    processed = preprocess_image(image)
    preds = model_instance.predict(processed)
    predicted_index = np.argmax(preds, axis=1)[0]
    return index_to_name.get(predicted_index, "Unknown")

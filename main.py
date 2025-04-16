from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
from model_utils import predict

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS if you want to allow requests from mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    logger.info("🔍 Received a prediction request")

    try:
        contents = await file.read()
        logger.info(f"📦 Received file: {file.filename}, size: {len(contents)} bytes")

        # Open and process the image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info("🖼️ Image loaded successfully")

        # Get the prediction
        prediction = predict(image)
        logger.info(f"✅ Prediction: {prediction}")

        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        return {"error": str(e)}

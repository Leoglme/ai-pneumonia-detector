import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictionResponse
from .inference import KNNPneumoniaService, CNNSPneumoniaService

app = FastAPI(title="Pneumonia KNN API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pneumonia-api")

# Charger le service au démarrage (1 seule fois)
KNN_SERVICE: KNNPneumoniaService | None = None
CNN_SERVICE: CNNSPneumoniaService | None = None


@app.on_event("startup")
def on_startup():
    global KNN_SERVICE, CNN_SERVICE
    KNN_SERVICE = KNNPneumoniaService()
    CNN_SERVICE = CNNSPneumoniaService()
    logger.info("KNN Pneumonia KNN_SERVICE loaded. Image size=%s", KNN_SERVICE.image_size)
    logger.info("CNN loaded. Image size=%s", CNN_SERVICE.image_size if CNN_SERVICE else None)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/knn/predict", response_model=PredictionResponse)
async def predict(
        file: UploadFile = File(...),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Seuil de classification pour PNEUMONIA"),
):
    if KNN_SERVICE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith(("image/", "application/octet-stream")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        content = await file.read()
        p_normal, p_pneumonia = KNN_SERVICE.predict_proba(content)
        label = KNN_SERVICE.predict_label(p_pneumonia, threshold=threshold)

        return PredictionResponse(
            predicted_class=label,
            probability_pneumonia=p_pneumonia,
            probability_normal=p_normal,
            threshold=threshold,
            metadata={
                "image_size": KNN_SERVICE.image_size,
                "class_map": KNN_SERVICE.class_map,
                "filename": file.filename,
                "content_type": file.content_type,
            },
        )
    except Exception as e:
        logger.exception("Prediction failure")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    
@app.post("/api/cnn/predict", response_model=PredictionResponse)
async def predict_cnn(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Seuil de classification pour PNEUMONIA"),
):
    if CNN_SERVICE is None:
        raise HTTPException(status_code=503, detail="CNN model not loaded")

    if not (file.content_type and file.content_type.startswith(("image/", "application/octet-stream"))):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        content = await file.read()
        p_normal, p_pneumonia = CNN_SERVICE.predict_proba(content)
        label = CNN_SERVICE.predict_label(p_pneumonia, threshold=threshold)

        return PredictionResponse(
            predicted_class=label,
            probability_pneumonia=p_pneumonia,
            probability_normal=p_normal,
            threshold=threshold,
            metadata={
                "image_size": CNN_SERVICE.image_size,
                "rescale": CNN_SERVICE.rescale,
                "class_map": CNN_SERVICE.idx_to_class,
                "filename": file.filename,
                "content_type": file.content_type,
            },
        )
    except Exception as e:
        logger.exception("CNN Prediction failure")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


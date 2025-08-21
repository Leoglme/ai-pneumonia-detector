import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictionResponse
from .inference import KNNPneumoniaService

app = FastAPI(title="Pneumonia KNN API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pneumonia-api")

# Charger le service au démarrage (1 seule fois)
SERVICE: KNNPneumoniaService | None = None


@app.on_event("startup")
def on_startup():
    global SERVICE
    SERVICE = KNNPneumoniaService()
    logger.info("KNN Pneumonia service loaded. Image size=%s", SERVICE.image_size)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/knn/predict", response_model=PredictionResponse)
async def predict(
        file: UploadFile = File(...),
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Seuil de classification pour PNEUMONIA"),
):
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith(("image/", "application/octet-stream")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        content = await file.read()
        p_normal, p_pneumonia = SERVICE.predict_proba(content)
        label = SERVICE.predict_label(p_pneumonia, threshold=threshold)

        return PredictionResponse(
            predicted_class=label,
            probability_pneumonia=p_pneumonia,
            probability_normal=p_normal,
            threshold=threshold,
            metadata={
                "image_size": SERVICE.image_size,
                "class_map": SERVICE.class_map,
                "filename": file.filename,
                "content_type": file.content_type,
            },
        )
    except Exception as e:
        logger.exception("Prediction failure")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

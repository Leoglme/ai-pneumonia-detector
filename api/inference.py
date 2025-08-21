import json
import joblib
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple

# chemins par défaut
MODEL_PATH = "models/knn/knn_model.joblib"
SCALER_PATH = "scalers/knn/scaler.joblib"
META_PATH = "knn_meta.json"

# Valeurs de secours si pas de meta
DEFAULT_IMAGE_SIZE = (64, 64)
CLASS_MAP = {0: "NORMAL", 1: "PNEUMONIA"}

class KNNPneumoniaService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.image_size = DEFAULT_IMAGE_SIZE
        self.class_map = CLASS_MAP

        # essaie de charger les metadatas si dispo
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "image_size" in meta and isinstance(meta["image_size"], (list, tuple)):
                self.image_size = tuple(meta["image_size"])
            if "class_map" in meta:
                self.class_map = {int(k): v for k, v in meta["class_map"].items()}
        except Exception:
            # pas bloquant : on garde les defaults
            pass

    def _preprocess_bytes(self, content: bytes) -> np.ndarray:
        """
        Reproduit le prétraitement de DataHandler._load_image :
        - resize
        - 3 canaux (si N&B)
        - flatten
        - normalisation /255
        - reshape (1, n_features) pour scikit
        """
        with Image.open(BytesIO(content)) as img:
            img = img.resize(self.image_size)
            arr = np.array(img)

            if arr.ndim == 2:
                arr = np.stack((arr,) * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.concatenate((arr, arr, arr), axis=-1)

            arr = arr.flatten() / 255.0
            return arr.reshape(1, -1)

    def predict_proba(self, content: bytes) -> Tuple[float, float]:
        """
        Retourne (proba_normal, proba_pneumonia)
        """
        x = self._preprocess_bytes(content)
        x = self.scaler.transform(x)
        proba = self.model.predict_proba(x)[0]  # [p0, p1]
        return float(proba[0]), float(proba[1])

    def predict_label(self, p_pneumonia: float, threshold: float = 0.5) -> str:
        idx = 1 if p_pneumonia >= threshold else 0
        return self.class_map.get(idx, str(idx))

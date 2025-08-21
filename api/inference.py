import json
import joblib
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple
from pathlib import Path
import tensorflow as tf

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


# --- CNN service ---
ROOT = Path(__file__).resolve().parents[1]
CNN_MODEL_PATH = ROOT / "models" / "cnn" / "model.keras"
CNN_META_PATH = ROOT / "models" / "cnn" / "meta.json"


class CNNSPneumoniaService:
    """
    Loads a Keras CNN (.keras) and performs inference with the same preprocessing as training:
    - RGB conversion
    - resize to image_size
    - rescale by 'rescale' meta (default 1/255)
    """

    def __init__(self):
        # charge le modèle Keras sans recompilation
        self.model = tf.keras.models.load_model(str(CNN_MODEL_PATH), compile=False)

        # valeurs par défaut si meta.json manquant
        self.image_size = (256, 256)
        self.rescale = 1.0 / 255.0
        # mapping index->label (0: NORMAL, 1: PNEUMONIA) par défaut
        self.idx_to_class = {0: "NORMAL", 1: "PNEUMONIA"}

        # charge les métadonnées si dispo
        try:
            with open(CNN_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "image_size" in meta:
                self.image_size = tuple(meta["image_size"])
            if "rescale" in meta:
                self.rescale = float(meta["rescale"])
            if "class_indices" in meta and isinstance(meta["class_indices"], dict):
                # meta: {"NORMAL": 0, "PNEUMONIA": 1} -> inverse
                self.idx_to_class = {int(v): k for k, v in meta["class_indices"].items()}
        except Exception:
            # pas bloquant : garde les defaults
            pass

    def _preprocess_bytes(self, content: bytes) -> np.ndarray:
        from PIL import Image
        from io import BytesIO

        with Image.open(BytesIO(content)) as img:
            img = img.convert("RGB").resize(self.image_size)
            arr = np.array(img, dtype=np.float32) * self.rescale
        # shape (1, H, W, 3)
        return np.expand_dims(arr, axis=0)

    def predict_proba(self, content: bytes) -> tuple[float, float]:
        """
        Retourne (p_normal, p_pneumonia) ; la sortie du modèle est sigmoid = p_pneumonia.
        """
        x = self._preprocess_bytes(content)
        p_pneumonia = float(self.model.predict(x, verbose=0)[0][0])
        p_normal = 1.0 - p_pneumonia
        return p_normal, p_pneumonia

    def predict_label(self, p_pneumonia: float, threshold: float = 0.5) -> str:
        idx = 1 if p_pneumonia >= threshold else 0
        # fallback si meta manquant
        return self.idx_to_class.get(idx, "PNEUMONIA" if idx == 1 else "NORMAL")

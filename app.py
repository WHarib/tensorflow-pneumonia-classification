import io, base64, os
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

import tensorflow as tf
from tensorflow import keras
from keras import layers

# ---------------- CONFIG ----------------
IMG_SIZE   = (150, 150)                 # Amy’s model size
N_CLASSES  = 2
THRESHOLD  = 0.5
EXPECT_KEY = os.getenv("EXPECT_KEY")    # optional simple auth
MODEL_PATH = os.getenv("MODEL_PATH", "xray_model.h5")  # local .h5
# ----------------------------------------

app = FastAPI(title="Pneumonia Detection API (AmyJang CNN)", docs_url="/docs", redoc_url="/redoc")
_model: Optional[keras.Model] = None


def ensure_auth(x_api_key: Optional[str]):
    if EXPECT_KEY and x_api_key != EXPECT_KEY:
        raise HTTPException(401, "Invalid or missing API key.")


def build_model() -> keras.Model:
    """Rebuild the CNN that matches xray_model.h5 (weights-only)."""
    inp = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1./255, name="rescaling")(inp)

    x = layers.Conv2D(32, 3, activation="relu", padding="same", name="conv2d")(x)
    x = layers.MaxPooling2D(2, 2, name="max_pooling2d")(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same", name="conv2d_1")(x)
    x = layers.MaxPooling2D(2, 2, name="max_pooling2d_1")(x)

    x = layers.Dropout(0.2, name="dropout")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense")(x)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="dense_3")(x)
    return keras.Model(inp, out)


def get_model() -> keras.Model:
    global _model
    if _model is None:
        m = build_model()
        # Load by name to tolerate extra aug layers / different ordering
        m.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        _model = m
    return _model


def preprocess(pil: Image.Image) -> np.ndarray:
    arr = np.array(pil.resize(IMG_SIZE))  # rescaling layer will /255
    return np.expand_dims(arr.astype("float32"), 0)


def overlay(pil: Image.Image, label: str) -> str:
    colour = (255, 0, 0) if label == "PNEUMONIA" else (0, 255, 0)
    buf = io.BytesIO()
    Image.blend(pil, Image.new("RGB", pil.size, colour), 0.25).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
async def root():
    return (
        "<h1>Pneumonia Detection API – AmyJang CNN</h1>"
        "<p>POST an X-ray to <code>/predict</code>.</p>"
    )


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    ensure_auth(x_api_key)

    pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    model = get_model()

    preds = model.predict(preprocess(pil))[0]
    pneu_score = float(preds[1])  # assuming order [NORMAL, PNEUMONIA]; adjust if needed
    label = "PNEUMONIA" if pneu_score > THRESHOLD else "NORMAL"
    conf = round(pneu_score if label == "PNEUMONIA" else 1 - pneu_score, 4)

    return JSONResponse({
        "diagnosis": label,
        "confidence": conf,
        "raw_score": pneu_score,
        "processed_image": overlay(pil, label)
    })

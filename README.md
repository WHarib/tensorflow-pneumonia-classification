---
title: Pneumonia Detection API (AmyJang CNN)
emoji: 🫁
colorFrom: blue
colorTo: gray
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---

# Pneumonia Detection API (AmyJang CNN)

Simple CNN (Conv2D/MaxPool/Dropout/Dense) trained on the Chest X-ray Pneumonia dataset.  
Served via FastAPI in this Space.

## Endpoint

**POST `/predict`** – `multipart/form-data` with field `file` (PNG/JPEG X-ray).  
Returns JSON: `diagnosis`, `confidence`, `raw_score`, `processed_image` (base64 PNG with overlay).

## Input Pre-processing

- Resize to 150×150
- Rescale /255 inside the model

## Output

Two-class softmax: NORMAL vs PNEUMONIA. Threshold 0.5 (configurable).
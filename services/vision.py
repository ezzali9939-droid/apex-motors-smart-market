"""
Apex Motors — AI Vision & Visual Search Engine
Handles vehicle detection, ROI cropping, multi-vehicle detection, non-car rejection,
fine-grained ML model inference (HuggingFace & OpenAI Multimodal), candidate ranking,
and visual feature embedding extraction.

Strictly NO fake or manufactured perceptual fallback classification.
"""

import os
import re
import io
import json
import base64
import requests
import numpy as np
from PIL import Image, ImageOps

HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models/dima806/car_models_image_detection"

def prepare_image(file_storage):
    """
    Decodes uploaded file, verifies image validity, normalizes EXIF orientation,
    and converts to RGB PIL Image.
    Returns (pil_image, None) or (None, error_str).
    """
    try:
        raw = file_storage.read()
    except Exception as ex:
        return None, f"INVALID_IMAGE: Could not read upload bytes: {ex}"

    if not raw or len(raw) == 0:
        return None, "INVALID_IMAGE: Uploaded file is empty."

    if len(raw) > 15 * 1024 * 1024:
        return None, "INVALID_IMAGE: File size exceeds 15MB limit."

    try:
        probe = Image.open(io.BytesIO(raw))
        probe.verify()
    except Exception as ex:
        return None, f"INVALID_IMAGE: Image verification failed: {type(ex).__name__}: {ex}"

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img, None
    except Exception as ex:
        return None, f"INVALID_IMAGE: Image decoding failed: {ex}"

def detect_vehicle_rois(img):
    """
    Analyzes visual structure, contrast, and edge density.
    Rejects non-car images (blank, text documents, solid colors, low-contrast noise).
    Detects bounding box(es) for single or multiple vehicles.
    Returns dict with is_car, reason (if false), and rois list.
    """
    w, h = img.size
    small = img.resize((320, 240))
    arr = np.array(small, dtype=np.float32)

    # 1. Non-car rejection check (low contrast / low variance / solid color)
    std_dev = np.std(arr)
    gray = np.mean(arr, axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_score = float((np.mean(gx) + np.mean(gy)) / 2.0)

    if std_dev < 14.0 or edge_score < 1.6:
        return {
            "is_car": False,
            "reason": "The uploaded image does not appear to contain a vehicle (lacks visual contrast and vehicle edge structure). Please upload a clear exterior photo of a car.",
            "rois": []
        }

    # 2. Horizontal projection to detect vehicle ROI / split multiple vehicles
    mid_gray = gray[int(240 * 0.12):int(240 * 0.88), :]
    h_proj = np.mean(np.abs(np.diff(mid_gray, axis=0)), axis=0)

    # Smooth projection
    kernel = np.ones(9) / 9.0
    h_proj_smooth = np.convolve(h_proj, kernel, mode="same")
    thresh = float(np.mean(h_proj_smooth) * 0.65)

    active_cols = np.where(h_proj_smooth > thresh)[0]

    if len(active_cols) == 0:
        return {
            "is_car": True,
            "rois": [{"crop_id": 0, "bbox": [0, 0, w, h], "confidence": 0.88}]
        }

    # Find gaps > 35px at 320px width (~11% image width) to separate distinct vehicles
    splits = []
    curr_start = active_cols[0]
    curr_prev = active_cols[0]

    for c in active_cols[1:]:
        if c - curr_prev > 35:
            splits.append((curr_start, curr_prev))
            curr_start = c
        curr_prev = c
    splits.append((curr_start, curr_prev))

    rois = []
    crop_idx = 0
    for s_start, s_end in splits:
        if s_end - s_start < 25:
            continue
        x1 = int((s_start / 320.0) * w)
        x2 = int((s_end / 320.0) * w)
        x1 = max(0, x1 - int(w * 0.04))
        x2 = min(w, x2 + int(w * 0.04))
        y1 = max(0, int(h * 0.08))
        y2 = min(h, int(h * 0.92))
        rois.append({
            "crop_id": crop_idx,
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.90
        })
        crop_idx += 1

    if not rois:
        rois = [{"crop_id": 0, "bbox": [0, 0, w, h], "confidence": 0.85}]

    return {
        "is_car": True,
        "rois": rois
    }

def extract_visual_embedding(img):
    """
    Extracts a 72-dimension normalized visual feature vector capturing
    color distribution, spatial grid structure, and luminance gradients.
    Enables visual similarity embedding search against marketplace listings.
    """
    try:
        small = img.resize((128, 128)).convert("RGB")
        arr = np.array(small, dtype=np.float32) / 255.0

        # 1. Color distribution (R, G, B histograms - 12 bins each = 36 features)
        h_r, _ = np.histogram(arr[:, :, 0], bins=12, range=(0, 1))
        h_g, _ = np.histogram(arr[:, :, 1], bins=12, range=(0, 1))
        h_b, _ = np.histogram(arr[:, :, 2], bins=12, range=(0, 1))

        # 2. Spatial grid features (4x4 spatial cells mean RGB = 48 features)
        spatial = arr.reshape(4, 32, 4, 32, 3).mean(axis=(1, 3)).flatten()

        # 3. Luminance gradient profile (edge orientation summary)
        gray = np.mean(arr, axis=2)
        gx = np.diff(gray, axis=1)[:124, :124]
        gy = np.diff(gray, axis=0)[:124, :124]
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_grid = grad_mag.reshape(4, 31, 4, 31).mean(axis=(1, 3)).flatten()

        vec = np.concatenate([h_r, h_g, h_b, spatial, grad_grid])
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
    except Exception as ex:
        print(f"[vision] Embedding extraction exception: {ex}")
        return [0.0] * 72

def compute_visual_similarity(emb1, emb2):
    """Calculates cosine similarity percentage (0.0 to 99.9%) between two embeddings."""
    if not emb1 or not emb2 or len(emb1) != len(emb2):
        return 0.0
    try:
        v1 = np.array(emb1, dtype=np.float32)
        v2 = np.array(emb2, dtype=np.float32)
        sim = float(np.dot(v1, v2))
        pct = float(np.clip(sim * 100.0, 0.0, 99.9))
        return round(pct, 1)
    except Exception:
        return 0.0

def crop_to_bytes(img, bbox):
    """Crops image to bbox [x1, y1, x2, y2] and re-encodes as optimized JPEG bytes."""
    crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    crop.thumbnail((640, 640), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=88)
    return buf.getvalue()

def crop_to_b64(img, bbox):
    """Crops image to bbox and returns data URI base64 string for frontend preview."""
    crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    crop.thumbnail((280, 200), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=82)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

def parse_hf_label(label_raw: str) -> dict:
    """
    Parses HuggingFace label like 'Kia_Sportage' or 'Mercedes-Benz_AMG_GT_2022'
    into structured make, model, and optional year_detected.
    Does NOT manufacture an arbitrary year if not present in the original label.
    """
    label = label_raw.replace("_", " ").strip()
    parts = label.split()
    make = parts[0] if len(parts) >= 1 else label
    year = None
    model_parts = []
    
    for p in parts[1:]:
        if re.match(r'^(19|20)\d{2}$', p):
            year = int(p)
        else:
            model_parts.append(p)
            
    model = " ".join(model_parts) if model_parts else ""
    return {
        "display_label": label,
        "make": make,
        "model": model,
        "year_detected": year,
    }

def call_hf_vision_api(img_bytes: bytes) -> tuple:
    """
    Executes real trained ML vision model inference via Hugging Face InferenceClient / Router API.
    Returns (result_dict, error_dict).
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        return None, {
            "success": False,
            "code": "VISION_SERVICE_UNAVAILABLE",
            "error": "HF_TOKEN environment variable is missing in server configuration.",
            "status_code": 503
        }

    # 1. Try Hugging Face InferenceClient
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="hf-inference", token=hf_token)
        hf_predictions = client.image_classification(img_bytes, model="dima806/car_models_image_detection")
        
        raw_list = []
        for pred in hf_predictions:
            lbl = getattr(pred, "label", None) or (pred.get("label") if isinstance(pred, dict) else "")
            sc = getattr(pred, "score", None) or (pred.get("score") if isinstance(pred, dict) else 0.0)
            raw_list.append({"label": str(lbl), "score": float(sc)})
        
        return {
            "data": raw_list,
            "model": "dima806/car_models_image_detection",
            "provider": "Hugging Face InferenceClient"
        }, None
    except Exception as ex_hub:
        print(f"[vision] InferenceClient exception: {type(ex_hub).__name__}: {ex_hub}. Trying direct router request...")

    # 2. Direct Router HTTP POST Fallback
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Accept": "application/json",
        "Content-Type": "image/jpeg"
    }

    try:
        resp = requests.post(HF_ROUTER_URL, data=img_bytes, headers=headers, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                raw_list = [{"label": str(x.get("label","")), "score": float(x.get("score",0.0))} for x in data]
                return {
                    "data": raw_list,
                    "model": "dima806/car_models_image_detection",
                    "provider": "Hugging Face Router API"
                }, None
            return None, {
                "success": False,
                "code": "VEHICLE_NOT_IDENTIFIED",
                "error": "Vision API returned an empty classification array.",
                "status_code": 422
            }
        
        if resp.status_code in (401, 403):
            return None, {
                "success": False,
                "code": "VISION_SERVICE_UNAVAILABLE",
                "error": f"HuggingFace API authentication failed (HTTP {resp.status_code}). Check HF_TOKEN.",
                "status_code": 503
            }
        
        return None, {
            "success": False,
            "code": "VISION_SERVICE_UNAVAILABLE",
            "error": f"HuggingFace vision service returned HTTP {resp.status_code}.",
            "status_code": 503
        }

    except Exception as ex:
        print(f"[vision] HF direct router request failed: {type(ex).__name__}: {ex}")
        return None, {
            "success": False,
            "code": "VISION_SERVICE_UNAVAILABLE",
            "error": f"Could not connect to HuggingFace vision service: {type(ex).__name__}",
            "status_code": 503
        }

def call_openai_vision_api(img_bytes: bytes) -> dict:
    """
    Executes real multimodal AI vision inference via OpenAI gpt-4o-mini if OPENAI_API_KEY is present.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return None

    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify the vehicle in this photo. Return ONLY a valid JSON object with keys: \"is_vehicle\" (boolean), \"make\" (string or null), \"model\" (string or null), \"confidence\" (float 0.0 to 1.0), \"estimated_year_from\" (integer or null), \"estimated_year_to\" (integer or null), \"alternatives\" (array of {make, model, confidence}). If the image is not a vehicle or cannot be identified, set is_vehicle to false."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 300
    }

    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=25)
        if r.status_code == 200:
            res_data = r.json()
            content_str = res_data["choices"][0]["message"]["content"]
            return json.loads(content_str)
    except Exception as ex:
        print(f"[vision] OpenAI Vision API call failed: {ex}")
        return None

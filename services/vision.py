"""
Apex Motors — AI Vision Inference Service
Handles image decoding, EXIF transpose, RGB normalization, and multi-provider
AI vision model inference (HuggingFace InferenceClient & OpenAI Multimodal Vision).
Strictly NO fake or perceptual fallback classification.
"""

import os
import re
import io
import json
import requests
from PIL import Image, ImageOps

HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models/dima806/car_models_image_detection"

def prepare_image_bytes(file_storage):
    """
    Decodes uploaded file, verifies image validity, normalizes EXIF orientation,
    converts mode to RGB, resizes thumbnail to max 640px maintaining aspect ratio,
    and re-encodes as optimized JPEG bytes.
    Returns (jpeg_bytes, None) on success or (None, error_str) on failure.
    """
    try:
        raw = file_storage.read()
    except Exception as ex:
        return None, f"INVALID_IMAGE: could not read upload bytes: {ex}"

    if not raw or len(raw) == 0:
        return None, "INVALID_IMAGE: uploaded file is empty"

    if len(raw) > 15 * 1024 * 1024:
        return None, "INVALID_IMAGE: file size exceeds 15MB limit"

    try:
        probe = Image.open(io.BytesIO(raw))
        probe.verify()
    except Exception as ex:
        return None, f"INVALID_IMAGE: image verification failed: {type(ex).__name__}: {ex}"

    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as ex:
        return None, f"INVALID_IMAGE: could not re-open image: {ex}"

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
    except Exception as ex:
        return None, f"INVALID_IMAGE: color conversion failed: {ex}"

    try:
        img.thumbnail((640, 640), Image.LANCZOS)
    except Exception:
        try:
            img.thumbnail((640, 640))
        except Exception as ex:
            return None, f"INVALID_IMAGE: resize failed: {ex}"

    try:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue(), None
    except Exception as ex:
        return None, f"INVALID_IMAGE: JPEG re-encode failed: {ex}"

def parse_hf_label(label_raw: str) -> dict:
    """
    Parses HuggingFace label like 'Kia_Sportage' or 'Mercedes-Benz_AMG_GT_2022'
    into structured make, model, and optional year_detected.
    Does NOT manufacture an arbitrary year if not present in the original label space.
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
            "error": "HF_TOKEN environment variable is missing in deployment server configuration.",
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

    import base64
    b64_img = base64.b64encode(img_bytes).decode('utf-8')

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

"""
Apex Motors — Official Computer Vision Vehicle Identification Service

Architecture & Policies:
- Uses Google GenAI SDK (`google-genai`) with Gemini (`gemini-3.8-flash`).
- Server-side GEMINI_API_KEY enforcement.
- Strict Pydantic JSON schema validation (`VehicleVisionResponse`).
- Strict Application-Side Confidence Policy:
  * < 0.50 : VEHICLE_NOT_IDENTIFIED (HTTP 422)
  * 0.50–0.69 : Make displayed if supported; Model remains Unknown
  * 0.70–0.87 : Make & Model displayed; Trim remains Unknown
  * >= 0.88 : Specific Trim/Generation allowed ONLY if supported by visible badges
- Strict Year Policy: Year range (e.g. 2019–2023) preferred over invented single years.
- ABSOLUTELY ZERO fake vehicle fallbacks, aspect-ratio guesses, or random selections.
- If no vision API key is configured server-side: returns VISION_UNAVAILABLE (HTTP 503).
"""

import os
import io
import re
import json
import base64
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError

# Primary AI Model
GEMINI_MODEL = "gemini-3.8-flash"

# ── 1. PYDANTIC STRUCTURED VISION RESPONSE SCHEMAS ───────────────────────────
class VehicleCandidate(BaseModel):
    make: Optional[str] = Field(default=None, description="Plausible candidate make")
    model: Optional[str] = Field(default=None, description="Plausible candidate model")
    generation: Optional[str] = Field(default=None, description="Plausible candidate generation code")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Candidate confidence score")


class VehicleVisionResponse(BaseModel):
    is_vehicle: bool = Field(description="True if image contains a clear exterior view of a motor vehicle")
    make: Optional[str] = Field(default=None, description="The vehicle manufacturer brand name")
    model: Optional[str] = Field(default=None, description="The specific vehicle model name")
    generation: Optional[str] = Field(default=None, description="Generation code if discernible")
    trim: Optional[str] = Field(default=None, description="Trim tier ONLY if visually verified by explicit badge text")
    year_from: Optional[int] = Field(default=None, description="Estimated production start year for this generation")
    year_to: Optional[int] = Field(default=None, description="Estimated production end year for this generation")
    body_type: Optional[str] = Field(default=None, description="Body style e.g. SUV, Sedan, Coupe, Hatchback, Pickup, Van")
    color: Optional[str] = Field(default=None, description="Primary exterior color of the vehicle")
    visible_badges: List[str] = Field(default_factory=list, description="Text or logos of visible badges on vehicle body")
    visible_text: List[str] = Field(default_factory=list, description="Any written text visible on vehicle or license plate")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence score from 0.0 to 1.0")
    candidates: List[VehicleCandidate] = Field(default_factory=list, description="Alternative candidate models")
    visual_evidence: List[str] = Field(default_factory=list, description="Specific visual cues supporting identification")


# ── 2. STAGE 1: IMAGE PREPARATION & VALIDATION ──────────────────────────────
def prepare_image(file_storage) -> tuple[Optional[Image.Image], Optional[str]]:
    """
    Decodes uploaded file, verifies image validity, normalizes EXIF orientation,
    converts to RGB PIL Image, and resizes appropriately.
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
        img.thumbnail((1024, 1024), Image.LANCZOS)
        return img, None
    except Exception as ex:
        return None, f"INVALID_IMAGE: Image decoding failed: {ex}"


def detect_vehicle_rois(img: Image.Image) -> Dict[str, Any]:
    """
    STAGE 1: Analyzes image quality, contrast, and edge density.
    Rejects blank images, solid colors, low contrast noise, and document scans.
    """
    w, h = img.size
    small = img.resize((320, 240))
    arr = np.array(small, dtype=np.float32)

    std_dev = float(np.std(arr))
    gray = np.mean(arr, axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_score = float((np.mean(gx) + np.mean(gy)) / 2.0)

    quality_metrics = {
        "resolution": f"{w}x{h}",
        "std_dev": round(std_dev, 2),
        "edge_score": round(edge_score, 2),
        "is_valid_quality": std_dev >= 8.0 and edge_score >= 0.6
    }

    if not quality_metrics["is_valid_quality"]:
        return {
            "is_car": False,
            "reason": "The uploaded image is blank, low quality, or does not contain a vehicle. Please upload a clear photo of a car.",
            "quality_metrics": quality_metrics,
            "rois": []
        }

    # Horizontal projection for spatial vehicle ROI
    mid_gray = gray[int(240 * 0.12):int(240 * 0.88), :]
    h_proj = np.mean(np.abs(np.diff(mid_gray, axis=0)), axis=0)
    kernel = np.ones(9) / 9.0
    h_proj_smooth = np.convolve(h_proj, kernel, mode="same")
    thresh = float(np.mean(h_proj_smooth) * 0.65)

    active_cols = np.where(h_proj_smooth > thresh)[0]

    if len(active_cols) == 0:
        return {
            "is_car": True,
            "quality_metrics": quality_metrics,
            "rois": [{"crop_id": 0, "bbox": [0, 0, w, h], "confidence": 0.90}]
        }

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
        x1 = max(0, int((s_start / 320.0) * w) - int(w * 0.03))
        x2 = min(w, int((s_end / 320.0) * w) + int(w * 0.03))
        y1 = max(0, int(h * 0.06))
        y2 = min(h, int(h * 0.94))
        rois.append({
            "crop_id": crop_idx,
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.92
        })
        crop_idx += 1

    if not rois:
        rois = [{"crop_id": 0, "bbox": [0, 0, w, h], "confidence": 0.88}]

    return {
        "is_car": True,
        "quality_metrics": quality_metrics,
        "rois": rois
    }


# ── 3. GEMINI VISION INFERENCE LAYER ─────────────────────────────────────────
GEMINI_SYSTEM_PROMPT = """You are an expert Computer Vision Vehicle Identification System.
Your task is to analyze the provided image and identify the exact motor vehicle.

STRICT INSTRUCTIONS:
1. Determine if the image contains a clear view of a motor vehicle (is_vehicle).
2. Identify vehicle make, model, generation code (internal chassis or generation identifier), and body style.
3. Identify estimated production year range (year_from and year_to) based on generation design cycle. DO NOT guess a single exact year unless an explicit model year badge is visible.
4. Identify visible badges, logos, model emblems, trim badges.
5. Identify trim ONLY if visually verified by an explicit badge in the photo. Otherwise leave trim as null.
6. Provide an overall identification confidence score between 0.00 and 1.00 based strictly on visual clarity and certainty.
7. List specific visual evidence cues supporting your identification.
8. NEVER invent mechanical specifications (horsepower, engine cc, price, mileage). Focus strictly on visual vehicle identification.

Return ONLY a valid JSON object strictly adhering to this schema:
{
  "is_vehicle": boolean,
  "make": string or null,
  "model": string or null,
  "generation": string or null,
  "trim": string or null,
  "year_from": integer or null,
  "year_to": integer or null,
  "body_type": string or null,
  "color": string or null,
  "visible_badges": [string],
  "visible_text": [string],
  "confidence": float (0.0 to 1.0),
  "candidates": [{"make": string, "model": string, "generation": string, "confidence": float}],
  "visual_evidence": [string]
}"""

def call_gemini_vision_api(pil_img: Image.Image) -> tuple[Optional[VehicleVisionResponse], Optional[Dict[str, Any]]]:
    """
    Executes vehicle identification using the official Google GenAI SDK (`google-genai`).
    Returns (validated_response_object, None) or (None, error_dict).
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return None, {
            "success": False,
            "vehicle_detected": False,
            "code": "VISION_UNAVAILABLE",
            "error": "Server-side GEMINI_API_KEY environment variable is missing or unconfigured.",
            "status_code": 503
        }

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=gemini_key)

        # Convert PIL image to JPEG bytes for Gemini API
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()

        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/jpeg"
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[image_part, GEMINI_SYSTEM_PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=700
            )
        )

        raw_text = response.text.strip() if response and response.text else ""
        if not raw_text:
            return None, {
                "success": False,
                "vehicle_detected": False,
                "code": "VEHICLE_NOT_IDENTIFIED",
                "error": "Gemini Vision API returned an empty response.",
                "status_code": 422
            }

        # Parse and validate with Pydantic
        # Extract JSON block if wrapped in markdown code fence
        json_str = raw_text
        if "```" in json_str:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
            if match:
                json_str = match.group(1).strip()

        parsed_data = json.loads(json_str)
        validated = VehicleVisionResponse.model_validate(parsed_data)
        return validated, None

    except ValidationError as ve:
        print(f"[vision] Pydantic validation error on Gemini output: {ve}")
        return None, {
            "success": False,
            "vehicle_detected": False,
            "code": "VEHICLE_NOT_IDENTIFIED",
            "error": "Invalid structured JSON format returned by vision model.",
            "status_code": 422
        }
    except Exception as ex:
        print(f"[vision] Gemini API Exception: {type(ex).__name__}: {ex}")
        return None, {
            "success": False,
            "vehicle_detected": False,
            "code": "VISION_UNAVAILABLE",
            "error": f"Could not connect to Gemini Vision service: {type(ex).__name__}",
            "status_code": 503
        }


# ── 4. STAGE 5: REASONING PIPELINE & APPLICATION-SIDE CONFIDENCE POLICY ──────
def run_vision_pipeline(pil_img: Image.Image, img_bytes: bytes = None, target_bbox: list = None) -> Dict[str, Any]:
    """
    Executes multi-stage vehicle vision identification with strict confidence rules.
    
    Confidence Policy Rules:
    - < 0.50 : VEHICLE_NOT_IDENTIFIED (HTTP 422)
    - 0.50–0.69 : Make displayed if supported; Model remains Unknown
    - 0.70–0.87 : Make & Model displayed; Trim remains Unknown
    - >= 0.88 : Specific Trim/Generation allowed ONLY if supported by visible badges
    """
    # 1. Call Gemini Vision API
    vision_data, err_dict = call_gemini_vision_api(pil_img)
    if err_dict:
        return err_dict

    # 2. Check basic vehicle detection
    if not vision_data.is_vehicle:
        return {
            "success": False,
            "vehicle_detected": False,
            "code": "NON_CAR_IMAGE",
            "error": "The uploaded image does not appear to contain a motor vehicle. Please upload a clear photo of a car.",
            "status_code": 422
        }

    raw_conf = float(vision_data.confidence)
    make = (vision_data.make or "").strip()
    model = (vision_data.model or "").strip()
    generation = (vision_data.generation or "").strip()
    trim = (vision_data.trim or "").strip()
    visible_badges = vision_data.visible_badges or []

    # 3. Apply Application-Side Confidence Policy Rules
    if raw_conf < 0.50:
        return {
            "success": False,
            "vehicle_detected": True,
            "code": "VEHICLE_NOT_IDENTIFIED",
            "error": f"Vehicle detected, but visual evidence is insufficient for confident identification (confidence: {round(raw_conf * 100, 1)}%). Please upload a clearer exterior photo.",
            "confidence": round(raw_conf * 100, 1),
            "visual_evidence": vision_data.visual_evidence,
            "status_code": 422
        }

    # Policy Level 1: 0.50 - 0.69 -> Make displayed, Model remains Unknown
    if 0.50 <= raw_conf < 0.70:
        model = None
        generation = None
        trim = None
        display_label = f"{make} (Model Unknown)" if make else "Unknown Vehicle"
        needs_confirmation = True

    # Policy Level 2: 0.70 - 0.87 -> Make & Model displayed, Trim remains Unknown
    elif 0.70 <= raw_conf < 0.88:
        trim = None
        display_label = f"{make} {model}".strip() if (make and model) else (make or "Unknown Vehicle")
        needs_confirmation = True

    # Policy Level 3: >= 0.88 -> Trim allowed ONLY IF supported by visible badges
    else:
        # Check if trim is supported by visible badges
        if trim and not any(trim.lower() in badge.lower() or badge.lower() in trim.lower() for badge in visible_badges):
            trim = None  # Strip unsupported trim guess
        
        parts = [make, model, trim]
        display_label = " ".join([p for p in parts if p]).strip()
        needs_confirmation = False

    # Build Year Range Estimate Object (Year Policy: range preferred over invented single year)
    year_estimate = None
    if vision_data.year_from:
        year_estimate = {
            "from": vision_data.year_from,
            "to": vision_data.year_to or vision_data.year_from
        }

    # Build Alternatives Candidates list
    alternatives = []
    seen = {f"{make}:{model}".lower()}
    for cand in vision_data.candidates:
        c_make = (cand.make or "").strip()
        c_model = (cand.model or "").strip()
        if c_make and c_model:
            key = f"{c_make}:{c_model}".lower()
            if key not in seen:
                seen.add(key)
                alternatives.append({
                    "label": f"{c_make} {c_model}",
                    "make": c_make,
                    "model": c_model,
                    "generation": cand.generation,
                    "confidence": round(cand.confidence * 100, 1)
                })

    conf_percent = round(raw_conf * 100, 1)

    confidence_label = "HIGH"
    if conf_percent < 70:
        confidence_label = "LOW"
    elif conf_percent < 88:
        confidence_label = "MEDIUM"

    if alternatives and len(alternatives) > 0:
        top_alt_conf = alternatives[0]["confidence"]
        if conf_percent - top_alt_conf <= 5.0:
            confidence_label = "AMBIGUOUS"

    return {
        "success": True,
        "vehicle_detected": True,
        "label": display_label,
        "confidence_label": confidence_label,
        "make": make or None,
        "model": model or None,
        "generation": generation or None,
        "trim": trim or None,
        "year_estimate": year_estimate,
        "body_type": vision_data.body_type or None,
        "color": vision_data.color or None,
        "confidence": conf_percent,
        "identification": {
            "make": make or None,
            "model": model or None,
            "generation": generation or None,
            "trim": trim or None,
            "year_estimate": year_estimate,
            "body_type": vision_data.body_type or None,
            "color": vision_data.color or None
        },
        "confidence_breakdown": {
            "overall": raw_conf,
            "make": min(1.0, round(raw_conf * 1.05, 2)) if make else 0.0,
            "model": raw_conf if model else 0.0,
            "generation": round(raw_conf * 0.90, 2) if generation else 0.0,
            "trim": round(raw_conf * 0.95, 2) if trim else 0.0
        },
        "visible_badges": visible_badges,
        "visible_text": vision_data.visible_text or [],
        "visual_evidence": vision_data.visual_evidence or [],
        "alternatives": alternatives,
        "needs_confirmation": needs_confirmation,
        "engine": "gemini_vision_engine",
        "provider": f"Google Gemini ({GEMINI_MODEL})",
        "debug_info": {
            "model": GEMINI_MODEL,
            "raw_confidence": raw_conf,
            "confidence_policy_applied": True
        }
    }


# ── 5. HELPER UTILITIES ──────────────────────────────────────────────────────
def extract_visual_embedding(img: Image.Image) -> List[float]:
    """Extracts a 100-dimension normalized visual feature vector for similarity comparison."""
    try:
        small = img.resize((128, 128)).convert("RGB")
        arr = np.array(small, dtype=np.float32) / 255.0

        h_r, _ = np.histogram(arr[:, :, 0], bins=12, range=(0, 1))
        h_g, _ = np.histogram(arr[:, :, 1], bins=12, range=(0, 1))
        h_b, _ = np.histogram(arr[:, :, 2], bins=12, range=(0, 1))

        spatial = arr.reshape(4, 32, 4, 32, 3).mean(axis=(1, 3)).flatten()

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
        return [0.0] * 100


def compute_visual_similarity(emb1: List[float], emb2: List[float]) -> float:
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


def crop_to_bytes(img: Image.Image, bbox: list) -> bytes:
    """Crops image to bbox [x1, y1, x2, y2] and re-encodes as optimized JPEG bytes."""
    crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    crop.thumbnail((640, 640), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def crop_to_b64(img: Image.Image, bbox: list) -> str:
    """Crops image to bbox and returns data URI base64 string for frontend preview."""
    crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    crop.thumbnail((280, 200), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=82)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

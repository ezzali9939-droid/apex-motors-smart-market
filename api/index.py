import os
import re
import io
import sys
import time
import hashlib
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from flask import Flask, request, jsonify, send_from_directory, Response
import requests
from bs4 import BeautifulSoup
import numpy as np

# Add project root directory to sys.path for services imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from services.vision import (
    prepare_image, detect_vehicle_rois, crop_to_bytes, crop_to_b64,
    extract_visual_embedding, compute_visual_similarity, run_vision_pipeline
)
from services.knowledge_base import (
    resolve_vehicle_specs, parse_search_query, normalize_brand, VEHICLE_KNOWLEDGE_BASE
)
from services.marketplace import (
    normalize_listing, apply_marketplace_filters
)

try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None
    ImageOps = None

app = Flask(__name__)

DETAIL_URL_PATTERN = re.compile(r'/(?:car|new-car)/[^\?#]*?\d{5,}$', re.IGNORECASE)
ARABIC_TO_ENGLISH_DIGITS = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669", "0123456789")

def normalize_digits(text: str) -> str:
    return (text or "").translate(ARABIC_TO_ENGLISH_DIGITS)

def is_valid_vehicle_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    if "hatla2ee.com" not in parsed.netloc.lower():
        return False
    return bool(DETAIL_URL_PATTERN.search(parsed.path)) and "teraz/" not in parsed.path.lower()

# ── AUTOCOMPLETE CATALOG ──────────────────────────────────────────────────────
_CATALOG = [
    ("Toyota","Corolla",[2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Toyota","Yaris",[2020,2021,2022,2023,2024],"Hatchback"),
    ("Toyota","Camry",[2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Toyota","Fortuner",[2016,2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Toyota","Land Cruiser",[2015,2016,2017,2018,2019,2020,2021],"SUV"),
    ("Toyota","Hilux",[2016,2017,2018,2019,2020,2021,2022],"Pickup"),
    ("Toyota","C-HR",[2018,2019,2020,2021,2022],"SUV"),
    ("Toyota","Rush",[2018,2019,2020,2021,2022,2023],"SUV"),
    ("Kia","Sportage",[2019,2020,2021,2022,2023,2024],"SUV"),
    ("Kia","Cerato",[2018,2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Kia","Picanto",[2017,2018,2019,2020,2021,2022,2023],"Hatchback"),
    ("Kia","Sorento",[2016,2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Kia","Rio",[2017,2018,2019,2020,2021,2022],"Sedan"),
    ("Kia","Stinger",[2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Kia","K5",[2020,2021,2022,2023,2024],"Sedan"),
    ("Hyundai","Elantra",[2018,2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Hyundai","Tucson",[2016,2017,2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("Hyundai","Accent",[2018,2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Hyundai","i10",[2018,2019,2020,2021,2022,2023],"Hatchback"),
    ("Hyundai","i20",[2018,2019,2020,2021,2022,2023],"Hatchback"),
    ("Hyundai","Santa Fe",[2016,2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Hyundai","Sonata",[2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Hyundai","Creta",[2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("Mercedes-Benz","C 180",[2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Mercedes-Benz","C 200",[2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Mercedes-Benz","E 200",[2017,2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Mercedes-Benz","GLC 200",[2016,2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Mercedes-Benz","AMG GT",[2015,2016,2017,2018,2019,2020,2021,2022,2023],"Coupe"),
    ("BMW","320i",[2018,2019,2020,2021,2022,2023,2024],"Sedan"),
    ("BMW","318i",[2019,2020,2021,2022,2023,2024],"Sedan"),
    ("BMW","X5",[2019,2020,2021,2022,2023,2024],"SUV"),
    ("BMW","520i",[2017,2018,2019,2020,2021,2022,2023],"Sedan"),
    ("BMW","X1",[2016,2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Nissan","Sunny",[2017,2018,2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Nissan","Sentra",[2013,2014,2015,2016,2017,2018,2019,2020],"Sedan"),
    ("Nissan","Qashqai",[2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Audi","A4",[2016,2017,2018,2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Audi","Q5",[2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("Volkswagen","Golf",[2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"Hatchback"),
    ("Volkswagen","Tiguan",[2017,2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("MG","ZS",[2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("MG","MG5",[2020,2021,2022,2023,2024],"Sedan"),
]

def _build_autocomplete():
    entries = []
    for make, model, years, body_type in _CATALOG:
        entries.append({
            "make": make, "model": model, "years": years, "body_type": body_type,
            "display": f"{make} {model}", "search_key": f"{make} {model}".lower(), "year": None
        })
        for yr in sorted(years, reverse=True)[:6]:
            entries.append({
                "make": make, "model": model, "years": years, "body_type": body_type,
                "display": f"{make} {model} {yr}", "search_key": f"{make} {model} {yr}".lower(), "year": yr
            })
    return entries

AUTOCOMPLETE_DATA = _build_autocomplete()

# ── MARKETPLACE SCRAPER & TTL CACHE ───────────────────────────────────────────
class DualPlatformMarketScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
        }
        self.base_url = "https://eg.hatla2ee.com"

    def scrape_hatla2ee(self, brand: str, model: str = None, page: int = 1) -> list:
        records = []
        if not brand:
            return []
        clean_b = brand.lower().strip()
        clean_m = model.lower().strip() if model else ""
        target_url = f"{self.base_url}/ar/car/{clean_b}"
        if clean_m:
            target_url += f"/{clean_m.replace(' ', '-')}"
        if page > 1:
            target_url += f"/page/{page}"
        try:
            resp = self.session.get(target_url, headers=self.headers, timeout=6)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.content, "html.parser")
            records_by_url = {}
            cards = soup.find_all(lambda tag: tag.name in ["div","article","section"] and tag.get("class") and "bg-card" in tag.get("class"))
            if not cards:
                cards = soup.find_all(lambda tag: tag.name in ["div","article","section"] and tag.get("class") and any("unit" in c.lower() or "card" in c.lower() for c in tag.get("class")))
            for card in cards:
                try:
                    detail_a = None
                    for a in card.find_all("a", href=True):
                        href = a["href"].strip()
                        if DETAIL_URL_PATTERN.search(href) and "teraz/" not in href.lower():
                            detail_a = a
                            text = a.get_text(strip=True)
                            if text and not any(k in text.lower() for k in ["slide","previous","next","عرض الكل"]):
                                break
                    if not detail_a:
                        continue
                    raw_href = detail_a["href"].strip()
                    full_link = urljoin(self.base_url, raw_href)
                    if full_link in records_by_url:
                        continue
                    title_text = ""
                    for a in card.find_all("a", href=True):
                        t = a.get_text(strip=True)
                        if t and not any(k in t.lower() for k in ["slide","previous","next","عرض الكل"]):
                            title_text = t
                            break
                    image_url = None
                    for img in card.find_all("img"):
                        src = img.get("src") or img.get("data-src") or ""
                        if "listing_image" in src or "hatla2ee.com/listing" in src:
                            image_url = src
                            break
                    if not image_url:
                        for img in card.find_all("img"):
                            src = img.get("src") or img.get("data-src") or ""
                            if src and not any(x in src for x in ["logo","icon","agency"]):
                                image_url = src
                                break
                    card_text_space = normalize_digits(card.get_text(" ", strip=True))
                    card_text_bar = normalize_digits(card.get_text(" | ", strip=True))
                    price = None
                    p_match = re.search(r'([\d,]{4,12})\s*(?:جنيه|EGP|ج\.م|L\.E)', card_text_space)
                    if p_match:
                        try:
                            p_val = float(p_match.group(1).replace(",","").replace(" ",""))
                            if p_val > 10000:
                                price = p_val
                        except ValueError:
                            pass
                    year = None
                    y_match = re.search(r"\b(19\d{2}|20\d{2})\b", card_text_space)
                    if y_match:
                        year = int(y_match.group(1))
                    mileage = None
                    km_match = re.search(r"([\d,]{1,8})\s*(?:کم|كم|km|كيلومتر|كيلو)", card_text_space, re.IGNORECASE)
                    if km_match:
                        try:
                            mileage = float(km_match.group(1).replace(",","").strip())
                        except ValueError:
                            pass
                    elif re.search(r"\b0\s*(?:کم|كم|km)\b", card_text_space, re.IGNORECASE):
                        mileage = 0.0
                    transmission = "Manual" if any(t in card_text_space for t in ["يدوي","مانيوال","Manual"]) else "Automatic"
                    fuel_type = "Benzine"
                    if "هجين" in card_text_space or "Hybrid" in card_text_space:
                        fuel_type = "Hybrid"
                    elif "كهرباء" in card_text_space or "Electric" in card_text_space:
                        fuel_type = "Electric"
                    elif "غاز" in card_text_space or "Gas" in card_text_space:
                        fuel_type = "Gas"
                    elif "ديزل" in card_text_space or "Diesel" in card_text_space:
                        fuel_type = "Diesel"
                    condition_tag = "Fabrika" if "فابريكا" in card_text_space else "Used"
                    
                    location = "Cairo"
                    known_locs = ["القاهرة","الجيزة","الإسكندرية","التجمع","المهندسين","دمياط","منوفية","الشرقية","الدقهلية","الغربية","أسيوط","سوهاج","المنيا","بني سويف","الفيوم","إسماعيلية","السويس","بورسعيد"]
                    for tok in [t.strip() for t in card_text_bar.split("|") if t.strip()]:
                        if any(loc in tok for loc in known_locs):
                            location = tok
                            break
                    rec_title = title_text if len(title_text) >= 3 else f"{brand.title()} {(model or '').title()} {year or ''}".strip()
                    records_by_url[full_link] = {
                        "name": rec_title, "brand": brand.title(),
                        "model": (model or "Model").title(),
                        "price": price, "year": year,
                        "mileage": mileage, "location": location,
                        "transmission": transmission, "fuel_type": fuel_type,
                        "car_condition": "New" if mileage == 0 else "Used",
                        "condition_tag": condition_tag,
                        "source": "Hatla2ee", "item_url": full_link, "image_url": image_url
                    }
                except Exception:
                    pass
            records = list(records_by_url.values())
        except Exception as e:
            print("Scraping Exception:", e)
        return records

live_engine = DualPlatformMarketScraper()

# ── IN-MEMORY TTL CACHE FOR MARKET SCRAPING ──────────────────────────────────
_SCRAPE_CACHE: Dict[str, Tuple[float, list]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes cache

def cached_scrape_hatla2ee(brand: str, model: str = None, page: int = 1) -> list:
    cache_key = f"{brand.lower()}:{model.lower() if model else ''}:{page}"
    now = time.time()
    if cache_key in _SCRAPE_CACHE:
        timestamp, records = _SCRAPE_CACHE[cache_key]
        if now - timestamp < CACHE_TTL_SECONDS:
            return records
    records = live_engine.scrape_hatla2ee(brand, model, page)
    _SCRAPE_CACHE[cache_key] = (now, records)
    return records

# ── MARKET PRICE STATISTICS ENGINE (NO FAKE VALUATIONS) ──────────────────────
def compute_market_price_stats(ads: list) -> Dict[str, Any]:
    prices = [float(r["price"]) for r in ads if r.get("price") and float(r["price"]) > 0]
    if not prices or len(prices) == 0:
        return {
            "status": "INSUFFICIENT_MARKET_DATA",
            "sample_size": 0,
            "minimum_price": None,
            "maximum_price": None,
            "median_price": None,
            "average_price": None,
            "currency": "EGP",
            "source": "Hatla2ee Egyptian Used Vehicle Market"
        }
    if len(prices) < 2:
        val = int(prices[0])
        return {
            "status": "INSUFFICIENT_MARKET_DATA",
            "sample_size": len(prices),
            "minimum_price": val,
            "maximum_price": val,
            "median_price": val,
            "average_price": val,
            "currency": "EGP",
            "source": "Hatla2ee Egyptian Used Vehicle Market"
        }
    sorted_p = sorted(prices)
    n = len(sorted_p)
    median = int(sorted_p[n // 2]) if n % 2 != 0 else int((sorted_p[n // 2 - 1] + sorted_p[n // 2]) / 2)
    return {
        "status": "VERIFIED_MARKET_DATA",
        "sample_size": n,
        "minimum_price": int(min(sorted_p)),
        "maximum_price": int(max(sorted_p)),
        "median_price": median,
        "average_price": int(sum(sorted_p) / n),
        "currency": "EGP",
        "source": "Hatla2ee Egyptian Used Vehicle Market"
    }

def calculate_match_score(query: str, item_name: str, brand: str, model: str, year: Optional[int]) -> Optional[float]:
    if not query:
        return None
    q_norm = normalize_digits(query.lower().strip())
    if q_norm in {"kia","toyota","mercedes","hyundai","bmw","nissan","audi"}:
        return None
    q_tokens = set(re.findall(r"\w+", q_norm))
    t_tokens = set(re.findall(r"\w+", normalize_digits(f"{item_name} {brand} {model} {year or ''}".lower())))
    if not q_tokens:
        return None
    overlap = len(q_tokens & t_tokens)
    score = (overlap / len(q_tokens)) * 100.0
    if brand and brand.lower() in q_norm:
        score = max(score, 88.0)
    if model and model.lower() in q_norm:
        score = max(score, 94.0)
    if year and str(year) in q_norm:
        score = min(score + 4.0, 99.8)
    return round(min(score, 99.8), 1)

def process_search_results(ads: list, query: str = "", market_stats: Dict[str, Any] = None) -> list:
    if not ads:
        return []
    
    med_price = market_stats.get("median_price") if market_stats else None
    
    out = []
    for idx, r in enumerate(ads):
        url = r.get("item_url")
        valid_url = is_valid_vehicle_url(url)
        price_val = float(r["price"]) if r.get("price") and float(r["price"]) > 0 else None
        
        # Calculate real market relative deal label (no fake depreciation)
        deal_label = None
        if price_val and med_price and med_price > 0:
            pct = (price_val - med_price) / med_price
            if pct <= -0.08:
                deal_label = "Great Deal 🔥"
            elif pct >= 0.12:
                deal_label = "Overpriced ⚠️"
            else:
                deal_label = "Fair Market Price ⚖️"

        out.append({
            "name": str(r.get("name","Vehicle")),
            "brand": str(r.get("brand","")),
            "model": str(r.get("model","")),
            "price": price_val,
            "predicted_fair_price": med_price,
            "valuation_source": "Market Median" if med_price else None,
            "deal_label": deal_label,
            "year": int(r["year"]) if r.get("year") else None,
            "mileage": float(r["mileage"]) if r.get("mileage") is not None else None,
            "location": str(r.get("location","Cairo")),
            "transmission": str(r.get("transmission","Automatic")),
            "fuel_type": str(r.get("fuel_type","Benzine")),
            "match_score": calculate_match_score(query, str(r.get("name","")), str(r.get("brand","")), str(r.get("model","")), int(r["year"]) if r.get("year") else None),
            "item_url": url if valid_url else None,
            "has_valid_url": valid_url,
            "image_url": r.get("image_url")
        })
    return out

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Apex Motors API",
        "valuation_mode": "Market-Based Verified Pricing",
        "vision_provider": "Google Gemini Vision API (gemini-3.8-flash)"
    })

@app.route("/api/suggest", methods=["GET"])
def suggest():
    raw = request.args.get("q","").strip()
    if len(raw) < 1:
        return jsonify({"suggestions":[]})
    q_norm = normalize_digits(raw).lower().strip()
    q_words = re.findall(r"\w+", q_norm)
    if not q_words:
        return jsonify({"suggestions":[]})
    results = []; seen = set()
    for entry in AUTOCOMPLETE_DATA:
        key = entry["search_key"]
        all_match = all(any(kw.startswith(w) for kw in key.split()) or w in key for w in q_words)
        if all_match:
            score = 100 if key.startswith(q_norm) else (80 if q_norm in key else 60)
            display = entry["display"]
            if display not in seen:
                seen.add(display)
                results.append({"display":display,"make":entry["make"],"model":entry["model"],"year":entry["year"],"body_type":entry["body_type"],"score":score})
    results.sort(key=lambda x: (-x["score"],-(x["year"] or 0)))
    final = results[:8]
    for r in final:
        del r["score"]
    return jsonify({"suggestions":final,"query":raw})

# ── CONSOLIDATED SPECIFICATION LOOKUP (SINGLE SOURCE OF TRUTH) ────────────────
@app.route("/api/vehicle-specs", methods=["GET"])
def vehicle_specs():
    make = request.args.get("make","").strip()
    model = request.args.get("model","").strip()
    year_raw = request.args.get("year","").strip()
    trim = request.args.get("trim","").strip() or None
    year = int(year_raw) if year_raw.isdigit() else None

    if not make or not model:
        return jsonify({
            "success": False,
            "specs_status": "not_verified",
            "error": "make and model query parameters are required"
        }), 400

    resolved = resolve_vehicle_specs(make, model, year=year, trim=trim)
    
    if not resolved.get("found"):
        return jsonify({
            "success": False,
            "specs_status": "not_verified",
            "make": make,
            "model": model,
            "year_requested": year,
            "error": f"Verified specifications for '{make} {model}' are not available in the database.",
            "spec_source": "Unconfirmed Specification Lookup"
        }), 200

    resolved["success"] = True
    resolved["specs_status"] = "verified"
    return jsonify(resolved), 200

@app.route("/api/image-proxy", methods=["GET"])
def image_proxy():
    raw_url = request.args.get("url","").strip()
    if not raw_url:
        return "Missing url parameter", 400
    ALLOWED_HOSTS = {"legion-images.hatla2ee.com","img.hatla2ee.com","cdn.hatla2ee.com"}
    try:
        parsed = urlparse(raw_url)
        if parsed.scheme not in ("http","https"):
            return "Forbidden", 403
        if parsed.netloc not in ALLOWED_HOSTS:
            return "Forbidden", 403
    except Exception:
        return "Bad URL", 400
    try:
        upstream = requests.get(raw_url, timeout=6, stream=True)
        content_type = upstream.headers.get("Content-Type","image/jpeg")
        if not content_type.startswith("image/"):
            return "Not an image", 400
        return Response(upstream.content, status=upstream.status_code,
                        headers={"Content-Type":content_type,"Access-Control-Allow-Origin":"*","Cache-Control":"public, max-age=3600"})
    except Exception as e:
        print("Image proxy error:", e)
        return "Upstream error", 502

# ── VISION HEALTH DIAGNOSTIC ──────────────────────────────────────────────────
@app.route("/api/vision-health", methods=["GET"])
def vision_health():
    gemini_key = os.environ.get("GEMINI_API_KEY")
    status_str = "ready" if gemini_key else "unconfigured"
    active_provider = "Google Gemini Vision API (gemini-3.8-flash)" if gemini_key else "None (Unconfigured)"

    return jsonify({
        "service": "vehicle-vision",
        "status": status_str,
        "active_provider": active_provider,
        "model": "gemini-3.8-flash",
        "env_vars": {
            "GEMINI_API_KEY": "configured" if gemini_key else "missing"
        },
        "pil_available": HAS_PIL
    })

# ── IMAGE CLASSIFICATION (GEMINI DRIVEN) ──────────────────────────────────────
@app.route("/api/classify", methods=["POST"])
def classify_image():
    # 1. PIL presence check
    if not HAS_PIL or Image is None:
        return jsonify({
            "success": False,
            "vehicle_detected": False,
            "code": "INVALID_IMAGE",
            "error": "Server-side image processing library unavailable (Pillow not installed)."
        }), 500

    # 2. Upload presence check
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "vehicle_detected": False,
            "code": "INVALID_IMAGE",
            "error": "No image file received. The field name must be 'image'."
        }), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "vehicle_detected": False,
            "code": "INVALID_IMAGE",
            "error": "Empty file upload received."
        }), 400

    original_bytes = file.read()
    file.seek(0)
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()

    selected_crop_id_raw = request.form.get("selected_crop_id")
    selected_crop_id = int(selected_crop_id_raw) if (selected_crop_id_raw is not None and selected_crop_id_raw.isdigit()) else None

    # 3. Image decode & EXIF transpose & size check
    pil_img, prep_error = prepare_image(file)
    if prep_error:
        return jsonify({
            "success": False,
            "vehicle_detected": False,
            "code": "INVALID_IMAGE",
            "error": prep_error
        }), 400

    # 4. Vehicle ROI & Non-car Quality Rejection
    detection_res = detect_vehicle_rois(pil_img)
    if not detection_res.get("is_car"):
        return jsonify({
            "success": False,
            "vehicle_detected": False,
            "code": "NON_CAR_IMAGE",
            "error": detection_res.get("reason", "The uploaded image does not contain a motor vehicle.")
        }), 422

    rois = detection_res.get("rois", [])

    # Multi-vehicle crop selection flow
    if len(rois) > 1 and selected_crop_id is None:
        vehicle_crops = []
        for r_item in rois:
            c_id = r_item["crop_id"]
            bbox = r_item["bbox"]
            vehicle_crops.append({
                "crop_id": c_id,
                "bbox": bbox,
                "preview_b64": crop_to_b64(pil_img, bbox)
            })
        return jsonify({
            "success": True,
            "vehicle_detected": True,
            "multiple_vehicles_detected": True,
            "vehicle_count": len(vehicle_crops),
            "vehicles": vehicle_crops,
            "message": "Multiple vehicles detected in photo. Please select which vehicle to analyze."
        }), 200

    target_roi = rois[0] if (selected_crop_id is None or selected_crop_id >= len(rois)) else rois[selected_crop_id]
    target_bbox = target_roi["bbox"]

    img_bytes = crop_to_bytes(pil_img, target_bbox)
    cropped_pil = pil_img.crop((target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]))

    gemini_sha256 = hashlib.sha256(img_bytes).hexdigest()
    
    # DEBUG SAVE - DEVELOPMENT ONLY
    try:
        with open("debug_last_gemini_input.jpg", "wb") as df:
            df.write(img_bytes)
    except Exception as e:
        print("Could not write debug image:", e)

    # 5. Execute Gemini Multi-Stage Recognition Pipeline
    vision_res = run_vision_pipeline(cropped_pil, img_bytes, target_bbox)

    # Add diagnostic fields
    vision_res["diagnostics"] = {
        "original_sha256": original_sha256,
        "gemini_sha256": gemini_sha256
    }

    if not vision_res.get("success") or not vision_res.get("vehicle_detected"):
        status_code = vision_res.get("status_code", 422)
        return jsonify(vision_res), status_code

    return jsonify(vision_res), 200

# ── LIVE MARKETPLACE SEARCH ───────────────────────────────────────────────────
@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q","").strip()
    page = int(request.args.get("page", 1))

    def _float(k):
        v = request.args.get(k)
        try: return float(v) if v else None
        except: return None

    def _int(k):
        v = request.args.get(k)
        try: return int(v) if v else None
        except: return None

    filter_params = {
        "min_price": _float("min_price"), "max_price": _float("max_price"),
        "min_year": _int("min_year"), "max_year": _int("max_year"),
        "mileage_preset": request.args.get("mileage_preset") or None,
        "min_mileage": _float("min_mileage"), "max_mileage": _float("max_mileage"),
        "transmission": request.args.get("transmission") or None,
        "fuel_type": request.args.get("fuel_type") or None,
        "body_type": request.args.get("body_type") or None,
    }
    has_filters = any(v is not None for v in filter_params.values())
    
    if not query:
        return jsonify({
            "query": "", "brand": "", "model": "", "page": page,
            "has_more": False, "results": [], "filters_active": False,
            "price_stats": compute_market_price_stats([])
        })

    parsed = parse_search_query(query)
    detected_brand = parsed["make"]
    detected_model = parsed["model"]
    
    if parsed["year"] and not filter_params.get("min_year") and not filter_params.get("max_year"):
        filter_params["min_year"] = parsed["year"]
        filter_params["max_year"] = parsed["year"]
        has_filters = True

    all_raw = []
    seen_urls = set()
    max_pages = 2

    for extra_page in range(page, page + max_pages):
        batch = cached_scrape_hatla2ee(detected_brand, detected_model, page=extra_page)
        if not batch:
            break
        for item in batch:
            u = item.get("item_url")
            if u not in seen_urls:
                seen_urls.add(u)
                all_raw.append(item)
        if len(apply_marketplace_filters(all_raw, filter_params)) >= 24:
            break

    filtered = apply_marketplace_filters(all_raw, filter_params)
    broader_matches = False
    
    # If filters resulted in 0 items, broaden matching safely
    if not filtered and all_raw:
        filtered = all_raw[:24]
        broader_matches = True

    price_stats = compute_market_price_stats(filtered)
    formatted = process_search_results(filtered[:24], query=query, market_stats=price_stats)

    return jsonify({
        "query": query,
        "brand": detected_brand or "",
        "model": detected_model or "",
        "parsed_year": parsed["year"],
        "page": page,
        "has_more": len(all_raw) >= 20,
        "total_loaded": len(formatted),
        "filters_active": has_filters,
        "broader_matches": broader_matches,
        "price_stats": price_stats,
        "results": formatted
    })

@app.route("/", methods=["GET"])
def serve_index():
    public_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
    return send_from_directory(public_dir, "index.html")

@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    public_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
    if os.path.exists(os.path.join(public_dir, path)):
        return send_from_directory(public_dir, path)
    return send_from_directory(public_dir, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

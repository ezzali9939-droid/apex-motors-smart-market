import os
import re
import io
import sys
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify, send_from_directory
import requests
from bs4 import BeautifulSoup
import numpy as np
import joblib

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


# ── Valuation engine stub ────────────────────────────────────────────────────
class ApexProductionValuationEngine:
    def __init__(self, model, num_cols, cat_cols, medians):
        self.model = model
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.medians = medians

sys.modules['__main__'].ApexProductionValuationEngine = ApexProductionValuationEngine

try:
    from catboost import Pool, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    Pool = None
    CatBoostRegressor = None

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

DETAIL_URL_PATTERN = re.compile(r'/(?:car|new-car)/[^\?#]*?\d{5,}$', re.IGNORECASE)
ARABIC_TO_ENGLISH_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_digits(text: str) -> str:
    if not text:
        return ""
    return text.translate(ARABIC_TO_ENGLISH_DIGITS)


def is_valid_vehicle_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    if "hatla2ee.com" not in parsed.netloc.lower():
        return False
    return bool(DETAIL_URL_PATTERN.search(parsed.path)) and "teraz/" not in parsed.path.lower()


# ── Market scraper ───────────────────────────────────────────────────────────
class DualPlatformMarketScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
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
            resp = self.session.get(target_url, headers=self.headers, timeout=12)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                records_by_url = {}

                cards = soup.find_all(
                    lambda tag: tag.name in ['div', 'article', 'section']
                    and tag.get('class')
                    and 'bg-card' in tag.get('class')
                )
                if not cards:
                    cards = soup.find_all(
                        lambda tag: tag.name in ['div', 'article', 'section']
                        and tag.get('class')
                        and any('unit' in c.lower() or 'card' in c.lower() for c in tag.get('class'))
                    )

                for card in cards:
                    try:
                        detail_a = None
                        for a in card.find_all("a", href=True):
                            href = a["href"].strip()
                            if DETAIL_URL_PATTERN.search(href) and "teraz/" not in href.lower():
                                detail_a = a
                                text = a.get_text(strip=True)
                                if text and not any(k in text.lower() for k in ["slide", "previous", "next", "عرض الكل"]):
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
                            if t and not any(k in t.lower() for k in ["slide", "previous", "next", "عرض الكل"]):
                                title_text = t
                                break

                        # ── Image extraction ─────────────────────────────────
                        image_url = None
                        for img in card.find_all("img"):
                            src = img.get("src") or img.get("data-src") or ""
                            if "listing_image" in src or "hatla2ee.com/listing" in src:
                                image_url = src
                                break
                        if not image_url:
                            for img in card.find_all("img"):
                                src = img.get("src") or img.get("data-src") or ""
                                if src and not any(x in src for x in ["logo", "icon", "agency"]):
                                    image_url = src
                                    break

                        card_text_space = normalize_digits(card.get_text(" ", strip=True))
                        card_text_bar   = normalize_digits(card.get_text(" | ", strip=True))

                        # ── Price ─────────────────────────────────────────────
                        price = None
                        p_match = re.search(r'([\d,]{4,12})\s*(?:جنيه|EGP|ج\.م|L\.E)', card_text_space)
                        if p_match:
                            try:
                                p_val = float(p_match.group(1).replace(',', '').replace(' ', ''))
                                if p_val > 10000:
                                    price = p_val
                            except ValueError:
                                price = None

                        # ── Year ──────────────────────────────────────────────
                        year = None
                        y_match = re.search(r'\b(19\d{2}|20\d{2})\b', card_text_space)
                        if y_match:
                            year = int(y_match.group(1))

                        # ── Mileage (None = unknown, 0.0 = zero km) ──────────
                        mileage = None
                        km_match = re.search(r'([\d,]{1,8})\s*(?:کم|كم|km|كيلومتر|كيلو)', card_text_space, re.IGNORECASE)
                        if km_match:
                            try:
                                km_str = km_match.group(1).replace(',', '').replace(' ', '').strip()
                                mileage = float(km_str)
                            except ValueError:
                                mileage = None
                        elif re.search(r'\b0\s*(?:کم|كم|km)\b', card_text_space, re.IGNORECASE):
                            mileage = 0.0

                        transmission = "Automatic"
                        if any(t in card_text_space for t in ["يدوي", "مانيوال", "Manual"]):
                            transmission = "Manual"

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

                        # ── Body type (best-effort from listing text) ─────────
                        body_type = None
                        name_lower = title_text.lower()
                        if any(k in name_lower for k in ["suv", "كروس", "cross", "سبورت", "sport"]):
                            body_type = "SUV"
                        elif any(k in name_lower for k in ["هاتش", "hatch"]):
                            body_type = "Hatchback"
                        elif any(k in name_lower for k in ["سيدان", "sedan", "saloon"]):
                            body_type = "Sedan"
                        elif any(k in name_lower for k in ["بيك اب", "pickup", "truck"]):
                            body_type = "Pickup"
                        elif any(k in name_lower for k in ["ميني فان", "van", "minivan"]):
                            body_type = "Van"
                        elif any(k in name_lower for k in ["كوبيه", "coupe"]):
                            body_type = "Coupe"

                        tokens = [t.strip() for t in card_text_bar.split('|') if t.strip()]
                        location = "Cairo"
                        known_locs = [
                            "القاهرة", "الجيزة", "الإسكندرية", "التجمع", "المهندسين",
                            "دمياط", "منوفية", "الشرقية", "الدقهلية", "الغربية", "أسيوط",
                            "سوهاج", "المنيا", "بني سويف", "الفيوم", "إسماعيلية",
                            "السويس", "بورسعيد"
                        ]
                        for tok in tokens:
                            if any(loc in tok for loc in known_locs):
                                location = tok
                                break

                        rec_title = (
                            title_text if len(title_text) >= 3
                            else f"{brand.title()} {model.title() if model else ''} {year or ''}".strip()
                        )

                        records_by_url[full_link] = {
                            "name": rec_title,
                            "brand": brand.title(),
                            "model": model.title() if model else "Model",
                            "price": price,
                            "year": year if year else 2024,
                            "mileage": mileage,
                            "location": location,
                            "transmission": transmission,
                            "fuel_type": fuel_type,
                            "body_type": body_type,
                            "car_condition": "New" if mileage == 0 else "Used",
                            "condition_tag": condition_tag,
                            "trim_tier": "Topline",
                            "source": "Hatla2ee Market",
                            "item_url": full_link,
                            "image_url": image_url
                        }
                    except Exception:
                        pass

                records = list(records_by_url.values())
        except Exception as e:
            print("Scraping Exception:", e)

        return records


live_engine = DualPlatformMarketScraper()

# ── Load CatBoost model (optional) ──────────────────────────────────────────
val_engine = None
cb_model_obj = None

if HAS_CATBOOST:
    if os.path.exists("catboost_model.cbm"):
        try:
            cb_model_obj = CatBoostRegressor()
            cb_model_obj.load_model("catboost_model.cbm")
        except Exception:
            cb_model_obj = None

    if cb_model_obj is None and os.path.exists("apex_catboost_valuation.joblib"):
        try:
            val_engine = joblib.load("apex_catboost_valuation.joblib")
            cb_model_obj = getattr(val_engine, 'model', None)
        except Exception:
            val_engine = None


# ── Server-side filter application ───────────────────────────────────────────
def apply_filters(ads: list, params: dict) -> list:
    """
    Filter scraped listings by user-supplied criteria.
    Missing mileage (None) is never treated as zero — excluded from zero/range filters.
    """
    min_price      = params.get('min_price')
    max_price      = params.get('max_price')
    min_year       = params.get('min_year')
    max_year       = params.get('max_year')
    mileage_preset = params.get('mileage_preset')   # zero | under_1000 | under_100000
    min_mileage    = params.get('min_mileage')
    max_mileage    = params.get('max_mileage')
    f_transmission = params.get('transmission')
    f_fuel         = params.get('fuel_type')
    f_body         = params.get('body_type')

    out = []
    for r in ads:
        price    = r.get('price')
        year     = r.get('year')
        mileage  = r.get('mileage')   # None = unknown, 0.0 = zero km

        # ── Price ──────────────────────────────────────────────────────────
        if min_price is not None and price is not None and price < min_price:
            continue
        if max_price is not None and price is not None and price > max_price:
            continue

        # ── Year ───────────────────────────────────────────────────────────
        if min_year is not None and year is not None and year < min_year:
            continue
        if max_year is not None and year is not None and year > max_year:
            continue

        # ── Mileage preset (missing mileage is NEVER treated as zero) ──────
        if mileage_preset == 'zero':
            if mileage is None or mileage != 0.0:
                continue
        elif mileage_preset == 'under_1000':
            if mileage is None or mileage >= 1000:
                continue
        elif mileage_preset == 'under_100000':
            if mileage is None or mileage >= 100000:
                continue

        # ── Custom mileage range (skip if mileage unknown) ────────────────
        if min_mileage is not None:
            if mileage is None or mileage < min_mileage:
                continue
        if max_mileage is not None:
            if mileage is None or mileage > max_mileage:
                continue

        # ── Transmission ───────────────────────────────────────────────────
        if f_transmission and r.get('transmission', '').lower() != f_transmission.lower():
            continue

        # ── Fuel type ──────────────────────────────────────────────────────
        if f_fuel and r.get('fuel_type', '').lower() != f_fuel.lower():
            continue

        # ── Body type (best-effort) ────────────────────────────────────────
        if f_body and r.get('body_type'):
            if r['body_type'].lower() != f_body.lower():
                continue

        out.append(r)
    return out


# ── Match scoring ────────────────────────────────────────────────────────────
def calculate_match_score(query: str, item_name: str, brand: str, model: str, year) -> float | None:
    if not query:
        return None
    q_norm = normalize_digits(query.lower().strip())
    if q_norm in ["kia", "toyota", "mercedes", "hyundai", "bmw", "nissan", "audi",
                  "كيا", "تويوتا", "مرسيدس", "هيونداي"]:
        return None

    q_tokens = set(re.findall(r'\w+', q_norm))
    target_text = normalize_digits(f"{item_name} {brand} {model} {year or ''}".lower())
    t_tokens = set(re.findall(r'\w+', target_text))
    if not q_tokens:
        return None
    overlap = len(q_tokens.intersection(t_tokens))
    score = (overlap / len(q_tokens)) * 100.0
    if brand.lower() in q_norm:
        score = max(score, 88.0)
    if model.lower() in q_norm:
        score = max(score, 94.0)
    if year and str(year) in q_norm:
        score = min(score + 4.0, 99.8)
    return round(min(score, 99.8), 1)


# ── Rule-based price estimate ─────────────────────────────────────────────────
def predict_fallback_fair_price(brand, model, year, mileage, transmission='Automatic', condition_tag='Fabrika'):
    brand     = str(brand or '').lower().strip()
    model_str = str(model or '').lower().strip()
    year      = int(year) if year else None
    if year is None:
        return None

    mileage = float(mileage) if mileage is not None else 100000.0

    base_market_prices = {
        ('kia', 'sportage'): 2400000.0,   ('toyota', 'corolla'): 1650000.0,
        ('hyundai', 'tucson'): 2350000.0, ('mercedes', 'c180'): 3200000.0,
        ('bmw', '320i'): 3100000.0,       ('nissan', 'sunny'): 850000.0,
        ('hyundai', 'elantra'): 1400000.0,('kia', 'cerato'): 1300000.0,
        ('mg', 'mg'): 1200000.0,          ('renault', 'megane'): 1350000.0,
        ('chevrolet', 'optra'): 750000.0,
    }

    base_2026 = base_market_prices.get((brand, model_str))
    if not base_2026:
        brand_bases = {
            'mercedes': 3000000, 'bmw': 2900000, 'audi': 2800000,
            'kia': 1800000, 'hyundai': 1700000, 'toyota': 1750000, 'nissan': 900000
        }
        base_2026 = float(brand_bases.get(brand, 0))
        if base_2026 == 0:
            return None

    age = max(0, 2026 - year)
    depreciated = base_2026 * ((1.0 - 0.075) ** age)
    expected_km = age * 15000.0
    km_adj = -((mileage - expected_km) * 1.5)
    val = depreciated + km_adj
    if str(transmission).lower() == 'manual':
        val *= 0.93
    if str(condition_tag).lower() == 'fabrika':
        val *= 1.03
    return float(round(max(val, 150000.0), 0))


# ── Enrich results ───────────────────────────────────────────────────────────
def process_search_results(ads: list, query: str = "") -> list:
    if not ads:
        return []

    predicted_prices = [None] * len(ads)

    if HAS_CATBOOST and Pool is not None and cb_model_obj is not None:
        try:
            num_cols = ['year', 'mileage', 'car_age', 'km_per_year']
            cat_cols = ['brand', 'model', 'location', 'transmission', 'fuel_type', 'car_condition', 'condition_tag', 'trim_tier']
            medians  = {'year': 2016.0, 'mileage': 122000.0, 'car_age': 10.0, 'km_per_year': 11600.0}
            current_year = 2026
            data_matrix = []
            for r in ads:
                yr    = int(r.get('year')) if r.get('year') else 2024
                raw_km = r.get('mileage')
                km    = float(raw_km) if raw_km is not None else medians['mileage']
                age   = max(0, current_year - yr)
                km_py = km / max(1, age) if age > 0 else km
                row = [
                    float(yr), float(km), float(age), float(km_py),
                    str(r.get('brand', 'Kia')).title(),
                    str(r.get('model', 'Sportage')).title(),
                    str(r.get('location', 'Cairo')).title(),
                    str(r.get('transmission', 'Automatic')).title(),
                    str(r.get('fuel_type', 'Benzine')).title(),
                    "New" if raw_km == 0 else "Used",
                    str(r.get('condition_tag', 'Fabrika')).title(),
                    str(r.get('trim_tier', 'Topline')).title()
                ]
                data_matrix.append(row)

            cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
            pool = Pool(data=data_matrix, cat_features=cat_indices)
            preds_log = cb_model_obj.predict(pool)
            preds_egp = np.expm1(preds_log)
            predicted_prices = [
                float(np.round(p, 0)) if not np.isnan(p) and p > 0 else None
                for p in preds_egp
            ]
        except Exception as e:
            print("CatBoost valuation error:", e)
            predicted_prices = [None] * len(ads)

    valuation_source = "CatBoost" if (HAS_CATBOOST and cb_model_obj is not None) else "rule-based"

    for idx, r in enumerate(ads):
        if idx >= len(predicted_prices) or predicted_prices[idx] is None:
            fb = predict_fallback_fair_price(
                brand=r.get('brand'), model=r.get('model'),
                year=r.get('year'), mileage=r.get('mileage'),
                transmission=r.get('transmission'), condition_tag=r.get('condition_tag')
            )
            if idx < len(predicted_prices):
                predicted_prices[idx] = fb
            else:
                predicted_prices.append(fb)

    out_records = []
    for idx, r in enumerate(ads):
        url       = r.get("item_url")
        valid_url = is_valid_vehicle_url(url)
        price_val = float(r['price']) if r.get('price') and r['price'] > 0 else None
        fair_val  = predicted_prices[idx] if idx < len(predicted_prices) else None

        deal_label = None
        if fair_val and price_val:
            pct = (price_val - fair_val) / fair_val
            if pct <= -0.05:   deal_label = "Great Deal 🔥"
            elif pct >= 0.08:  deal_label = "Overpriced ⚠️"
            else:              deal_label = "Fair Market Price ⚖️"

        mileage_val = float(r['mileage']) if r.get('mileage') is not None else None

        out_records.append({
            "name":                str(r.get("name", "Vehicle")),
            "brand":               str(r.get("brand", "")),
            "model":               str(r.get("model", "")),
            "body_type":           r.get("body_type"),
            "price":               price_val,
            "predicted_fair_price":fair_val,
            "valuation_source":    valuation_source if fair_val is not None else None,
            "deal_label":          deal_label,
            "year":                int(r.get("year", 2024)) if r.get("year") else None,
            "mileage":             mileage_val,
            "location":            str(r.get("location", "Cairo")),
            "transmission":        str(r.get("transmission", "Automatic")),
            "fuel_type":           str(r.get("fuel_type", "Benzine")),
            "match_score":         calculate_match_score(
                                       query=query,
                                       item_name=str(r.get("name", "")),
                                       brand=str(r.get("brand", "")),
                                       model=str(r.get("model", "")),
                                       year=int(r.get("year")) if r.get("year") else None
                                   ),
            "item_url":            url if valid_url else None,
            "has_valid_url":       valid_url,
            "image_url":           r.get("image_url")
        })

    return out_records


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Apex Motors API",
        "catboost_loaded": cb_model_obj is not None,
        "valuation_mode": "CatBoost" if cb_model_obj is not None else "rule-based"
    })


@app.route("/api/classify", methods=["POST"])
def classify_image():
    if not HAS_PIL or Image is None:
        return jsonify({"success": False, "error": "Image processing unavailable."}), 500

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        image.verify()
        image = Image.open(io.BytesIO(img_bytes))
        image.thumbnail((640, 640))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        img_bytes = buf.getvalue()
    except Exception:
        return jsonify({"success": False, "error": "Invalid or corrupted image. Please upload a clear JPG/PNG car photo."}), 400

    try:
        hf_headers = {"Accept": "application/json", "Content-Type": "image/jpeg"}
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            hf_headers["Authorization"] = f"Bearer {hf_token}"

        api_url = "https://router.huggingface.co/hf-inference/v1/models/dima806/car_models_image_detection"
        hf_resp = requests.post(api_url, data=img_bytes, headers=hf_headers, timeout=10)

        if hf_resp.status_code == 200:
            res_json = hf_resp.json()
            if isinstance(res_json, list) and len(res_json) > 0:
                top   = res_json[0]
                label = top.get("label", "").replace("_", " ").strip()
                score = float(top.get("score", 0.0))

                if score >= 0.15 and label:
                    alternatives = []
                    for item in res_json[1:5]:
                        alt_label = item.get("label", "").replace("_", " ").strip()
                        alt_score = float(item.get("score", 0.0))
                        if alt_label and alt_score >= 0.06:
                            alternatives.append({"label": alt_label, "confidence": round(alt_score * 100, 1)})

                    return jsonify({
                        "success":      True,
                        "label":        label,
                        "confidence":   round(score * 100, 1),
                        "alternatives": alternatives,
                        "model":        "dima806/car_models_image_detection",
                        "note":         "Identification based on visual features only. Year, trim, and mechanical condition cannot be determined from appearance."
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error":   "Vehicle not identified with sufficient confidence. Try a clearer exterior photo.",
                        "raw_confidence": round(score * 100, 1)
                    }), 422

        elif hf_resp.status_code == 503:
            return jsonify({"success": False, "error": "Vision model is loading. Please retry in 20 seconds.", "retry": True}), 503
        else:
            return jsonify({"success": False, "error": f"Vision API returned status {hf_resp.status_code}. Try text search."}), 502

    except requests.Timeout:
        return jsonify({"success": False, "error": "Vision API timed out. Try again or use text search."}), 504
    except Exception as e:
        print("Image classification error:", e)
        return jsonify({"success": False, "error": "Image classification failed. Please use text search."}), 500


@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    page  = int(request.args.get("page", 1))

    # ── Filter params ────────────────────────────────────────────────────────
    def _float(key):
        v = request.args.get(key)
        try: return float(v) if v else None
        except: return None

    def _int(key):
        v = request.args.get(key)
        try: return int(v) if v else None
        except: return None

    filter_params = {
        'min_price':      _float('min_price'),
        'max_price':      _float('max_price'),
        'min_year':       _int('min_year'),
        'max_year':       _int('max_year'),
        'mileage_preset': request.args.get('mileage_preset') or None,
        'min_mileage':    _float('min_mileage'),
        'max_mileage':    _float('max_mileage'),
        'transmission':   request.args.get('transmission') or None,
        'fuel_type':      request.args.get('fuel_type') or None,
        'body_type':      request.args.get('body_type') or None,
    }
    has_filters = any(v is not None for v in filter_params.values())

    if not query:
        return jsonify({"query": "", "brand": "", "model": "", "page": page,
                        "has_more": False, "results": [], "filters_active": has_filters})

    q_lower = query.lower()

    brands_dict = {
        "kia": "kia", "كيا": "kia",
        "mercedes": "mercedes", "مرسيدس": "mercedes", "مرسيدس-بنز": "mercedes",
        "hyundai": "hyundai", "هيونداي": "hyundai",
        "toyota": "toyota", "تويوتا": "toyota",
        "bmw": "bmw", "بي ام": "bmw", "بي إم": "bmw", "بي ام دبليو": "bmw",
        "nissan": "nissan", "نيسان": "nissan",
        "audi": "audi", "أودي": "audi",
        "mitsubishi": "mitsubishi", "ميتسوبيشي": "mitsubishi",
        "chevrolet": "chevrolet", "شيفروليه": "chevrolet", "شفروليه": "chevrolet",
        "renault": "renault", "رينو": "renault",
        "peugeot": "peugeot", "بيجو": "peugeot",
        "mg": "mg", "ام جي": "mg", "إم جي": "mg",
        "chery": "chery", "شيري": "chery",
        "skoda": "skoda", "سكودا": "skoda",
        "volkswagen": "volkswagen", "فولكس": "volkswagen", "فولكس فاجن": "volkswagen", "vw": "volkswagen",
        "fiat": "fiat", "فيات": "fiat",
        "jeep": "jeep", "جيب": "jeep",
        "ford": "ford", "فورد": "ford",
        "honda": "honda", "هوندا": "honda",
        "mazda": "mazda", "مازدا": "mazda",
        "suzuki": "suzuki", "سوزوكي": "suzuki",
        "opel": "opel", "أوبل": "opel",
        "subaru": "subaru", "سوبارو": "subaru",
        "byd": "byd", "بي واي دي": "byd"
    }

    models_dict = {
        "sportage": "sportage", "سبورتاج": "sportage",
        "corolla": "corolla", "كورولا": "corolla",
        "tucson": "tucson", "توسان": "tucson",
        "c180": "c180", "c-class": "c180", "c 180": "c180", "cla": "cla", "e200": "e200",
        "sunny": "sunny", "صني": "sunny",
        "cerato": "cerato", "سيراتو": "cerato",
        "elantra": "elantra", "النترا": "elantra", "إلنترا": "elantra",
        "accent": "accent", "اكسنت": "accent", "أكسنت": "accent",
        "pegas": "pegas", "بيجاس": "pegas",
        "yaris": "yaris", "ياريس": "yaris",
        "fortuner": "fortuner", "فورتشنر": "fortuner",
        "320i": "320i", "320": "320i", "520i": "520i", "520": "520i",
        "megane": "megane", "ميجان": "megane",
        "lanos": "lanos", "لانس": "lanos",
        "optra": "optra", "أوبترا": "optra"
    }

    detected_brand = None
    detected_model = None

    for k, v in brands_dict.items():
        if k in q_lower:
            detected_brand = v
            break

    for k, v in models_dict.items():
        if k in q_lower:
            detected_model = v
            break

    if not detected_brand:
        words = [w for w in re.findall(r'\w+', q_lower) if not w.isdigit()]
        detected_brand = words[0] if words else query

    # ── Scrape with overflow to compensate for filtered-out items ────────────
    all_raw = []
    seen_urls: set = set()
    max_pages_to_try = 3 if has_filters else 2

    for extra_page in range(page, page + max_pages_to_try):
        batch = live_engine.scrape_hatla2ee(detected_brand, detected_model, page=extra_page)
        if not batch:
            break
        for item in batch:
            u = item.get("item_url")
            if u not in seen_urls:
                seen_urls.add(u)
                all_raw.append(item)
        filtered_so_far = apply_filters(all_raw, filter_params)
        if len(filtered_so_far) >= 24:
            break

    filtered = apply_filters(all_raw, filter_params)

    # ── Price range stats (before slicing) ───────────────────────────────────
    prices = [r['price'] for r in filtered if r.get('price')]
    price_stats = None
    if prices:
        price_stats = {
            "min": int(min(prices)),
            "max": int(max(prices)),
            "median": int(sorted(prices)[len(prices) // 2]),
            "count": len(prices)
        }

    filtered = filtered[:24]
    has_more = len(filtered) >= 20

    formatted = process_search_results(filtered, query=query)

    return jsonify({
        "query":          query,
        "brand":          detected_brand,
        "model":          detected_model or "",
        "page":           page,
        "has_more":       has_more,
        "total_loaded":   len(formatted),
        "filters_active": has_filters,
        "price_stats":    price_stats,
        "results":        formatted
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

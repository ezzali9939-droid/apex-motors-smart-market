import os
import re
import io
import sys
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import joblib

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

# Register ApexProductionValuationEngine in __main__ for joblib unpickling
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

try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

VISION_MODEL_NAME = "dima806/car_models_image_detection"
img_processor, car_vision_model = None, None

if HAS_TORCH:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
        car_vision_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL_NAME).to(device)
        car_vision_model.eval()
    except Exception:
        img_processor, car_vision_model = None, None

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

    def scrape_hatla2ee(self, brand: str, model: str = None) -> list:
        records = []
        if not brand:
            return []

        clean_b = brand.lower().strip()
        clean_m = model.lower().strip() if model else ""
        
        target_url = f"{self.base_url}/ar/car/{clean_b}"
        if clean_m:
            target_url += f"/{clean_m.replace(' ', '-')}"

        try:
            resp = self.session.get(target_url, headers=self.headers, timeout=12)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                records_by_url = {}

                # Look for top-level listing cards
                cards = soup.find_all(lambda tag: tag.name in ['div', 'article', 'section'] and tag.get('class') and 'bg-card' in tag.get('class'))
                if not cards:
                    cards = soup.find_all(lambda tag: tag.name in ['div', 'article', 'section'] and tag.get('class') and any('unit' in c.lower() or 'card' in c.lower() for c in tag.get('class')))

                for card in cards:
                    try:
                        # Find valid detail link
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

                        # Extract Title
                        title_text = ""
                        for a in card.find_all("a", href=True):
                            t = a.get_text(strip=True)
                            if t and not any(k in t.lower() for k in ["slide", "previous", "next", "عرض الكل"]):
                                title_text = t
                                break

                        card_text_space = normalize_digits(card.get_text(" ", strip=True))
                        card_text_bar = normalize_digits(card.get_text(" | ", strip=True))

                        # Listed Price
                        price = None
                        p_match = re.search(r'([\d,]{4,12})\s*(?:جنيه|EGP|ج\.م|L\.E)', card_text_space)
                        if p_match:
                            try:
                                p_val = float(p_match.group(1).replace(',', '').replace(' ', ''))
                                if p_val > 10000:
                                    price = p_val
                            except ValueError:
                                price = None

                        # Year
                        year = None
                        y_match = re.search(r'\b(19\d{2}|20\d{2})\b', card_text_space)
                        if y_match:
                            year = int(y_match.group(1))

                        # Mileage
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

                        # Transmission
                        transmission = "Automatic"
                        if any(t in card_text_space for t in ["يدوي", "مانيوال", "Manual"]):
                            transmission = "Manual"

                        # Fuel Type
                        fuel_type = "Benzine"
                        if "هجين" in card_text_space or "Hybrid" in card_text_space:
                            fuel_type = "Hybrid"
                        elif "كهرباء" in card_text_space or "Electric" in card_text_space:
                            fuel_type = "Electric"
                        elif "غاز" in card_text_space or "Gas" in card_text_space:
                            fuel_type = "Gas"

                        # Condition Tag
                        condition_tag = "Fabrika" if "فابريكا" in card_text_space else "Used"

                        # Location
                        tokens = [t.strip() for t in card_text_bar.split('|') if t.strip()]
                        location = "Cairo"
                        known_locs = ["القاهرة", "الجيزة", "الإسكندرية", "التجمع", "المهندسين", "دمياط", "منوفية", "الشرقية", "الدقهلية", "الغربية", "أسيوط", "سوهاج", "المنيا", "بني سويف", "الفيوم", "إسماعيلية", "السويس", "بورسعيد"]
                        for tok in tokens:
                            if any(loc in tok for loc in known_locs):
                                location = tok
                                break

                        rec_title = title_text if len(title_text) >= 3 else f"{brand.title()} {model.title() if model else ''} {year or ''}".strip()

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
                            "car_condition": "New" if mileage == 0 else "Used",
                            "condition_tag": condition_tag,
                            "trim_tier": "Topline",
                            "source": "Hatla2ee Market",
                            "item_url": full_link
                        }
                    except Exception:
                        pass

                records = list(records_by_url.values())
        except Exception as e:
            print("Scraping Exception:", e)

        return records

live_engine = DualPlatformMarketScraper()

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

def calculate_match_score(query: str, item_name: str, brand: str, model: str, year: int | None) -> float:
    if not query:
        return 0.0
    q_norm = normalize_digits(query.lower())
    q_tokens = set(re.findall(r'\w+', q_norm))
    target_text = normalize_digits(f"{item_name} {brand} {model} {year or ''}".lower())
    t_tokens = set(re.findall(r'\w+', target_text))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(t_tokens))
    score = (overlap / len(q_tokens)) * 100.0
    if brand.lower() in q_norm:
        score = max(score, 88.0)
    if model.lower() in q_norm:
        score = max(score, 94.0)
    if year and str(year) in q_norm:
        score = min(score + 4.0, 99.8)
    return round(min(score, 99.8), 1)

def process_search_results(df: pd.DataFrame, query: str = "") -> list:
    if df.empty:
        return []
    df = df.copy().reset_index(drop=True)

    predicted_prices = [None] * len(df)

    if HAS_CATBOOST and Pool is not None and cb_model_obj is not None:
        try:
            eval_df = df.copy()
            current_year = 2026
            
            years = pd.to_numeric(eval_df['year'], errors='coerce').fillna(2024)
            raw_mileages = pd.to_numeric(eval_df['mileage'], errors='coerce')
            mileages = raw_mileages.fillna(122000.0)
            
            eval_df['car_age'] = (current_year - years).clip(lower=0)
            eval_df['km_per_year'] = np.where(eval_df['car_age'] > 0, mileages / eval_df['car_age'].replace(0, 1), mileages)
            
            eval_df['fuel_type'] = eval_df.get('fuel_type', pd.Series(['Benzine']*len(df))).fillna('Benzine')
            eval_df['car_condition'] = np.where(raw_mileages == 0, 'New', 'Used')
            eval_df['condition_tag'] = eval_df.get('condition_tag', pd.Series(['Fabrika']*len(df))).fillna('Fabrika')
            eval_df['trim_tier'] = eval_df.get('trim_tier', pd.Series(['Topline']*len(df))).fillna('Topline')
            
            num_cols = ['year', 'mileage', 'car_age', 'km_per_year']
            cat_cols = ['brand', 'model', 'location', 'transmission', 'fuel_type', 'car_condition', 'condition_tag', 'trim_tier']
            medians = {'year': 2016.0, 'mileage': 122000.0, 'car_age': 10.0, 'km_per_year': 11600.0}

            for c in num_cols:
                eval_df[c] = pd.to_numeric(eval_df[c], errors="coerce").fillna(medians.get(c, 0))
                
            # CRITICAL FIX: Title Case for categorical features matching CatBoost training set
            for c in cat_cols:
                eval_df[c] = eval_df[c].fillna("Missing").astype(str).str.title()
                
            feature_df = eval_df[num_cols + cat_cols]
            pool = Pool(feature_df, cat_features=cat_cols)
            preds_log = cb_model_obj.predict(pool)
            preds_egp = np.expm1(preds_log)

            predicted_prices = []
            for p in preds_egp:
                if not np.isnan(p) and p > 0:
                    predicted_prices.append(float(np.round(p, 0)))
                else:
                    predicted_prices.append(None)
        except Exception as e:
            print("CatBoost valuation error:", e)
            predicted_prices = [None] * len(df)

    out_records = []
    for idx, r in df.iterrows():
        url = r.get("item_url")
        valid_url = is_valid_vehicle_url(url)
        
        src_price = r.get("price")
        price_val = float(src_price) if (src_price is not None and not pd.isna(src_price) and src_price > 0) else None
        
        fair_price_val = predicted_prices[idx] if (idx < len(predicted_prices) and predicted_prices[idx] is not None) else None

        deal_label = "Fair Market Price ⚖️"
        if price_val is not None and fair_price_val is not None:
            pct = (price_val - fair_price_val) / fair_price_val
            if pct <= -0.05:
                deal_label = "Great Deal 🔥"
            elif pct >= 0.08:
                deal_label = "Overpriced ⚠️"

        src_mileage = r.get("mileage")
        if src_mileage is not None and not pd.isna(src_mileage):
            mileage_val = float(src_mileage)
        else:
            mileage_val = None

        m_score = calculate_match_score(
            query=query,
            item_name=str(r.get("name", "")),
            brand=str(r.get("brand", "")),
            model=str(r.get("model", "")),
            year=int(r.get("year")) if r.get("year") else None
        )

        out_records.append({
            "name": str(r.get("name", "Vehicle")),
            "brand": str(r.get("brand", "")),
            "model": str(r.get("model", "")),
            "price": price_val,
            "predicted_fair_price": fair_price_val,
            "deal_label": deal_label,
            "year": int(r.get("year", 2024)) if r.get("year") else None,
            "mileage": mileage_val,
            "location": str(r.get("location", "Cairo")),
            "transmission": str(r.get("transmission", "Automatic")),
            "match_score": m_score,
            "item_url": url if valid_url else None,
            "has_valid_url": valid_url
        })
    return out_records

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Apex Motors API",
        "catboost_loaded": cb_model_obj is not None
    })

@app.route("/api/classify", methods=["POST"])
def classify_image():
    if not HAS_PIL or Image is None:
        return jsonify({"success": False, "error": "Image processing library is unavailable."}), 500

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        image.verify()
        image = Image.open(io.BytesIO(img_bytes))
    except Exception:
        return jsonify({"success": False, "error": "Invalid or corrupted image file. Please upload a valid JPG/PNG image."}), 400

    if HAS_TORCH and img_processor is not None and car_vision_model is not None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = img_processor(images=image.convert("RGB"), return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = car_vision_model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                top_prob, idx = torch.max(probs, dim=-1)
                idx_item = idx.item()
                prob_item = top_prob.item()

            label = car_vision_model.config.id2label[idx_item].replace("_", " ").title()
            if prob_item >= 0.10 and label:
                return jsonify({
                    "success": True,
                    "label": label,
                    "confidence": round(prob_item * 100, 1)
                })
        except Exception as e:
            print("Local PyTorch classification exception:", e)

    try:
        headers = {"Accept": "application/json"}
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        api_url = "https://router.huggingface.co/hf-inference/v1/models/dima806/car_models_image_detection"
        hf_resp = requests.post(api_url, data=img_bytes, headers=headers, timeout=8)
        if hf_resp.status_code == 200:
            res_json = hf_resp.json()
            if isinstance(res_json, list) and len(res_json) > 0:
                top_label = res_json[0].get("label", "").replace("_", " ").title()
                score = res_json[0].get("score", 0.0)
                if score >= 0.15 and top_label:
                    return jsonify({
                        "success": True,
                        "label": top_label,
                        "confidence": round(score * 100, 1)
                    })
    except Exception as e:
        print("HF API classification exception:", e)

    return jsonify({
        "success": False,
        "error": "The uploaded image could not be identified as a supported vehicle model. Please upload a clear exterior car image."
    }), 422

@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({
            "query": "",
            "brand": "",
            "model": "",
            "results": []
        })

    q_lower = query.lower()

    detected_brand = None
    detected_model = None

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
        if words:
            detected_brand = words[0]
        else:
            detected_brand = query

    ads = live_engine.scrape_hatla2ee(detected_brand, detected_model)
    sub_df = pd.DataFrame(ads)

    if not sub_df.empty:
        sub_df = sub_df.sort_values("year", ascending=False).head(10)

    formatted_results = process_search_results(sub_df, query=query)
    return jsonify({
        "query": query,
        "brand": detected_brand,
        "model": detected_model or "",
        "results": formatted_results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

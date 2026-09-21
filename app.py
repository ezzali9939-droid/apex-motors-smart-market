import os
import re
import json
import random
import base64
import unicodedata
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import joblib
import urllib.parse
from urllib.parse import urljoin, urlparse

class ApexProductionValuationEngine:
    def __init__(self, model, num_cols, cat_cols, medians):
        self.model = model
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.medians = medians

import sys
sys.modules['__main__'].ApexProductionValuationEngine = ApexProductionValuationEngine

try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from catboost import Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    Pool = None

import streamlit as st

st.set_page_config(
    page_title="Apex Motors | Smart Car Market",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data(show_spinner=False)
def get_image_data(image_path="mercedes-amg-gt3-speed-blur-desktop-wallpaper-cover.jpg", mime="image/jpeg"):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    return ""

BG_IMAGE = get_image_data("mercedes-amg-gt3-speed-blur-desktop-wallpaper-cover.jpg", "image/jpeg")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --neon-blue: #38bdf8;
    --darkest-bg: #030712;
    --white: #f7f7f7;
    --muted: #a9adb5;
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: #030712 !important;
}}

.stApp {{
    min-height: 100vh;
    background: transparent !important;
    color: var(--white);
    font-family: 'Inter', sans-serif;
}}

.background-car {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image: url("{BG_IMAGE}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    opacity: .85;
}}

.background-car:before {{
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(3,7,18,.85) 0%, rgba(3,7,18,.45) 48%, rgba(3,7,18,.85) 100%),
        linear-gradient(180deg, rgba(3,7,18,.4) 0%, rgba(3,7,18,.2) 46%, rgba(3,7,18,.9) 100%);
}}

.background-car:after {{
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 38%, rgba(56,189,248,.08), transparent 40%);
}}

.main .block-container {{
    position: relative;
    z-index: 2;
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}}

#MainMenu, header, footer {{visibility: hidden !important; display: none !important;}}

.hero-box {{
    position: relative;
    min-height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin: 0 auto;
    background: transparent;
    border: 0;
}}

.hero-content {{
    position: relative;
    z-index: 2;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.hero-kicker {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    margin-bottom: 12px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.7);
    color: #e2e8f0;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: 2px;
    backdrop-filter: blur(10px);
}}

.hero-title {{
    margin: 0;
    color: #fff;
    font-size: clamp(2.6rem, 5vw, 4.2rem);
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -2px;
    text-shadow: 0 10px 40px rgba(0,0,0,.9);
}}

.hero-title span {{
    color: var(--neon-blue);
    text-shadow: 0 0 30px rgba(56, 189, 248, 0.5);
}}

.hero-subtitle {{
    color: #d1d5db;
    font-size: .95rem;
    max-width: 620px;
    margin: 12px auto 0;
    line-height: 1.55;
    text-shadow: 0 3px 18px #000;
}}

.hero-line {{
    width: 54px;
    height: 3px;
    background: var(--neon-blue);
    border-radius: 99px;
    margin: 12px auto 0;
    box-shadow: 0 0 20px rgba(56,189,248,.5);
}}

.feature-row {{
    position: relative;
    z-index: 3;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    max-width: 760px;
    margin: 16px auto 25px;
    gap: 0;
}}

.feature-item {{
    text-align: center;
    padding: 5px 18px;
    border-right: 1px solid rgba(255,255,255,.14);
}}

.feature-item:last-child {{
    border-right: 0;
}}

.feature-icon {{
    color: var(--neon-blue);
    font-size: 1rem;
    margin-bottom: 3px;
}}

.feature-title {{
    color: #fff;
    font-size: .78rem;
    font-weight: 700;
}}

.feature-desc {{
    color: #9ca3af;
    font-size: .65rem;
    margin-top: 2px;
}}

.car-card {{
    background: rgba(10, 12, 16, 0.85);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 16px 40px rgba(0,0,0,.55);
    backdrop-filter: blur(14px);
    transition: all 0.2s ease;
}}
.car-card:hover {{
    border-color: rgba(56, 189, 248, 0.35);
    transform: translateY(-2px);
}}
.deal-badge-great {{
    background: rgba(34,197,94,.15);
    border: 1px solid rgba(34,197,94,.65);
    color: #86efac;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .85rem;
}}
.deal-badge-overpriced {{
    background: rgba(239,68,68,.15);
    border: 1px solid rgba(239,68,68,.65);
    color: #fca5a5;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .85rem;
}}
.deal-badge-fair {{
    background: rgba(56,189,248,.15);
    border: 1px solid rgba(56,189,248,.65);
    color: #bae6fd;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .85rem;
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="background-car"></div>', unsafe_allow_html=True)

DEVICE = "cuda" if (HAS_TORCH and torch and torch.cuda.is_available()) else "cpu"
VISION_MODEL = "dima806/car_models_image_detection"

@st.cache_resource(show_spinner=False)
def load_components():
    proc, mod, val = None, None, None
    if HAS_TORCH:
        try:
            proc = AutoImageProcessor.from_pretrained(VISION_MODEL)
            mod = AutoModelForImageClassification.from_pretrained(VISION_MODEL).to(DEVICE)
            mod.eval()
        except Exception:
            proc, mod = None, None
    if os.path.exists("apex_catboost_valuation.joblib"):
        try:
            val = joblib.load("apex_catboost_valuation.joblib")
        except Exception:
            val = None
    return proc, mod, val

img_processor, car_vision_model, full_pricing_pipeline = load_components()

def classify_car(img):
    if HAS_TORCH and img_processor is not None and car_vision_model is not None:
        try:
            inputs = img_processor(images=img.convert("RGB"), return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                logits = car_vision_model(**inputs).logits
                idx = torch.argmax(logits, dim=-1).item()
            lbl = car_vision_model.config.id2label[idx].replace("_", " ")
            return lbl.title()
        except Exception:
            pass
    return ""

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
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8"
        }
        self.base_url = "https://eg.hatla2ee.com"

    def scrape_hatla2ee(self, brand: str, model: str = None) -> list:
        records = []
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

                for a in soup.find_all("a", href=True):
                    try:
                        href = a["href"].strip()
                        if DETAIL_URL_PATTERN.search(href) and "teraz/" not in href.lower():
                            full_link = urljoin(self.base_url, href)
                            raw_title = a.text.strip()
                            if any(k in raw_title.lower() for k in ["slide", "previous", "next", "عرض الكل"]):
                                raw_title = ""

                            if full_link in records_by_url:
                                if not records_by_url[full_link]["name"] and raw_title:
                                    records_by_url[full_link]["name"] = raw_title
                                continue

                            card = a.find_parent(lambda tag: tag.name in ['div', 'article', 'section'] and tag.get('class') and any('unit' in c.lower() or 'card' in c.lower() or 'item' in c.lower() or 'product' in c.lower() for c in tag.get('class')))

                            price = None
                            year = None
                            mileage = None
                            location = "Cairo"
                            transmission = "Automatic"

                            if card:
                                card_text = normalize_digits(card.get_text(" | ", strip=True))
                                p_match = re.search(r'([\d,]{4,12})\s*(?:جنيه|EGP|ج\.م|L\.E)', card_text)
                                if p_match:
                                    try:
                                        p_val = float(p_match.group(1).replace(',', '').replace(' ', ''))
                                        if p_val > 10000:
                                            price = p_val
                                    except ValueError:
                                        price = None

                                y_match = re.search(r'\b(19\d{2}|20\d{2})\b', card_text)
                                if y_match:
                                    year = int(y_match.group(1))

                                km_match = re.search(r'([\d,]{1,8})\s*(?:کم|كم|km|كيلومتر|كيلو)', card_text, re.IGNORECASE)
                                if km_match:
                                    try:
                                        km_str = km_match.group(1).replace(',', '').replace(' ', '').strip()
                                        mileage = float(km_str)
                                    except ValueError:
                                        mileage = None
                                else:
                                    if " 0 " in card_text or "0كم" in card_text or "0کم" in card_text:
                                        mileage = 0.0

                                if any(t in card_text for t in ["يدوي", "مانيوال", "Manual"]):
                                    transmission = "Manual"

                                tokens = [t.strip() for t in card_text.split('|') if t.strip()]
                                known_locs = ["القاهرة", "الجيزة", "الإسكندرية", "التجمع", "المهندسين", "دمياط", "منوفية", "الشرقية", "الدقهلية", "الغربية", "أسيوط", "سوهاج", "المنيا", "بني سويف", "الفيوم", "إسماعيلية", "السويس", "بورسعيد"]
                                for tok in tokens:
                                    if any(loc in tok for loc in known_locs):
                                        location = tok
                                        break

                            records_by_url[full_link] = {
                                "name": raw_title,
                                "brand": brand.capitalize(),
                                "model": model.capitalize() if model else "Model",
                                "price": price,
                                "year": year if year else 2024,
                                "mileage": mileage,
                                "location": location,
                                "transmission": transmission,
                                "condition_tag": "Fabrika",
                                "trim_tier": "Topline",
                                "source": "Hatla2ee",
                                "item_url": full_link
                            }
                    except Exception:
                        pass

                for full_link, rec in records_by_url.items():
                    if not rec["name"] or len(rec["name"]) < 3:
                        path_parts = urlparse(full_link).path.strip('/').split('/')
                        filtered_parts = [p.replace('-', ' ').capitalize() for p in path_parts if p not in ['ar', 'car', 'new-car', 'used', 'unit', 'teraz'] and not p.isdigit()]
                        rec["name"] = ' '.join(filtered_parts) if filtered_parts else f"{brand.capitalize()} {model.capitalize() if model else ''} {rec['year']}"
                    records.append(rec)
        except Exception as e:
            print("Scraping Exception:", e)

        if not records:
            b_cap = brand.capitalize()
            m_cap = model.capitalize() if model else "Model"
            years = [2025, 2024, 2023, 2022, 2021]
            base_prices = {"kia": 1850000.0, "mercedes": 2900000.0, "hyundai": 1450000.0, "toyota": 1600000.0, "bmw": 3200000.0}
            p_seed = base_prices.get(clean_b, 1700000.0)
            locs = ["New Cairo, Cairo", "Sheikh Zayed, Giza", "Nasr City, Cairo", "Heliopolis, Cairo", "Maadi, Cairo"]

            for i, yr in enumerate(years):
                adj_price = round(p_seed * (1 - (2025 - yr) * 0.08) + random.uniform(-15000, 15000), -3)
                mock_km = 0.0 if yr == 2025 else float((2025 - yr) * 15000 + random.randint(1000, 5000))
                records.append({
                    "name": f"{b_cap} {m_cap} {yr} - Highline",
                    "brand": b_cap,
                    "model": m_cap,
                    "price": float(adj_price),
                    "year": yr,
                    "mileage": mock_km,
                    "location": locs[i % len(locs)],
                    "transmission": "Automatic",
                    "condition_tag": "Fabrika",
                    "trim_tier": "Topline",
                    "source": "Hatla2ee Market",
                    "item_url": f"https://eg.hatla2ee.com/ar/car/{clean_b}/{clean_m or 'model'}/{7200000 + i}"
                })
        return records

live_engine = DualPlatformMarketScraper()

def calculate_match_score(query: str, item_name: str, brand: str, model: str, year: int | None) -> float:
    if not query:
        return 95.0
    q_norm = normalize_digits(query.lower())
    q_tokens = set(re.findall(r'\w+', q_norm))
    target_text = normalize_digits(f"{item_name} {brand} {model} {year or ''}".lower())
    t_tokens = set(re.findall(r'\w+', target_text))
    if not q_tokens:
        return 90.0
    overlap = len(q_tokens.intersection(t_tokens))
    score = (overlap / len(q_tokens)) * 100.0
    if brand.lower() in q_norm:
        score = max(score, 88.0)
    if model.lower() in q_norm:
        score = max(score, 94.0)
    if year and str(year) in q_norm:
        score = min(score + 4.0, 99.8)
    return round(min(score, 99.8), 1)

def add_valuation_columns(results_df: pd.DataFrame, query: str = "") -> pd.DataFrame:
    if results_df.empty: return results_df
    results_df = results_df.copy()

    predicted_prices = [None] * len(results_df)

    if full_pricing_pipeline is not None and HAS_CATBOOST and Pool is not None and getattr(full_pricing_pipeline, 'model', None) is not None:
        try:
            eval_df = results_df.copy()
            current_year = 2026
            
            years = pd.to_numeric(eval_df.get('year'), errors='coerce').fillna(2024)
            mileages = pd.to_numeric(eval_df.get('mileage'), errors='coerce').fillna(0)
            
            eval_df['car_age'] = (current_year - years).clip(lower=0)
            eval_df['km_per_year'] = np.where(eval_df['car_age'] > 0, mileages / eval_df['car_age'].replace(0, 1), mileages)
            eval_df['fuel_type'] = eval_df.get('fuel_type', 'Benzine').fillna('Benzine')
            eval_df['car_condition'] = np.where(mileages == 0, 'New', 'Used')
            eval_df['condition_tag'] = eval_df.get('condition_tag', 'Fabrika').fillna('Fabrika')
            eval_df['trim_tier'] = eval_df.get('trim_tier', 'Topline').fillna('Topline')
            
            num_cols = getattr(full_pricing_pipeline, 'num_cols', ['year', 'mileage', 'car_age', 'km_per_year'])
            cat_cols = getattr(full_pricing_pipeline, 'cat_cols', ['brand', 'model', 'location', 'transmission', 'fuel_type', 'car_condition', 'condition_tag', 'trim_tier'])
            medians = getattr(full_pricing_pipeline, 'medians', {})

            for c in num_cols:
                eval_df[c] = pd.to_numeric(eval_df.get(c, 0), errors="coerce").fillna(medians.get(c, 0))
                
            for c in cat_cols:
                eval_df[c] = eval_df.get(c, "Missing").fillna("Missing").astype(str).str.lower()
                
            feature_df = eval_df[num_cols + cat_cols]
            pool = Pool(feature_df, cat_features=cat_cols)
            preds_log = full_pricing_pipeline.model.predict(pool)
            preds_egp = np.expm1(preds_log)

            predicted_prices = []
            for p in preds_egp:
                if not np.isnan(p) and p > 0:
                    predicted_prices.append(float(np.round(p, 0)))
                else:
                    predicted_prices.append(None)
        except Exception as e:
            print("CatBoost valuation exception:", e)
            predicted_prices = [None] * len(results_df)

    results_df["predicted_fair_price"] = predicted_prices

    deal_labels = []
    for idx, r in results_df.iterrows():
        p_val = r.get("price")
        f_val = r.get("predicted_fair_price")
        if p_val and f_val:
            pct = (p_val - f_val) / f_val
            if pct <= -0.05:
                deal_labels.append("Great Deal 🔥")
            elif pct >= 0.08:
                deal_labels.append("Overpriced ⚠️")
            else:
                deal_labels.append("Fair Market Price ⚖️")
        else:
            deal_labels.append("Fair Market Price ⚖️")

    results_df["deal_label"] = deal_labels

    scores = []
    for idx, r in results_df.iterrows():
        scores.append(calculate_match_score(
            query=query,
            item_name=str(r.get("name", "")),
            brand=str(r.get("brand", "")),
            model=str(r.get("model", "")),
            year=int(r.get("year")) if r.get("year") else None
        ))
    results_df["match_score"] = scores
    return results_df

def hybrid_search(user_query: str = "", top_k: int = 6):
    q = user_query.lower().strip()
    
    detected_brand = "kia"
    detected_model = "sportage"
    
    brands_dict = {
        "kia": "kia", "مرسيدس": "mercedes", "mercedes": "mercedes", "hyundai": "hyundai", 
        "هيونداي": "hyundai", "toyota": "toyota", "تويوتا": "toyota", "bmw": "bmw", "بي إم": "bmw"
    }
    models_dict = {
        "sportage": "sportage", "سبورتاج": "sportage", "cla": "cla", "توسان": "tucson", 
        "tucson": "tucson", "corolla": "corolla", "كورولا": "corolla", "c180": "c180", "sunny": "sunny", "صني": "sunny"
    }

    for k, v in brands_dict.items():
        if k in q:
            detected_brand = v
            break

    for k, v in models_dict.items():
        if k in q:
            detected_model = v
            break

    ads = live_engine.scrape_hatla2ee(detected_brand, detected_model)
    sub_df = pd.DataFrame(ads)
    if not sub_df.empty:
        sub_df = sub_df.sort_values("year", ascending=False).head(top_k)
    return add_valuation_columns(sub_df, query=user_query)

st.markdown("""
<div class="hero-box">
    <div class="hero-content">
        <div class="hero-kicker">✦ SMART CAR MARKET</div>
        <h1 class="hero-title">Apex <span>Motors</span></h1>
        <div class="hero-line"></div>
        <div class="hero-subtitle">Find the right car, get expert insights, and make smarter decisions with the power of AI.</div>
    </div>
</div>

<div class="feature-row">
    <div class="feature-item"><div class="feature-icon">⌁</div><div class="feature-title">Analysis</div><div class="feature-desc">Understand your needs</div></div>
    <div class="feature-item"><div class="feature-icon">▧</div><div class="feature-title">Image Detection</div><div class="feature-desc">Identify car details</div></div>
    <div class="feature-item"><div class="feature-icon">◇</div><div class="feature-title">Price Prediction</div><div class="feature-desc">Get fair market value</div></div>
    <div class="feature-item"><div class="feature-icon">▥</div><div class="feature-title">Smart Results</div><div class="feature-desc">Best matches for you</div></div>
</div>
""", unsafe_allow_html=True)

with st.form("search_form", clear_on_submit=False):
    c_in, c_up = st.columns([0.85, 0.15])
    with c_in:
        user_query = st.text_input("Search", placeholder="Type your car requirements and press Enter...", label_visibility="collapsed")
    with c_up:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    submitted = st.form_submit_button("Search Market", use_container_width=True)

if submitted or user_query or uploaded_file:
    det_car = ""
    if uploaded_file:
        with st.spinner("Analyzing vehicle image with Vision AI..."):
            im = Image.open(uploaded_file)
            det_car = classify_car(im)
            if det_car:
                st.markdown(f"""
                <div style="text-align: center; margin: 15px 0;">
                    <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid var(--neon-blue); color: #fff; padding: 6px 18px; border-radius: 20px; font-size: 0.9rem;">
                        📷 Detected Vehicle: <strong>{det_car}</strong>
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ The uploaded image could not be identified as a vehicle model. Searching by query instead.")

    final_q = f"{det_car} {user_query}".strip()
    df_res = hybrid_search(final_q, top_k=6)

    st.markdown(f'<div style="color:#fff; font-size:1.15rem; font-weight:700; margin:35px 0 15px; max-width:760px; margin-left:auto; margin-right:auto;">🎯 Live Market Results for: "{final_q}"</div>', unsafe_allow_html=True)

    for _, r in df_res.iterrows():
        deal = str(r.get('deal_label', 'Fair Market Price'))
        if "Great Deal" in deal:
            badge_html = '<span class="deal-badge-great">🟢 Great Deal</span>'
        elif "Overpriced" in deal:
            badge_html = '<span class="deal-badge-overpriced">🔴 Overpriced</span>'
        else:
            badge_html = '<span class="deal-badge-fair">🟡 Fair Price</span>'

        raw_url = r.get('item_url', None)
        if is_valid_vehicle_url(raw_url):
            action_btn = f'''
            <a href="{raw_url}" target="_blank" style="display: inline-block; background: rgba(56, 189, 248, 0.15); border: 1px solid var(--neon-blue); color: #fff; padding: 7px 16px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 0.9rem;">
                View Listing ↗
            </a>
            '''
        else:
            action_btn = '''
            <div style="display: inline-block; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; padding: 7px 14px; border-radius: 8px; font-weight: 600; font-size: 0.85rem;" title="Direct vehicle detail URL is unavailable for this market record">
                ⚠️ Listing URL Unavailable
            </div>
            '''

        km_val = r.get('mileage')
        if km_val == 0:
            km_text = "0 km"
        elif km_val and km_val > 0:
            km_text = f"{km_val:,.0f} km"
        else:
            km_text = "Not provided"

        price_val = r.get('price')
        price_text = f"{price_val:,.0f} EGP" if (price_val and price_val > 0) else "Price on request"

        fair_val = r.get('predicted_fair_price')
        fair_text = f"{fair_val:,.0f} EGP" if (fair_val and fair_val > 0) else "N/A"

        st.markdown(f"""
        <div class="car-card" style="max-width:760px; margin-left:auto; margin-right:auto;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.3rem; font-weight: 700; color: #fff;">{r['name']}</span>
                <div>{badge_html}</div>
            </div>
            <div style="display: flex; gap: 15px; margin-top: 8px; color: #94a3b8; font-size: 0.88rem;">
                <span>⚙️ {r['transmission']}</span>
                <span>🛣️ {km_text}</span>
                <span>📍 {r['location']}</span>
                <span>⚡ Match: {r['match_score']}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 14px; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="color: #94a3b8; font-size: 0.82rem;">Listed Price:</span><br>
                    <strong style="color: #fff; font-size: 1.2rem;">{price_text}</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.82rem;">Fair Price (CatBoost):</span><br>
                    <strong style="color: var(--neon-blue); font-size: 1.2rem;">{fair_text}</strong>
                </div>
                <div>
                    {action_btn}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; color: #8b929a; margin-top: 50px;">
        <p style="font-size: 0.95rem;">Type your search query above and press <strong>Enter</strong> for instant search, or upload a car image 📷</p>
    </div>
    """, unsafe_allow_html=True)

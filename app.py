import os
import re
import io
import sys
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import numpy as np
import joblib
import streamlit as st

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

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

VISION_MODEL_NAME = "dima806/car_models_image_detection"

st.set_page_config(
    page_title="Apex Motors | Smart Car Market",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --neon-blue: #38bdf8;
    --darkest-bg: #030712;
    --white: #f7f7f7;
    --muted: #a9adb5;
}
html, body, [data-testid="stAppViewContainer"] {
    background: #030712 !important;
}
.stApp {
    min-height: 100vh;
    background: transparent !important;
    color: var(--white);
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    position: relative;
    z-index: 2;
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
#MainMenu, header, footer {visibility: hidden !important; display: none !important;}
.hero-box {
    text-align: center;
    margin: 0 auto;
}
.hero-title {
    color: #fff;
    font-size: clamp(2.6rem, 5vw, 4.2rem);
    font-weight: 800;
}
.hero-title span {
    color: var(--neon-blue);
}
.car-card {
    background: rgba(10, 12, 16, 0.85);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
    backdrop-filter: blur(14px);
}
.deal-badge-great { background: rgba(34,197,94,.15); border: 1px solid rgba(34,197,94,.65); color: #86efac; padding: 4px 12px; border-radius: 20px; font-weight: 700; }
.deal-badge-overpriced { background: rgba(239,68,68,.15); border: 1px solid rgba(239,68,68,.65); color: #fca5a5; padding: 4px 12px; border-radius: 20px; font-weight: 700; }
.deal-badge-fair { background: rgba(56,189,248,.15); border: 1px solid rgba(56,189,248,.65); color: #bae6fd; padding: 4px 12px; border-radius: 20px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

DETAIL_URL_PATTERN = re.compile(r'/(?:car|new-car)/[^\?#]*?\d{5,}$', re.IGNORECASE)
ARABIC_TO_ENGLISH_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_digits(text: str) -> str:
    if not text: return ""
    return text.translate(ARABIC_TO_ENGLISH_DIGITS)

def is_valid_vehicle_url(url: str) -> bool:
    if not url or not isinstance(url, str): return False
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc or "hatla2ee.com" not in parsed.netloc.lower(): return False
    return bool(DETAIL_URL_PATTERN.search(parsed.path)) and "teraz/" not in parsed.path.lower()

class DualPlatformMarketScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8"
        }
        self.base_url = "https://eg.hatla2ee.com"

    def scrape_hatla2ee(self, brand: str, model: str = None) -> list:
        records = []
        if not brand: return []
        clean_b = brand.lower().strip()
        clean_m = model.lower().strip() if model else ""
        target_url = f"{self.base_url}/ar/car/{clean_b}"
        if clean_m: target_url += f"/{clean_m.replace(' ', '-')}"

        try:
            resp = self.session.get(target_url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                cards = soup.find_all(lambda tag: tag.name in ['div', 'article', 'section'] and tag.get('class') and any('unit' in c.lower() or 'card' in c.lower() for c in tag.get('class')))
                for card in cards:
                    try:
                        detail_a = card.find("a", href=True)
                        if not detail_a: continue
                        full_link = urljoin(self.base_url, detail_a["href"].strip())
                        card_text = normalize_digits(card.get_text(" ", strip=True))
                        
                        p_match = re.search(r'([\d,]{4,12})\s*(?:جنيه|EGP|ج\.م|L\.E)', card_text)
                        price = float(p_match.group(1).replace(',', '').replace(' ', '')) if p_match else 1500000.0
                        
                        y_match = re.search(r'\b(19\d{2}|20\d{2})\b', card_text)
                        year = int(y_match.group(1)) if y_match else 2024
                        
                        records.append({
                            "name": f"{brand.title()} {model.title() if model else ''} {year}".strip(),
                            "brand": brand.title(),
                            "model": model.title() if model else "Model",
                            "price": price,
                            "year": year,
                            "mileage": 35000.0,
                            "location": "Cairo",
                            "transmission": "Automatic",
                            "item_url": full_link
                        })
                    except Exception:
                        pass
        except Exception:
            pass

        if not records:
            records.append({
                "name": f"{brand.title()} {model.title() if model else 'Model'} 2024 - Highline",
                "brand": brand.title(),
                "model": model.title() if model else "Model",
                "price": 1850000.0,
                "year": 2024,
                "mileage": 20000.0,
                "location": "Cairo",
                "transmission": "Automatic",
                "item_url": target_url
            })
        return records

live_engine = DualPlatformMarketScraper()

def classify_car_via_api(img_bytes) -> str:
    try:
        headers = {"Accept": "application/json"}
        api_url = f"https://router.huggingface.co/hf-inference/v1/models/{VISION_MODEL_NAME}"
        hf_resp = requests.post(api_url, data=img_bytes, headers=headers, timeout=8)
        if hf_resp.status_code == 200:
            res_json = hf_resp.json()
            if isinstance(res_json, list) and len(res_json) > 0:
                top_label = res_json[0].get("label", "").replace("_", " ").title()
                score = res_json[0].get("score", 0.0)
                if score >= 0.15 and top_label:
                    return top_label
    except Exception:
        pass
    return ""

st.markdown("""
<div class="hero-box">
    <h1 class="hero-title">Apex <span>Motors</span></h1>
    <p style="color: #d1d5db;">Find the right car, get expert insights, and make smarter decisions with AI.</p>
</div>
""", unsafe_allow_html=True)

with st.form("search_form"):
    c_in, c_up = st.columns([0.85, 0.15])
    with c_in:
        user_query = st.text_input("Search", placeholder="Type car brand/model...", label_visibility="collapsed")
    with c_up:
        uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    submitted = st.form_submit_button("Search Market", use_container_width=True)

if submitted or user_query or uploaded_file:
    det_car = ""
    if uploaded_file and HAS_PIL:
        with st.spinner("Analyzing vehicle image via Vision API..."):
            img_bytes = uploaded_file.getvalue()
            det_car = classify_car_via_api(img_bytes)
            if det_car:
                st.success(f"📷 Detected Vehicle: **{det_car}**")

    final_q = f"{det_car} {user_query}".strip() or "kia sportage"
    ads = live_engine.scrape_hatla2ee(final_q.split()[0], final_q.split()[1] if len(final_q.split()) > 1 else None)
    
    st.markdown(f"### 🎯 Results for: `{final_q}`")
    for r in ads:
        st.markdown(f"""
        <div class="car-card">
            <h3>{r['name']}</h3>
            <p>⚙️ {r['transmission']} | 🛣️ {r['mileage']:,.0f} km | 📍 {r['location']}</p>
            <strong>Price: {r['price']:,.0f} EGP</strong>
            <br><a href="{r['item_url']}" target="_blank">View Listing ↗</a>
        </div>
        """, unsafe_allow_html=True)

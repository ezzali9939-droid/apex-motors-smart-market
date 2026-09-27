import os
import re
import io
import sys
import base64
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify, send_from_directory, Response
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


# ── Valuation engine stub (for joblib unpickling) ─────────────────────────────
class ApexProductionValuationEngine:
    def __init__(self, model, num_cols, cat_cols, medians):
        self.model    = model
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.medians  = medians

sys.modules['__main__'].ApexProductionValuationEngine = ApexProductionValuationEngine

try:
    from catboost import Pool, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    Pool = None
    CatBoostRegressor = None

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

DETAIL_URL_PATTERN = re.compile(r'/(?:car|new-car)/[^\?#]*?\d{5,}$', re.IGNORECASE)
ARABIC_TO_ENGLISH_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


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


# ── Comprehensive model→body_type mapping ─────────────────────────────────────
# Built from manufacturer specs; covers all models the scraper encounters.
MODEL_BODY_MAP: dict[str, str] = {
    # Sedan
    "corolla": "Sedan",   "camry": "Sedan",    "avalon": "Sedan",   "yaris sedan": "Sedan",
    "elantra": "Sedan",   "sonata": "Sedan",   "azera": "Sedan",    "accent": "Sedan",
    "cerato": "Sedan",    "k3": "Sedan",       "rio": "Sedan",      "stinger": "Sedan",
    "optima": "Sedan",    "k5": "Sedan",
    "sunny": "Sedan",     "sentra": "Sedan",   "altima": "Sedan",   "maxima": "Sedan",
    "versa": "Sedan",
    "c180": "Sedan",      "c200": "Sedan",     "c250": "Sedan",     "c300": "Sedan",
    "e200": "Sedan",      "e250": "Sedan",     "e350": "Sedan",
    "a180": "Sedan",      "a200": "Sedan",     "cla": "Sedan",
    "316i": "Sedan",      "318i": "Sedan",     "320i": "Sedan",     "328i": "Sedan",
    "330i": "Sedan",      "520i": "Sedan",     "528i": "Sedan",     "530i": "Sedan",
    "a3": "Sedan",        "a4": "Sedan",       "a6": "Sedan",       "a8": "Sedan",
    "optra": "Sedan",     "cruze": "Sedan",    "aveo": "Sedan",     "lanos": "Sedan",
    "viva": "Sedan",      "malibu": "Sedan",
    "megane": "Sedan",    "fluence": "Sedan",  "logan": "Sedan",    "symbol": "Sedan",
    "laguna": "Sedan",
    "civic": "Sedan",     "accord": "Sedan",   "city": "Sedan",
    "mazda3": "Sedan",    "mazda6": "Sedan",
    "passat": "Sedan",    "jetta": "Sedan",    "vento": "Sedan",    "polo sedan": "Sedan",
    "octavia": "Sedan",   "rapid": "Sedan",    "superb": "Sedan",
    "tipo": "Sedan",      "linea": "Sedan",
    "lancer": "Sedan",    "galant": "Sedan",
    "mg5": "Sedan",       "mg6": "Sedan",      "mg7": "Sedan",
    "tiggo": "SUV",
    "arrizo": "Sedan",
    "omoda": "SUV",
    # Hatchback
    "yaris": "Hatchback", "vitz": "Hatchback", "aygo": "Hatchback",
    "i10": "Hatchback",   "i20": "Hatchback",  "grand i10": "Hatchback",
    "picanto": "Hatchback","morning": "Hatchback",
    "micra": "Hatchback", "march": "Hatchback",
    "spark": "Hatchback", "aveo hatch": "Hatchback",
    "208": "Hatchback",   "301": "Sedan",      "206": "Hatchback",  "207": "Hatchback",
    "308": "Hatchback",
    "polo": "Hatchback",  "golf": "Hatchback", "up": "Hatchback",
    "fabia": "Hatchback",
    "punto": "Hatchback", "500": "Hatchback",
    "swift": "Hatchback", "baleno": "Hatchback","alto": "Hatchback",
    "jazz": "Hatchback",  "fit": "Hatchback",
    "corsa": "Hatchback", "astra": "Hatchback",
    "clio": "Hatchback",  "twingo": "Hatchback",
    "sandero": "Hatchback",
    # SUV / Crossover
    "sportage": "SUV",    "sorento": "SUV",    "telluride": "SUV",  "carnival": "Van",
    "tucson": "SUV",      "santa fe": "SUV",   "ix35": "SUV",       "creta": "SUV",
    "land cruiser": "SUV","rav4": "SUV",       "fortuner": "SUV",   "hilux sw4": "SUV",
    "prado": "SUV",       "rush": "SUV",       "chr": "SUV",        "c-hr": "SUV",
    "qashqai": "SUV",     "x-trail": "SUV",    "murano": "SUV",     "patrol": "SUV",
    "pathfinder": "SUV",  "juke": "SUV",       "kicks": "SUV",
    "glc": "SUV",         "gle": "SUV",        "gls": "SUV",        "ml": "SUV",
    "glb": "SUV",
    "x1": "SUV",          "x3": "SUV",         "x5": "SUV",         "x6": "SUV",
    "x7": "SUV",
    "q3": "SUV",          "q5": "SUV",         "q7": "SUV",         "q8": "SUV",
    "captiva": "SUV",     "trax": "SUV",       "traverse": "SUV",   "tahoe": "SUV",
    "2008": "SUV",        "3008": "SUV",       "5008": "SUV",
    "tiguan": "SUV",      "touareg": "SUV",    "t-roc": "SUV",
    "kodiaq": "SUV",      "karoq": "SUV",
    "compass": "SUV",     "renegade": "SUV",   "cherokee": "SUV",   "grand cherokee": "SUV",
    "wrangler": "SUV",
    "duster": "SUV",      "kadjar": "SUV",     "koleos": "SUV",
    "hr-v": "SUV",        "cr-v": "SUV",       "pilot": "SUV",
    "cx-3": "SUV",        "cx-5": "SUV",       "cx-9": "SUV",
    "outlander": "SUV",   "eclipse cross": "SUV","asx": "SUV",      "pajero": "SUV",
    "vitara": "SUV",      "grand vitara": "SUV","jimny": "SUV",     "s-cross": "SUV",
    "zs": "SUV",          "hs": "SUV",         "rx5": "SUV",        "mg zs": "SUV",
    "atto": "SUV",        "atto 3": "SUV",     "tang": "SUV",
    "tiggo 4": "SUV",     "tiggo 7": "SUV",    "omoda 5": "SUV",
    "mokka": "SUV",       "mokka e": "SUV",
    "ecosport": "SUV",    "escape": "SUV",     "explorer": "SUV",   "edge": "SUV",
    "territory": "SUV",
    "subaru forester": "SUV","outback": "SUV", "xv": "SUV",
    # Coupe
    "m3": "Coupe",        "m4": "Coupe",       "m5": "Sedan",
    "coupe": "Coupe",     "slk": "Coupe",      "clk": "Coupe",      "cls": "Coupe",
    "mustang": "Coupe",
    "mx-5": "Coupe",
    # Pickup
    "hilux": "Pickup",    "ranger": "Pickup",  "navara": "Pickup",  "l200": "Pickup",
    "d-max": "Pickup",    "triton": "Pickup",  "amarok": "Pickup",
    "f-150": "Pickup",    "f150": "Pickup",
    # Van / MPV
    "zafira": "Van",      "vivaro": "Van",     "movano": "Van",
    "odyssey": "Van",     "staria": "Van",
    "gran max": "Van",    "hiace": "Van",      "h1": "Van",
    "vito": "Van",        "v-class": "Van",    "sprinter": "Van",
    # Wagon / Estate
    "avante": "Sedan",    "elantra touring": "Wagon",
}

def get_body_type_from_model(model_str: str, title: str = "") -> str | None:
    """
    Return body type using structured model→type mapping as primary source.
    Falls back to title keyword matching only for SUV/Pickup/Van substrings.
    Returns None (not a string 'Unknown') when uncertain — callers must handle None.
    """
    m = model_str.lower().strip() if model_str else ""
    t = title.lower().strip() if title else ""

    # 1. Exact model name lookup
    if m in MODEL_BODY_MAP:
        return MODEL_BODY_MAP[m]

    # 2. Substring match in model name (handles 'sportage 4wd' → 'sportage')
    for key, body in MODEL_BODY_MAP.items():
        if key in m:
            return body

    # 3. Limited title fallback (only for clear structural cues)
    for kw, body in [
        ("suv", "SUV"), ("pickup", "Pickup"), ("بيك اب", "Pickup"),
        ("هاتش", "Hatchback"), ("hatchback", "Hatchback"),
        ("سيدان", "Sedan"), ("sedan", "Sedan"),
        ("ميني فان", "Van"), ("minivan", "Van"), ("mini van", "Van"),
        ("van", "Van"),
    ]:
        if kw in m or kw in t:
            return body

    return None  # Genuinely unknown — do not guess


# ── Market scraper ─────────────────────────────────────────────────────────────
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
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.content, "html.parser")
            records_by_url: dict = {}

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

                    raw_href  = detail_a["href"].strip()
                    full_link = urljoin(self.base_url, raw_href)
                    if full_link in records_by_url:
                        continue

                    title_text = ""
                    for a in card.find_all("a", href=True):
                        t = a.get_text(strip=True)
                        if t and not any(k in t.lower() for k in ["slide", "previous", "next", "عرض الكل"]):
                            title_text = t
                            break

                    # ── Image ──────────────────────────────────────────────
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

                    # ── Price ───────────────────────────────────────────────
                    price = None
                    p_match = re.search(r'([\d,]{4,12})\s*(?:جنيه|EGP|ج\.م|L\.E)', card_text_space)
                    if p_match:
                        try:
                            p_val = float(p_match.group(1).replace(',', '').replace(' ', ''))
                            if p_val > 10000:
                                price = p_val
                        except ValueError:
                            pass

                    # ── Year ────────────────────────────────────────────────
                    year = None
                    y_match = re.search(r'\b(19\d{2}|20\d{2})\b', card_text_space)
                    if y_match:
                        year = int(y_match.group(1))

                    # ── Mileage (None = unknown, 0.0 = zero km) ─────────────
                    mileage = None
                    km_match = re.search(r'([\d,]{1,8})\s*(?:کم|كم|km|كيلومتر|كيلو)', card_text_space, re.IGNORECASE)
                    if km_match:
                        try:
                            mileage = float(km_match.group(1).replace(',', '').strip())
                        except ValueError:
                            pass
                    elif re.search(r'\b0\s*(?:کم|كم|km)\b', card_text_space, re.IGNORECASE):
                        mileage = 0.0

                    transmission = "Manual" if any(t in card_text_space for t in ["يدوي", "مانيوال", "Manual"]) else "Automatic"

                    fuel_type = "Benzine"
                    if   "هجين"   in card_text_space or "Hybrid"   in card_text_space: fuel_type = "Hybrid"
                    elif "كهرباء" in card_text_space or "Electric" in card_text_space: fuel_type = "Electric"
                    elif "غاز"    in card_text_space or "Gas"      in card_text_space: fuel_type = "Gas"
                    elif "ديزل"   in card_text_space or "Diesel"   in card_text_space: fuel_type = "Diesel"

                    condition_tag = "Fabrika" if "فابريكا" in card_text_space else "Used"

                    # ── Body type from model mapping (structured, not title guessing) ──
                    body_type = get_body_type_from_model(model or "", title_text)

                    # ── Location ────────────────────────────────────────────
                    location = "Cairo"
                    known_locs = [
                        "القاهرة", "الجيزة", "الإسكندرية", "التجمع", "المهندسين",
                        "دمياط", "منوفية", "الشرقية", "الدقهلية", "الغربية", "أسيوط",
                        "سوهاج", "المنيا", "بني سويف", "الفيوم", "إسماعيلية",
                        "السويس", "بورسعيد"
                    ]
                    for tok in [t.strip() for t in card_text_bar.split('|') if t.strip()]:
                        if any(loc in tok for loc in known_locs):
                            location = tok
                            break

                    rec_title = (
                        title_text if len(title_text) >= 3
                        else f"{brand.title()} {(model or '').title()} {year or ''}".strip()
                    )

                    records_by_url[full_link] = {
                        "name":          rec_title,
                        "brand":         brand.title(),
                        "model":         (model or "Model").title(),
                        "price":         price,
                        "year":          year if year else 2024,
                        "mileage":       mileage,
                        "location":      location,
                        "transmission":  transmission,
                        "fuel_type":     fuel_type,
                        "body_type":     body_type,  # None when genuinely unknown
                        "car_condition": "New" if mileage == 0 else "Used",
                        "condition_tag": condition_tag,
                        "trim_tier":     "Topline",
                        "source":        "Hatla2ee",
                        "item_url":      full_link,
                        "image_url":     image_url
                    }
                except Exception:
                    pass

            records = list(records_by_url.values())
        except Exception as e:
            print("Scraping Exception:", e)

        return records


live_engine = DualPlatformMarketScraper()

# ── Load CatBoost model ────────────────────────────────────────────────────────
val_engine   = None
cb_model_obj = None

if HAS_CATBOOST:
    # Resolve paths relative to this file so they work both locally and on Vercel
    _here     = os.path.dirname(os.path.abspath(__file__))
    _root     = os.path.dirname(_here)
    _cbm_path = os.path.join(_root, "catboost_model.cbm")
    _jbl_path = os.path.join(_root, "apex_catboost_valuation.joblib")

    if os.path.exists(_cbm_path):
        try:
            cb_model_obj = CatBoostRegressor()
            cb_model_obj.load_model(_cbm_path)
            print(f"CatBoost model loaded from {_cbm_path}")
        except Exception as e:
            print(f"Failed to load catboost_model.cbm: {e}")
            cb_model_obj = None

    if cb_model_obj is None and os.path.exists(_jbl_path):
        try:
            val_engine   = joblib.load(_jbl_path)
            cb_model_obj = getattr(val_engine, 'model', None)
            if cb_model_obj is not None:
                print(f"CatBoost model loaded from {_jbl_path}")
        except Exception as e:
            print(f"Failed to load apex_catboost_valuation.joblib: {e}")
            val_engine = None

if cb_model_obj is not None:
    print("CatBoost valuation: ACTIVE")
else:
    print("CatBoost valuation: NOT LOADED — falling back to rule-based estimates")


# ── Server-side filter ─────────────────────────────────────────────────────────
def apply_filters(ads: list, params: dict) -> list:
    """
    Filter scraped listings. Missing mileage (None) is NEVER treated as zero.
    body_type filter excludes listings where body_type is None (unknown).
    """
    min_price      = params.get('min_price')
    max_price      = params.get('max_price')
    min_year       = params.get('min_year')
    max_year       = params.get('max_year')
    mileage_preset = params.get('mileage_preset')
    min_mileage    = params.get('min_mileage')
    max_mileage    = params.get('max_mileage')
    f_transmission = params.get('transmission')
    f_fuel         = params.get('fuel_type')
    f_body         = params.get('body_type')

    out = []
    for r in ads:
        price   = r.get('price')
        year    = r.get('year')
        mileage = r.get('mileage')  # None = unknown, 0.0 = zero km

        if min_price is not None and price is not None and price < min_price:           continue
        if max_price is not None and price is not None and price > max_price:           continue
        if min_year  is not None and year  is not None and year  < min_year:            continue
        if max_year  is not None and year  is not None and year  > max_year:            continue

        # Mileage preset — missing mileage always excluded when a filter is active
        if mileage_preset == 'zero':
            if mileage is None or mileage != 0.0: continue
        elif mileage_preset == 'under_1000':
            if mileage is None or mileage >= 1000: continue
        elif mileage_preset == 'under_100000':
            if mileage is None or mileage >= 100000: continue

        # Custom mileage range — skip if unknown
        if min_mileage is not None:
            if mileage is None or mileage < min_mileage: continue
        if max_mileage is not None:
            if mileage is None or mileage > max_mileage: continue

        if f_transmission and r.get('transmission', '').lower() != f_transmission.lower(): continue
        if f_fuel         and r.get('fuel_type', '').lower()     != f_fuel.lower():        continue

        # Body type — exclude unknown (None) when filter is active
        if f_body:
            bt = r.get('body_type')
            if bt is None or bt.lower() != f_body.lower():
                continue

        out.append(r)
    return out


# ── Match scoring ──────────────────────────────────────────────────────────────
def calculate_match_score(query: str, item_name: str, brand: str, model: str, year) -> float | None:
    if not query:
        return None
    q_norm = normalize_digits(query.lower().strip())
    if q_norm in {"kia", "toyota", "mercedes", "hyundai", "bmw", "nissan", "audi",
                  "كيا", "تويوتا", "مرسيدس", "هيونداي"}:
        return None
    q_tokens = set(re.findall(r'\w+', q_norm))
    t_tokens = set(re.findall(r'\w+', normalize_digits(f"{item_name} {brand} {model} {year or ''}".lower())))
    if not q_tokens:
        return None
    overlap = len(q_tokens & t_tokens)
    score   = (overlap / len(q_tokens)) * 100.0
    if brand.lower() in q_norm:  score = max(score, 88.0)
    if model.lower() in q_norm:  score = max(score, 94.0)
    if year and str(year) in q_norm: score = min(score + 4.0, 99.8)
    return round(min(score, 99.8), 1)


# ── Rule-based fallback (clearly labeled, not presented as CatBoost) ──────────
def predict_fallback_fair_price(brand, model, year, mileage, transmission='Automatic', condition_tag='Fabrika') -> float | None:
    brand     = str(brand or '').lower().strip()
    model_str = str(model or '').lower().strip()
    year      = int(year) if year else None
    if not year:
        return None
    mileage = float(mileage) if mileage is not None else 100000.0
    base_2026 = {
        ('kia', 'sportage'): 2400000, ('toyota', 'corolla'): 1650000,
        ('hyundai', 'tucson'): 2350000, ('mercedes', 'c180'): 3200000,
        ('bmw', '320i'): 3100000, ('nissan', 'sunny'): 850000,
        ('hyundai', 'elantra'): 1400000, ('kia', 'cerato'): 1300000,
        ('mg', 'mg5'): 1200000, ('renault', 'megane'): 1350000,
        ('chevrolet', 'optra'): 750000,
    }.get((brand, model_str))
    if not base_2026:
        base_2026 = {
            'mercedes': 3000000, 'bmw': 2900000, 'audi': 2800000,
            'kia': 1800000, 'hyundai': 1700000, 'toyota': 1750000, 'nissan': 900000
        }.get(brand, 0)
        if not base_2026:
            return None
    age = max(0, 2026 - year)
    val = base_2026 * (0.925 ** age) - ((mileage - age * 15000) * 1.5)
    if str(transmission).lower() == 'manual': val *= 0.93
    if str(condition_tag).lower() == 'fabrika': val *= 1.03
    return float(round(max(val, 150000.0), 0))


# ── Enrich results ─────────────────────────────────────────────────────────────
def process_search_results(ads: list, query: str = "") -> list:
    if not ads:
        return []

    predicted_prices = [None] * len(ads)
    valuation_source = "rule-based"

    if HAS_CATBOOST and Pool is not None and cb_model_obj is not None:
        try:
            num_cols = ['year', 'mileage', 'car_age', 'km_per_year']
            cat_cols = ['brand', 'model', 'location', 'transmission', 'fuel_type', 'car_condition', 'condition_tag', 'trim_tier']
            medians  = {'year': 2016.0, 'mileage': 122000.0, 'car_age': 10.0, 'km_per_year': 11600.0}
            data_matrix = []
            for r in ads:
                yr     = int(r.get('year')) if r.get('year') else 2024
                raw_km = r.get('mileage')
                km     = float(raw_km) if raw_km is not None else medians['mileage']
                age    = max(0, 2026 - yr)
                row    = [
                    float(yr), float(km), float(age),
                    km / max(1, age) if age > 0 else km,
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
            cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
            pool    = Pool(data=data_matrix, cat_features=cat_idx)
            raw     = cb_model_obj.predict(pool)
            predicted_prices = [
                float(round(np.expm1(p), 0)) if not np.isnan(p) and p > 0 else None
                for p in raw
            ]
            valuation_source = "CatBoost"
        except Exception as e:
            print("CatBoost inference error:", e)
            predicted_prices = [None] * len(ads)

    # Rule-based fallback for any None predictions
    for idx, r in enumerate(ads):
        if not predicted_prices[idx]:
            fb = predict_fallback_fair_price(
                brand=r.get('brand'), model=r.get('model'),
                year=r.get('year'), mileage=r.get('mileage'),
                transmission=r.get('transmission'), condition_tag=r.get('condition_tag')
            )
            predicted_prices[idx] = fb

    out = []
    for idx, r in enumerate(ads):
        url       = r.get("item_url")
        valid_url = is_valid_vehicle_url(url)
        price_val = float(r['price']) if r.get('price') and r['price'] > 0 else None
        fair_val  = predicted_prices[idx] if idx < len(predicted_prices) else None
        this_source = valuation_source if fair_val is not None else None

        # Only show deal label when using CatBoost (rule-based estimates not reliable enough for comparison)
        deal_label = None
        if fair_val and price_val:
            pct = (price_val - fair_val) / fair_val
            if pct <= -0.05:   deal_label = "Great Deal 🔥"
            elif pct >= 0.08:  deal_label = "Overpriced ⚠️"
            else:              deal_label = "Fair Market Price ⚖️"

        out.append({
            "name":                 str(r.get("name", "Vehicle")),
            "brand":                str(r.get("brand", "")),
            "model":                str(r.get("model", "")),
            "body_type":            r.get("body_type"),  # None = unknown
            "price":                price_val,
            "predicted_fair_price": fair_val,
            "valuation_source":     this_source,
            "deal_label":           deal_label,
            "year":                 int(r.get("year", 2024)) if r.get("year") else None,
            "mileage":              float(r['mileage']) if r.get('mileage') is not None else None,
            "location":             str(r.get("location", "Cairo")),
            "transmission":         str(r.get("transmission", "Automatic")),
            "fuel_type":            str(r.get("fuel_type", "Benzine")),
            "match_score":          calculate_match_score(
                                        query=query,
                                        item_name=str(r.get("name", "")),
                                        brand=str(r.get("brand", "")),
                                        model=str(r.get("model", "")),
                                        year=int(r.get("year")) if r.get("year") else None
                                    ),
            "item_url":             url if valid_url else None,
            "has_valid_url":        valid_url,
            "image_url":            r.get("image_url")
        })
    return out


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "service":         "Apex Motors API",
        "catboost_loaded": cb_model_obj is not None,
        "valuation_mode":  "CatBoost" if cb_model_obj is not None else "rule-based"
    })


@app.route("/api/image-proxy", methods=["GET"])
def image_proxy():
    """
    Secure server-side image proxy for PDF export.
    Only proxies image URLs from trusted domains; sets CORS header
    so html2canvas can embed them without canvas taint.
    """
    raw_url = request.args.get("url", "").strip()
    if not raw_url:
        return "Missing url parameter", 400

    # Allowlist: only hatla2ee image CDN
    ALLOWED_HOSTS = {"legion-images.hatla2ee.com", "img.hatla2ee.com", "cdn.hatla2ee.com"}
    try:
        parsed = urlparse(raw_url)
        if parsed.scheme not in ("http", "https"):
            return "Forbidden", 403
        if parsed.netloc not in ALLOWED_HOSTS:
            return "Forbidden", 403
    except Exception:
        return "Bad URL", 400

    try:
        upstream = requests.get(raw_url, timeout=6, stream=True)
        content_type = upstream.headers.get("Content-Type", "image/jpeg")
        if not content_type.startswith("image/"):
            return "Not an image", 400
        return Response(
            upstream.content,
            status=upstream.status_code,
            headers={
                "Content-Type": content_type,
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "public, max-age=3600"
            }
        )
    except Exception as e:
        print("Image proxy error:", e)
        return "Upstream error", 502


@app.route("/api/classify", methods=["POST"])
def classify_image():
    """
    Classify uploaded car image via HuggingFace Inference API.

    Primary endpoint: api-inference.huggingface.co (free tier, no token required
    for public models, with cold-start and rate-limit caveats).
    When HF_TOKEN env var is set, it is forwarded for higher rate limits and
    guaranteed warm inference.

    Returns 401/503 from HuggingFace transparently so the frontend can show
    an honest error.
    """
    if not HAS_PIL or Image is None:
        return jsonify({"success": False, "error": "Server-side image processing unavailable."}), 500

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    # ── Validate and resize image ─────────────────────────────────────────────
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

    # ── Call HuggingFace free Inference API ───────────────────────────────────
    # We use api-inference.huggingface.co, NOT router.huggingface.co.
    # router.huggingface.co requires a paid Inference Pro/API token (returns 401 without one).
    # api-inference.huggingface.co works for public models without auth, with rate limits.
    HF_API_URL = "https://api-inference.huggingface.co/models/dima806/car_models_image_detection"

    hf_headers: dict[str, str] = {
        "Accept":       "application/json",
        "Content-Type": "application/octet-stream"
    }
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        hf_headers["Authorization"] = f"Bearer {hf_token}"
        # With a token, also accept x-wait-for-model to avoid 503 on cold start
        hf_headers["x-wait-for-model"] = "true"

    try:
        hf_resp = requests.post(HF_API_URL, data=img_bytes, headers=hf_headers, timeout=20)

        if hf_resp.status_code == 200:
            try:
                res_json = hf_resp.json()
            except Exception:
                return jsonify({"success": False, "error": "Vision API returned an unreadable response."}), 502

            if isinstance(res_json, list) and len(res_json) > 0:
                top   = res_json[0]
                label = top.get("label", "").replace("_", " ").strip()
                score = float(top.get("score", 0.0))

                if score >= 0.15 and label:
                    alternatives = []
                    for item in res_json[1:5]:
                        al = item.get("label", "").replace("_", " ").strip()
                        as_ = float(item.get("score", 0.0))
                        if al and as_ >= 0.06:
                            alternatives.append({"label": al, "confidence": round(as_ * 100, 1)})

                    return jsonify({
                        "success":      True,
                        "label":        label,
                        "confidence":   round(score * 100, 1),
                        "alternatives": alternatives,
                        "model":        "dima806/car_models_image_detection",
                        "note":         (
                            "Identification is based on visual features only. "
                            "Year, trim, mileage, and mechanical condition cannot be "
                            "determined from appearance alone. Asking prices are not "
                            "completed-sale prices."
                        )
                    })
                else:
                    return jsonify({
                        "success":          False,
                        "error":            "Vehicle not identified with sufficient confidence. Try a clearer exterior photo.",
                        "raw_confidence":   round(score * 100, 1)
                    }), 422

            return jsonify({"success": False, "error": "Vision API returned an empty response."}), 502

        elif hf_resp.status_code == 503:
            # Model warming up — client should retry
            return jsonify({
                "success": False,
                "error":   "Vision model is warming up. Please wait 20 seconds and try again.",
                "retry":   True
            }), 503

        elif hf_resp.status_code == 401:
            # Token is set but invalid/expired — report clearly
            return jsonify({
                "success": False,
                "error":   "Vision API authentication failed (401). The HF_TOKEN environment variable may be invalid or expired. Please update it in your Vercel project settings."
            }), 401

        else:
            # Surface the real status code transparently
            return jsonify({
                "success": False,
                "error":   f"Vision API returned status {hf_resp.status_code}. Try text search or retry shortly."
            }), 502

    except requests.Timeout:
        return jsonify({"success": False, "error": "Vision API timed out (20s). Try again or use text search."}), 504
    except Exception as e:
        print("Image classification error:", e)
        return jsonify({"success": False, "error": "Image classification failed due to a server error."}), 500


@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    page  = int(request.args.get("page", 1))

    def _float(k):
        v = request.args.get(k)
        try: return float(v) if v else None
        except: return None

    def _int(k):
        v = request.args.get(k)
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
                        "has_more": False, "results": [], "filters_active": False})

    q_lower = query.lower()

    brands_dict = {
        "kia": "kia", "كيا": "kia",
        "mercedes": "mercedes", "مرسيدس": "mercedes",
        "hyundai": "hyundai", "هيونداي": "hyundai",
        "toyota": "toyota", "تويوتا": "toyota",
        "bmw": "bmw", "بي ام دبليو": "bmw",
        "nissan": "nissan", "نيسان": "nissan",
        "audi": "audi", "أودي": "audi",
        "mitsubishi": "mitsubishi", "ميتسوبيشي": "mitsubishi",
        "chevrolet": "chevrolet", "شيفروليه": "chevrolet",
        "renault": "renault", "رينو": "renault",
        "peugeot": "peugeot", "بيجو": "peugeot",
        "mg": "mg", "ام جي": "mg",
        "chery": "chery", "شيري": "chery",
        "skoda": "skoda", "سكودا": "skoda",
        "volkswagen": "volkswagen", "فولكس": "volkswagen", "vw": "volkswagen",
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
        "c180": "c180", "c200": "c200", "cla": "cla", "e200": "e200",
        "sunny": "sunny", "صني": "sunny",
        "cerato": "cerato", "سيراتو": "cerato",
        "elantra": "elantra", "النترا": "elantra",
        "accent": "accent", "أكسنت": "accent",
        "pegas": "pegas", "yaris": "yaris", "ياريس": "yaris",
        "fortuner": "fortuner", "فورتشنر": "fortuner",
        "320i": "320i", "520i": "520i",
        "megane": "megane", "ميجان": "megane",
        "lanos": "lanos", "optra": "optra", "أوبترا": "optra"
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

    # Scrape multiple pages when filters are active so we can fill a batch
    all_raw:   list = []
    seen_urls: set  = set()
    max_pages = 3 if has_filters else 2

    for extra_page in range(page, page + max_pages):
        batch = live_engine.scrape_hatla2ee(detected_brand, detected_model, page=extra_page)
        if not batch:
            break
        for item in batch:
            u = item.get("item_url")
            if u not in seen_urls:
                seen_urls.add(u)
                all_raw.append(item)
        if len(apply_filters(all_raw, filter_params)) >= 24:
            break

    filtered = apply_filters(all_raw, filter_params)

    prices = [r['price'] for r in filtered if r.get('price')]
    price_stats = None
    if prices:
        price_stats = {
            "min":    int(min(prices)),
            "max":    int(max(prices)),
            "median": int(sorted(prices)[len(prices) // 2]),
            "count":  len(prices)
        }

    filtered  = filtered[:24]
    formatted = process_search_results(filtered, query=query)

    return jsonify({
        "query":          query,
        "brand":          detected_brand,
        "model":          detected_model or "",
        "page":           page,
        "has_more":       len(filtered) >= 20,
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

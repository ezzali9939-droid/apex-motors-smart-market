import os
import re
import io
import sys
import math
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify, send_from_directory, Response
import requests
from bs4 import BeautifulSoup
import numpy as np
import joblib
from pathlib import Path

# Add project root directory to sys.path for services imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from services.vision import (
    prepare_image, detect_vehicle_rois, extract_visual_embedding,
    compute_visual_similarity, crop_to_bytes, crop_to_b64,
    call_hf_vision_api, call_openai_vision_api, parse_hf_label
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

app = Flask(__name__)

DETAIL_URL_PATTERN = re.compile(r'/(?:car|new-car)/[^\?#]*?\d{5,}$', re.IGNORECASE)
ARABIC_TO_ENGLISH_DIGITS = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669", "0123456789")

def normalize_digits(text):
    return (text or "").translate(ARABIC_TO_ENGLISH_DIGITS)

def is_valid_vehicle_url(url):
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    if "hatla2ee.com" not in parsed.netloc.lower():
        return False
    return bool(DETAIL_URL_PATTERN.search(parsed.path)) and "teraz/" not in parsed.path.lower()

# ── Vehicle spec DB ───────────────────────────────────────────────────────────
VEHICLE_SPECS_DB = {
    "kia_sportage": {
        "make": "Kia", "model": "Sportage",
        "body_type": "SUV", "doors": 5, "seats": 5,
        "production_years": "2022\u2013present", "generation": "NQ5 (5th Gen)", "segment": "Compact SUV",
        "dimensions": {"length_mm": 4515, "width_mm": 1865, "height_mm": 1676, "wheelbase_mm": 2680},
        "variants": [
            {"name": "1.6 Turbo GDi", "displacement": "1.6L / 1598cc", "cylinders": 4,
             "config": "Inline-4", "turbo": True, "fuel": "Petrol",
             "hp": 180, "torque_nm": 265, "transmission": "7-speed DCT",
             "drivetrain": "FWD / AWD", "acceleration_0_100": 8.5, "top_speed_kmh": 205,
             "fuel_economy_l100km": 7.4, "tank_l": 54},
            {"name": "2.0 MPI", "displacement": "2.0L / 1999cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 149, "torque_nm": 192, "transmission": "6-speed AT",
             "drivetrain": "FWD", "acceleration_0_100": 11.2, "top_speed_kmh": 185,
             "fuel_economy_l100km": 8.9, "tank_l": 54},
            {"name": "1.6 Turbo Hybrid", "displacement": "1.6L / 1598cc", "cylinders": 4,
             "config": "Inline-4", "turbo": True, "fuel": "Petrol Hybrid",
             "hp": 230, "torque_nm": 350, "transmission": "6-speed AT",
             "drivetrain": "AWD", "acceleration_0_100": 8.0, "top_speed_kmh": 193,
             "fuel_economy_l100km": 6.1, "tank_l": 42},
        ]
    },
    "kia_cerato": {
        "make": "Kia", "model": "Cerato",
        "body_type": "Sedan", "doors": 4, "seats": 5,
        "production_years": "2018\u2013present", "generation": "BD (4th Gen)", "segment": "Compact Sedan",
        "dimensions": {"length_mm": 4640, "width_mm": 1800, "height_mm": 1450, "wheelbase_mm": 2700},
        "variants": [
            {"name": "1.6 MPI", "displacement": "1.6L / 1591cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 128, "torque_nm": 157, "transmission": "6-speed AT / 6-speed MT",
             "drivetrain": "FWD", "acceleration_0_100": 10.9, "top_speed_kmh": 185,
             "fuel_economy_l100km": 7.2, "tank_l": 50},
            {"name": "2.0 MPI", "displacement": "2.0L / 1999cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 147, "torque_nm": 179, "transmission": "6-speed AT",
             "drivetrain": "FWD", "acceleration_0_100": 9.7, "top_speed_kmh": 200,
             "fuel_economy_l100km": 8.0, "tank_l": 50},
        ]
    },
    "toyota_corolla": {
        "make": "Toyota", "model": "Corolla",
        "body_type": "Sedan", "doors": 4, "seats": 5,
        "production_years": "2019\u2013present", "generation": "E210 (12th Gen)", "segment": "Compact Sedan",
        "dimensions": {"length_mm": 4630, "width_mm": 1780, "height_mm": 1435, "wheelbase_mm": 2700},
        "variants": [
            {"name": "1.6 Dual VVT-i", "displacement": "1.6L / 1598cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 132, "torque_nm": 160, "transmission": "6-speed AT / CVT",
             "drivetrain": "FWD", "acceleration_0_100": 10.8, "top_speed_kmh": 188,
             "fuel_economy_l100km": 7.0, "tank_l": 50},
            {"name": "1.8 VVT-i", "displacement": "1.8L / 1798cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 138, "torque_nm": 172, "transmission": "CVT",
             "drivetrain": "FWD", "acceleration_0_100": 10.2, "top_speed_kmh": 200,
             "fuel_economy_l100km": 6.8, "tank_l": 50},
            {"name": "1.8 Hybrid", "displacement": "1.8L / 1798cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol Hybrid",
             "hp": 121, "torque_nm": 142, "transmission": "e-CVT",
             "drivetrain": "FWD", "acceleration_0_100": 11.0, "top_speed_kmh": 180,
             "fuel_economy_l100km": 4.5, "tank_l": 43},
        ]
    },
    "hyundai_tucson": {
        "make": "Hyundai", "model": "Tucson",
        "body_type": "SUV", "doors": 5, "seats": 5,
        "production_years": "2020\u2013present", "generation": "NX4 (4th Gen)", "segment": "Compact SUV",
        "dimensions": {"length_mm": 4500, "width_mm": 1865, "height_mm": 1665, "wheelbase_mm": 2680},
        "variants": [
            {"name": "1.6 Turbo GDi", "displacement": "1.6L / 1598cc", "cylinders": 4,
             "config": "Inline-4", "turbo": True, "fuel": "Petrol",
             "hp": 150, "torque_nm": 253, "transmission": "7-speed DCT",
             "drivetrain": "FWD / AWD", "acceleration_0_100": 9.5, "top_speed_kmh": 195,
             "fuel_economy_l100km": 7.8, "tank_l": 54},
            {"name": "2.0 MPI", "displacement": "2.0L / 1999cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 154, "torque_nm": 192, "transmission": "6-speed AT",
             "drivetrain": "FWD", "acceleration_0_100": 11.0, "top_speed_kmh": 185,
             "fuel_economy_l100km": 9.3, "tank_l": 54},
        ]
    },
    "hyundai_elantra": {
        "make": "Hyundai", "model": "Elantra",
        "body_type": "Sedan", "doors": 4, "seats": 5,
        "production_years": "2020\u2013present", "generation": "CN7 (7th Gen)", "segment": "Compact Sedan",
        "dimensions": {"length_mm": 4680, "width_mm": 1825, "height_mm": 1415, "wheelbase_mm": 2720},
        "variants": [
            {"name": "1.6 MPI", "displacement": "1.6L / 1591cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 123, "torque_nm": 151, "transmission": "6-speed AT / 6-speed MT",
             "drivetrain": "FWD", "acceleration_0_100": 11.0, "top_speed_kmh": 185,
             "fuel_economy_l100km": 7.1, "tank_l": 47},
        ]
    },
    "mercedes_c180": {
        "make": "Mercedes-Benz", "model": "C 180",
        "body_type": "Sedan", "doors": 4, "seats": 5,
        "production_years": "2021\u2013present", "generation": "W206 (5th Gen)", "segment": "Executive Sedan",
        "dimensions": {"length_mm": 4751, "width_mm": 1821, "height_mm": 1438, "wheelbase_mm": 2865},
        "variants": [
            {"name": "1.5 EQ Boost", "displacement": "1.5L / 1496cc", "cylinders": 4,
             "config": "Inline-4", "turbo": True, "fuel": "Petrol Mild Hybrid",
             "hp": 170, "torque_nm": 250, "transmission": "9-speed 9G-Tronic",
             "drivetrain": "RWD", "acceleration_0_100": 8.0, "top_speed_kmh": 240,
             "fuel_economy_l100km": 6.4, "tank_l": 66},
        ]
    },
    "mercedes_c200": {
        "make": "Mercedes-Benz", "model": "C 200",
        "body_type": "Sedan", "doors": 4, "seats": 5,
        "production_years": "2021\u2013present", "generation": "W206 (5th Gen)", "segment": "Executive Sedan",
        "dimensions": {"length_mm": 4751, "width_mm": 1821, "height_mm": 1438, "wheelbase_mm": 2865},
        "variants": [
            {"name": "1.5 EQ Boost", "displacement": "1.5L / 1496cc", "cylinders": 4,
             "config": "Inline-4", "turbo": True, "fuel": "Petrol Mild Hybrid",
             "hp": 204, "torque_nm": 300, "transmission": "9-speed 9G-Tronic",
             "drivetrain": "RWD", "acceleration_0_100": 7.1, "top_speed_kmh": 250,
             "fuel_economy_l100km": 6.5, "tank_l": 66},
        ]
    },
    "bmw_320i": {
        "make": "BMW", "model": "320i",
        "body_type": "Sedan", "doors": 4, "seats": 5,
        "production_years": "2018\u2013present", "generation": "G20 (7th Gen)", "segment": "Executive Sedan",
        "dimensions": {"length_mm": 4709, "width_mm": 1827, "height_mm": 1435, "wheelbase_mm": 2851},
        "variants": [
            {"name": "2.0 TwinPower Turbo", "displacement": "2.0L / 1998cc", "cylinders": 4,
             "config": "Inline-4", "turbo": True, "fuel": "Petrol",
             "hp": 184, "torque_nm": 300, "transmission": "8-speed Steptronic",
             "drivetrain": "RWD / xDrive AWD", "acceleration_0_100": 7.1, "top_speed_kmh": 240,
             "fuel_economy_l100km": 6.1, "tank_l": 59},
        ]
    },
    "bmw_318i": {
        "make": "BMW", "model": "318i",
        "body_type": "Sedan", "doors": 4, "seats": 5,
        "production_years": "2020\u2013present", "generation": "G20 (7th Gen)", "segment": "Executive Sedan",
        "dimensions": {"length_mm": 4709, "width_mm": 1827, "height_mm": 1435, "wheelbase_mm": 2851},
        "variants": [
            {"name": "1.5 TwinPower Turbo", "displacement": "1.5L / 1499cc", "cylinders": 3,
             "config": "Inline-3", "turbo": True, "fuel": "Petrol",
             "hp": 156, "torque_nm": 250, "transmission": "7-speed DCT",
             "drivetrain": "RWD", "acceleration_0_100": 8.8, "top_speed_kmh": 225,
             "fuel_economy_l100km": 5.9, "tank_l": 59},
        ]
    },
    "nissan_sunny": {
        "make": "Nissan", "model": "Sunny",
        "body_type": "Sedan", "doors": 4, "seats": 5,
        "production_years": "2020\u2013present", "generation": "N18 (3rd Gen)", "segment": "Subcompact Sedan",
        "dimensions": {"length_mm": 4490, "width_mm": 1730, "height_mm": 1505, "wheelbase_mm": 2600},
        "variants": [
            {"name": "1.6 HR16DE", "displacement": "1.6L / 1598cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 115, "torque_nm": 156, "transmission": "CVT / 5-speed MT",
             "drivetrain": "FWD", "acceleration_0_100": 12.5, "top_speed_kmh": 170,
             "fuel_economy_l100km": 7.0, "tank_l": 41},
        ]
    },
    "toyota_yaris": {
        "make": "Toyota", "model": "Yaris",
        "body_type": "Hatchback", "doors": 5, "seats": 5,
        "production_years": "2020\u2013present", "generation": "XP210 (4th Gen)", "segment": "Subcompact Hatchback",
        "dimensions": {"length_mm": 3940, "width_mm": 1745, "height_mm": 1500, "wheelbase_mm": 2550},
        "variants": [
            {"name": "1.5 Dual VVT-i", "displacement": "1.5L / 1490cc", "cylinders": 3,
             "config": "Inline-3", "turbo": False, "fuel": "Petrol",
             "hp": 120, "torque_nm": 145, "transmission": "CVT / 5-speed MT",
             "drivetrain": "FWD", "acceleration_0_100": 10.1, "top_speed_kmh": 175,
             "fuel_economy_l100km": 5.9, "tank_l": 42},
        ]
    },
    "volkswagen_golf": {
        "make": "Volkswagen", "model": "Golf",
        "body_type": "Hatchback", "doors": 5, "seats": 5,
        "production_years": "2020\u2013present", "generation": "Mk8 (8th Gen)", "segment": "Compact Hatchback",
        "dimensions": {"length_mm": 4284, "width_mm": 1789, "height_mm": 1456, "wheelbase_mm": 2620},
        "variants": [
            {"name": "1.0 TSI", "displacement": "1.0L / 999cc", "cylinders": 3,
             "config": "Inline-3", "turbo": True, "fuel": "Petrol",
             "hp": 110, "torque_nm": 200, "transmission": "6-speed AT / 6-speed MT",
             "drivetrain": "FWD", "acceleration_0_100": 10.1, "top_speed_kmh": 196,
             "fuel_economy_l100km": 5.3, "tank_l": 50},
            {"name": "1.5 TSI", "displacement": "1.5L / 1498cc", "cylinders": 4,
             "config": "Inline-4", "turbo": True, "fuel": "Petrol",
             "hp": 150, "torque_nm": 250, "transmission": "7-speed DSG",
             "drivetrain": "FWD", "acceleration_0_100": 8.5, "top_speed_kmh": 225,
             "fuel_economy_l100km": 5.9, "tank_l": 50},
        ]
    },
    "renault_megane": {
        "make": "Renault", "model": "Megane",
        "body_type": "Sedan", "doors": 4, "seats": 5,
        "production_years": "2016\u20132022", "generation": "Mk4 (4th Gen)", "segment": "Compact Sedan",
        "dimensions": {"length_mm": 4359, "width_mm": 1814, "height_mm": 1445, "wheelbase_mm": 2669},
        "variants": [
            {"name": "1.6 SCe", "displacement": "1.6L / 1598cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 115, "torque_nm": 156, "transmission": "6-speed AT / 5-speed MT",
             "drivetrain": "FWD", "acceleration_0_100": 11.5, "top_speed_kmh": 180,
             "fuel_economy_l100km": 7.2, "tank_l": 47},
        ]
    },
    "mg_zs": {
        "make": "MG", "model": "ZS",
        "body_type": "SUV", "doors": 5, "seats": 5,
        "production_years": "2017\u2013present", "generation": "2nd Gen Facelift", "segment": "Subcompact SUV",
        "dimensions": {"length_mm": 4314, "width_mm": 1809, "height_mm": 1665, "wheelbase_mm": 2585},
        "variants": [
            {"name": "1.5 VTi", "displacement": "1.5L / 1498cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 106, "torque_nm": 150, "transmission": "CVT / 5-speed MT",
             "drivetrain": "FWD", "acceleration_0_100": 11.5, "top_speed_kmh": 175,
             "fuel_economy_l100km": 7.3, "tank_l": 45},
        ]
    },
    "chevrolet_optra": {
        "make": "Chevrolet", "model": "Optra",
        "body_type": "Sedan", "doors": 4, "seats": 5,
        "production_years": "2003\u20132010 (Egypt market continues)", "generation": "J200", "segment": "Compact Sedan",
        "dimensions": {"length_mm": 4480, "width_mm": 1730, "height_mm": 1440, "wheelbase_mm": 2605},
        "variants": [
            {"name": "1.6 LS", "displacement": "1.6L / 1598cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 109, "torque_nm": 150, "transmission": "4-speed AT / 5-speed MT",
             "drivetrain": "FWD", "acceleration_0_100": 12.5, "top_speed_kmh": 175,
             "fuel_economy_l100km": 8.5, "tank_l": 52},
        ]
    },
    "toyota_fortuner": {
        "make": "Toyota", "model": "Fortuner",
        "body_type": "SUV", "doors": 5, "seats": 7,
        "production_years": "2015\u2013present", "generation": "AN160 (2nd Gen)", "segment": "Mid-size SUV",
        "dimensions": {"length_mm": 4795, "width_mm": 1855, "height_mm": 1835, "wheelbase_mm": 2745},
        "variants": [
            {"name": "2.7 VVT-i", "displacement": "2.7L / 2694cc", "cylinders": 4,
             "config": "Inline-4", "turbo": False, "fuel": "Petrol",
             "hp": 165, "torque_nm": 245, "transmission": "6-speed AT",
             "drivetrain": "4WD", "acceleration_0_100": 12.0, "top_speed_kmh": 175,
             "fuel_economy_l100km": 11.5, "tank_l": 80},
            {"name": "2.4 Diesel", "displacement": "2.4L / 2393cc", "cylinders": 4,
             "config": "Inline-4", "turbo": True, "fuel": "Diesel",
             "hp": 163, "torque_nm": 420, "transmission": "6-speed AT",
             "drivetrain": "4WD", "acceleration_0_100": 13.5, "top_speed_kmh": 175,
             "fuel_economy_l100km": 8.5, "tank_l": 80},
        ]
    },
}

# ── Autocomplete catalog ──────────────────────────────────────────────────────
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
    ("Mercedes","C 180",[2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Mercedes","C 200",[2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Mercedes","E 200",[2017,2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Mercedes","GLC 200",[2016,2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Mercedes","A 200",[2018,2019,2020,2021,2022,2023],"Hatchback"),
    ("Mercedes","CLA 200",[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"Sedan"),
    ("BMW","320i",[2018,2019,2020,2021,2022,2023,2024],"Sedan"),
    ("BMW","318i",[2019,2020,2021,2022,2023,2024],"Sedan"),
    ("BMW","316i",[2012,2013,2014,2015,2016,2017,2018,2019],"Sedan"),
    ("BMW","520i",[2017,2018,2019,2020,2021,2022,2023],"Sedan"),
    ("BMW","X1",[2016,2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("BMW","X3",[2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("BMW","M3",[2020,2021,2022,2023,2024],"Sedan"),
    ("Nissan","Sunny",[2017,2018,2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Nissan","Sentra",[2013,2014,2015,2016,2017,2018,2019,2020],"Sedan"),
    ("Nissan","Qashqai",[2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Nissan","X-Trail",[2014,2015,2016,2017,2018,2019,2020,2021],"SUV"),
    ("Nissan","Micra",[2017,2018,2019,2020,2021,2022],"Hatchback"),
    ("Audi","A3",[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Audi","A4",[2016,2017,2018,2019,2020,2021,2022,2023,2024],"Sedan"),
    ("Audi","A6",[2016,2017,2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Audi","Q3",[2019,2020,2021,2022,2023,2024],"SUV"),
    ("Audi","Q5",[2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("Chevrolet","Optra",[2004,2005,2006,2007,2008,2009,2010],"Sedan"),
    ("Chevrolet","Cruze",[2011,2012,2013,2014,2015,2016,2017],"Sedan"),
    ("Chevrolet","Aveo",[2006,2007,2008,2009,2010,2011,2012],"Sedan"),
    ("Chevrolet","Spark",[2010,2011,2012,2013,2014,2015,2016],"Hatchback"),
    ("Chevrolet","Captiva",[2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018],"SUV"),
    ("Renault","Megane",[2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021],"Sedan"),
    ("Renault","Symbol",[2012,2013,2014,2015,2016,2017,2018,2019,2020],"Sedan"),
    ("Renault","Logan",[2013,2014,2015,2016,2017,2018,2019,2020,2021],"Sedan"),
    ("Renault","Duster",[2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022],"SUV"),
    ("Peugeot","208",[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"Hatchback"),
    ("Peugeot","301",[2013,2014,2015,2016,2017,2018,2019,2020,2021],"Sedan"),
    ("Peugeot","2008",[2019,2020,2021,2022,2023,2024],"SUV"),
    ("Peugeot","3008",[2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("MG","ZS",[2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("MG","MG5",[2020,2021,2022,2023,2024],"Sedan"),
    ("MG","MG6",[2012,2013,2014,2015,2016,2017],"Sedan"),
    ("MG","HS",[2019,2020,2021,2022,2023],"SUV"),
    ("Volkswagen","Polo",[2018,2019,2020,2021,2022,2023,2024],"Hatchback"),
    ("Volkswagen","Golf",[2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"Hatchback"),
    ("Volkswagen","Passat",[2016,2017,2018,2019,2020,2021,2022],"Sedan"),
    ("Volkswagen","Tiguan",[2017,2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("Skoda","Octavia",[2015,2016,2017,2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Skoda","Rapid",[2013,2014,2015,2016,2017,2018,2019,2020,2021],"Sedan"),
    ("Skoda","Fabia",[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"Hatchback"),
    ("Ford","EcoSport",[2014,2015,2016,2017,2018,2019,2020,2021],"SUV"),
    ("Ford","Explorer",[2016,2017,2018,2019,2020,2021,2022],"SUV"),
    ("Ford","Ranger",[2015,2016,2017,2018,2019,2020,2021,2022,2023],"Pickup"),
    ("Ford","Focus",[2012,2013,2014,2015,2016,2017,2018,2019],"Hatchback"),
    ("Honda","Civic",[2016,2017,2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Honda","Accord",[2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Honda","HR-V",[2015,2016,2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Honda","CR-V",[2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Mazda","Mazda3",[2015,2016,2017,2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Mazda","CX-5",[2016,2017,2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("Mitsubishi","Lancer",[2008,2009,2010,2011,2012,2013,2014,2015,2016,2017],"Sedan"),
    ("Mitsubishi","Outlander",[2013,2014,2015,2016,2017,2018,2019,2020,2021,2022],"SUV"),
    ("Mitsubishi","Pajero",[2015,2016,2017,2018,2019,2020,2021,2022],"SUV"),
    ("Jeep","Grand Cherokee",[2014,2015,2016,2017,2018,2019,2020,2021,2022],"SUV"),
    ("Jeep","Wrangler",[2018,2019,2020,2021,2022,2023,2024],"SUV"),
    ("Jeep","Compass",[2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("BYD","Atto 3",[2022,2023,2024],"SUV"),
    ("BYD","Seal",[2022,2023,2024],"Sedan"),
    ("Chery","Tiggo 4",[2019,2020,2021,2022,2023,2024],"SUV"),
    ("Chery","Tiggo 7",[2019,2020,2021,2022,2023,2024],"SUV"),
    ("Chery","Omoda 5",[2022,2023,2024],"SUV"),
    ("Chery","Arrizo 5",[2018,2019,2020,2021,2022,2023],"Sedan"),
    ("Suzuki","Swift",[2018,2019,2020,2021,2022,2023,2024],"Hatchback"),
    ("Suzuki","Vitara",[2015,2016,2017,2018,2019,2020,2021,2022,2023],"SUV"),
    ("Suzuki","Jimny",[2019,2020,2021,2022,2023,2024],"SUV"),
    ("Opel","Corsa",[2015,2016,2017,2018,2019,2020,2021,2022,2023],"Hatchback"),
    ("Opel","Astra",[2016,2017,2018,2019,2020,2021,2022],"Hatchback"),
    ("Opel","Mokka",[2021,2022,2023,2024],"SUV"),
    ("Fiat","Tipo",[2016,2017,2018,2019,2020,2021,2022,2023,2024],"Sedan"),
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

# ── Body type map ─────────────────────────────────────────────────────────────
MODEL_BODY_MAP = {
    "corolla":"Sedan","camry":"Sedan","avalon":"Sedan","elantra":"Sedan","sonata":"Sedan",
    "accent":"Sedan","cerato":"Sedan","k3":"Sedan","rio":"Sedan","stinger":"Sedan",
    "optima":"Sedan","k5":"Sedan","sunny":"Sedan","sentra":"Sedan","altima":"Sedan",
    "maxima":"Sedan","versa":"Sedan","c180":"Sedan","c200":"Sedan","c250":"Sedan",
    "c300":"Sedan","e200":"Sedan","e250":"Sedan","e350":"Sedan","a180":"Sedan",
    "a200":"Sedan","cla":"Sedan","316i":"Sedan","318i":"Sedan","320i":"Sedan",
    "328i":"Sedan","330i":"Sedan","520i":"Sedan","528i":"Sedan","530i":"Sedan",
    "a3":"Sedan","a4":"Sedan","a6":"Sedan","a8":"Sedan","optra":"Sedan","cruze":"Sedan",
    "aveo":"Sedan","lanos":"Sedan","viva":"Sedan","malibu":"Sedan","megane":"Sedan",
    "fluence":"Sedan","logan":"Sedan","symbol":"Sedan","laguna":"Sedan","civic":"Sedan",
    "accord":"Sedan","city":"Sedan","mazda3":"Sedan","mazda6":"Sedan","passat":"Sedan",
    "jetta":"Sedan","vento":"Sedan","octavia":"Sedan","rapid":"Sedan","superb":"Sedan",
    "tipo":"Sedan","linea":"Sedan","lancer":"Sedan","galant":"Sedan","mg5":"Sedan",
    "mg6":"Sedan","mg7":"Sedan","arrizo":"Sedan","pegas":"Sedan",
    "yaris":"Hatchback","vitz":"Hatchback","aygo":"Hatchback","i10":"Hatchback",
    "i20":"Hatchback","grand i10":"Hatchback","picanto":"Hatchback","morning":"Hatchback",
    "micra":"Hatchback","march":"Hatchback","spark":"Hatchback","208":"Hatchback",
    "301":"Sedan","206":"Hatchback","207":"Hatchback","308":"Hatchback","polo":"Hatchback",
    "golf":"Hatchback","up":"Hatchback","fabia":"Hatchback","punto":"Hatchback",
    "500":"Hatchback","swift":"Hatchback","baleno":"Hatchback","alto":"Hatchback",
    "jazz":"Hatchback","fit":"Hatchback","corsa":"Hatchback","astra":"Hatchback",
    "clio":"Hatchback","twingo":"Hatchback","sandero":"Hatchback","focus":"Hatchback",
    "sportage":"SUV","sorento":"SUV","telluride":"SUV","tucson":"SUV","santa fe":"SUV",
    "ix35":"SUV","creta":"SUV","land cruiser":"SUV","rav4":"SUV","fortuner":"SUV",
    "prado":"SUV","rush":"SUV","chr":"SUV","c-hr":"SUV","qashqai":"SUV","x-trail":"SUV",
    "murano":"SUV","patrol":"SUV","pathfinder":"SUV","juke":"SUV","kicks":"SUV",
    "glc":"SUV","gle":"SUV","gls":"SUV","ml":"SUV","glb":"SUV","x1":"SUV","x3":"SUV",
    "x5":"SUV","x6":"SUV","x7":"SUV","q3":"SUV","q5":"SUV","q7":"SUV","q8":"SUV",
    "captiva":"SUV","trax":"SUV","traverse":"SUV","tahoe":"SUV","2008":"SUV","3008":"SUV",
    "5008":"SUV","tiguan":"SUV","touareg":"SUV","t-roc":"SUV","kodiaq":"SUV","karoq":"SUV",
    "compass":"SUV","renegade":"SUV","cherokee":"SUV","grand cherokee":"SUV","wrangler":"SUV",
    "duster":"SUV","kadjar":"SUV","koleos":"SUV","hr-v":"SUV","cr-v":"SUV","pilot":"SUV",
    "cx-3":"SUV","cx-5":"SUV","cx-9":"SUV","outlander":"SUV","eclipse cross":"SUV",
    "asx":"SUV","pajero":"SUV","vitara":"SUV","grand vitara":"SUV","jimny":"SUV",
    "s-cross":"SUV","zs":"SUV","hs":"SUV","rx5":"SUV","atto":"SUV","atto 3":"SUV",
    "tang":"SUV","tiggo 4":"SUV","tiggo 7":"SUV","omoda 5":"SUV","omoda":"SUV",
    "mokka":"SUV","ecosport":"SUV","escape":"SUV","explorer":"SUV","edge":"SUV",
    "territory":"SUV","sonet":"SUV","tiggo":"SUV",
    "m3":"Coupe","m4":"Coupe","coupe":"Coupe","slk":"Coupe","clk":"Coupe","cls":"Coupe",
    "mustang":"Coupe","mx-5":"Coupe",
    "hilux":"Pickup","ranger":"Pickup","navara":"Pickup","l200":"Pickup","d-max":"Pickup",
    "triton":"Pickup","amarok":"Pickup","f-150":"Pickup","f150":"Pickup",
    "carnival":"Van","zafira":"Van","vivaro":"Van","odyssey":"Van","staria":"Van",
    "gran max":"Van","hiace":"Van","h1":"Van","vito":"Van","v-class":"Van","sprinter":"Van",
}

def get_body_type_from_model(model_str, title=""):
    m = model_str.lower().strip() if model_str else ""
    t = title.lower().strip() if title else ""
    if m in MODEL_BODY_MAP:
        return MODEL_BODY_MAP[m]
    for key, body in MODEL_BODY_MAP.items():
        if key in m:
            return body
    for kw, body in [("suv","SUV"),("pickup","Pickup"),("hatchback","Hatchback"),("sedan","Sedan"),("van","Van")]:
        if kw in m or kw in t:
            return body
    return None

# ── Market scraper ─────────────────────────────────────────────────────────────
class DualPlatformMarketScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
        }
        self.base_url = "https://eg.hatla2ee.com"

    def scrape_hatla2ee(self, brand, model=None, page=1):
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
                            if text and not any(k in text.lower() for k in ["slide","previous","next","\u0639\u0631\u0636 \u0627\u0644\u0643\u0644"]):
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
                        if t and not any(k in t.lower() for k in ["slide","previous","next","\u0639\u0631\u0636 \u0627\u0644\u0643\u0644"]):
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
                    p_match = re.search(r'([\d,]{4,12})\s*(?:\u062c\u0646\u064a\u0647|EGP|\u062c\.\u0645|L\.E)', card_text_space)
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
                    km_match = re.search(r"([\d,]{1,8})\s*(?:\u06a9\u0645|\u0643\u0645|km|\u0643\u064a\u0644\u0648\u0645\u062a\u0631|\u0643\u064a\u0644\u0648)", card_text_space, re.IGNORECASE)
                    if km_match:
                        try:
                            mileage = float(km_match.group(1).replace(",","").strip())
                        except ValueError:
                            pass
                    elif re.search(r"\b0\s*(?:\u06a9\u0645|\u0643\u0645|km)\b", card_text_space, re.IGNORECASE):
                        mileage = 0.0
                    transmission = "Manual" if any(t in card_text_space for t in ["\u064a\u062f\u0648\u064a","\u0645\u0627\u0646\u064a\u0648\u0627\u0644","Manual"]) else "Automatic"
                    fuel_type = "Benzine"
                    if "\u0647\u062c\u064a\u0646" in card_text_space or "Hybrid" in card_text_space:
                        fuel_type = "Hybrid"
                    elif "\u0643\u0647\u0631\u0628\u0627\u0621" in card_text_space or "Electric" in card_text_space:
                        fuel_type = "Electric"
                    elif "\u063a\u0627\u0632" in card_text_space or "Gas" in card_text_space:
                        fuel_type = "Gas"
                    elif "\u062f\u064a\u0632\u0644" in card_text_space or "Diesel" in card_text_space:
                        fuel_type = "Diesel"
                    condition_tag = "Fabrika" if "\u0641\u0627\u0628\u0631\u064a\u0643\u0627" in card_text_space else "Used"
                    body_type = get_body_type_from_model(model or "", title_text)
                    location = "Cairo"
                    known_locs = ["\u0627\u0644\u0642\u0627\u0647\u0631\u0629","\u0627\u0644\u062c\u064a\u0632\u0629","\u0627\u0644\u0625\u0633\u0643\u0646\u062f\u0631\u064a\u0629","\u0627\u0644\u062a\u062c\u0645\u0639","\u0627\u0644\u0645\u0647\u0646\u062f\u0633\u064a\u0646","\u062f\u0645\u064a\u0627\u0637","\u0645\u0646\u0648\u0641\u064a\u0629","\u0627\u0644\u0634\u0631\u0642\u064a\u0629","\u0627\u0644\u062f\u0642\u0647\u0644\u064a\u0629","\u0627\u0644\u063a\u0631\u0628\u064a\u0629","\u0623\u0633\u064a\u0648\u0637","\u0633\u0648\u0647\u0627\u062c","\u0627\u0644\u0645\u0646\u064a\u0627","\u0628\u0646\u064a \u0633\u0648\u064a\u0641","\u0627\u0644\u0641\u064a\u0648\u0645","\u0625\u0633\u0645\u0627\u0639\u064a\u0644\u064a\u0629","\u0627\u0644\u0633\u0648\u064a\u0633","\u0628\u0648\u0631\u0633\u0639\u064a\u062f"]
                    for tok in [t.strip() for t in card_text_bar.split("|") if t.strip()]:
                        if any(loc in tok for loc in known_locs):
                            location = tok
                            break
                    rec_title = title_text if len(title_text) >= 3 else f"{brand.title()} {(model or '').title()} {year or ''}".strip()
                    records_by_url[full_link] = {
                        "name": rec_title, "brand": brand.title(),
                        "model": (model or "Model").title(),
                        "price": price, "year": year if year else 2024,
                        "mileage": mileage, "location": location,
                        "transmission": transmission, "fuel_type": fuel_type,
                        "body_type": body_type,
                        "car_condition": "New" if mileage == 0 else "Used",
                        "condition_tag": condition_tag, "trim_tier": "Topline",
                        "source": "Hatla2ee", "item_url": full_link, "image_url": image_url
                    }
                except Exception:
                    pass
            records = list(records_by_url.values())
        except Exception as e:
            print("Scraping Exception:", e)
        return records

live_engine = DualPlatformMarketScraper()

# ── Load CatBoost ─────────────────────────────────────────────────────────────
val_engine = None
cb_model_obj = None
if HAS_CATBOOST:
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    _cbm_path = os.path.join(_root, "catboost_model.cbm")
    _jbl_path = os.path.join(_root, "apex_catboost_valuation.joblib")
    if os.path.exists(_cbm_path):
        try:
            cb_model_obj = CatBoostRegressor()
            cb_model_obj.load_model(_cbm_path)
        except Exception as e:
            print(f"Failed catboost_model.cbm: {e}")
            cb_model_obj = None
    if cb_model_obj is None and os.path.exists(_jbl_path):
        try:
            val_engine = joblib.load(_jbl_path)
            cb_model_obj = getattr(val_engine, "model", None)
        except Exception as e:
            print(f"Failed apex_catboost_valuation.joblib: {e}")
print("CatBoost valuation: ACTIVE" if cb_model_obj else "CatBoost valuation: NOT LOADED")

def apply_filters(ads, params):
    min_price = params.get("min_price"); max_price = params.get("max_price")
    min_year = params.get("min_year"); max_year = params.get("max_year")
    mileage_preset = params.get("mileage_preset")
    min_mileage = params.get("min_mileage"); max_mileage = params.get("max_mileage")
    f_trans = params.get("transmission"); f_fuel = params.get("fuel_type"); f_body = params.get("body_type")
    out = []
    for r in ads:
        price = r.get("price"); year = r.get("year"); mileage = r.get("mileage")
        if min_price is not None and price is not None and price < min_price: continue
        if max_price is not None and price is not None and price > max_price: continue
        if min_year is not None and year is not None and year < min_year: continue
        if max_year is not None and year is not None and year > max_year: continue
        if mileage_preset == "zero":
            if mileage is None or mileage != 0.0: continue
        elif mileage_preset == "under_1000":
            if mileage is None or mileage >= 1000: continue
        elif mileage_preset == "under_100000":
            if mileage is None or mileage >= 100000: continue
        if min_mileage is not None and (mileage is None or mileage < min_mileage): continue
        if max_mileage is not None and (mileage is None or mileage > max_mileage): continue
        if f_trans and r.get("transmission","").lower() != f_trans.lower(): continue
        if f_fuel and r.get("fuel_type","").lower() != f_fuel.lower(): continue
        if f_body:
            bt = r.get("body_type")
            if bt is None or bt.lower() != f_body.lower(): continue
        out.append(r)
    return out

def calculate_match_score(query, item_name, brand, model, year):
    if not query: return None
    q_norm = normalize_digits(query.lower().strip())
    if q_norm in {"kia","toyota","mercedes","hyundai","bmw","nissan","audi"}: return None
    q_tokens = set(re.findall(r"\w+", q_norm))
    t_tokens = set(re.findall(r"\w+", normalize_digits(f"{item_name} {brand} {model} {year or ''}".lower())))
    if not q_tokens: return None
    overlap = len(q_tokens & t_tokens)
    score = (overlap / len(q_tokens)) * 100.0
    if brand.lower() in q_norm: score = max(score, 88.0)
    if model.lower() in q_norm: score = max(score, 94.0)
    if year and str(year) in q_norm: score = min(score + 4.0, 99.8)
    return round(min(score, 99.8), 1)

def predict_fallback_fair_price(brand, model, year, mileage, transmission="Automatic", condition_tag="Fabrika"):
    brand = str(brand or "").lower().strip()
    model_str = str(model or "").lower().strip()
    year = int(year) if year else None
    if not year: return None
    mileage = float(mileage) if mileage is not None else 100000.0
    base = {("kia","sportage"):2400000,("toyota","corolla"):1650000,("hyundai","tucson"):2350000,
            ("mercedes","c180"):3200000,("bmw","320i"):3100000,("nissan","sunny"):850000,
            ("hyundai","elantra"):1400000,("kia","cerato"):1300000,
            ("mg","mg5"):1200000,("renault","megane"):1350000,("chevrolet","optra"):750000}.get((brand,model_str))
    if not base:
        base = {"mercedes":3000000,"bmw":2900000,"audi":2800000,"kia":1800000,
                "hyundai":1700000,"toyota":1750000,"nissan":900000}.get(brand,0)
    if not base: return None
    age = max(0, 2026 - year)
    val = base * (0.925 ** age) - ((mileage - age * 15000) * 1.5)
    if str(transmission).lower() == "manual": val *= 0.93
    if str(condition_tag).lower() == "fabrika": val *= 1.03
    return float(round(max(val, 150000.0), 0))

def process_search_results(ads, query=""):
    if not ads: return []
    predicted = [None] * len(ads)
    vsource = "rule-based"
    if HAS_CATBOOST and Pool is not None and cb_model_obj is not None:
        try:
            num_cols = ["year","mileage","car_age","km_per_year"]
            cat_cols = ["brand","model","location","transmission","fuel_type","car_condition","condition_tag","trim_tier"]
            medians = {"year":2016.0,"mileage":122000.0,"car_age":10.0,"km_per_year":11600.0}
            matrix = []
            for r in ads:
                yr = int(r.get("year")) if r.get("year") else 2024
                raw_km = r.get("mileage")
                km = float(raw_km) if raw_km is not None else medians["mileage"]
                age = max(0, 2026 - yr)
                matrix.append([float(yr),float(km),float(age),km/max(1,age) if age>0 else km,
                    str(r.get("brand","Kia")).title(),str(r.get("model","Sportage")).title(),
                    str(r.get("location","Cairo")).title(),str(r.get("transmission","Automatic")).title(),
                    str(r.get("fuel_type","Benzine")).title(),"New" if raw_km==0 else "Used",
                    str(r.get("condition_tag","Fabrika")).title(),str(r.get("trim_tier","Topline")).title()])
            cat_idx = list(range(len(num_cols), len(num_cols)+len(cat_cols)))
            pool = Pool(data=matrix, cat_features=cat_idx)
            raw = cb_model_obj.predict(pool)
            predicted = [float(round(np.expm1(p),0)) if not np.isnan(p) and p>0 else None for p in raw]
            vsource = "CatBoost"
        except Exception as e:
            print("CatBoost inference error:", e)
            predicted = [None]*len(ads)
    for idx, r in enumerate(ads):
        if not predicted[idx]:
            predicted[idx] = predict_fallback_fair_price(r.get("brand"),r.get("model"),r.get("year"),r.get("mileage"),r.get("transmission"),r.get("condition_tag"))
    out = []
    for idx, r in enumerate(ads):
        url = r.get("item_url")
        valid_url = is_valid_vehicle_url(url)
        price_val = float(r["price"]) if r.get("price") and r["price"]>0 else None
        fair_val = predicted[idx] if idx<len(predicted) else None
        this_src = vsource if fair_val is not None else None
        deal = None
        if fair_val and price_val:
            pct = (price_val - fair_val) / fair_val
            if pct <= -0.05: deal = "Great Deal \U0001f525"
            elif pct >= 0.08: deal = "Overpriced \u26a0\ufe0f"
            else: deal = "Fair Market Price \u2696\ufe0f"
        out.append({
            "name": str(r.get("name","Vehicle")), "brand": str(r.get("brand","")),
            "model": str(r.get("model","")), "body_type": r.get("body_type"),
            "price": price_val, "predicted_fair_price": fair_val,
            "valuation_source": this_src, "deal_label": deal,
            "year": int(r.get("year",2024)) if r.get("year") else None,
            "mileage": float(r["mileage"]) if r.get("mileage") is not None else None,
            "location": str(r.get("location","Cairo")),
            "transmission": str(r.get("transmission","Automatic")),
            "fuel_type": str(r.get("fuel_type","Benzine")),
            "match_score": calculate_match_score(query,str(r.get("name","")),str(r.get("brand","")),str(r.get("model","")),int(r.get("year")) if r.get("year") else None),
            "item_url": url if valid_url else None, "has_valid_url": valid_url,
            "image_url": r.get("image_url")
        })
    return out

def normalize_query(query):
    q = normalize_digits(query.strip())
    q = re.sub(r"\s+", " ", q)
    brands_map = {
        "kia":"kia","\u0643\u064a\u0627":"kia","mercedes":"mercedes","mercedes-benz":"mercedes",
        "\u0645\u0631\u0633\u064a\u062f\u0633":"mercedes","hyundai":"hyundai","\u0647\u064a\u0648\u0646\u062f\u0627\u064a":"hyundai",
        "toyota":"toyota","\u062a\u0648\u064a\u0648\u062a\u0627":"toyota","bmw":"bmw","nissan":"nissan","\u0646\u064a\u0633\u0627\u0646":"nissan",
        "audi":"audi","\u0623\u0648\u062f\u064a":"audi","mitsubishi":"mitsubishi","chevrolet":"chevrolet",
        "renault":"renault","\u0631\u064a\u0646\u0648":"renault","peugeot":"peugeot","mg":"mg",
        "chery":"chery","skoda":"skoda","volkswagen":"volkswagen","vw":"volkswagen",
        "fiat":"fiat","jeep":"jeep","ford":"ford","honda":"honda","mazda":"mazda",
        "suzuki":"suzuki","opel":"opel","byd":"byd","haval":"haval","changan":"changan","geely":"geely",
    }
    models_map = {
        "sportage":"sportage","spo":"sportage","corolla":"corolla","tucson":"tucson",
        "c180":"c180","c 180":"c180","c200":"c200","c 200":"c200","e200":"e200","e 200":"e200",
        "cla":"cla","glc":"glc","sunny":"sunny","cerato":"cerato","elantra":"elantra",
        "accent":"accent","yaris":"yaris","fortuner":"fortuner","320i":"320i","318i":"318i",
        "316i":"316i","520i":"520i","megane":"megane","optra":"optra","zs":"zs","mg5":"mg5",
        "mg6":"mg6","golf":"golf","polo":"polo","tiguan":"tiguan","passat":"passat",
        "octavia":"octavia","rapid":"rapid","duster":"duster","qashqai":"qashqai",
        "lancer":"lancer","sorento":"sorento","picanto":"picanto","hilux":"hilux",
        "prado":"prado","civic":"civic","accord":"accord","cruze":"cruze","aveo":"aveo",
        "rav4":"rav4",
    }
    q_lower = q.lower()
    detected_brand = None; detected_model = None; detected_year = None
    year_match = re.search(r"\b(19\d{2}|20[012]\d)\b", q_lower)
    if year_match:
        detected_year = int(year_match.group(1))
    else:
        short_year = re.search(r"\b([1-9]\d)\b", q_lower)
        if short_year:
            yr_raw = int(short_year.group(1))
            if 10 <= yr_raw <= 30:
                detected_year = 2000 + yr_raw
    for k, v in brands_map.items():
        if re.search(r"\b" + re.escape(k) + r"\b", q_lower) or k in q_lower:
            detected_brand = v; break
    for k, v in models_map.items():
        if re.search(r"\b" + re.escape(k) + r"\b", q_lower) or k in q_lower:
            detected_model = v; break
    if not detected_brand:
        words = [w for w in re.findall(r"[a-zA-Z\u0600-\u06FF]+", q_lower) if not w.isdigit()]
        detected_brand = words[0] if words else q_lower
    return {"make": detected_brand, "model": detected_model, "year": detected_year, "normalized_query": q}

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","service":"Apex Motors API","catboost_loaded":cb_model_obj is not None,"valuation_mode":"CatBoost" if cb_model_obj else "rule-based"})

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
    for r in final: del r["score"]
    return jsonify({"suggestions":final,"query":raw})

@app.route("/api/vehicle-specs", methods=["GET"])
def vehicle_specs():
    make = request.args.get("make","").strip().lower()
    model = request.args.get("model","").strip().lower()
    year = request.args.get("year","").strip()
    if not make or not model:
        return jsonify({"success":False,"error":"make and model are required"}), 400
    model_clean = re.sub(r"\s+","",model)
    key_variants = [f"{make}_{model}",f"{make}_{model_clean}",f"{make}_{model.replace('-','').replace(' ','')}"]
    spec = None
    for k in key_variants:
        if k in VEHICLE_SPECS_DB:
            spec = VEHICLE_SPECS_DB[k]; break
    if spec is None:
        for db_key, db_val in VEHICLE_SPECS_DB.items():
            db_make, _, db_model = db_key.partition("_")
            if db_make == make and (model in db_model or db_model in model or model_clean in db_model):
                spec = db_val; break
    if spec is None:
        return jsonify({"success":False,"error":f"No specification data found for {make} {model}. Try searching the marketplace directly.","make":make.title(),"model":model.title(),"year":year or None}), 404
    variants_out = []
    for v in spec.get("variants",[]):
        variants_out.append({
            "name": v["name"],
            "displacement": {"value":v["displacement"],"source":"verified"},
            "cylinders": {"value":v["cylinders"],"source":"verified"},
            "config": {"value":v["config"],"source":"verified"},
            "turbo": {"value":v["turbo"],"source":"verified"},
            "fuel": {"value":v["fuel"],"source":"verified"},
            "hp": {"value":v["hp"],"source":"verified"},
            "torque_nm": {"value":v["torque_nm"],"source":"verified"},
            "transmission": {"value":v["transmission"],"source":"verified"},
            "drivetrain": {"value":v["drivetrain"],"source":"verified"},
            "acceleration_0_100": {"value":v.get("acceleration_0_100"),"source":"verified" if v.get("acceleration_0_100") else "unknown"},
            "top_speed_kmh": {"value":v.get("top_speed_kmh"),"source":"verified" if v.get("top_speed_kmh") else "unknown"},
            "fuel_economy_l100km": {"value":v.get("fuel_economy_l100km"),"source":"verified" if v.get("fuel_economy_l100km") else "unknown"},
            "tank_l": {"value":v.get("tank_l"),"source":"verified" if v.get("tank_l") else "unknown"},
        })
    dims = spec.get("dimensions",{})
    return jsonify({
        "success":True,"make":spec["make"],"model":spec["model"],
        "generation":{"value":spec.get("generation",""),"source":"verified"},
        "body_type":{"value":spec.get("body_type",""),"source":"verified"},
        "doors":{"value":spec.get("doors"),"source":"verified"},
        "seats":{"value":spec.get("seats"),"source":"verified"},
        "production_years":{"value":spec.get("production_years",""),"source":"verified"},
        "segment":{"value":spec.get("segment",""),"source":"verified"},
        "dimensions":{
            "length_mm":{"value":dims.get("length_mm"),"source":"verified" if dims.get("length_mm") else "unknown"},
            "width_mm":{"value":dims.get("width_mm"),"source":"verified" if dims.get("width_mm") else "unknown"},
            "height_mm":{"value":dims.get("height_mm"),"source":"verified" if dims.get("height_mm") else "unknown"},
            "wheelbase_mm":{"value":dims.get("wheelbase_mm"),"source":"verified" if dims.get("wheelbase_mm") else "unknown"},
        },
        "variants":variants_out,"year_requested":year or None,
        "spec_source":"Apex Motors Verified Specification Database",
    })

@app.route("/api/image-proxy", methods=["GET"])
def image_proxy():
    raw_url = request.args.get("url","").strip()
    if not raw_url: return "Missing url parameter", 400
    ALLOWED_HOSTS = {"legion-images.hatla2ee.com","img.hatla2ee.com","cdn.hatla2ee.com"}
    try:
        parsed = urlparse(raw_url)
        if parsed.scheme not in ("http","https"): return "Forbidden", 403
        if parsed.netloc not in ALLOWED_HOSTS: return "Forbidden", 403
    except Exception:
        return "Bad URL", 400
    try:
        upstream = requests.get(raw_url, timeout=6, stream=True)
        content_type = upstream.headers.get("Content-Type","image/jpeg")
        if not content_type.startswith("image/"): return "Not an image", 400
        return Response(upstream.content, status=upstream.status_code,
                       headers={"Content-Type":content_type,"Access-Control-Allow-Origin":"*","Cache-Control":"public, max-age=3600"})
    except Exception as e:
        print("Image proxy error:", e)
        return "Upstream error", 502

# ── Vision health diagnostic ──────────────────────────────────────────────────
@app.route("/api/vision-health", methods=["GET"])
def vision_health():
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")
    vision_key = os.environ.get("VISION_API_KEY") or os.environ.get("GOOGLE_VISION_API_KEY")
    
    token_status = "configured" if hf_token else "missing"
    openai_key_status = "configured" if openai_key else "missing"
    vision_key_status = "configured" if vision_key else "missing"

    active_provider = "HuggingFace Hub Inference Client" if hf_token else (
        "OpenAI Vision API" if openai_key else (
            "Google Cloud Vision API" if vision_key else "None (Unconfigured)"
        )
    )

    probe_result = "not_probed"
    probe_status = 200

    if hf_token:
        try:
            from PIL import Image as _PIL_Image
            buf = io.BytesIO()
            _PIL_Image.new("RGB", (32, 32), (200, 100, 50)).save(buf, format="JPEG")
            probe_bytes = buf.getvalue()
            headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "image/jpeg"}
            resp = requests.post("https://router.huggingface.co/hf-inference/models/dima806/car_models_image_detection", data=probe_bytes, headers=headers, timeout=12)
            probe_status = resp.status_code
            probe_result = "200_ok" if resp.status_code == 200 else f"http_{resp.status_code}"
        except Exception as ex:
            probe_result = f"error: {type(ex).__name__}: {ex}"

    status_str = "ready" if (hf_token and probe_status == 200) or openai_key or vision_key else "unconfigured"

    return jsonify({
        "service": "vehicle-vision",
        "status": status_str,
        "active_provider": active_provider,
        "model": "dima806/car_models_image_detection",
        "env_vars": {
            "HF_TOKEN": token_status,
            "OPENAI_API_KEY": openai_key_status,
            "VISION_API_KEY": vision_key_status,
        },
        "probe_status": probe_status,
        "probe_result": probe_result,
        "pil_available": HAS_PIL
    })

# ── Image classification ───────────────────────────────────────────────────────
HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models/dima806/car_models_image_detection"

def _prepare_image_bytes(file_storage):
    """Read, decode, normalize orientation, convert to RGB, resize, re-encode as JPEG.
    Returns (jpeg_bytes, None) on success or (None, error_str) on failure."""
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
        from PIL import ImageOps
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

def _parse_hf_label(label_raw):
    """Parse HF label like 'Kia_Sportage' or 'Kia_Sportage_2022' into structured make/model/year.
    Does NOT manufacture or append an arbitrary year if not present in original label."""
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

def _call_hf_vision_api(img_bytes):
    """Executes real trained model inference via Hugging Face InferenceClient / Router API.
    Returns (result_dict, error_dict)."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        return None, {"success": False, "code": "VISION_SERVICE_UNAVAILABLE", "error": "HF_TOKEN environment variable is missing.", "status_code": 503}

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="hf-inference", token=hf_token)
        hf_predictions = client.image_classification(img_bytes, model="dima806/car_models_image_detection")
        
        # Convert InferenceClient output objects to dicts
        raw_list = []
        for pred in hf_predictions:
            lbl = getattr(pred, "label", None) or (pred.get("label") if isinstance(pred, dict) else "")
            sc = getattr(pred, "score", None) or (pred.get("score") if isinstance(pred, dict) else 0.0)
            raw_list.append({"label": str(lbl), "score": float(sc)})
        
        return {"data": raw_list, "model": "dima806/car_models_image_detection", "provider": "Hugging Face InferenceClient"}, None
    except Exception as ex_hub:
        print(f"[vision] InferenceClient failed: {type(ex_hub).__name__}: {ex_hub}. Trying direct router request...")

    # Direct Router HTTP POST fallback
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
                return {"data": raw_list, "model": "dima806/car_models_image_detection", "provider": "Hugging Face Router API"}, None
            return None, {"success": False, "code": "VEHICLE_NOT_IDENTIFIED", "error": "Vision API returned empty result.", "status_code": 422}
        
        if resp.status_code in (401, 403):
            return None, {"success": False, "code": "VISION_SERVICE_UNAVAILABLE", "error": f"HuggingFace API authentication failed (HTTP {resp.status_code}). Check HF_TOKEN.", "status_code": 503}
        
        return None, {"success": False, "code": "VISION_SERVICE_UNAVAILABLE", "error": f"HuggingFace vision service returned HTTP {resp.status_code}.", "status_code": 503}

    except Exception as ex:
        print(f"[vision] HF direct router request failed: {type(ex).__name__}: {ex}")
        return None, {"success": False, "code": "VISION_SERVICE_UNAVAILABLE", "error": f"Could not connect to vision AI provider: {type(ex).__name__}", "status_code": 503}

def _call_openai_vision_api(img_bytes):
    """Executes real multimodal AI vision inference via OpenAI gpt-4o-mini if OPENAI_API_KEY is present."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return None

    import base64
    import json
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
            parsed_json = json.loads(content_str)
            return parsed_json
    except Exception as ex:
        print(f"[vision] OpenAI Vision API call failed: {ex}")
        return None

@app.route("/api/classify", methods=["POST"])
def classify_image():
    # ── 1. PIL check ──────────────────────────────────────────────────────────
    if not HAS_PIL or Image is None:
        return jsonify({
            "success": False,
            "code": "INVALID_IMAGE",
            "error": "Server-side image processing library unavailable (PIL/Pillow not installed)."
        }), 500

    # ── 2. Upload presence check ──────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "code": "INVALID_IMAGE",
            "error": "No image file received. The field name must be 'image'."
        }), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "code": "INVALID_IMAGE",
            "error": "Empty file upload received."
        }), 400

    # Optional selected crop ID from frontend when multiple cars are detected
    selected_crop_id_raw = request.form.get("selected_crop_id")
    selected_crop_id = int(selected_crop_id_raw) if (selected_crop_id_raw is not None and selected_crop_id_raw.isdigit()) else None

    # ── 3. Image decode & EXIF transpose ─────────────────────────────────────
    pil_img, prep_error = prepare_image(file)
    if prep_error:
        print(f"[classify] Image prep failed: {prep_error}")
        return jsonify({
            "success": False,
            "code": "INVALID_IMAGE",
            "error": prep_error
        }), 400

    # ── 4. Vehicle Detection & Non-Car Rejection ──────────────────────────────
    detection_res = detect_vehicle_rois(pil_img)
    if not detection_res.get("is_car"):
        return jsonify({
            "success": False,
            "code": "NON_CAR_IMAGE",
            "error": detection_res.get("reason", "The uploaded image does not appear to contain a vehicle. Please upload a clear exterior photo of a car.")
        }), 422

    rois = detection_res.get("rois", [])

    # Multi-vehicle detection flow: if multiple cars detected and user hasn't chosen one yet
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
            "multiple_vehicles_detected": True,
            "vehicle_count": len(vehicle_crops),
            "vehicles": vehicle_crops,
            "message": "Multiple vehicles detected in photo. Please select which vehicle to analyze."
        }), 200

    # Single vehicle or selected crop ROI
    target_roi = rois[0] if (selected_crop_id is None or selected_crop_id >= len(rois)) else rois[selected_crop_id]
    target_bbox = target_roi["bbox"]

    # Crop to vehicle ROI & re-encode
    img_bytes = crop_to_bytes(pil_img, target_bbox)
    cropped_pil = pil_img.crop((target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]))

    # ── 5. Visual Similarity Feature Embedding ────────────────────────────────
    visual_emb = extract_visual_embedding(cropped_pil)

    # ── 6. Real AI Model Inference Pipeline ───────────────────────────────────
    # A. Try Hugging Face Inference API / Client first if HF_TOKEN is present
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        ai_result, ai_err = call_hf_vision_api(img_bytes)
        if ai_result and ai_result.get("data"):
            raw_list = ai_result["data"]
            top = raw_list[0]
            top_label = top.get("label", "").strip()
            top_score = float(top.get("score", 0.0))

            # Strictly enforce confidence threshold: score < 0.40 -> VEHICLE_NOT_IDENTIFIED
            if top_score < 0.40 or not top_label:
                return jsonify({
                    "success": False,
                    "code": "VEHICLE_NOT_IDENTIFIED",
                    "error": "Vehicle not identified with sufficient confidence. Try uploading a clearer exterior photo showing the front or rear of the car.",
                    "raw_confidence": round(top_score * 100, 1)
                }), 422

            parsed_top = parse_hf_label(top_label)

            # Top-K real predictions with exact model output probabilities
            alternatives = []
            for item in raw_list[1:6]:
                alt_label = item.get("label", "").strip()
                alt_score = float(item.get("score", 0.0))
                if alt_label and alt_score >= 0.03:
                    parsed_alt = parse_hf_label(alt_label)
                    alternatives.append({
                        "label": parsed_alt["display_label"],
                        "make": parsed_alt["make"],
                        "model": parsed_alt["model"],
                        "year_detected": parsed_alt["year_detected"],
                        "confidence": round(alt_score * 100, 1)
                    })

            is_uncertain = bool(top_score < 0.75 and len(alternatives) > 0)

            return jsonify({
                "success": True,
                "is_uncertain": is_uncertain,
                "label": parsed_top["display_label"],
                "make": parsed_top["make"],
                "model": parsed_top["model"],
                "year_detected": parsed_top["year_detected"],
                "confidence": round(top_score * 100, 1),
                "alternatives": alternatives,
                "visual_embedding": visual_emb,
                "engine": "huggingface",
                "model_used": ai_result.get("model", "dima806/car_models_image_detection"),
                "provider": ai_result.get("provider", "Hugging Face Inference API"),
                "note": "Uncertain identification — please select candidate model below." if is_uncertain else "Visual identification via trained computer vision model. Exact specs require separate lookup."
            }), 200
        
        if ai_err:
            return jsonify(ai_err), ai_err.get("status_code", 503)

    # B. Try OpenAI Multimodal Vision API if OPENAI_API_KEY is present
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        oai_res = call_openai_vision_api(img_bytes)
        if oai_res and oai_res.get("is_vehicle") and oai_res.get("make"):
            conf = float(oai_res.get("confidence", 0.85))
            if conf < 0.40:
                return jsonify({
                    "success": False,
                    "code": "VEHICLE_NOT_IDENTIFIED",
                    "error": "Vehicle not identified with sufficient confidence.",
                    "raw_confidence": round(conf * 100, 1)
                }), 422

            make_str = str(oai_res["make"])
            model_str = str(oai_res.get("model", ""))
            label_str = f"{make_str} {model_str}".strip()

            alts = []
            for alt in oai_res.get("alternatives", []):
                m_a = alt.get("make", "")
                md_a = alt.get("model", "")
                c_a = float(alt.get("confidence", 0.1))
                alts.append({
                    "label": f"{m_a} {md_a}".strip(),
                    "make": m_a,
                    "model": md_a,
                    "year_detected": None,
                    "confidence": round(c_a * 100, 1)
                })

            is_uncertain = bool(conf < 0.75 and len(alts) > 0)

            return jsonify({
                "success": True,
                "is_uncertain": is_uncertain,
                "label": label_str,
                "make": make_str,
                "model": model_str,
                "year_detected": None,
                "estimated_year_range": [oai_res.get("estimated_year_from"), oai_res.get("estimated_year_to")],
                "confidence": round(conf * 100, 1),
                "alternatives": alts,
                "visual_embedding": visual_emb,
                "engine": "openai_vision",
                "model_used": "gpt-4o-mini",
                "provider": "OpenAI Multimodal Vision API",
                "note": "Uncertain identification — please select candidate model below." if is_uncertain else "Visual identification via OpenAI Multimodal Vision AI."
            }), 200

    # C. No AI Provider Credentials Configured / Offline
    return jsonify({
        "success": False,
        "code": "VISION_SERVICE_UNAVAILABLE",
        "error": "Vehicle analysis service is temporarily unavailable. AI vision credentials (HF_TOKEN or OPENAI_API_KEY) are missing or offline."
    }), 503

@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q","").strip()
    page = int(request.args.get("page",1))
    def _float(k):
        v = request.args.get(k)
        try: return float(v) if v else None
        except: return None
    def _int(k):
        v = request.args.get(k)
        try: return int(v) if v else None
        except: return None
    filter_params = {
        "min_price":_float("min_price"),"max_price":_float("max_price"),
        "min_year":_int("min_year"),"max_year":_int("max_year"),
        "mileage_preset":request.args.get("mileage_preset") or None,
        "min_mileage":_float("min_mileage"),"max_mileage":_float("max_mileage"),
        "transmission":request.args.get("transmission") or None,
        "fuel_type":request.args.get("fuel_type") or None,
        "body_type":request.args.get("body_type") or None,
    }
    has_filters = any(v is not None for v in filter_params.values())
    if not query:
        return jsonify({"query":"","brand":"","model":"","page":page,"has_more":False,"results":[],"filters_active":False})
    parsed = normalize_query(query)
    detected_brand = parsed["make"]; detected_model = parsed["model"]
    if parsed["year"] and not filter_params.get("min_year") and not filter_params.get("max_year"):
        filter_params["min_year"] = parsed["year"]; filter_params["max_year"] = parsed["year"]; has_filters = True
    all_raw = []; seen_urls = set()
    max_pages = 3 if has_filters else 2
    for extra_page in range(page, page + max_pages):
        batch = live_engine.scrape_hatla2ee(detected_brand, detected_model, page=extra_page)
        if not batch: break
        for item in batch:
            u = item.get("item_url")
            if u not in seen_urls:
                seen_urls.add(u); all_raw.append(item)
        if len(apply_filters(all_raw, filter_params)) >= 24: break
    filtered = apply_filters(all_raw, filter_params)
    prices = [r["price"] for r in filtered if r.get("price")]
    price_stats = None
    if prices:
        price_stats = {"min":int(min(prices)),"max":int(max(prices)),"median":int(sorted(prices)[len(prices)//2]),"count":len(prices)}
    filtered = filtered[:24]
    formatted = process_search_results(filtered, query=query)
    return jsonify({
        "query":query,"brand":detected_brand,"model":detected_model or "",
        "parsed_year":parsed["year"],"page":page,"has_more":len(filtered)>=20,
        "total_loaded":len(formatted),"filters_active":has_filters,
        "price_stats":price_stats,"results":formatted
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

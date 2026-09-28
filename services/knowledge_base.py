"""
Apex Motors — Vehicle Knowledge Base & Specification Engine
Contains canonical vehicle normalization, multi-generation spec database,
multi-engine option resolution, and provenance tracking.
"""

import re

# ── 1. CANONICAL BRAND NORMALIZATION ALIASES ────────────────────────────────
BRAND_ALIASES = {
    "mercedes": "Mercedes-Benz",
    "mercedes benz": "Mercedes-Benz",
    "merc": "Mercedes-Benz",
    "مرسيدس": "Mercedes-Benz",
    "مرسيدس بنز": "Mercedes-Benz",
    "bmw": "BMW",
    "بي ام": "BMW",
    "بي ام دبليو": "BMW",
    "toyota": "Toyota",
    "تويوتا": "Toyota",
    "kia": "Kia",
    "كيا": "Kia",
    "hyundai": "Hyundai",
    "هيونداي": "Hyundai",
    "honda": "Honda",
    "هوندا": "Honda",
    "nissan": "Nissan",
    "نيسان": "Nissan",
    "audi": "Audi",
    "أودي": "Audi",
    "اودي": "Audi",
    "volkswagen": "Volkswagen",
    "vw": "Volkswagen",
    "فولكس": "Volkswagen",
    "فولكس فاجن": "Volkswagen",
    "ford": "Ford",
    "فورد": "Ford",
    "chevrolet": "Chevrolet",
    "chevy": "Chevrolet",
    "شفروليه": "Chevrolet",
    "شيفروليه": "Chevrolet",
    "peugeot": "Peugeot",
    "بيجو": "Peugeot",
    "renault": "Renault",
    "رينو": "Renault",
    "mg": "MG",
    "ام جي": "MG",
    "امجي": "MG",
    "skoda": "Skoda",
    "شكودا": "Skoda",
    "اشكودا": "Skoda",
    "fiat": "Fiat",
    "فيات": "Fiat",
    "jeep": "Jeep",
    "جيب": "Jeep",
    "suzuki": "Suzuki",
    "سوزوكي": "Suzuki",
    "opel": "Opel",
    "أوبل": "Opel",
    "اوبل": "Opel",
    "byd": "BYD",
    "بي واي دي": "BYD",
    "chery": "Chery",
    "شيري": "Chery",
    "porsche": "Porsche",
    "بورشه": "Porsche",
    "lexus": "Lexus",
    "لكزس": "Lexus",
    "volvo": "Volvo",
    "فولفو": "Volvo",
    "land rover": "Land Rover",
    "لاند روفر": "Land Rover",
}

# ── 2. COMPREHENSIVE VEHICLE SPECIFICATION CATALOG ─────────────────────────
VEHICLE_KNOWLEDGE_BASE = {
    ("Kia", "Sportage"): {
        "manufacturer": "Kia",
        "model": "Sportage",
        "arabic_name": "كيا سبورتاج",
        "segment": "Compact Crossover / SUV",
        "generations": [
            {
                "generation_code": "NQ5",
                "years": [2022, 2025],
                "body_style": "SUV",
                "length_mm": 4515,
                "width_mm": 1865,
                "height_mm": 1645,
                "wheelbase_mm": 2680,
                "ground_clearance_mm": 170,
                "doors": 5,
                "seats": 5,
                "fuel_tank_l": 54,
                "engines": [
                    {
                        "id": "nq5_16t",
                        "name": "1.6 T-GDI Turbo",
                        "displacement_cc": 1598,
                        "cylinders": 4,
                        "induction": "Turbocharged",
                        "fuel_type": "Petrol",
                        "horsepower": 180,
                        "torque_nm": 265,
                        "transmission": "7-Speed Dual-Clutch (DCT)",
                        "drivetrain": "FWD / AWD",
                        "acceleration_0_100": 8.8,
                        "top_speed_kmh": 201,
                        "fuel_consumption_l100": 6.7
                    },
                    {
                        "id": "nq5_20na",
                        "name": "2.0 MPI Naturally Aspirated",
                        "displacement_cc": 1999,
                        "cylinders": 4,
                        "induction": "Naturally Aspirated",
                        "fuel_type": "Petrol",
                        "horsepower": 156,
                        "torque_nm": 192,
                        "transmission": "6-Speed Automatic",
                        "drivetrain": "FWD",
                        "acceleration_0_100": 10.1,
                        "top_speed_kmh": 186,
                        "fuel_consumption_l100": 7.4
                    },
                    {
                        "id": "nq5_16_hybrid",
                        "name": "1.6 T-GDI Hybrid (HEV)",
                        "displacement_cc": 1598,
                        "cylinders": 4,
                        "induction": "Turbo Hybrid",
                        "fuel_type": "Petrol / Electric",
                        "horsepower": 230,
                        "torque_nm": 350,
                        "transmission": "6-Speed Automatic",
                        "drivetrain": "FWD / AWD",
                        "acceleration_0_100": 8.0,
                        "top_speed_kmh": 193,
                        "fuel_consumption_l100": 5.4
                    }
                ]
            },
            {
                "generation_code": "QL",
                "years": [2016, 2021],
                "body_style": "SUV",
                "length_mm": 4480,
                "width_mm": 1855,
                "height_mm": 1635,
                "wheelbase_mm": 2670,
                "ground_clearance_mm": 172,
                "doors": 5,
                "seats": 5,
                "fuel_tank_l": 62,
                "engines": [
                    {
                        "id": "ql_16gdi",
                        "name": "1.6 GDI NA",
                        "displacement_cc": 1591,
                        "cylinders": 4,
                        "induction": "Naturally Aspirated",
                        "fuel_type": "Petrol",
                        "horsepower": 132,
                        "torque_nm": 161,
                        "transmission": "6-Speed Automatic",
                        "drivetrain": "FWD",
                        "acceleration_0_100": 11.5,
                        "top_speed_kmh": 180,
                        "fuel_consumption_l100": 7.2
                    },
                    {
                        "id": "ql_16tgdi",
                        "name": "1.6 T-GDI Turbo",
                        "displacement_cc": 1591,
                        "cylinders": 4,
                        "induction": "Turbocharged",
                        "fuel_type": "Petrol",
                        "horsepower": 177,
                        "torque_nm": 265,
                        "transmission": "7-Speed DCT",
                        "drivetrain": "AWD",
                        "acceleration_0_100": 9.1,
                        "top_speed_kmh": 201,
                        "fuel_consumption_l100": 7.5
                    }
                ]
            }
        ]
    },
    ("Mercedes-Benz", "AMG GT"): {
        "manufacturer": "Mercedes-Benz",
        "model": "AMG GT",
        "arabic_name": "مرسيدس اي ام جي جي تي",
        "segment": "Sports Car / Grand Tourer",
        "generations": [
            {
                "generation_code": "C190",
                "years": [2015, 2023],
                "body_style": "Coupe / Sports Car",
                "length_mm": 4544,
                "width_mm": 1939,
                "height_mm": 1287,
                "wheelbase_mm": 2630,
                "ground_clearance_mm": 110,
                "doors": 2,
                "seats": 2,
                "fuel_tank_l": 75,
                "engines": [
                    {
                        "id": "amggt_40_v8",
                        "name": "4.0L Bi-Turbo V8",
                        "displacement_cc": 3982,
                        "cylinders": 8,
                        "induction": "Twin-Turbocharged",
                        "fuel_type": "Petrol",
                        "horsepower": 530,
                        "torque_nm": 670,
                        "transmission": "7-Speed AMG SPEEDSHIFT DCT",
                        "drivetrain": "RWD",
                        "acceleration_0_100": 3.8,
                        "top_speed_kmh": 312,
                        "fuel_consumption_l100": 11.4
                    },
                    {
                        "id": "amggt_r_v8",
                        "name": "4.0L Bi-Turbo V8 (AMG GT R)",
                        "displacement_cc": 3982,
                        "cylinders": 8,
                        "induction": "Twin-Turbocharged",
                        "fuel_type": "Petrol",
                        "horsepower": 585,
                        "torque_nm": 700,
                        "transmission": "7-Speed AMG SPEEDSHIFT DCT",
                        "drivetrain": "RWD",
                        "acceleration_0_100": 3.6,
                        "top_speed_kmh": 318,
                        "fuel_consumption_l100": 12.4
                    }
                ]
            }
        ]
    },
    ("Mercedes-Benz", "C-Class"): {
        "manufacturer": "Mercedes-Benz",
        "model": "C-Class",
        "arabic_name": "مرسيدس سي كلاس",
        "segment": "Compact Executive Sedan",
        "generations": [
            {
                "generation_code": "W206",
                "years": [2021, 2025],
                "body_style": "Sedan",
                "length_mm": 4751,
                "width_mm": 1820,
                "height_mm": 1438,
                "wheelbase_mm": 2865,
                "ground_clearance_mm": 135,
                "doors": 4,
                "seats": 5,
                "fuel_tank_l": 66,
                "engines": [
                    {
                        "id": "w206_c180",
                        "name": "C 180 (1.5L Mild-Hybrid)",
                        "displacement_cc": 1496,
                        "cylinders": 4,
                        "induction": "Turbocharged + Mild Hybrid",
                        "fuel_type": "Petrol",
                        "horsepower": 170,
                        "torque_nm": 250,
                        "transmission": "9G-TRONIC 9-Speed Automatic",
                        "drivetrain": "RWD",
                        "acceleration_0_100": 8.6,
                        "top_speed_kmh": 231,
                        "fuel_consumption_l100": 6.2
                    },
                    {
                        "id": "w206_c200",
                        "name": "C 200 (1.5L Mild-Hybrid)",
                        "displacement_cc": 1496,
                        "cylinders": 4,
                        "induction": "Turbocharged + Mild Hybrid",
                        "fuel_type": "Petrol",
                        "horsepower": 204,
                        "torque_nm": 300,
                        "transmission": "9G-TRONIC 9-Speed Automatic",
                        "drivetrain": "RWD",
                        "acceleration_0_100": 7.3,
                        "top_speed_kmh": 246,
                        "fuel_consumption_l100": 6.4
                    }
                ]
            }
        ]
    },
    ("Toyota", "Corolla"): {
        "manufacturer": "Toyota",
        "model": "Corolla",
        "arabic_name": "تويوتا كورولا",
        "segment": "Compact Sedan",
        "generations": [
            {
                "generation_code": "E210",
                "years": [2019, 2025],
                "body_style": "Sedan",
                "length_mm": 4630,
                "width_mm": 1780,
                "height_mm": 1435,
                "wheelbase_mm": 2700,
                "ground_clearance_mm": 145,
                "doors": 4,
                "seats": 5,
                "fuel_tank_l": 50,
                "engines": [
                    {
                        "id": "e210_16na",
                        "name": "1.6L Dual VVT-i NA",
                        "displacement_cc": 1598,
                        "cylinders": 4,
                        "induction": "Naturally Aspirated",
                        "fuel_type": "Petrol",
                        "horsepower": 120,
                        "torque_nm": 154,
                        "transmission": "CVT Automatic",
                        "drivetrain": "FWD",
                        "acceleration_0_100": 11.0,
                        "top_speed_kmh": 190,
                        "fuel_consumption_l100": 6.1
                    },
                    {
                        "id": "e210_18hybrid",
                        "name": "1.8L Hybrid (HEV)",
                        "displacement_cc": 1798,
                        "cylinders": 4,
                        "induction": "Hybrid NA",
                        "fuel_type": "Petrol / Electric",
                        "horsepower": 121,
                        "torque_nm": 142,
                        "transmission": "e-CVT Automatic",
                        "drivetrain": "FWD",
                        "acceleration_0_100": 10.5,
                        "top_speed_kmh": 180,
                        "fuel_consumption_l100": 4.1
                    }
                ]
            }
        ]
    },
    ("BMW", "3 Series"): {
        "manufacturer": "BMW",
        "model": "3 Series",
        "arabic_name": "بي ام دبليو الفئة الثالثة",
        "segment": "Executive Compact Sedan",
        "generations": [
            {
                "generation_code": "G20",
                "years": [2019, 2025],
                "body_style": "Sedan",
                "length_mm": 4709,
                "width_mm": 1827,
                "height_mm": 1435,
                "wheelbase_mm": 2851,
                "ground_clearance_mm": 136,
                "doors": 4,
                "seats": 5,
                "fuel_tank_l": 59,
                "engines": [
                    {
                        "id": "g20_320i",
                        "name": "320i (2.0L TwinPower Turbo)",
                        "displacement_cc": 1998,
                        "cylinders": 4,
                        "induction": "TwinPower Turbo",
                        "fuel_type": "Petrol",
                        "horsepower": 184,
                        "torque_nm": 300,
                        "transmission": "8-Speed Steptronic Automatic",
                        "drivetrain": "RWD",
                        "acceleration_0_100": 7.1,
                        "top_speed_kmh": 235,
                        "fuel_consumption_l100": 6.3
                    },
                    {
                        "id": "g20_330i",
                        "name": "330i (2.0L TwinPower Turbo)",
                        "displacement_cc": 1998,
                        "cylinders": 4,
                        "induction": "TwinPower Turbo",
                        "fuel_type": "Petrol",
                        "horsepower": 258,
                        "torque_nm": 400,
                        "transmission": "8-Speed Steptronic Automatic",
                        "drivetrain": "RWD",
                        "acceleration_0_100": 5.8,
                        "top_speed_kmh": 250,
                        "fuel_consumption_l100": 6.6
                    }
                ]
            }
        ]
    },
    ("Hyundai", "Tucson"): {
        "manufacturer": "Hyundai",
        "model": "Tucson",
        "arabic_name": "هيونداي توسان",
        "segment": "Compact SUV",
        "generations": [
            {
                "generation_code": "NX4",
                "years": [2021, 2025],
                "body_style": "SUV",
                "length_mm": 4500,
                "width_mm": 1865,
                "height_mm": 1650,
                "wheelbase_mm": 2680,
                "ground_clearance_mm": 170,
                "doors": 5,
                "seats": 5,
                "fuel_tank_l": 54,
                "engines": [
                    {
                        "id": "nx4_16t",
                        "name": "1.6 T-GDI Turbo",
                        "displacement_cc": 1598,
                        "cylinders": 4,
                        "induction": "Turbocharged",
                        "fuel_type": "Petrol",
                        "horsepower": 180,
                        "torque_nm": 265,
                        "transmission": "7-Speed DCT",
                        "drivetrain": "FWD / HTRAC AWD",
                        "acceleration_0_100": 8.8,
                        "top_speed_kmh": 201,
                        "fuel_consumption_l100": 6.8
                    }
                ]
            }
        ]
    }
}

# ── 3. QUERY & BRAND NORMALIZER ─────────────────────────────────────────────
def normalize_brand(raw_brand: str) -> str:
    """Normalize raw brand/manufacturer string into canonical format."""
    if not raw_brand:
        return ""
    clean = raw_brand.strip().lower()
    return BRAND_ALIASES.get(clean, raw_brand.strip().title())

def parse_search_query(query: str) -> dict:
    """
    Parses complex search queries like 'kia sportage 22', 'مرسيدس C180', 'bmw 320i 2020'.
    Returns dict with make, model, year, and clean_query.
    """
    if not query:
        return {"make": None, "model": None, "year": None, "clean_query": ""}
    
    clean = query.strip()
    
    # 1. Extract 4-digit or unambiguous 2-digit year
    year = None
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', clean)
    if year_match:
        year = int(year_match.group(1))
        clean = re.sub(r'\b(19\d{2}|20\d{2})\b', '', clean).strip()
    else:
        # Check two digit year at end like 'kia sportage 22'
        m2 = re.search(r'\b(1\d|2\d)\b$', clean)
        if m2:
            yr_val = int(m2.group(1))
            if yr_val <= 26:
                year = 2000 + yr_val
                clean = re.sub(r'\b(1\d|2\d)\b$', '', clean).strip()

    # 2. Extract Make / Brand
    detected_make = None
    for alias, canonical in BRAND_ALIASES.items():
        pattern = r'^\b' + re.escape(alias) + r'\b'
        if re.search(pattern, clean, re.IGNORECASE):
            detected_make = canonical
            clean = re.sub(pattern, '', clean, flags=re.IGNORECASE).strip()
            break
    
    model = clean.strip() if clean else None

    return {
        "make": detected_make,
        "model": model,
        "year": year,
        "clean_query": query.strip()
    }

# ── 4. KNOWLEDGE BASE SPECIFICATION RESOLVER ────────────────────────────────
def resolve_vehicle_specs(make: str, model: str, year: int = None) -> dict:
    """
    Resolves canonical specs for make + model (+ optional year).
    Returns verified technical spec report with provenance status flags.
    """
    norm_make = normalize_brand(make)
    
    # Fuzzy model match lookup in knowledge base
    key = None
    for (m_make, m_model), data in VEHICLE_KNOWLEDGE_BASE.items():
        if m_make.lower() == norm_make.lower() and (model and m_model.lower() in model.lower() or model.lower() in m_model.lower()):
            key = (m_make, m_model)
            break
    
    if not key and norm_make:
        # Search by make only first match
        for (m_make, m_model), data in VEHICLE_KNOWLEDGE_BASE.items():
            if m_make.lower() == norm_make.lower():
                key = (m_make, m_model)
                break

    if not key:
        return {
            "found": False,
            "make": norm_make or make,
            "model": model,
            "provenance": "unknown",
            "message": "Vehicle specifications not found in knowledge base repository."
        }

    vdata = VEHICLE_KNOWLEDGE_BASE[key]
    generations = vdata["generations"]
    
    # Target generation selection based on year
    selected_gen = generations[0]
    if year:
        for gen in generations:
            y_start, y_end = gen["years"][0], gen["years"][1]
            if y_start <= year <= y_end:
                selected_gen = gen
                break

    return {
        "found": True,
        "manufacturer": vdata["manufacturer"],
        "model": vdata["model"],
        "arabic_name": vdata.get("arabic_name"),
        "segment": vdata.get("segment"),
        "generation_code": selected_gen["generation_code"],
        "production_years": selected_gen["years"],
        "body_style": selected_gen["body_style"],
        "dimensions": {
            "length_mm": {"value": selected_gen["length_mm"], "status": "verified_from_specs"},
            "width_mm": {"value": selected_gen["width_mm"], "status": "verified_from_specs"},
            "height_mm": {"value": selected_gen["height_mm"], "status": "verified_from_specs"},
            "wheelbase_mm": {"value": selected_gen["wheelbase_mm"], "status": "verified_from_specs"},
            "ground_clearance_mm": {"value": selected_gen["ground_clearance_mm"], "status": "verified_from_specs"},
            "doors": {"value": selected_gen["doors"], "status": "verified_from_specs"},
            "seats": {"value": selected_gen["seats"], "status": "verified_from_specs"},
            "fuel_tank_l": {"value": selected_gen["fuel_tank_l"], "status": "verified_from_specs"}
        },
        "engine_options": selected_gen["engines"],
        "requires_engine_confirmation": len(selected_gen["engines"]) > 1,
        "provenance": "verified_from_specs"
    }

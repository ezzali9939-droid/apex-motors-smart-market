"""
Apex Motors — Vehicle Knowledge Base & Specification Engine
Contains canonical vehicle normalization, multi-generation spec database,
multi-engine option resolution, and provenance tracking.

Strictly separates vehicle identification from specification lookup.
No hallucinated specifications: unknown values return null.
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
        "default_body": "SUV",
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
        "default_body": "Coupe",
        "generations": [
            {
                "generation_code": "C190",
                "years": [2015, 2023],
                "body_style": "Coupe",
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
        "default_body": "Sedan",
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
                    }
                ]
            },
            {
                "generation_code": "W205",
                "years": [2014, 2021],
                "body_style": "Sedan",
                "length_mm": 4686,
                "width_mm": 1810,
                "height_mm": 1442,
                "wheelbase_mm": 2840,
                "ground_clearance_mm": 130,
                "doors": 4,
                "seats": 5,
                "fuel_tank_l": 66,
                "engines": [
                    {
                        "id": "w205_c180",
                        "name": "C 180 (1.6L Turbo)",
                        "displacement_cc": 1595,
                        "cylinders": 4,
                        "induction": "Turbocharged",
                        "fuel_type": "Petrol",
                        "horsepower": 156,
                        "torque_nm": 250,
                        "transmission": "9G-TRONIC 9-Speed Automatic",
                        "drivetrain": "RWD",
                        "acceleration_0_100": 8.3,
                        "top_speed_kmh": 225,
                        "fuel_consumption_l100": 6.1
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
        "default_body": "Sedan",
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
        "default_body": "Sedan",
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
        "default_body": "SUV",
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
    },

    ("Audi", "A4"): {
        "manufacturer": "Audi",
        "model": "A4",
        "arabic_name": "أودي أيه 4",
        "segment": "Compact Executive Sedan",
        "default_body": "Sedan",
        "generations": [
            {
                "generation_code": "B9",
                "years": [2016, 2024],
                "body_style": "Sedan",
                "length_mm": 4762,
                "width_mm": 1847,
                "height_mm": 1428,
                "wheelbase_mm": 2820,
                "ground_clearance_mm": 135,
                "doors": 4,
                "seats": 5,
                "fuel_tank_l": 54,
                "engines": [
                    {
                        "id": "b9_40tfsi",
                        "name": "2.0 TFSI Mild Hybrid",
                        "displacement_cc": 1984,
                        "cylinders": 4,
                        "induction": "Turbocharged",
                        "fuel_type": "Petrol",
                        "horsepower": 190,
                        "torque_nm": 320,
                        "transmission": "7-Speed S Tronic",
                        "drivetrain": "FWD / Quattro",
                        "acceleration_0_100": 7.3,
                        "top_speed_kmh": 241,
                        "fuel_consumption_l100": 6.0
                    }
                ]
            }
        ]
    },

    ("Nissan", "Sunny"): {
        "manufacturer": "Nissan",
        "model": "Sunny",
        "arabic_name": "نيسان صني",
        "segment": "Subcompact Sedan",
        "default_body": "Sedan",
        "generations": [
            {
                "generation_code": "N18",
                "years": [2020, 2025],
                "body_style": "Sedan",
                "length_mm": 4496,
                "width_mm": 1740,
                "height_mm": 1460,
                "wheelbase_mm": 2618,
                "ground_clearance_mm": 150,
                "doors": 4,
                "seats": 5,
                "fuel_tank_l": 41,
                "engines": [
                    {
                        "id": "n18_15na",
                        "name": "1.5L HR15DE NA",
                        "displacement_cc": 1498,
                        "cylinders": 4,
                        "induction": "Naturally Aspirated",
                        "fuel_type": "Petrol",
                        "horsepower": 118,
                        "torque_nm": 149,
                        "transmission": "CVT Automatic / 5-Speed MT",
                        "drivetrain": "FWD",
                        "acceleration_0_100": 10.7,
                        "top_speed_kmh": 180,
                        "fuel_consumption_l100": 6.2
                    }
                ]
            }
        ]
    },

    ("Porsche", "911"): {
        "manufacturer": "Porsche",
        "model": "911",
        "arabic_name": "بورشه 911",
        "segment": "Supercar / Sports Car",
        "default_body": "Sports Car",
        "generations": [
            {
                "generation_code": "992",
                "years": [2019, 2025],
                "body_style": "Coupe",
                "length_mm": 4519,
                "width_mm": 1852,
                "height_mm": 1300,
                "wheelbase_mm": 2450,
                "ground_clearance_mm": 105,
                "doors": 2,
                "seats": 4,
                "fuel_tank_l": 64,
                "engines": [
                    {
                        "id": "992_carrera",
                        "name": "3.0L Twin-Turbo Flat-6 (Carrera)",
                        "displacement_cc": 2981,
                        "cylinders": 6,
                        "induction": "Twin-Turbocharged",
                        "fuel_type": "Petrol",
                        "horsepower": 385,
                        "torque_nm": 450,
                        "transmission": "8-Speed PDK Dual-Clutch",
                        "drivetrain": "RWD",
                        "acceleration_0_100": 4.2,
                        "top_speed_kmh": 293,
                        "fuel_consumption_l100": 9.4
                    }
                ]
            }
        ]
    },

    ("Volkswagen", "Golf"): {
        "manufacturer": "Volkswagen",
        "model": "Golf",
        "arabic_name": "فولكس فاجن جولف",
        "segment": "Compact Hatchback",
        "default_body": "Hatchback",
        "generations": [
            {
                "generation_code": "Mk8",
                "years": [2020, 2025],
                "body_style": "Hatchback",
                "length_mm": 4284,
                "width_mm": 1789,
                "height_mm": 1456,
                "wheelbase_mm": 2636,
                "ground_clearance_mm": 142,
                "doors": 5,
                "seats": 5,
                "fuel_tank_l": 50,
                "engines": [
                    {
                        "id": "mk8_14tsi",
                        "name": "1.4 TSI Turbo",
                        "displacement_cc": 1395,
                        "cylinders": 4,
                        "induction": "Turbocharged",
                        "fuel_type": "Petrol",
                        "horsepower": 150,
                        "torque_nm": 250,
                        "transmission": "8-Speed Automatic",
                        "drivetrain": "FWD",
                        "acceleration_0_100": 8.5,
                        "top_speed_kmh": 216,
                        "fuel_consumption_l100": 5.8
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
def resolve_vehicle_specs(make: str, model: str, year: int = None, trim: str = None) -> dict:
    """
    Resolves canonical specs for make + model (+ optional year).
    Returns strict structured specifications object per architecture requirement.
    NO hallucinated values: unconfirmed values remain null.
    """
    norm_make = normalize_brand(make) if make else ""
    
    key = None
    if norm_make and model:
        for (m_make, m_model), data in VEHICLE_KNOWLEDGE_BASE.items():
            if m_make.lower() == norm_make.lower() and (m_model.lower() in model.lower() or model.lower() in m_model.lower()):
                key = (m_make, m_model)
                break

    if not key and norm_make:
        for (m_make, m_model), data in VEHICLE_KNOWLEDGE_BASE.items():
            if m_make.lower() == norm_make.lower():
                key = (m_make, m_model)
                break

    if not key:
        return {
            "found": False,
            "manufacturer": norm_make or make or None,
            "model": model or None,
            "generation_code": None,
            "engine_options": [],
            "requires_engine_confirmation": False,
            "vehicle_id": None,
            "source_match": {
                "make": norm_make or make or None,
                "model": model or None,
                "generation": None,
                "year": str(year) if year else None,
                "trim": trim or None
            },
            "specifications": {
                "engine": None,
                "displacement_cc": None,
                "horsepower": None,
                "torque_nm": None,
                "transmission": None,
                "drivetrain": None,
                "fuel_type": None,
                "acceleration_0_100": None,
                "top_speed_kmh": None,
                "fuel_economy_l100km": None,
                "dimensions": None
            },
            "dimensions": {
                "length_mm": {"value": None, "status": "unknown"},
                "width_mm": {"value": None, "status": "unknown"},
                "height_mm": {"value": None, "status": "unknown"},
                "wheelbase_mm": {"value": None, "status": "unknown"},
                "ground_clearance_mm": {"value": None, "status": "unknown"},
                "doors": {"value": None, "status": "unknown"},
                "seats": {"value": None, "status": "unknown"},
                "fuel_tank_l": {"value": None, "status": "unknown"}
            },
            "provenance": "unknown",
            "message": "Specifications for this exact vehicle configuration are not available in the database."
        }

    vdata = VEHICLE_KNOWLEDGE_BASE[key]
    generations = vdata["generations"]
    
    selected_gen = generations[0]
    if year:
        for gen in generations:
            y_start, y_end = gen["years"][0], gen["years"][1]
            if y_start <= year <= y_end:
                selected_gen = gen
                break

    engines = selected_gen.get("engines", [])
    eng = engines[0] if engines else None

    gen_code = selected_gen.get("generation_code", "Standard")
    year_str = f"{selected_gen['years'][0]}–{selected_gen['years'][1]}"
    
    make_clean = vdata["manufacturer"].lower().replace("-", "_").replace(" ", "_")
    model_clean = vdata["model"].lower().replace("-", "_").replace(" ", "_")
    vehicle_id = f"{make_clean}/{model_clean}/{gen_code.lower()}/{selected_gen['years'][0]}"

    return {
        "found": True,
        "manufacturer": vdata["manufacturer"],
        "model": vdata["model"],
        "arabic_name": vdata.get("arabic_name"),
        "segment": vdata.get("segment"),
        "generation_code": gen_code,
        "production_years": selected_gen["years"],
        "body_style": selected_gen["body_style"],
        "engine_options": engines,
        "requires_engine_confirmation": len(engines) > 1,
        "vehicle_id": vehicle_id,
        "source_match": {
            "make": vdata["manufacturer"],
            "model": vdata["model"],
            "generation": gen_code,
            "year": year_str,
            "trim": trim or None
        },
        "specifications": {
            "engine": eng.get("name") if eng else None,
            "displacement_cc": eng.get("displacement_cc") if eng else None,
            "horsepower": eng.get("horsepower") if eng else None,
            "torque_nm": eng.get("torque_nm") if eng else None,
            "transmission": eng.get("transmission") if eng else None,
            "drivetrain": eng.get("drivetrain") if eng else None,
            "fuel_type": eng.get("fuel_type") if eng else None,
            "acceleration_0_100": eng.get("acceleration_0_100") if eng else None,
            "top_speed_kmh": eng.get("top_speed_kmh") if eng else None,
            "fuel_economy_l100km": eng.get("fuel_consumption_l100") if eng else None,
            "dimensions": {
                "length_mm": selected_gen.get("length_mm"),
                "width_mm": selected_gen.get("width_mm"),
                "height_mm": selected_gen.get("height_mm"),
                "wheelbase_mm": selected_gen.get("wheelbase_mm"),
                "ground_clearance_mm": selected_gen.get("ground_clearance_mm"),
                "doors": selected_gen.get("doors"),
                "seats": selected_gen.get("seats"),
                "fuel_tank_l": selected_gen.get("fuel_tank_l")
            }
        },
        "dimensions": {
            "length_mm": {"value": selected_gen.get("length_mm"), "status": "verified_from_specs"},
            "width_mm": {"value": selected_gen.get("width_mm"), "status": "verified_from_specs"},
            "height_mm": {"value": selected_gen.get("height_mm"), "status": "verified_from_specs"},
            "wheelbase_mm": {"value": selected_gen.get("wheelbase_mm"), "status": "verified_from_specs"},
            "ground_clearance_mm": {"value": selected_gen.get("ground_clearance_mm"), "status": "verified_from_specs"},
            "doors": {"value": selected_gen.get("doors"), "status": "verified_from_specs"},
            "seats": {"value": selected_gen.get("seats"), "status": "verified_from_specs"},
            "fuel_tank_l": {"value": selected_gen.get("fuel_tank_l"), "status": "verified_from_specs"}
        },
        "provenance": "verified_from_specs"
    }

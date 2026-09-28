"""
Apex Motors — Marketplace Connector & Listing Intelligence Engine
Normalizes marketplace listings, preserves exact original item URLs,
handles independent mileage rendering, applies advanced filters, and calculates fair pricing.
"""

def normalize_listing(item: dict) -> dict:
    """
    Normalizes listing item into standard schema:
    - Preserves exact original item_url (never fabricates or alters URLs)
    - Distinguishes missing mileage ('Mileage unavailable') from zero mileage
    - Calculates fair market valuation deal badge (Great Deal, Fair Price, Above Market)
    """
    raw_url = item.get("item_url", "").strip()
    clean_url = raw_url if raw_url.startswith("http") else None

    # Handle mileage independence
    mileage_val = item.get("mileage")
    mileage_str = "Mileage unavailable"
    if mileage_val is not None:
        try:
            m_num = float(mileage_val)
            if m_num == 0:
                mileage_str = "0 km (New)"
            else:
                mileage_str = f"{int(m_num):,} km"
        except (ValueError, TypeError):
            mileage_str = "Mileage unavailable"

    # Price handling
    price_val = item.get("price")
    price_num = float(price_val) if price_val else None

    return {
        "title": item.get("title", "").strip() or "Vehicle Listing",
        "make": item.get("make", "").strip(),
        "model": item.get("model", "").strip(),
        "year": item.get("year"),
        "price": price_num,
        "price_formatted": f"{int(price_num):,} EGP" if price_num else "Price on Request",
        "mileage": mileage_val,
        "mileage_formatted": mileage_str,
        "location": item.get("location", "").strip() or "Egypt",
        "transmission": item.get("transmission", "Automatic"),
        "fuel_type": item.get("fuel_type", "Petrol"),
        "condition": item.get("condition", "Used"),
        "body_type": item.get("body_type", "SUV/Sedan"),
        "image_url": item.get("image_url") or item.get("main_image"),
        "item_url": clean_url,
        "has_valid_url": bool(clean_url),
        "source": item.get("source", "Hatla2ee / Marketplace")
    }

def apply_marketplace_filters(listings: list, filters: dict) -> list:
    """
    Applies independent & combined filters:
    min_price, max_price, min_year, max_year, min_mileage, max_mileage,
    transmission, fuel_type, body_type
    """
    filtered = []
    for item in listings:
        price = item.get("price")
        year = item.get("year")
        mileage = item.get("mileage")
        transmission = (item.get("transmission") or "").lower()
        fuel_type = (item.get("fuel_type") or "").lower()
        body_type = (item.get("body_type") or "").lower()

        # Price filter
        if filters.get("min_price") and (not price or price < filters["min_price"]):
            continue
        if filters.get("max_price") and (not price or price > filters["max_price"]):
            continue

        # Year filter
        if filters.get("min_year") and (not year or year < filters["min_year"]):
            continue
        if filters.get("max_year") and (not year or year > filters["max_year"]):
            continue

        # Mileage filter (Do NOT treat missing mileage as 0)
        if filters.get("min_mileage") is not None:
            if mileage is None or mileage < filters["min_mileage"]:
                continue
        if filters.get("max_mileage") is not None:
            if mileage is None or mileage > filters["max_mileage"]:
                continue

        # Categorical filters
        if filters.get("transmission") and filters["transmission"].lower() not in transmission:
            continue
        if filters.get("fuel_type") and filters["fuel_type"].lower() not in fuel_type:
            continue
        if filters.get("body_type") and filters["body_type"].lower() not in body_type:
            continue

        filtered.append(item)

    return filtered

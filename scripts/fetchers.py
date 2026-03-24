"""
Dataset fetchers – one function per dataset.
Each returns a pandas DataFrame with an image_url column (if applicable).
Results are cached as CSVs in data/{dataset_name}/source.csv.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

CACHE_TIMEOUT = 86400 * 7  # re-download after 7 days


def _cache_path(data_dir: Path, name: str) -> Path:
    return data_dir / name / "source.csv"


def _is_cached(path: Path) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < CACHE_TIMEOUT


# ── Pokemon ──────────────────────────────────────────────────

POKEMON_CSV_URL = "https://raw.githubusercontent.com/lgreski/pokemonData/master/Pokemon.csv"
POKEMON_SPRITE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{}.png"


def fetch_pokemon(data_dir: Path) -> pd.DataFrame:
    cache = _cache_path(data_dir, "pokemon")
    if _is_cached(cache):
        print("  Using cached Pokemon data.")
        return pd.read_csv(cache)

    print("  Downloading Pokemon dataset...")
    df = pd.read_csv(POKEMON_CSV_URL)

    # Normalize column names for this source
    if "Type1" in df.columns:
        df = df.rename(columns={"Type1": "Type 1", "Type2": "Type 2", "ID": "pokedex_num"})
    elif "#" in df.columns:
        df = df.rename(columns={"#": "pokedex_num"})
    else:
        df["pokedex_num"] = range(1, len(df) + 1)

    df["pokedex_num"] = df["pokedex_num"].astype(int)
    df["image_url"] = df["pokedex_num"].apply(lambda n: POKEMON_SPRITE.format(n))

    # Make Generation a string for categorical use
    df["Generation_str"] = "Gen " + df["Generation"].astype(str)

    # Legendary column may or may not exist
    if "Legendary" in df.columns:
        df["Legendary"] = df["Legendary"].map({True: "Yes", False: "No"})
    else:
        df["Legendary"] = "No"

    # Fill NaN in Type 2
    df["Type 2"] = df["Type 2"].fillna("None")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  Saved {len(df)} Pokemon.")
    return df


# ── MoMA ─────────────────────────────────────────────────────

MOMA_CSV_URL = "https://media.githubusercontent.com/media/MuseumofModernArt/collection/main/Artworks.csv"


def fetch_moma(data_dir: Path) -> pd.DataFrame:
    cache = _cache_path(data_dir, "moma")
    if _is_cached(cache):
        print("  Using cached MoMA data.")
        return pd.read_csv(cache)

    print("  Downloading MoMA Artworks dataset (may take a moment)...")
    df = pd.read_csv(MOMA_CSV_URL)

    # Normalize image column name
    if "ImageURL" in df.columns and "ThumbnailURL" not in df.columns:
        df["ThumbnailURL"] = df["ImageURL"]

    # Keep only rows with a thumbnail
    df = df[df["ThumbnailURL"].notna() & (df["ThumbnailURL"].str.startswith("http"))].copy()

    # Clean columns
    df["Artist"] = df["Artist"].fillna("Unknown")
    df["Nationality"] = df["Nationality"].fillna("Unknown")
    df["Gender"] = df["Gender"].fillna("Unknown")
    df["Date"] = df["Date"].fillna("")
    df["Medium"] = df["Medium"].fillna("Unknown")
    df["Classification"] = df["Classification"].fillna("Other")
    df["Department"] = df["Department"].fillna("Other")

    # Simplify nationality: take first if multiple
    df["Nationality"] = df["Nationality"].str.strip("()").str.split(",").str[0].str.strip()

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  Saved {len(df)} MoMA artworks with thumbnails.")
    return df


# ── Art Institute of Chicago (replaces Met Museum – better API) ──

AIC_API_URL = "https://api.artic.edu/api/v1/artworks"
AIC_IMAGE_URL = "https://www.artic.edu/iiif/2/{}/full/400,/0/default.jpg"


def fetch_met_museum(data_dir: Path) -> pd.DataFrame:
    """Fetches from Art Institute of Chicago API (more reliable than Met API)."""
    cache = _cache_path(data_dir, "met_museum")
    if _is_cached(cache):
        print("  Using cached Art Institute of Chicago data.")
        return pd.read_csv(cache)

    print("  Fetching Art Institute of Chicago artworks...")
    fields = "id,title,artist_display,date_display,medium_display,department_title,place_of_origin,artwork_type_title,style_title,classification_title,image_id"

    results = []
    page = 1
    max_pages = 30  # ~3000 artworks (100 per page)

    while page <= max_pages:
        try:
            r = requests.get(AIC_API_URL, params={
                "fields": fields,
                "limit": 100,
                "page": page,
                "query[term][is_public_domain]": "true",
            }, timeout=30)
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                break
            for item in data:
                img_id = item.get("image_id")
                if not img_id:
                    continue
                results.append({
                    "title": item.get("title", "Untitled"),
                    "artistDisplayName": item.get("artist_display", "Unknown"),
                    "objectDate": item.get("date_display", ""),
                    "medium": item.get("medium_display", "Unknown"),
                    "department": item.get("department_title", "Other"),
                    "culture": item.get("place_of_origin", "") or "Unknown",
                    "classification": item.get("classification_title", "") or "Other",
                    "artworkType": item.get("artwork_type_title", "") or "Other",
                    "style": item.get("style_title", "") or "Unknown",
                    "primaryImageSmall": AIC_IMAGE_URL.format(img_id),
                })
            print(f"    Page {page}: {len(data)} items, {len(results)} with images total")
            page += 1
        except Exception as e:
            print(f"    Page {page} failed: {e}")
            break

    df = pd.DataFrame(results)

    for col in ["culture", "classification", "department", "artworkType", "style", "artistDisplayName"]:
        if col in df.columns:
            df[col] = df[col].replace("", "Unknown").fillna("Unknown")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  Saved {len(df)} Art Institute of Chicago artworks.")
    return df


# ── Movies (IMDb Top 1000) ───────────────────────────────────

IMDB_CSV_URLS = [
    "https://raw.githubusercontent.com/krishna-koly/IMDB_TOP_1000/main/imdb_top_1000.csv",
    "https://raw.githubusercontent.com/WictorDalbosco/Computer-Visualization/main/imdb_top_1000.csv",
    "https://raw.githubusercontent.com/daffaregenta/Recommender-System/main/imdb_top_1000.csv",
]


def fetch_movies(data_dir: Path) -> pd.DataFrame:
    cache = _cache_path(data_dir, "movies")
    if _is_cached(cache):
        print("  Using cached Movies data.")
        return pd.read_csv(cache)

    df = None
    for url in IMDB_CSV_URLS:
        try:
            print(f"  Trying {url} ...")
            df = pd.read_csv(url)
            if "Poster_Link" in df.columns and "Series_Title" in df.columns:
                break
        except Exception as e:
            print(f"    Failed: {e}")
            df = None

    if df is None or df.empty:
        raise RuntimeError("Could not download IMDb Top 1000 dataset from any known URL.")

    # Clean Runtime: "120 min" -> 120
    if "Runtime" in df.columns:
        df["Runtime_min"] = df["Runtime"].astype(str).str.extract(r"(\d+)").astype(float)

    # Clean Gross: "100,000,000" -> 100000000
    if "Gross" in df.columns:
        df["Gross"] = df["Gross"].astype(str).str.replace(",", "").replace("nan", "")

    # Primary genre
    if "Genre" in df.columns:
        df["Genre_Primary"] = df["Genre"].str.split(",").str[0].str.strip()

    # Fill NaN
    df["Meta_score"] = df["Meta_score"].fillna(0).astype(float)
    df["Certificate"] = df["Certificate"].fillna("Unrated")
    df["No_of_Votes"] = df["No_of_Votes"].fillna(0).astype(float)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  Saved {len(df)} movies.")
    return df


# ── E-commerce (DummyJSON) ───────────────────────────────────

DUMMYJSON_URL = "https://dummyjson.com/products?limit=0"


def fetch_ecommerce(data_dir: Path) -> pd.DataFrame:
    cache = _cache_path(data_dir, "ecommerce")
    if _is_cached(cache):
        print("  Using cached E-commerce data.")
        return pd.read_csv(cache)

    print("  Fetching products from DummyJSON API...")
    r = requests.get(DUMMYJSON_URL, timeout=30)
    r.raise_for_status()
    products = r.json().get("products", [])

    rows = []
    for p in products:
        rows.append({
            "id": p["id"],
            "title": p["title"],
            "description": p.get("description", ""),
            "price": p["price"],
            "discountPercentage": p.get("discountPercentage", 0),
            "rating": p.get("rating", 0),
            "stock": p.get("stock", 0),
            "brand": p.get("brand", "Unknown"),
            "category": p.get("category", ""),
            "thumbnail": p.get("thumbnail", ""),
        })

    df = pd.DataFrame(rows)
    df["brand"] = df["brand"].fillna("Unknown")
    df["category"] = df["category"].str.replace("-", " ").str.title()

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  Saved {len(df)} products.")
    return df



# ── dispatcher ───────────────────────────────────────────────

FETCHERS = {
    "pokemon": fetch_pokemon,
    "moma": fetch_moma,
    "met_museum": fetch_met_museum,
    "movies": fetch_movies,
    "ecommerce": fetch_ecommerce,
}


def fetch_dataset(name: str, data_dir: Path) -> pd.DataFrame:
    fn = FETCHERS.get(name)
    if not fn:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(FETCHERS.keys())}")
    return fn(data_dir)

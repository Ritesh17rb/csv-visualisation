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


# ── The Metropolitan Museum of Art ────────────────────────────

MET_SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"


def _fetch_met_object(obj_id):
    """Fetch a single Met object; returns dict or None."""
    try:
        r = requests.get(MET_OBJECT_URL.format(obj_id), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_met_museum(data_dir: Path) -> pd.DataFrame:
    """Fetches artworks from the Met Museum Open Access API."""
    cache = _cache_path(data_dir, "met_museum")
    if _is_cached(cache):
        print("  Using cached Met Museum data.")
        return pd.read_csv(cache)

    print("  Searching Met Museum for public-domain artworks with images...")
    r = requests.get(MET_SEARCH_URL, params={
        "hasImages": "true",
        "isPublicDomain": "true",
        "q": "painting",
    }, timeout=30)
    r.raise_for_status()
    all_ids = r.json().get("objectIDs", [])
    print(f"    Found {len(all_ids)} object IDs, fetching details...")

    # Shuffle and take a larger pool to ensure enough have images
    import random
    rng = random.Random(42)
    rng.shuffle(all_ids)
    pool = all_ids[:4000]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_fetch_met_object, oid): oid for oid in pool}
        for fut in as_completed(futures):
            obj = fut.result()
            if not obj:
                continue
            img = obj.get("primaryImageSmall", "")
            if not img:
                continue
            results.append({
                "title": obj.get("title", "Untitled"),
                "artistDisplayName": obj.get("artistDisplayName", "") or "Unknown",
                "objectDate": obj.get("objectDate", ""),
                "medium": obj.get("medium", "") or "Unknown",
                "department": obj.get("department", "") or "Other",
                "culture": obj.get("culture", "") or "Unknown",
                "classification": obj.get("classification", "") or "Other",
                "artworkType": obj.get("objectName", "") or "Other",
                "primaryImageSmall": img,
            })
            if len(results) % 200 == 0:
                print(f"    {len(results)} artworks with images so far...")
            if len(results) >= 2500:
                break

    print(f"    Collected {len(results)} artworks with images")
    df = pd.DataFrame(results)

    for col in ["culture", "classification", "department", "artworkType", "artistDisplayName"]:
        if col in df.columns:
            df[col] = df[col].replace("", "Unknown").fillna("Unknown")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  Saved {len(df)} Met Museum artworks.")
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



# ── Anime (MyAnimeList via Jikan) ────────────────────────

JIKAN_TOP_URL = "https://api.jikan.moe/v4/top/anime"


def fetch_anime(data_dir: Path) -> pd.DataFrame:
    cache = _cache_path(data_dir, "anime")
    if _is_cached(cache):
        print("  Using cached Anime data.")
        return pd.read_csv(cache)

    print("  Fetching top anime from Jikan API (MyAnimeList)...")
    rows = []
    page = 1
    while len(rows) < 1500:
        print(f"    Page {page} ({len(rows)} so far)...")
        r = requests.get(JIKAN_TOP_URL, params={"page": page, "limit": 25}, timeout=30)
        if r.status_code == 429:
            time.sleep(2)
            continue
        r.raise_for_status()
        data = r.json()
        items = data.get("data", [])
        if not items:
            break
        for a in items:
            img = ""
            images = a.get("images", {})
            jpg = images.get("jpg", {})
            img = jpg.get("large_image_url") or jpg.get("image_url", "")

            genres = ", ".join(g["name"] for g in a.get("genres", []))
            demographics = ", ".join(g["name"] for g in a.get("demographics", []))
            studios = ", ".join(s["name"] for s in a.get("studios", []))
            genre_primary = a["genres"][0]["name"] if a.get("genres") else "Unknown"

            rows.append({
                "title": a.get("title", ""),
                "title_english": a.get("title_english") or a.get("title", ""),
                "score": a.get("score") or 0,
                "scored_by": a.get("scored_by") or 0,
                "members": a.get("members") or 0,
                "episodes": a.get("episodes") or 0,
                "type": a.get("type") or "Unknown",
                "source": a.get("source") or "Unknown",
                "status": a.get("status") or "Unknown",
                "rating": a.get("rating") or "Unknown",
                "genres": genres or "Unknown",
                "genre_primary": genre_primary,
                "demographics": demographics or "Unknown",
                "studios": studios or "Unknown",
                "year": a.get("year") or 0,
                "image_url": img,
            })
        page += 1
        # Jikan rate limit: ~3 req/s
        time.sleep(0.5)

        if not data.get("pagination", {}).get("has_next_page", False):
            break

    print(f"    Collected {len(rows)} anime entries")
    df = pd.DataFrame(rows)

    # Clean up
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["rating"] = df["rating"].fillna("Unknown")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"  Saved {len(df)} anime.")
    return df


# ── dispatcher ───────────────────────────────────────────────

FETCHERS = {
    "pokemon": fetch_pokemon,
    "moma": fetch_moma,
    "met_museum": fetch_met_museum,
    "movies": fetch_movies,
    "ecommerce": fetch_ecommerce,
    "anime": fetch_anime,
}


def fetch_dataset(name: str, data_dir: Path) -> pd.DataFrame:
    fn = FETCHERS.get(name)
    if not fn:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(FETCHERS.keys())}")
    return fn(data_dir)

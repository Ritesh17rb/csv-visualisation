"""
CSV Visualisation – Build Pipeline
===================================
Builds one or all datasets: fetches data, computes embeddings,
projects to 2D via UMAP, clusters, and writes vis-data.json.

Usage:
    python scripts/build_dataset.py --dataset pokemon
    python scripts/build_dataset.py --all
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fetchers import fetch_dataset  # noqa: E402

DATA_DIR = ROOT / "data"

AVAILABLE_DATASETS = ["pokemon", "moma", "met_museum", "movies", "ecommerce"]

try:
    from google import genai
except ImportError:
    genai = None


# ── Gemini text embedding ──────────────────────────────────

def row_to_text(row: pd.Series, columns: list[str]) -> str:
    """Convert a DataFrame row to a descriptive text sentence for embedding."""
    parts = []
    for col in columns:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                parts.append(f"{col}: {val}")
    return ". ".join(parts)


def build_gemini_vectors(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Embed each row as text via Gemini embedding API."""
    if genai is None:
        raise ImportError("google-genai is required for text-gemini mode. Install: pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required for text-gemini mode")

    client = genai.Client(api_key=api_key)
    model = cfg.get("gemini_embedding_model", "gemini-embedding-001")
    output_dims = cfg.get("gemini_output_dims", 768)
    emb_cols = [c for c in cfg["embedding_columns"] if c in df.columns]

    # Convert rows to text
    texts = [row_to_text(row, emb_cols) for _, row in df.iterrows()]
    print(f"  Converting {len(texts)} rows to text (sample: {texts[0][:120]}...)")

    # Embed in batches — use 50 per call to stay within free-tier rate limits
    BATCH_SIZE = 50
    all_embeddings = []

    # Check for cached embeddings
    cache_path = Path(cfg.get("_out_dir", "")) / "embeddings_cache.json"
    cached = {}
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached_list = json.load(f)
            cached = {item["text_hash"]: item["embedding"] for item in cached_list}
            print(f"  Found {len(cached)} cached embeddings")
        except Exception:
            cached = {}

    import hashlib
    text_hashes = [hashlib.md5(t.encode()).hexdigest() for t in texts]

    # Find which texts need embedding
    to_embed_indices = [i for i, h in enumerate(text_hashes) if h not in cached]
    to_embed_texts = [texts[i] for i in to_embed_indices]

    if to_embed_texts:
        print(f"  Embedding {len(to_embed_texts)} texts via Gemini ({model}, {output_dims}d)...", flush=True)
        import time as _time
        import re as _re
        total_batches = (len(to_embed_texts) - 1) // BATCH_SIZE + 1

        for batch_start in range(0, len(to_embed_texts), BATCH_SIZE):
            batch = to_embed_texts[batch_start:batch_start + BATCH_SIZE]
            batch_indices = to_embed_indices[batch_start:batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1

            for attempt in range(8):
                try:
                    result = client.models.embed_content(
                        model=model,
                        contents=batch,
                        config={"output_dimensionality": output_dims},
                    )
                    for idx, emb in zip(batch_indices, result.embeddings):
                        cached[text_hashes[idx]] = emb.values
                    print(f"    Batch {batch_num}/{total_batches}: {len(batch)} texts embedded", flush=True)
                    # Small delay between batches to avoid rate limits
                    if batch_start + BATCH_SIZE < len(to_embed_texts):
                        _time.sleep(2)
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        # Extract retry delay from error if available
                        m = _re.search(r"retry\s*(?:in|Delay['\"]:\s*['\"])\s*(\d+)", err_str, _re.IGNORECASE)
                        wait = int(m.group(1)) + 2 if m else min(60, 2 ** attempt * 10)
                        print(f"    Rate limited (attempt {attempt + 1}/8), waiting {wait}s...", flush=True)
                        _time.sleep(wait)
                    else:
                        raise
            else:
                print(f"    WARNING: Skipping batch {batch_num} after 8 failed attempts — will use numerical fallback", flush=True)
                break  # skip this batch, use numerical fallback for missing

            # Save cache after each batch (resumable)
            if cache_path.parent.exists():
                cache_list = [{"text_hash": h, "embedding": v} for h, v in cached.items()]
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_list, f)

        print(f"  Cached {len(cached)} embeddings", flush=True)
    else:
        print(f"  All {len(texts)} embeddings found in cache")

    # Assemble final matrix — if some embeddings are missing, use numerical fallback
    missing = [i for i in range(len(texts)) if text_hashes[i] not in cached]
    if missing:
        print(f"  WARNING: {len(missing)} rows missing Gemini embeddings, using numerical fallback for them")
        # Build numerical vectors for fallback, padded/truncated to match embedding dim
        num_vecs = build_numerical_vectors(df, cfg)
        # Pad or truncate numerical vectors to match output_dims
        if num_vecs.shape[1] < output_dims:
            num_vecs = np.hstack([num_vecs, np.zeros((len(num_vecs), output_dims - num_vecs.shape[1]))])
        elif num_vecs.shape[1] > output_dims:
            num_vecs = num_vecs[:, :output_dims]
        all_embeddings = []
        for i in range(len(texts)):
            h = text_hashes[i]
            if h in cached:
                all_embeddings.append(cached[h])
            else:
                all_embeddings.append(num_vecs[i].tolist())
    else:
        all_embeddings = [cached[text_hashes[i]] for i in range(len(texts))]
    return np.array(all_embeddings, dtype=np.float64)


# ── helpers ──────────────────────────────────────────────────

def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_config(name: str) -> dict:
    mod = importlib.import_module(f"configs.{name}")
    return mod.CONFIG


# ── column detection ─────────────────────────────────────────

def detect_column_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    return "categorical"


def build_column_meta(df: pd.DataFrame, cfg: dict) -> dict:
    meta = {}
    all_cols = set(
        cfg.get("embedding_columns", [])
        + cfg.get("color_columns", [])
        + cfg.get("filter_columns", [])
        + cfg.get("tooltip_columns", [])
    )
    for col in all_cols:
        if col not in df.columns:
            continue
        ctype = detect_column_type(df[col])
        info: dict = {"type": ctype, "name": col}
        if ctype == "numerical":
            info["min"] = float(df[col].min())
            info["max"] = float(df[col].max())
        else:
            vals = sorted(df[col].dropna().unique().tolist(), key=str)
            if len(vals) > 50:
                vals = vals[:50]
            info["values"] = vals
        meta[col] = info
    return meta


# ── numerical embedding ─────────────────────────────────────

def build_numerical_vectors(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    emb_cols = [c for c in cfg["embedding_columns"] if c in df.columns]
    parts = []
    for col in emb_cols:
        if detect_column_type(df[col]) == "numerical":
            vals = df[col].values.astype(np.float64).reshape(-1, 1)
            parts.append(StandardScaler().fit_transform(vals))
        else:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].astype(str))
            n_classes = len(le.classes_)
            onehot = np.zeros((len(df), n_classes), dtype=np.float64)
            onehot[np.arange(len(df)), encoded] = 1.0
            parts.append(onehot)
    return np.hstack(parts)


# ── dimensionality reduction ────────────────────────────────

def reduce_vectors(vectors: np.ndarray, cfg: dict) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros((len(vectors), 2), dtype=np.float32)
    scaled = StandardScaler().fit_transform(vectors)
    if cfg.get("reducer", "umap") == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(scaled).astype(np.float32)
    n_neighbors = min(cfg.get("umap_n_neighbors", 15), len(vectors) - 1)
    reducer = UMAP(
        n_components=2,
        n_neighbors=max(5, n_neighbors),
        min_dist=cfg.get("umap_min_dist", 0.1),
        spread=cfg.get("umap_spread", 2.0),
        metric=cfg.get("umap_metric", "euclidean"),
        random_state=42,
    )
    return reducer.fit_transform(scaled).astype(np.float32)


# ── clustering ──────────────────────────────────────────────

def cluster_vectors(vectors: np.ndarray, cfg: dict) -> np.ndarray:
    if len(vectors) < 3:
        return np.zeros(len(vectors), dtype=int)
    n = cfg.get("n_clusters", 6)
    if not n or n <= 0:
        n = min(8, max(3, len(vectors) // 200))
    n = min(n, len(vectors))
    return KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(vectors)


# ── color palette ───────────────────────────────────────────

PALETTE = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#6366f1",
    "#84cc16", "#e11d48", "#0ea5e9", "#d946ef", "#a3e635",
    "#fb923c", "#2dd4bf", "#818cf8", "#fbbf24", "#34d399",
]


def build_color_map(values: list[str]) -> dict[str, str]:
    unique = sorted(set(values), key=str)
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(unique)}


# ── output ──────────────────────────────────────────────────

def write_vis_data(
    df: pd.DataFrame,
    points: np.ndarray,
    clusters: np.ndarray,
    col_meta: dict,
    cfg: dict,
    out_path: Path,
) -> None:
    color_cols = [c for c in cfg.get("color_columns", []) if c in df.columns]
    filter_cols = [c for c in cfg.get("filter_columns", []) if c in df.columns]
    tooltip_cols = [c for c in cfg.get("tooltip_columns", []) if c in df.columns]
    default_color = cfg.get("default_color", "")
    if default_color not in color_cols and color_cols:
        default_color = color_cols[0]

    range_col = cfg.get("range_column", "")
    label_col = cfg.get("label_column", "")
    image_col = cfg.get("image_column", "")

    # Color maps
    color_maps = {}
    for col in color_cols:
        if col_meta.get(col, {}).get("type") == "categorical":
            color_maps[col] = build_color_map(df[col].astype(str).tolist())

    # Build points
    items = []
    for i in range(len(df)):
        row = df.iloc[i]
        point: dict = {
            "x": float(points[i, 0]),
            "y": float(points[i, 1]),
            "cluster": int(clusters[i]),
        }

        # Label
        if label_col and label_col in df.columns:
            point["label"] = str(row[label_col])
        else:
            point["label"] = f"Row {i + 1}"

        # Range value
        if range_col and range_col in df.columns:
            val = row[range_col]
            point["rangeVal"] = float(val) if pd.notna(val) else 0.0
        else:
            point["rangeVal"] = float(i + 1)

        # Image URL
        if image_col and image_col in df.columns:
            img = row[image_col]
            point["image"] = str(img) if pd.notna(img) else ""

        # All relevant columns
        for col in set(color_cols + filter_cols + tooltip_cols):
            if col in df.columns:
                val = row[col]
                if pd.isna(val):
                    point[col] = ""
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    point[col] = round(float(val), 4)
                else:
                    point[col] = str(val)

        items.append(point)

    range_vals = [p["rangeVal"] for p in items]
    range_min = min(range_vals) if range_vals else 0
    range_max = max(range_vals) if range_vals else 1

    payload = {
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "datasetName": cfg.get("name", ""),
            "displayName": cfg.get("display_name", ""),
            "totalRows": len(items),
            "embeddingMode": cfg.get("embedding_mode", "numerical"),
            "columns": col_meta,
            "colorColumns": color_cols,
            "filterColumns": filter_cols,
            "tooltipColumns": tooltip_cols,
            "defaultColor": default_color,
            "colorMaps": color_maps,
            "rangeColumn": range_col if range_col and range_col in df.columns else "_index",
            "rangeLabel": range_col if range_col and range_col in df.columns else "Row",
            "hasImages": bool(image_col and image_col in df.columns),
        },
        "domains": {
            "x": [float(points[:, 0].min()), float(points[:, 0].max())],
            "y": [float(points[:, 1].min()), float(points[:, 1].max())],
            "range": [range_min, range_max],
        },
        "points": items,
    }

    save_json(out_path, payload)
    print(f"  Wrote {out_path} ({len(items)} points)")


# ── build one dataset ───────────────────────────────────────

def build_one(name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Building: {name}")
    print(f"{'='*60}")

    cfg = load_config(name)
    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Fetching data ...")
    df = fetch_dataset(name, DATA_DIR)
    print(f"  Raw rows: {len(df)}")

    # Sample
    sample_size = cfg.get("sample_size", 0)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=cfg.get("sample_seed", 42)).reset_index(drop=True)
        print(f"  Sampled to {len(df)} rows")

    # Drop NaN in embedding columns
    emb_cols = [c for c in cfg["embedding_columns"] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=emb_cols).reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with NaN")

    print(f"[2/5] Detecting columns ... ({len(df)} rows)")
    col_meta = build_column_meta(df, cfg)

    emb_mode = cfg.get("embedding_mode", "numerical")
    print(f"[3/5] Building embeddings ({emb_mode}) ...")
    if emb_mode == "text-gemini":
        cfg["_out_dir"] = str(out_dir)
        vectors = build_gemini_vectors(df, cfg)
    else:
        vectors = build_numerical_vectors(df, cfg)
    print(f"  Vector shape: {vectors.shape}")

    print("[4/5] Reducing to 2D ...")
    points = reduce_vectors(vectors, cfg)

    print("[5/5] Clustering & writing ...")
    clusters = cluster_vectors(vectors, cfg)
    write_vis_data(df, points, clusters, col_meta, cfg, out_dir / "vis-data.json")
    print(f"  Done: {name}")


# ── main ────────────────────────────────────────────────────

def main() -> None:
    load_env(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Build CSV visualisation datasets.")
    parser.add_argument("--dataset", choices=AVAILABLE_DATASETS, help="Build a single dataset")
    parser.add_argument("--all", action="store_true", help="Build all datasets")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Specify --dataset NAME or --all")

    datasets = AVAILABLE_DATASETS if args.all else [args.dataset]

    # Also write a manifest of all built datasets
    built = []
    for name in datasets:
        try:
            build_one(name)
            cfg = load_config(name)
            built.append({
                "name": name,
                "displayName": cfg.get("display_name", name),
                "path": f"data/{name}/vis-data.json",
            })
        except Exception as e:
            print(f"\n  ERROR building {name}: {e}")
            import traceback
            traceback.print_exc()

    # Write datasets manifest for the frontend
    manifest_path = DATA_DIR / "datasets.json"
    save_json(manifest_path, built)
    print(f"\nDataset manifest: {manifest_path} ({len(built)} datasets)")
    print("All done!")


if __name__ == "__main__":
    main()

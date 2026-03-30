from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from umap import UMAP

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

GEMINI_MODEL = "gemini-embedding-2-preview"
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent"
MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "nil", "unknown"}
TEXT_HINTS = (
    "title", "name", "description", "summary", "text", "content", "caption", "brand",
    "category", "genre", "artist", "author", "director", "tags", "keywords", "department",
)
LABEL_HINTS = ("title", "name", "label", "product", "movie", "song", "item")
IMAGE_HINTS = ("image_url", "image", "main_image", "thumbnail", "photo", "picture", "poster")
TIME_HINTS = ("year", "date", "time", "timestamp", "created", "updated", "released", "published")
ID_HINTS = ("id", "asin", "sku", "isbn", "upc", "gtin", "uuid", "url", "image", "thumbnail")
PALETTE = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#6366f1",
    "#84cc16", "#e11d48", "#0ea5e9", "#d946ef", "#a3e635",
    "#fb923c", "#2dd4bf", "#818cf8", "#fbbf24", "#34d399",
]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return str(int(value)) if float(value).is_integer() else str(round(float(value), 4))

    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return "" if text.lower() in MISSING_TOKENS else text


def nonempty_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda value: clean_text(value) != "")


def looks_like_id(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ID_HINTS)


def looks_jsonish(value: object) -> bool:
    text = clean_text(value)
    return bool(text) and text[0] in "[{" and text[-1] in "]}"


def extract_image_url(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                url = extract_image_url(item)
                if url:
                    return url
    match = re.search(r"https?://[^\s\"',\]]+", text)
    return match.group(0) if match else (text if text.startswith(("http://", "https://")) else "")


def row_to_text(row: pd.Series, columns: list[str], limit: int = 1400) -> str:
    parts: list[str] = []
    for col in columns:
        if col not in row.index:
            continue
        value = clean_text(row[col])
        if value:
            parts.append(f"{col}: {value[:240]}")
    text = ". ".join(parts)
    return text[:limit] if len(text) > limit else text


def infer_columns(df: pd.DataFrame, max_count: int = 5) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in df.columns:
        name = col.lower()
        if looks_like_id(name):
            continue
        series = df[col]
        coverage = float(nonempty_mask(series).mean())
        unique = int(series.nunique(dropna=True))
        if coverage < 0.25 or unique <= 1:
            continue

        score = coverage
        if pd.api.types.is_numeric_dtype(series):
            score -= 0.25
        else:
            sample = [clean_text(v) for v in series.head(25) if clean_text(v)]
            avg_len = sum(len(v) for v in sample) / max(1, len(sample))
            score += min(avg_len, 80) / 160
            if sample and sum(looks_jsonish(v) for v in sample) / len(sample) > 0.5:
                score -= 0.6

        for idx, hint in enumerate(TEXT_HINTS):
            if hint in name:
                score += 2.0 - idx * 0.08

        if unique >= len(df) * 0.98 and looks_like_id(name):
            continue
        scored.append((score, col))

    chosen = [col for _, col in sorted(scored, reverse=True)]
    if chosen:
        return chosen[:max_count]

    fallback = [col for col in df.columns if not looks_like_id(col.lower())]
    return fallback[: max_count or 1]


def infer_single_column(df: pd.DataFrame, hints: tuple[str, ...], validator=None) -> str:
    for hint in hints:
        for col in df.columns:
            if hint in col.lower():
                if validator is None or validator(df[col]):
                    return col
    return ""


def parse_timeline(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() >= 0.6:
        return dt.dt.year.astype("float64")
    return pd.to_numeric(series, errors="coerce")


def column_profile(series: pd.Series) -> dict[str, float | int]:
    values = [clean_text(v) for v in series.tolist()]
    nonempty = [v for v in values if v]
    unique = len(set(nonempty))
    coverage = len(nonempty) / len(values) if values else 0.0
    avg_len = sum(len(v) for v in nonempty[:100]) / max(1, min(len(nonempty), 100))
    return {"coverage": coverage, "unique": unique, "avg_len": avg_len}


def score_ui_column(name: str, profile: dict[str, float | int]) -> float:
    score = float(profile["coverage"]) * 3.0
    unique = int(profile["unique"])
    score -= abs(unique - 8) / 20
    lower = name.lower()
    for hint in ("category", "type", "genre", "group", "class", "department", "status", "brand"):
        if hint in lower:
            score += 0.6
    return score


def select_ui_columns(
    df: pd.DataFrame,
    requested: list[str],
    *,
    max_unique: int,
    min_coverage: float,
    limit: int,
    exclude: set[str],
    label: str,
) -> list[str]:
    def acceptable(col: str) -> bool:
        if col in exclude or col not in df.columns:
            return False
        lower = col.lower()
        if looks_like_id(lower):
            return False
        if any(token in lower for token in ("count", "rank", "rating", "price", "question", "review")):
            return False
        series = df[col]
        profile = column_profile(df[col])
        if int(profile["unique"]) < 2 or int(profile["unique"]) > max_unique:
            return False
        if float(profile["coverage"]) < min_coverage:
            return False
        if float(profile["avg_len"]) > 80:
            return False
        if pd.api.types.is_numeric_dtype(series) and int(profile["unique"]) > 6:
            return False
        return True

    selected = [col for col in requested if acceptable(col)]
    rejected = [col for col in requested if col not in selected]
    for col in rejected:
        profile = column_profile(df[col]) if col in df.columns else None
        if profile is None:
            print(f"  Ignoring {label} column '{col}' (missing)")
        else:
            print(
                f"  Ignoring {label} column '{col}' "
                f"(coverage={profile['coverage']:.2f}, unique={profile['unique']})"
            )

    if selected:
        return selected[:limit]

    scored: list[tuple[float, str]] = []
    for col in df.columns:
        if not acceptable(col):
            continue
        scored.append((score_ui_column(col, column_profile(df[col])), col))

    return [col for _, col in sorted(scored, reverse=True)[:limit]]


def build_color_map(values: list[str]) -> dict[str, str]:
    keys = sorted({value for value in values if value}, key=str)
    return {value: PALETTE[i % len(PALETTE)] for i, value in enumerate(keys)}


def gemini_embed_one(text: str, api_key: str, output_dim: int = 768) -> list[float]:
    url = GEMINI_EMBED_URL.format(GEMINI_MODEL) + f"?key={api_key}"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "model": f"models/{GEMINI_MODEL}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": output_dim,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]["values"]


def build_gemini_vectors(texts: list[str], cache_dir: Path) -> np.ndarray:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    cache_path = cache_dir / "embeddings_cache.json"
    cached: dict[str, list[float]] = {}
    if cache_path.exists():
        try:
            with cache_path.open(encoding="utf-8") as f:
                cached = {item["text_hash"]: item["embedding"] for item in json.load(f)}
        except Exception:
            cached = {}

    hashes = [hashlib.md5(text.encode("utf-8")).hexdigest() for text in texts]
    pending = [(h, t) for h, t in zip(hashes, texts) if h not in cached]
    if pending:
        print(f"  Building {len(pending)} Gemini embeddings...")
    else:
        print(f"  Using {len(texts)} cached Gemini embeddings")

    for index, (text_hash, text) in enumerate(pending, start=1):
        for attempt in range(8):
            try:
                cached[text_hash] = gemini_embed_one(text, api_key)
                break
            except Exception as exc:
                message = str(exc).lower()
                if any(token in message for token in ("429", "503", "timeout", "rate", "resource_exhausted")):
                    wait = min(60, 2 ** attempt * 5)
                    print(f"    Retry {attempt + 1}/8 after {wait}s")
                    _time.sleep(wait)
                    continue
                raise
        if index % 25 == 0 or index == len(pending):
            save_json(
                cache_path,
                [{"text_hash": key, "embedding": value} for key, value in cached.items()],
            )
            print(f"    {index}/{len(pending)} done")

    vectors = [cached[h] for h in hashes]
    return np.array(vectors, dtype=np.float32)


def build_vectors(df: pd.DataFrame, emb_cols: list[str], cache_dir: Path) -> np.ndarray:
    texts = [row_to_text(row, emb_cols) or f"row {idx + 1}" for idx, (_, row) in enumerate(df.iterrows())]
    sample = texts[0][:120] if texts else ""
    print(f"  Using embedding columns: {', '.join(emb_cols)}")
    print(f"  Sample text: {sample}...")
    return build_gemini_vectors(texts, cache_dir)


def reduce_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros((len(vectors), 2), dtype=np.float32)
    scaled = StandardScaler().fit_transform(vectors)
    n_neighbors = min(15, max(2, len(vectors) - 1))
    reducer = UMAP(
        n_components=2,
        n_neighbors=max(5, n_neighbors),
        min_dist=0.15,
        spread=1.5,
        metric="euclidean",
        random_state=42,
    )
    return reducer.fit_transform(scaled).astype(np.float32)


def cluster_vectors(vectors: np.ndarray, n_clusters: int) -> np.ndarray:
    if len(vectors) < 3:
        return np.zeros(len(vectors), dtype=int)
    target = n_clusters if n_clusters > 0 else min(8, max(3, len(vectors) // 200))
    target = min(target, len(vectors))
    return KMeans(n_clusters=target, random_state=42, n_init=10).fit_predict(vectors)


def build_column_meta(df: pd.DataFrame, columns: list[str], clusters: np.ndarray) -> dict:
    meta = {
        "cluster": {
            "type": "categorical",
            "name": "cluster",
            "values": [str(v) for v in sorted({int(v) for v in clusters.tolist()})],
        }
    }
    for col in columns:
        values = [clean_text(v) for v in df[col].tolist()]
        unique = sorted({value for value in values if value}, key=str)
        meta[col] = {"type": "categorical", "name": col, "values": unique[:50]}
    return meta


def write_vis_data(
    df: pd.DataFrame,
    points: np.ndarray,
    clusters: np.ndarray,
    *,
    color_cols: list[str],
    filter_cols: list[str],
    label_col: str,
    image_col: str,
    timeline_col: str,
    name: str,
    out_path: Path,
) -> None:
    tooltip_cols = [col for col in dict.fromkeys(color_cols + filter_cols) if col != "cluster"]
    col_meta = build_column_meta(df, [col for col in dict.fromkeys(color_cols + filter_cols) if col != "cluster"], clusters)

    color_maps = {"cluster": build_color_map([str(int(v)) for v in clusters.tolist()])}
    for col in color_cols:
        if col == "cluster":
            continue
        values = [clean_text(v) for v in df[col].tolist()]
        color_maps[col] = build_color_map(values)

    items = []
    for i in range(len(df)):
        row = df.iloc[i]
        point = {
            "x": float(points[i, 0]),
            "y": float(points[i, 1]),
            "cluster": int(clusters[i]),
            "label": clean_text(row[label_col]) or f"Row {i + 1}",
            "rangeVal": float(i + 1),
        }

        if timeline_col:
            value = row[timeline_col]
            point["rangeVal"] = float(value) if pd.notna(value) else float(i + 1)

        if image_col:
            point["image"] = extract_image_url(row[image_col])

        for col in tooltip_cols + [c for c in color_cols if c != "cluster"] + filter_cols:
            value = clean_text(row[col])
            point[col] = value

        items.append(point)

    range_vals = [p["rangeVal"] for p in items]
    payload = {
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "datasetName": name,
            "displayName": name,
            "totalRows": len(items),
            "columns": col_meta,
            "colorColumns": color_cols,
            "filterColumns": filter_cols,
            "tooltipColumns": tooltip_cols,
            "defaultColor": color_cols[0] if color_cols else "",
            "colorMaps": color_maps,
            "hasImages": bool(image_col),
            "hasTimeline": bool(timeline_col),
            "timelineColumn": timeline_col or "",
        },
        "domains": {
            "x": [float(points[:, 0].min()), float(points[:, 0].max())],
            "y": [float(points[:, 1].min()), float(points[:, 1].max())],
            "range": [min(range_vals), max(range_vals)],
        },
        "points": items,
    }
    save_json(out_path, payload)
    print(f"  Wrote {out_path} ({len(items)} points)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 2D CSV visualization.")
    parser.add_argument("--csv", required=True, help="Path to a CSV file")
    parser.add_argument("--embedding-columns", nargs="+", default=[], help="Columns used to build embeddings")
    parser.add_argument("--color-columns", nargs="+", default=[], help="Columns exposed for coloring")
    parser.add_argument("--filter-columns", nargs="+", default=[], help="Columns exposed for filtering")
    parser.add_argument("--label-column", default="", help="Primary label column")
    parser.add_argument("--image-column", default="", help="Image URL column")
    parser.add_argument("--timeline-column", default="", help="Date or year column")
    parser.add_argument("--clusters", type=int, default=0, help="Cluster count (0 = auto)")
    parser.add_argument("--name", default="", help="Dataset display name")
    parser.add_argument("--sample", type=int, default=0, help="Sample N rows before building")
    return parser.parse_args()


def main() -> None:
    load_env(ROOT / ".env")
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(clean_text)

    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    if not args.name:
        args.name = csv_path.stem.replace("_", " ").replace("-", " ").title()

    requested = set(args.embedding_columns + args.color_columns + args.filter_columns)
    for col in (args.label_column, args.image_column, args.timeline_column):
        if col:
            requested.add(col)
    missing = sorted(col for col in requested if col not in df.columns)
    if missing:
        raise SystemExit(f"Columns not found in CSV: {', '.join(missing)}")

    emb_cols = args.embedding_columns or infer_columns(df)
    emb_cols = [col for col in emb_cols if col in df.columns]
    if not emb_cols:
        raise SystemExit("No usable embedding columns found")

    label_col = args.label_column or infer_single_column(df, LABEL_HINTS, lambda s: nonempty_mask(s).mean() >= 0.3) or emb_cols[0]
    image_col = args.image_column or infer_single_column(df, IMAGE_HINTS, lambda s: s.map(extract_image_url).astype(bool).mean() >= 0.2)
    timeline_col = args.timeline_column or infer_single_column(df, TIME_HINTS)

    keep_mask = df[emb_cols].apply(lambda row: any(clean_text(v) for v in row), axis=1)
    dropped = int((~keep_mask).sum())
    if dropped:
        print(f"  Dropped {dropped} rows with no embedding data")
    df = df.loc[keep_mask].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No rows left after cleaning embedding columns")

    if timeline_col:
        parsed = parse_timeline(df[timeline_col])
        coverage = float(parsed.notna().mean())
        if coverage >= 0.6:
            df[timeline_col] = parsed
            print(f"  Timeline column: {timeline_col} ({coverage:.0%} usable)")
        else:
            print(f"  Ignoring timeline column '{timeline_col}' ({coverage:.0%} usable)")
            timeline_col = ""

    print(f"\n{'=' * 60}")
    print(f"  Building: {args.name}")
    print(f"  CSV: {csv_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"{'=' * 60}")

    out_dir = DATA_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building vectors ...")
    vectors = build_vectors(df, emb_cols, out_dir)
    print(f"  Vector shape: {vectors.shape}")

    print("[2/4] Reducing to 2D ...")
    points = reduce_vectors(vectors)

    print("[3/4] Clustering ...")
    clusters = cluster_vectors(vectors, args.clusters)

    exclude = {label_col, image_col, timeline_col}
    auto_colors = select_ui_columns(
        df,
        args.color_columns,
        max_unique=18,
        min_coverage=0.45,
        limit=3,
        exclude=exclude,
        label="color",
    )
    auto_filters = select_ui_columns(
        df,
        args.filter_columns,
        max_unique=24,
        min_coverage=0.45,
        limit=4,
        exclude=exclude,
        label="filter",
    )
    color_cols = ["cluster"] + [col for col in auto_colors if col != "cluster"]
    filter_cols = [col for col in auto_filters if col != "cluster"]

    print(f"  Label column: {label_col}")
    if image_col:
        print(f"  Image column: {image_col}")
    print(f"  Color columns: {', '.join(color_cols) if color_cols else '(none)'}")
    print(f"  Filter columns: {', '.join(filter_cols) if filter_cols else '(none)'}")

    print("[4/4] Writing output ...")
    write_vis_data(
        df,
        points,
        clusters,
        color_cols=color_cols,
        filter_cols=filter_cols,
        label_col=label_col,
        image_col=image_col,
        timeline_col=timeline_col,
        name=args.name,
        out_path=out_dir / "vis-data.json",
    )

    save_json(DATA_DIR / "datasets.json", [{"name": args.name, "displayName": args.name, "path": "data/output/vis-data.json"}])
    print("\nDone. Serve the repo with: python -m http.server 8000")


if __name__ == "__main__":
    main()

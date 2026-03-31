from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from .cache import EmbeddingCache
from .frontend import render_html

GEMINI_MODEL = "gemini-embedding-2-preview"
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent"
GEMINI_GEN_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
DEFAULT_OUTPUT = Path("dist/index.html")
DEFAULT_STATE_DB = Path(".csv-viz/embeddings.duckdb")
DEFAULT_STATE_PARQUET = Path(".csv-viz/embeddings.parquet")
DEFAULT_CLUSTER_NAMING_MODEL = "gemini-2.5-flash"
DEFAULT_MEDIA_MODEL = "gemini-2.5-flash"
MAX_INLINE_MEDIA_BYTES = 18 * 1024 * 1024

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


@dataclass(slots=True)
class CsvSource:
    label: str
    frame: pd.DataFrame
    csv_path: Path | None = None
    csv_url: str | None = None


@dataclass(slots=True)
class MediaRef:
    kind: str
    column: str
    raw_value: str
    display_url: str
    local_path: Path | None = None
    remote_url: str | None = None


@dataclass(slots=True)
class PreparedRow:
    row_index: int
    raw: dict[str, str]
    label: str
    text_payload: str
    audio_metadata_text: str
    images: list[MediaRef]
    audios: list[MediaRef]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            os.environ.setdefault("GEMINI_API_KEY", line.strip("\"'"))
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
    if not os.environ.get("GEMINI_API_KEY"):
        google_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if google_key:
            os.environ["GEMINI_API_KEY"] = google_key


def is_http_url(value: str) -> bool:
    return urlparse(str(value)).scheme in {"http", "https"}


def source_name(value: str) -> str:
    if is_http_url(value):
        tail = Path(urlparse(value).path).stem
        return tail or "Remote Dataset"
    return Path(value).stem


def load_csv_source(csv_input: str) -> CsvSource:
    if is_http_url(csv_input):
        response = requests.get(csv_input, timeout=60)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text), low_memory=False)
        return CsvSource(label=csv_input, frame=frame, csv_url=csv_input)
    csv_path = resolve_output_path(csv_input)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    return CsvSource(label=str(csv_path), frame=pd.read_csv(csv_path, low_memory=False), csv_path=csv_path)


def resolve_output_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path)


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


def default_label(raw: dict[str, str], label_col: str, emb_cols: list[str], image_cols: list[str], audio_cols: list[str], row_index: int) -> str:
    for col in [label_col, *emb_cols, *image_cols, *audio_cols]:
        value = clean_text(raw.get(col, ""))
        if value:
            return value[:140]
    return f"Row {row_index + 1}"


def resolve_media_ref(source: CsvSource, value: object, *, kind: str, column: str) -> MediaRef | None:
    raw = clean_text(value)
    if not raw:
        return None
    if is_http_url(raw):
        return MediaRef(kind=kind, column=column, raw_value=raw, display_url=raw, remote_url=raw)
    if source.csv_url:
        joined = urljoin(source.csv_url, raw)
        return MediaRef(kind=kind, column=column, raw_value=raw, display_url=joined, remote_url=joined)
    if not source.csv_path:
        return None
    path = Path(unquote(raw)).expanduser()
    if not path.is_absolute():
        path = (source.csv_path.parent / path).resolve()
    return MediaRef(kind=kind, column=column, raw_value=raw, display_url=path.as_uri(), local_path=path)


def media_signature(media: MediaRef) -> str:
    payload: dict[str, object] = {
        "kind": media.kind,
        "column": media.column,
        "raw_value": media.raw_value,
        "display_url": media.display_url,
    }
    if media.local_path:
        payload["local_path"] = str(media.local_path)
        if media.local_path.exists():
            stat = media.local_path.stat()
            payload["size"] = stat.st_size
            payload["mtime_ns"] = stat.st_mtime_ns
    if media.remote_url:
        payload["remote_url"] = media.remote_url
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def media_bytes(media: MediaRef) -> tuple[bytes, str]:
    if media.local_path:
        if not media.local_path.exists():
            raise FileNotFoundError(f"Missing {media.kind} file: {media.local_path}")
        data = media.local_path.read_bytes()
        mime = mimetypes.guess_type(media.local_path.name)[0] or "application/octet-stream"
    elif media.remote_url:
        response = requests.get(media.remote_url, timeout=60)
        response.raise_for_status()
        data = response.content
        mime = response.headers.get("content-type", "").split(";")[0] or mimetypes.guess_type(media.remote_url)[0] or "application/octet-stream"
    else:
        raise FileNotFoundError(f"Could not resolve {media.kind}: {media.raw_value}")
    if len(data) > MAX_INLINE_MEDIA_BYTES:
        raise ValueError(f"{media.kind} is too large for inline Gemini requests: {media.display_url}")
    return data, mime


def gemini_generate_text(parts: list[dict[str, object]], *, model: str = DEFAULT_MEDIA_MODEL, temperature: float = 0.2) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    for attempt in range(8):
        try:
            response = requests.post(
                GEMINI_GEN_URL.format(model) + f"?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": parts}],
                    "generationConfig": {"temperature": temperature},
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            texts: list[str] = []
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    text = clean_text(part.get("text", ""))
                    if text:
                        texts.append(text)
            if not texts:
                raise ValueError(f"No text returned from Gemini model {model}")
            return "\n".join(texts).strip()
        except Exception as exc:
            message = str(exc).lower()
            if attempt < 7 and any(token in message for token in ("429", "503", "timeout", "rate", "resource_exhausted")):
                wait = min(60, 2 ** attempt * 5)
                print(f"    Retrying text generation {attempt + 1}/8 after {wait}s")
                _time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Could not generate text with Gemini model {model}")


def describe_media(media: MediaRef, cache: EmbeddingCache, *, extra_text: str = "") -> str:
    cache_key = media_signature(media)
    cached = cache.fetch_text("media_summary", cache_key)
    if cached:
        return cached

    data, mime = media_bytes(media)
    prompt = (
        f"Describe this {media.kind} for semantic clustering in under 80 words. "
        "Focus on concrete entities, subject matter, mood, text, and distinguishing details. "
        "Return plain text only."
    )
    if extra_text:
        prompt += f"\nRelated metadata:\n{extra_text[:600]}"
    summary = gemini_generate_text(
        [
            {"text": prompt},
            {"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode("ascii")}},
        ]
    )
    cache.upsert_text("media_summary", cache_key, summary)
    return summary


def prepare_rows(
    source: CsvSource,
    frame: pd.DataFrame,
    *,
    emb_cols: list[str],
    image_cols: list[str],
    audio_cols: list[str],
    audio_metadata_cols: list[str],
    label_col: str,
) -> tuple[list[PreparedRow], int]:
    rows: list[PreparedRow] = []
    dropped = 0
    for index, (_, row) in enumerate(frame.iterrows()):
        raw = {col: clean_text(row[col]) for col in frame.columns}
        images = [media for col in image_cols if (media := resolve_media_ref(source, raw.get(col, ""), kind="image", column=col))]
        audios = [media for col in audio_cols if (media := resolve_media_ref(source, raw.get(col, ""), kind="audio", column=col))]
        text_payload = row_to_text(row, emb_cols)
        audio_metadata_text = row_to_text(row, [col for col in audio_metadata_cols if col in frame.columns and col not in emb_cols])
        if not text_payload and not images and not audios:
            dropped += 1
            continue
        rows.append(
            PreparedRow(
                row_index=index,
                raw=raw,
                label=default_label(raw, label_col, emb_cols, image_cols, audio_cols, index),
                text_payload=text_payload,
                audio_metadata_text=audio_metadata_text,
                images=images,
                audios=audios,
            )
        )
    return rows, dropped


def embedding_texts(rows: list[PreparedRow], cache: EmbeddingCache) -> list[str]:
    texts: list[str] = []
    for row in rows:
        parts = [row.text_payload] if row.text_payload else []
        for image in row.images:
            parts.append(f"{image.column}: {describe_media(image, cache)}")
        for audio in row.audios:
            parts.append(f"{audio.column}: {describe_media(audio, cache, extra_text=row.audio_metadata_text)}")
        text = "\n\n".join(part for part in parts if part).strip()
        texts.append(text or row.label)
    return texts


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


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_gemini_vectors(texts: list[str], cache: EmbeddingCache, output_dim: int = 768) -> np.ndarray:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    ordered_hashes = [text_hash(text) for text in texts]
    unique_texts: dict[str, str] = {}
    for hashed, text in zip(ordered_hashes, texts):
        unique_texts.setdefault(hashed, text)

    cached = cache.fetch_embeddings(unique_texts.keys(), model=GEMINI_MODEL, output_dim=output_dim)
    pending = [(hashed, unique_texts[hashed]) for hashed in unique_texts if hashed not in cached]
    if pending:
        print(f"  Building {len(pending)} Gemini embeddings...")
    else:
        print(f"  Using {len(unique_texts)} cached Gemini embeddings")

    built = dict(cached)
    for index, (hashed, text) in enumerate(pending, start=1):
        for attempt in range(8):
            try:
                embedding = gemini_embed_one(text, api_key, output_dim)
                cache.upsert_embedding(
                    text_hash=hashed,
                    model=GEMINI_MODEL,
                    output_dim=output_dim,
                    source_text=text,
                    embedding=embedding,
                )
                built[hashed] = embedding
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
            print(f"    {index}/{len(pending)} done")

    vectors = [built[hashed] for hashed in ordered_hashes]
    return np.array(vectors, dtype=np.float32)


def build_vectors(texts: list[str], emb_cols: list[str], cache: EmbeddingCache) -> np.ndarray:
    sample = texts[0][:120] if texts else ""
    print(f"  Using embedding columns: {', '.join(emb_cols) if emb_cols else '(media only)'}")
    print(f"  Sample text: {sample}...")
    return build_gemini_vectors(texts, cache)


def reduce_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros((len(vectors), 2), dtype=np.float32)
    scaled = StandardScaler().fit_transform(vectors)
    if len(vectors) < 5:
        if scaled.shape[1] >= 2:
            return scaled[:, :2].astype(np.float32)
        padded = np.pad(scaled, ((0, 0), (0, max(0, 2 - scaled.shape[1]))))
        return padded[:, :2].astype(np.float32)
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


def direct_cluster_labels(rows: list[PreparedRow], cluster_cols: list[str]) -> tuple[np.ndarray, dict[int, str]]:
    labels = []
    for row in rows:
        parts = []
        for col in cluster_cols:
            value = row.raw.get(col, "") or "(blank)"
            parts.append(value if len(cluster_cols) == 1 else f"{col}={value}")
        labels.append(" | ".join(parts))
    codes, uniques = pd.factorize(np.asarray(labels, dtype=object), sort=False)
    label_map = {int(index): str(value) for index, value in enumerate(uniques.tolist())}
    return codes.astype(int), label_map


def cluster_with_columns(vectors: np.ndarray, rows: list[PreparedRow], cluster_cols: list[str], n_clusters: int) -> tuple[np.ndarray, dict[int, str]]:
    if cluster_cols and "embeddings" not in cluster_cols:
        cluster_ids, label_map = direct_cluster_labels(rows, cluster_cols)
        return cluster_ids, label_map

    feature_blocks = [vectors.astype(np.float32)]
    meta_cols = [col for col in cluster_cols if col != "embeddings"]
    if meta_cols:
        encoded = pd.get_dummies(
            pd.DataFrame([{col: row.raw.get(col, "") or "(blank)" for col in meta_cols} for row in rows]),
            columns=meta_cols,
            dtype=float,
        ).to_numpy(dtype=np.float32)
        if encoded.size:
            feature_blocks.append(0.35 * encoded)
    features = np.hstack(feature_blocks) if len(feature_blocks) > 1 else feature_blocks[0]
    cluster_ids = cluster_vectors(features, n_clusters)
    label_map = {int(cluster_id): f"Cluster {int(cluster_id) + 1}" for cluster_id in sorted(set(cluster_ids.tolist()))}
    return cluster_ids, label_map


def cluster_name_payload(rows: list[PreparedRow], cluster_ids: np.ndarray, label_map: dict[int, str]) -> list[dict[str, object]]:
    payload = []
    for cluster_id in sorted(label_map):
        indices = np.where(cluster_ids == cluster_id)[0].tolist()
        sample = [
            {
                "label": rows[index].label,
                "text": rows[index].text_payload[:160],
            }
            for index in indices[:5]
        ]
        payload.append({"cluster_id": int(cluster_id), "label": label_map[cluster_id], "size": len(indices), "rows": sample})
    return payload


def maybe_name_clusters(
    rows: list[PreparedRow],
    cluster_ids: np.ndarray,
    label_map: dict[int, str],
    cache: EmbeddingCache,
    *,
    enabled: bool,
    model: str,
) -> dict[int, str]:
    if not enabled or not label_map or not all(name.startswith("Cluster ") for name in label_map.values()):
        return label_map

    payload = cluster_name_payload(rows, cluster_ids, label_map)
    cache_key = hashlib.sha256(json.dumps({"model": model, "payload": payload}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    cached = cache.fetch_text("cluster_names", cache_key)
    if cached:
        try:
            parsed = json.loads(cached)
            return {int(key): str(value) for key, value in parsed.items()}
        except Exception:
            pass

    prompt = "\n".join(
        [
            "Name each cluster for a semantic map.",
            "Return only JSON as an object whose keys are cluster ids and values are short names.",
            "Use 2 to 5 words when possible.",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ]
    )
    text = gemini_generate_text([{"text": prompt}], model=model, temperature=0.2)
    match = re.search(r"\{.*\}", text, re.S)
    parsed = json.loads(match.group(0) if match else text)
    names = {
        int(cluster_id): clean_text(name)[:48] or fallback
        for cluster_id, fallback in label_map.items()
        for name in [parsed.get(str(cluster_id), fallback)]
    }
    cache.upsert_text("cluster_names", cache_key, json.dumps(names, ensure_ascii=False, separators=(",", ":")))
    return names


def build_column_meta(rows: list[PreparedRow], columns: list[str], cluster_values: list[str]) -> dict:
    meta = {
        "cluster": {
            "type": "categorical",
            "name": "cluster",
            "values": sorted({value for value in cluster_values if value}, key=str),
        }
    }
    for col in columns:
        values = [clean_text(row.raw.get(col, "")) for row in rows]
        unique = sorted({value for value in values if value}, key=str)
        meta[col] = {"type": "categorical", "name": col, "values": unique[:50]}
    return meta


def build_vis_payload(
    rows: list[PreparedRow],
    points: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_labels: dict[int, str],
    *,
    color_cols: list[str],
    filter_cols: list[str],
    timeline_col: str,
    name: str,
    opacity: float,
    popup_style: str,
) -> dict:
    tooltip_cols = [col for col in dict.fromkeys(color_cols + filter_cols) if col != "cluster"]
    cluster_values = [cluster_labels[int(cluster_id)] for cluster_id in cluster_ids.tolist()]
    col_meta = build_column_meta(rows, [col for col in dict.fromkeys(color_cols + filter_cols) if col != "cluster"], cluster_values)

    color_maps = {"cluster": build_color_map(cluster_values)}
    for col in color_cols:
        if col == "cluster":
            continue
        values = [clean_text(row.raw.get(col, "")) for row in rows]
        color_maps[col] = build_color_map(values)

    items = []
    for i, prepared in enumerate(rows):
        cluster_label = cluster_labels[int(cluster_ids[i])]
        point = {
            "x": float(points[i, 0]),
            "y": float(points[i, 1]),
            "cluster": cluster_label,
            "clusterId": int(cluster_ids[i]),
            "label": prepared.label or f"Row {i + 1}",
            "rangeVal": float(i + 1),
            "audio": prepared.audios[0].display_url if prepared.audios else "",
            "audioCount": len(prepared.audios),
        }

        if timeline_col:
            value = prepared.raw.get(timeline_col, "")
            point["rangeVal"] = float(value) if pd.notna(value) else float(i + 1)

        if prepared.images:
            point["image"] = prepared.images[0].display_url

        for col in tooltip_cols + [c for c in color_cols if c != "cluster"] + filter_cols:
            value = clean_text(prepared.raw.get(col, ""))
            point[col] = value

        items.append(point)

    range_vals = [p["rangeVal"] for p in items]
    return {
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
            "opacity": opacity,
            "popupStyle": popup_style,
            "hasImages": any(bool(row.images) for row in rows),
            "hasAudio": any(bool(row.audios) for row in rows),
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


def dry_run_report(
    *,
    source_label: str,
    total_rows: int,
    usable_rows: int,
    dropped_rows: int,
    emb_cols: list[str],
    image_cols: list[str],
    audio_cols: list[str],
    audio_metadata_cols: list[str],
    cluster_cols: list[str],
    color_cols: list[str],
    filter_cols: list[str],
    label_col: str,
    timeline_col: str,
    cluster_names: bool,
    popup_style: str,
    output_path: Path,
    state_db_path: Path,
    state_parquet_path: Path,
) -> None:
    print("\n[dry-run]")
    print(f"  Source: {source_label}")
    print(f"  Rows: {usable_rows} usable / {total_rows} total")
    if dropped_rows:
        print(f"  Dropped rows: {dropped_rows}")
    print(f"  Embedding columns: {', '.join(emb_cols) if emb_cols else '(none)'}")
    print(f"  Image columns: {', '.join(image_cols) if image_cols else '(none)'}")
    print(f"  Audio columns: {', '.join(audio_cols) if audio_cols else '(none)'}")
    print(f"  Audio metadata columns: {', '.join(audio_metadata_cols) if audio_metadata_cols else '(none)'}")
    print(f"  Label column: {label_col}")
    print(f"  Timeline column: {timeline_col or '(none)'}")
    print(f"  Cluster columns: {', '.join(cluster_cols) if cluster_cols else '(none)'}")
    print(f"  Cluster naming: {'enabled' if cluster_names else 'disabled'}")
    print(f"  Popup style: {popup_style}")
    print(f"  Color columns: {', '.join(color_cols) if color_cols else '(none)'}")
    print(f"  Filter columns: {', '.join(filter_cols) if filter_cols else '(none)'}")
    print(f"  Output: {output_path}")
    print(f"  State DB: {state_db_path}")
    print(f"  State Parquet: {state_parquet_path}")


def build_visualisation(args) -> Path:
    load_env(Path.cwd() / ".env")

    output_path = resolve_output_path(args.output)
    state_db_path = resolve_output_path(args.state_db)
    state_parquet_path = resolve_output_path(args.state_parquet)

    source = load_csv_source(args.csv)
    df = source.frame
    for col in df.columns:
        df[col] = df[col].map(clean_text)

    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
    source.frame = df

    if not args.name:
        args.name = source_name(args.csv).replace("_", " ").replace("-", " ").title()

    requested = set(
        args.embedding_columns
        + args.image_columns
        + args.audio_columns
        + args.audio_metadata_columns
        + args.color_columns
        + args.filter_columns
        + [col for col in args.cluster_columns if col != "embeddings"]
    )
    for col in (args.label_column, args.image_column, args.timeline_column):
        if col:
            requested.add(col)
    missing = sorted(col for col in requested if col not in df.columns)
    if missing:
        raise SystemExit(f"Columns not found in CSV: {', '.join(missing)}")

    emb_cols = args.embedding_columns or infer_columns(df)
    emb_cols = [col for col in emb_cols if col in df.columns]
    image_cols = [col for col in args.image_columns if col in df.columns]
    if args.image_column and args.image_column in df.columns and args.image_column not in image_cols:
        image_cols = [args.image_column, *image_cols]
    audio_cols = [col for col in args.audio_columns if col in df.columns]
    audio_metadata_cols = [col for col in args.audio_metadata_columns if col in df.columns]
    if not emb_cols and not image_cols and not audio_cols:
        raise SystemExit("Provide at least one of --embedding-columns, --image-columns, or --audio-columns")

    label_col = args.label_column or infer_single_column(df, LABEL_HINTS, lambda s: nonempty_mask(s).mean() >= 0.3) or (emb_cols[0] if emb_cols else (image_cols[0] if image_cols else audio_cols[0]))
    image_col = image_cols[0] if image_cols else (args.image_column or infer_single_column(df, IMAGE_HINTS, lambda s: s.map(extract_image_url).astype(bool).mean() >= 0.2))
    timeline_col = args.timeline_column or infer_single_column(df, TIME_HINTS)

    rows, dropped = prepare_rows(
        source,
        df,
        emb_cols=emb_cols,
        image_cols=image_cols,
        audio_cols=audio_cols,
        audio_metadata_cols=audio_metadata_cols,
        label_col=label_col,
    )
    if dropped:
        print(f"  Dropped {dropped} rows with no embedding data")
    if not rows:
        raise SystemExit("No rows left after cleaning embedding inputs")
    df = pd.DataFrame([row.raw for row in rows])

    if timeline_col:
        parsed = parse_timeline(df[timeline_col])
        coverage = float(parsed.notna().mean())
        if coverage >= 0.6:
            df[timeline_col] = parsed
            for row, value in zip(rows, parsed.tolist(), strict=False):
                row.raw[timeline_col] = "" if pd.isna(value) else clean_text(value)
            print(f"  Timeline column: {timeline_col} ({coverage:.0%} usable)")
        else:
            print(f"  Ignoring timeline column '{timeline_col}' ({coverage:.0%} usable)")
            timeline_col = ""

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

    print(f"\n{'=' * 60}")
    print(f"  Building: {args.name}")
    print(f"  CSV: {source.label}")
    print(f"  Rows: {len(rows)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Output: {output_path}")
    print(f"  State DB: {state_db_path}")
    print(f"{'=' * 60}")

    print(f"  Label column: {label_col}")
    if image_col:
        print(f"  Image column: {image_col}")
    if audio_cols:
        print(f"  Audio columns: {', '.join(audio_cols)}")
    print(f"  Cluster columns: {', '.join(args.cluster_columns)}")
    print(f"  Color columns: {', '.join(color_cols) if color_cols else '(none)'}")
    print(f"  Filter columns: {', '.join(filter_cols) if filter_cols else '(none)'}")

    if args.dry_run:
        dry_run_report(
            source_label=source.label,
            total_rows=int(len(rows) + dropped),
            usable_rows=len(rows),
            dropped_rows=dropped,
            emb_cols=emb_cols,
            image_cols=image_cols,
            audio_cols=audio_cols,
            audio_metadata_cols=audio_metadata_cols,
            cluster_cols=args.cluster_columns,
            color_cols=color_cols,
            filter_cols=filter_cols,
            label_col=label_col,
            timeline_col=timeline_col,
            cluster_names=bool(args.cluster_names),
            popup_style=args.popup_style,
            output_path=output_path,
            state_db_path=state_db_path,
            state_parquet_path=state_parquet_path,
        )
        print("\nDone.")
        return output_path

    cache = EmbeddingCache(state_db_path)
    try:
        print("[1/4] Building vectors ...")
        vectors = build_vectors(embedding_texts(rows, cache), emb_cols, cache)
        print(f"  Vector shape: {vectors.shape}")

        print("[2/4] Reducing to 2D ...")
        points = reduce_vectors(vectors)

        print("[3/4] Clustering ...")
        clusters, cluster_labels = cluster_with_columns(vectors, rows, args.cluster_columns, args.clusters)
        cluster_labels = maybe_name_clusters(
            rows,
            clusters,
            cluster_labels,
            cache,
            enabled=bool(args.cluster_names),
            model=(args.cluster_naming_model or DEFAULT_CLUSTER_NAMING_MODEL).strip(),
        )

        print("[4/4] Writing standalone HTML ...")
        payload = build_vis_payload(
            rows,
            points,
            clusters,
            cluster_labels,
            color_cols=color_cols,
            filter_cols=filter_cols,
            timeline_col=timeline_col,
            name=args.name,
            opacity=args.opacity,
            popup_style=args.popup_style,
        )
        html = render_html(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        print(f"  Wrote {output_path}")

        if not args.no_export_parquet:
            cache.export_parquet(state_parquet_path)
            print(f"  Exported cache snapshot to {state_parquet_path}")
    finally:
        cache.close()

    print("\nDone.")
    return output_path

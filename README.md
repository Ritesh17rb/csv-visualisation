# CSV Data Explorer

An interactive scatter-plot visualisation tool that turns any CSV dataset into a 2D map where **similar rows land near each other**. Built with D3.js, Canvas, UMAP, and optional Gemini text embeddings.

**[Live Demo](https://your-username.github.io/csv-visualisation/)**

---

## Screenshots

| Scatter Plot | Brush Selection | Lightbox |
|---|---|---|
| Colour-coded dots positioned by similarity | Drag to select a region and inspect rows | Click any image to view full-size |

---

## Features

### Visualisation
- **UMAP / PCA projection** — high-dimensional data reduced to 2D coordinates preserving neighbourhood structure
- **KMeans clustering** — automatic grouping reveals hidden patterns in the data
- **Force-separated dots** — a physics simulation nudges overlapping points apart so every dot is visible
- **Canvas rendering** — smooth 60 fps even with thousands of points

### Interaction
- **Legend highlighting** — hover a legend item to spotlight that category; click to lock the highlight
- **Brush selection** — drag a rectangle to select a region; a popup shows the selected rows in grid or table view
- **Sortable table** — click any column header in the popup to sort ascending / descending
- **Tooltips** — hover any dot to see its details: label, cluster, and all configured tooltip columns
- **Lightbox** — click an image thumbnail to open a full-size lightbox; arrow keys navigate between images
- **Draggable popup** — grab the popup header to reposition it anywhere on screen
- **Dataset switcher** — pick any dataset from the dropdown to load a completely different visualisation

### Data Pipeline
- **Dual embedding modes:**
  - `numerical` — one-hot encodes categoricals, standardises numericals (no API key needed)
  - `text-gemini` — converts each row to a text sentence and embeds via Google Gemini for richer semantic similarity
- **Resumable Gemini batching** — embeddings are cached to disk; interrupted builds resume from where they left off
- **Rate-limit handling** — automatic exponential back-off on 429 errors with up to 8 retries

---

## Datasets

Five pre-configured datasets ship out of the box:

| Dataset | Source | Rows | Embedding | Colour By |
|---|---|---|---|---|
| **Pokemon** | [PokeAPI / Seaborn](https://github.com/lgreski/pokemonData) | ~800 | Gemini | Type 1 |
| **MoMA Collection** | [MoMA on GitHub](https://github.com/MuseumofModernArt/collection) | 3 000 | Gemini | Classification |
| **Art Institute of Chicago** | [AIC API](https://api.artic.edu) | ~2 000 | Gemini | Department |
| **IMDb Top 1000 Movies** | [IMDB CSV](https://github.com/krishna-koly/IMDB_TOP_1000) | ~1 000 | Gemini | Genre |
| **E-commerce Products** | [DummyJSON API](https://dummyjson.com) | ~190 | Gemini | Category |

Each dataset includes thumbnail images shown in tooltips, grid cards, and the lightbox viewer.

---

## Project Structure

```
csv-visualisation/
├── index.html                 # Single-page app entry point
├── assets/
│   ├── app.css                # All styles (dark theme, animations, modals)
│   └── app.js                 # D3 canvas renderer, interactions, data loading
├── configs/                   # One Python config per dataset
│   ├── pokemon.py
│   ├── moma.py
│   ├── met_museum.py
│   ├── movies.py
│   └── ecommerce.py
├── scripts/
│   ├── build_dataset.py       # Build pipeline: fetch → embed → UMAP → cluster → JSON
│   └── fetchers.py            # Per-dataset download & cleaning logic
├── data/
│   ├── datasets.json          # Manifest consumed by the frontend
│   └── <name>/
│       └── vis-data.json      # Pre-built visualisation payload
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API keys
└── .gitignore
```

### Frontend (`index.html` + `assets/`)

A single HTML page loads `datasets.json`, populates the dataset dropdown, and fetches the selected `vis-data.json`. The entire visualisation runs client-side — no backend needed at runtime.

### Build Pipeline (`scripts/`)

`build_dataset.py` orchestrates the full pipeline for each dataset:

```
CSV → Sample → Embed → UMAP 2D → KMeans → vis-data.json
```

1. **Fetch** — downloads or loads cached source CSV via `fetchers.py`
2. **Sample** — optionally sub-samples large datasets (configurable per dataset)
3. **Embed** — builds a feature vector per row:
   - *Numerical mode:* one-hot encodes categoricals, standardises numericals
   - *Gemini mode:* serialises each row to text, calls `gemini-embedding-001`, caches results
4. **Reduce** — UMAP (or PCA) projects vectors to 2D coordinates
5. **Cluster** — KMeans assigns each point to a cluster
6. **Write** — produces `vis-data.json` with all points, metadata, colour maps, and domains

### Dataset Configs (`configs/`)

Each config file exports a `CONFIG` dict controlling:

| Key | Purpose |
|---|---|
| `embedding_columns` | Which columns feed into the embedding vector |
| `color_columns` | Categorical columns available for colour-coding |
| `filter_columns` | Columns for dropdown filters (currently disabled in UI) |
| `tooltip_columns` | Columns shown on hover |
| `default_color` | Which colour column is active on load |
| `label_column` | Column used as the display name for each dot |
| `image_column` | Column containing image URLs for thumbnails |
| `range_column` | Column for the range slider (currently disabled in UI) |
| `embedding_mode` | `"numerical"` or `"text-gemini"` |
| `sample_size` | Max rows to use (0 = all) |
| `umap_*` | UMAP hyperparameters |
| `n_clusters` | Number of KMeans clusters |

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **pip** (comes with Python)
- A **Gemini API key** (free tier works) — only needed if using `text-gemini` embedding mode

### 1. Clone the repository

```bash
git clone https://github.com/your-username/csv-visualisation.git
cd csv-visualisation
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `umap-learn`, `requests`, `google-genai`

### 3. Set up your API key (optional)

Only required for `text-gemini` embedding mode:

```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your_key_here
```

Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Build the datasets

Build all datasets at once:

```bash
python scripts/build_dataset.py --all
```

Or build a single dataset:

```bash
python scripts/build_dataset.py --dataset pokemon
```

This generates `data/<name>/vis-data.json` files and the `data/datasets.json` manifest.

### 5. Serve locally

Any static file server works. The simplest option:

```bash
python -m http.server 8000
```

Then open **http://localhost:8000** in your browser.

---

## Deploying to GitHub Pages

The app is fully static — just HTML, CSS, JS, and JSON files. No server needed.

1. Push the repository to GitHub
2. Go to **Settings → Pages**
3. Set source to **Deploy from a branch** → select `main` (or `master`) → root `/`
4. Wait a minute for the build — your site will be live at `https://<user>.github.io/<repo>/`

**What gets deployed:**
- `index.html` — the app
- `assets/app.css`, `assets/app.js` — styles and logic
- `data/datasets.json` — dataset manifest
- `data/*/vis-data.json` — pre-built visualisation data

**What stays out** (via `.gitignore`):
- `data/*/source.csv` — large raw source files
- `data/*/embeddings_cache.json` — Gemini embedding caches
- `.env` — API keys
- `__pycache__/`, `*.log`, editor configs

---

## Adding a New Dataset

1. **Create a config** — add `configs/my_dataset.py`:

```python
CONFIG = {
    "name": "my_dataset",
    "display_name": "My Dataset",
    "sample_size": 2000,
    "sample_seed": 42,
    "embedding_columns": ["col_a", "col_b", "col_c"],
    "color_columns": ["col_a"],
    "filter_columns": ["col_a"],
    "tooltip_columns": ["col_a", "col_b", "col_c"],
    "default_color": "col_a",
    "range_column": "",
    "label_column": "col_b",
    "image_column": "",            # set to a column with URLs for thumbnails
    "embedding_mode": "numerical", # or "text-gemini"
    "reducer": "umap",
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.35,
    "umap_metric": "cosine",
    "n_clusters": 6,
}
```

2. **Create a fetcher** — add a `fetch_my_dataset()` function in `scripts/fetchers.py` and register it in the `FETCHERS` dict

3. **Register the dataset** — add `"my_dataset"` to the `AVAILABLE_DATASETS` list in `scripts/build_dataset.py`

4. **Build it:**

```bash
python scripts/build_dataset.py --dataset my_dataset
```

5. **Rebuild the manifest** (if adding to an existing set, run `--all` once to regenerate `datasets.json`)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   BUILD TIME (Python)                 │
│                                                      │
│  CSV ──→ Sample ──→ Embed ──→ UMAP 2D ──→ KMeans    │
│                       │                      │       │
│               ┌───────┴───────┐              │       │
│               │  numerical    │              │       │
│               │  OR           │              ▼       │
│               │  text-gemini  │        vis-data.json │
│               └───────────────┘                      │
└──────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│                  RUNTIME (Browser)                    │
│                                                      │
│  datasets.json ──→ Dropdown ──→ Fetch vis-data.json  │
│                                        │             │
│                        ┌───────────────┼─────────┐   │
│                        ▼               ▼         ▼   │
│                  D3 Scales      Force Layout  Legend  │
│                        │               │         │   │
│                        ▼               ▼         ▼   │
│                     Canvas ◄─── Quadtree ◄── Hover   │
│                        │                     Click   │
│                        ▼                     Brush   │
│                  Tooltip / Popup / Lightbox           │
└──────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Visualisation** | HTML5 Canvas + D3.js v7 |
| **Layout** | UMAP / PCA (via `umap-learn`, `scikit-learn`) |
| **Clustering** | KMeans (`scikit-learn`) |
| **Embeddings** | Google Gemini `gemini-embedding-001` |
| **Data processing** | Python, pandas, NumPy |
| **Styling** | Custom CSS (dark theme, no frameworks) |
| **Hosting** | GitHub Pages (static files) |

---

## Configuration Reference

### Embedding Modes

| Mode | Needs API Key | Quality | Speed |
|---|---|---|---|
| `numerical` | No | Good for numeric-heavy data | Fast |
| `text-gemini` | Yes (free tier OK) | Best for mixed/text data | Slower (API calls) |

### UMAP Parameters

| Parameter | Default | Effect |
|---|---|---|
| `umap_n_neighbors` | 15 | Higher = more global structure, lower = tighter clusters |
| `umap_min_dist` | 0.1 | Higher = more spread out, lower = denser clusters |
| `umap_metric` | `"euclidean"` | Distance metric (`"cosine"` recommended for Gemini embeddings) |

---

## License

MIT

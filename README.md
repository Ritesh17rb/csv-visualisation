# CSV Data Explorer

An interactive scatter-plot visualization that turns **any CSV** into a 2D map where similar rows land near each other. Uses Gemini embeddings, UMAP, and KMeans clustering.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

## Usage

```bash
python build.py --csv <file.csv> --embedding-columns <col1> <col2> ... --color-columns <col1> ...
```

Then serve and open in browser:

```bash
python -m http.server 8000
# Open http://localhost:8000
```

### All Options

| Flag | Required | Description |
|------|----------|-------------|
| `--csv` | Yes | Path to your CSV file |
| `--embedding-columns` | Yes | Columns to embed via Gemini (determines point positions) |
| `--color-columns` | No | Columns for color-coding the scatter plot |
| `--filter-columns` | No | Columns for dropdown filtering in the UI |
| `--label-column` | No | Column to use as point label on hover |
| `--image-column` | No | Column with image URLs for thumbnails |
| `--timeline-column` | No | Date/year column for timeline playback slider |
| `--clusters` | No | Number of clusters (default: 6) |
| `--name` | No | Display name (default: derived from filename) |
| `--sample` | No | Sample N rows (default: use all) |

### Examples

**Music collection with timeline:**
```bash
python build.py --csv music.csv \
  --embedding-columns title artist genre \
  --color-columns genre artist \
  --label-column title \
  --timeline-column year
```

**Product catalog with images:**
```bash
python build.py --csv products.csv \
  --embedding-columns name description category \
  --color-columns category \
  --filter-columns category brand \
  --label-column name \
  --image-column image_url \
  --clusters 8
```

## How It Works

1. **Load CSV** - Read your data with pandas
2. **Embed** - Convert selected columns to text, embed via Gemini API (768-dim vectors)
3. **UMAP** - Project to 2D preserving local structure
4. **Cluster** - KMeans groups similar points
5. **Visualize** - Interactive D3.js + Canvas scatter plot with hover, brush selection, legend, and optional timeline

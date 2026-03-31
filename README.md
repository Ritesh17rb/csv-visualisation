# CSV Visualisation

Build a standalone HTML semantic map from any local CSV or HTTP(S) CSV URL.

The generated file contains:

- the UI
- bundled JavaScript
- bundled CSS
- bundled D3
- the embedded dataset payload

No separate `assets/` or `data/` files are required at runtime.

## What It Does

The pipeline:

1. reads a CSV with `pandas`
2. turns selected text, image, and audio columns into embedding text with Gemini
3. stores embeddings and text summaries in a local DuckDB cache
4. exports the embedding cache to Parquet
5. reduces vectors to 2D with UMAP
6. clusters rows from embeddings or direct metadata columns
7. writes a single standalone HTML artifact

## Features

- packaged CLI: `csv-viz`
- runs locally with `uv run`
- runs without installation via `uvx --from git+https://...`
- local or remote CSV input
- dry-run mode
- text, image, and audio support
- resumable embeddings with DuckDB
- Parquet export
- cluster naming with Gemini
- popup styles: `auto`, `grid`, `list`, `table`

## Requirements

- Python 3.11+
- `uv`
- `GEMINI_API_KEY` in the environment or a local `.env`

Example `.env`:

```env
GEMINI_API_KEY=your_key_here
```

A `.env` containing just the raw Gemini key also works. `GOOGLE_API_KEY` is also accepted and mapped to `GEMINI_API_KEY`.

## Quick Start

Install and lock dependencies:

```bash
uv lock
```

Show CLI help:

```bash
uv run csv-viz --help
```

Build a standalone bundle from a local CSV:

```bash
uv run csv-viz music.csv \
  --embedding-columns title,artist,genre \
  --color-columns genre,artist \
  --filter-columns genre \
  --label-column title \
  --timeline-column year
```

Validate the run without calling Gemini:

```bash
uv run csv-viz music.csv \
  --embedding-columns title,artist,genre \
  --color-columns genre,artist \
  --filter-columns genre \
  --timeline-column year \
  --dry-run
```

Open the generated HTML directly in a browser.

## Multimodal Example

```bash
uv run csv-viz books.csv \
  --embedding-columns title,authors \
  --image-columns image_url \
  --color-columns language_code \
  --filter-columns language_code \
  --timeline-column year \
  --cluster-columns embeddings,language_code \
  --cluster-names \
  --popup-style grid
```

## Remote CSV Example

```bash
uv run csv-viz https://raw.githubusercontent.com/sanand0/embedumap/main/samples/blog-text.csv \
  --embedding-columns text \
  --color-columns primary_category,year \
  --filter-columns primary_category,year \
  --timeline-column year \
  --cluster-columns embeddings,primary_category \
  --cluster-names \
  --popup-style list
```

## `uvx` Usage

Run directly from GitHub without installing the project:

```bash
uvx --from git+https://github.com/Ritesh17rb/csv-visualisation csv-viz music.csv \
  --embedding-columns title,artist,genre \
  --color-columns genre
```

Generated artifacts are written relative to the caller's working directory, not the package directory.

## Important Flags

| Flag | Description |
|------|-------------|
| `csv_input` | Local CSV path or HTTP(S) CSV URL |
| `--embedding-columns` | Columns used to build embeddings |
| `--image-columns` | Image columns to summarize and display |
| `--audio-columns` | Audio columns to summarize and display |
| `--audio-metadata-columns` | Text columns included when describing audio |
| `--color-columns` | Columns exposed for coloring |
| `--filter-columns` | Columns exposed for filtering |
| `--cluster-columns` | Use `embeddings`, direct metadata columns, or both |
| `--cluster-names` | Ask Gemini for short cluster labels |
| `--cluster-naming-model` | Model used for cluster naming |
| `--label-column` | Hover and detail label column |
| `--image-column` | Legacy single image column alias |
| `--timeline-column` | Date or year column |
| `--popup-style` | Default detail view: `auto`, `grid`, `list`, or `table` |
| `--clusters` | Cluster count, `0` means auto |
| `--opacity` | Base point opacity in the plot |
| `--output` | Output HTML path |
| `--state-db` | DuckDB cache path |
| `--state-parquet` | Parquet export path |
| `--no-export-parquet` | Skip the Parquet export |
| `--dry-run` | Validate inputs without embedding |

## Output And Cache

Defaults:

- output HTML: `dist/index.html`
- DuckDB cache: `.csv-viz/embeddings.duckdb`
- Parquet export: `.csv-viz/embeddings.parquet`

Embeddings are keyed by text hash plus model metadata and stored in DuckDB. Media summaries and cluster names are cached there too.

On rerun:

- existing embeddings are reused
- existing image/audio summaries are reused
- existing cluster names are reused for the same cluster payload
- only missing rows are sent to Gemini

## Testing

Run unit tests:

```bash
uv run python -m unittest discover -s tests
```

Basic CLI validation:

```bash
uv run csv-viz --help
```

## Notes From Real Runs

Verified on real public datasets:

- Yelp reviews text dataset
- Goodreads/Goodbooks image-cover sample
- 500-row text run completed successfully

Current practical limit:

- large text datasets are workable
- large multimodal datasets can hit Gemini `generateContent` rate limits, especially when image/audio summarization and cluster naming are both enabled

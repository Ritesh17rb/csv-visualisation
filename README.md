# CSV Data Explorer

Build a standalone `index.html` bundle from any local CSV or HTTP(S) CSV URL. The bundle contains the UI, styles, JavaScript, D3 runtime, and dataset payload in one file.

The pipeline:

1. reads a CSV with `pandas`
2. turns selected text, image, and audio columns into embedding text with Gemini
3. persists embeddings and text summaries in a local `.duckdb` cache
4. exports the embedding cache to `.parquet`
5. projects vectors to 2D with UMAP
6. clusters rows from embeddings or direct metadata columns
7. writes a single self-contained HTML artifact

## Requirements

- Python 3.11+
- `uv`
- `GEMINI_API_KEY` in the environment or a local `.env`

Example `.env`:

```env
GEMINI_API_KEY=your_key_here
```

A `.env` containing just the raw Gemini key also works.

## Local Workflow With `uv`

Install dependencies and lock them:

```bash
uv lock
```

Build a standalone bundle:

```bash
uv run csv-viz music.csv \
  --embedding-columns title,artist,genre \
  --color-columns genre,artist \
  --filter-columns genre \
  --label-column title \
  --timeline-column year
```

Multimodal example:

```bash
uv run csv-viz songs.csv \
  --embedding-columns title,artist \
  --image-columns cover_url \
  --audio-columns preview_url \
  --audio-metadata-columns genre,album \
  --cluster-columns embeddings,genre \
  --cluster-names \
  --popup-style grid
```

Validate the plan without embedding:

```bash
uv run csv-viz music.csv \
  --embedding-columns title,artist,genre \
  --color-columns genre,artist \
  --filter-columns genre \
  --timeline-column year \
  --dry-run
```

Defaults:

- output HTML: `dist/index.html`
- DuckDB cache: `.csv-viz/embeddings.duckdb`
- Parquet export: `.csv-viz/embeddings.parquet`

Open the generated `dist/index.html` directly in a browser.

## `uvx` With A GitHub URL

Once the repo is pushed, the tool can run without installation:

```bash
uvx --from git+https://github.com/<owner>/<repo> csv-viz music.csv \
  --embedding-columns title,artist,genre \
  --color-columns genre
```

Because the tool may run from an ephemeral install, all generated artifacts are written relative to the caller's working directory, not the package directory.

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

## Resume Behavior

Embeddings are keyed by the row text hash plus model metadata and stored in DuckDB. Media summaries and cluster names are cached there too. On rerun:

- existing embeddings are reused
- existing image/audio summaries are reused
- existing cluster names are reused for the same cluster payload
- only missing rows are sent to Gemini
- the cache can be inspected via the `.duckdb` file or consumed from the exported `.parquet`

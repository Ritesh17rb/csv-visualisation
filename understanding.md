# Understanding This Application

## 1. What this project is

This project is a static CSV exploration app. It turns each row of a dataset into a point on a 2D scatter plot, where rows with similar content appear near each other.

The project has two user-facing pages:

- `index.html`: the main visual explorer
- `catalog.html`: a dataset browser and row viewer

There is no backend server at runtime. The browser only loads HTML, CSS, JavaScript, and prebuilt JSON files from the `data/` folder.

---

## 2. The main idea

We do the heavy work before the app runs in the browser.

At build time, Python scripts:

1. fetch raw dataset data
2. clean and normalize it
3. convert each row into a vector representation
4. reduce that high-dimensional vector into 2D coordinates
5. cluster similar rows
6. write the final output into `data/<dataset>/vis-data.json`

At runtime, the frontend:

1. loads `data/datasets.json`
2. lets the user choose a dataset
3. fetches that dataset's `vis-data.json`
4. draws all points on a canvas
5. enables hover, selection, highlighting, popup views, and image lightbox interactions

So the core pattern is:

`raw CSV/API data -> Python build pipeline -> vis-data.json -> browser visualization`

---

## 3. How the application works

### 3.1 Startup flow

When `index.html` opens:

- `assets/app.js` fetches `data/datasets.json`
- the dataset dropdown is populated
- the first dataset is loaded automatically
- the page title, stats, legend, and plot are updated from that dataset's metadata

When `catalog.html` opens:

- `assets/catalog.js` also fetches `data/datasets.json`
- it loads dataset previews to build catalog cards
- when a card is clicked, it opens a row browser for that dataset

---

### 3.2 What is inside `vis-data.json`

Each built dataset contains:

- `meta`: dataset name, display name, column metadata, color settings, tooltip columns, image availability, embedding mode
- `domains`: min/max values for `x`, `y`, and `range`
- `points`: one object per row, including:
  - `x`, `y`
  - `cluster`
  - `label`
  - `rangeVal`
  - optional `image`
  - selected dataset fields used by the UI

This means the frontend does not need to run UMAP, clustering, or embedding logic in the browser. It only renders already-prepared data.

---

### 3.3 Main explorer page (`index.html` + `assets/app.js`)

The main explorer shows a scatter plot where each dot is one row.

#### What happens after a dataset loads

`assets/app.js` does the following:

1. reads `meta`, `domains`, and `points`
2. stores all points in memory
3. creates D3 linear scales for `x` and `y`
4. maps each point to screen coordinates
5. runs a D3 force simulation to slightly separate overlapping points
6. builds a quadtree for fast nearest-point hover lookup
7. draws all visible points onto an HTML canvas

Canvas is used for performance, because it handles large point counts better than rendering every point as an SVG element.

#### Interactions implemented in the explorer

- Dataset switcher
  - changes between prebuilt datasets
- Legend highlighting
  - hover a legend value to focus matching points
  - click a legend value to lock the highlight
- Tooltip
  - moving over the plot finds the nearest point using a quadtree
  - the tooltip shows label, cluster, image, and configured fields
- Brush selection
  - dragging on the overlay selects points in that region
- Popup detail panel
  - selected rows open in a draggable popup
  - rows can be viewed in grid or table format
  - table headers can sort the selection
- Lightbox
  - clicking an image opens a larger preview
  - keyboard arrows navigate images
- Architecture modal
  - the info button opens a short explanation of the pipeline

#### Important implementation detail

There is some code support for color dropdowns, filters, and a range slider, but in the current HTML/UI those controls are removed or not rendered. The live behavior is mainly driven by:

- dataset selection
- legend interaction
- hover
- brush selection
- popup/table/grid views
- lightbox

---

### 3.4 Catalog page (`catalog.html` + `assets/catalog.js`)

The catalog page is a second way to explore the same data.

It works like this:

1. load every dataset from `data/datasets.json`
2. prefetch each dataset's JSON
3. build a preview card for each dataset
4. show row count, embedding mode, and key columns
5. open a viewer when a dataset card is clicked

Inside the viewer, the user can:

- search across rows
- toggle visible columns
- sort columns
- switch between table view and card view
- paginate large result sets
- open images in a lightbox

This page is useful when someone wants to inspect the actual records more directly instead of only seeing the scatter layout.

---

## 4. How we made it

### 4.1 Data source layer

We created dataset-specific fetchers in `scripts/fetchers.py`.

Each fetcher:

- downloads data from a remote CSV or API
- cleans the source fields
- creates consistent columns for the rest of the pipeline
- caches the result into `data/<dataset>/source.csv`

Current datasets are:

- `pokemon`
- `moma`
- `met_museum`
- `movies`
- `ecommerce`

One important detail: the `met_museum` dataset name is legacy. In the current code it actually fetches data from the Art Institute of Chicago API, not the Met Museum API.

---

### 4.2 Dataset configuration layer

Each dataset has a config file in `configs/`.

These config files define:

- display name
- sample size
- embedding columns
- color columns
- filter columns
- tooltip columns
- default color field
- range column
- label column
- image column
- embedding mode
- reducer settings
- cluster count

This is what makes the pipeline reusable. We did not hardcode the app for one CSV. We made a general pipeline and then described each dataset through config.

---

### 4.3 Embedding and vector creation

The build script is `scripts/build_dataset.py`.

It supports two embedding modes:

#### `text-gemini`

For most datasets, we turn each row into descriptive text using `row_to_text()`, then send batches to Gemini embeddings.

This gives a semantic vector for each row, which works well for mixed data like:

- text
- categories
- labels
- numeric values written into text form

The script also:

- reads `GEMINI_API_KEY` from `.env`
- caches embeddings in `embeddings_cache.json`
- retries with backoff on rate limits
- falls back to numerical vectors if some embeddings fail

#### `numerical`

The script can also build vectors locally by:

- standardizing numeric columns
- one-hot encoding categorical columns

This mode does not require an API key.

---

### 4.4 Dimensionality reduction and clustering

After vectors are built, the script:

- standardizes the full vector matrix
- reduces it to 2D using UMAP by default
- can use PCA if configured
- clusters rows using KMeans

Why we did this:

- UMAP gives a 2D layout where similar rows stay close together
- KMeans adds a simple machine-generated grouping
- the browser only has to render the output, not compute it

---

### 4.5 Writing frontend-ready JSON

The script writes `vis-data.json` with:

- metadata for the UI
- color maps for categorical fields
- numeric domains for plotting
- one frontend-ready point per row

It also writes `data/datasets.json`, which is the manifest both pages use to know which datasets are available.

---

## 5. What we built in this repo

In practical terms, this repo contains:

- a static scatter-plot explorer
- a static catalog/data-browser page
- a reusable Python dataset build pipeline
- configurable dataset definitions
- cached raw data support
- cached embedding support
- multiple sample datasets already built into JSON

The frontend stack is:

- HTML
- CSS
- vanilla JavaScript
- D3.js
- Canvas

The build/data stack is:

- Python
- pandas
- NumPy
- scikit-learn
- UMAP
- requests
- Google Gemini embeddings

---

## 6. Important files and their roles

- `index.html`
  - main visualization screen
- `catalog.html`
  - catalog and row browser
- `assets/app.js`
  - scatter plot rendering and interactions
- `assets/catalog.js`
  - catalog cards, table view, card view, search, pagination
- `assets/app.css`
  - explorer page styling
- `assets/catalog.css`
  - catalog page styling
- `scripts/fetchers.py`
  - download and clean each dataset
- `scripts/build_dataset.py`
  - build pipeline from raw data to visualization JSON
- `configs/*.py`
  - per-dataset behavior and metadata rules
- `data/datasets.json`
  - dataset manifest for the frontend
- `data/*/vis-data.json`
  - final visualization payloads used by the UI

---

## 7. End-to-end flow summary

### Build time

1. choose a dataset config
2. fetch raw data
3. clean and normalize columns
4. sample rows if needed
5. create embeddings/vectors
6. reduce vectors to 2D
7. cluster rows
8. write `vis-data.json`
9. update `datasets.json`

### Runtime

1. browser loads the static page
2. JavaScript loads `datasets.json`
3. selected dataset JSON is fetched
4. metadata updates the UI
5. points are mapped to canvas coordinates
6. the plot is rendered
7. user interacts through hover, legend, brush, popup, and lightbox features

---

## 8. Short conclusion

This application works by doing machine-learning-style preprocessing at build time and lightweight rendering at runtime.

What we did was:

- build a reusable dataset pipeline
- convert raw CSV/API data into 2D similarity maps
- create a browser-based explorer for visual discovery
- add a second catalog interface for direct row inspection
- keep the final deployment fully static, so it can run without a backend

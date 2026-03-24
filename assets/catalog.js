(async function () {
  "use strict";

  const ROWS_PER_PAGE = 50;
  const PALETTE = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#6366f1",
    "#84cc16", "#e11d48", "#0ea5e9", "#d946ef", "#a3e635",
  ];

  // ── DOM refs ─────────────────────────────────────────
  const cardsSection = document.getElementById("cardsSection");
  const cardsGrid    = document.getElementById("cardsGrid");
  const viewer       = document.getElementById("viewer");
  const viewerBack   = document.getElementById("viewerBack");
  const viewerTitle  = document.getElementById("viewerTitle");
  const viewerStats  = document.getElementById("viewerStats");
  const searchInput  = document.getElementById("searchInput");
  const searchCount  = document.getElementById("searchCount");
  const colChips     = document.getElementById("colChips");
  const tableWrap    = document.getElementById("tableWrap");
  const dataHead     = document.getElementById("dataHead");
  const dataBody     = document.getElementById("dataBody");
  const dataCards    = document.getElementById("dataCards");
  const pagination   = document.getElementById("pagination");
  const pageLoader   = document.getElementById("pageLoader");
  const lightbox     = document.getElementById("lightbox");
  const lbImg        = document.getElementById("lbImg");
  const lbLabel      = document.getElementById("lbLabel");

  // ── State ────────────────────────────────────────────
  let datasets   = [];
  let current    = null;   // { meta, points }
  let allCols    = [];
  let visCols    = [];
  let filtered   = [];
  let sortCol    = null;
  let sortAsc    = true;
  let page       = 0;
  let viewMode   = "table";

  // ── Helpers ──────────────────────────────────────────
  function fmt(v) {
    if (v == null || v === "") return "\u2014";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
    return String(v);
  }

  function isNum(v) { return typeof v === "number" && isFinite(v); }

  function colorForVal(val, map) {
    if (map && map[val]) return map[val];
    return "#475569";
  }

  // ── Load datasets manifest ───────────────────────────
  try {
    const res = await fetch("data/datasets.json");
    if (!res.ok) throw new Error(res.status);
    datasets = await res.json();
  } catch (e) {
    pageLoader.querySelector(".loader-text").textContent = "Failed to load datasets";
    pageLoader.querySelector(".loader-sub").textContent = e.message;
    return;
  }

  // ── Build dataset cards ──────────────────────────────
  async function buildCards() {
    cardsGrid.innerHTML = "";

    for (let i = 0; i < datasets.length; i++) {
      const ds = datasets[i];

      // Pre-fetch metadata for each dataset to show rich cards
      let meta = null, points = [];
      try {
        const res = await fetch(ds.path);
        const json = await res.json();
        meta = json.meta;
        points = json.points || [];
        ds._cache = json;
      } catch { /* ignore */ }

      const card = document.createElement("div");
      card.className = "ds-card";

      // Preview area — show sample images or coloured dots
      let previewHtml = "";
      const hasImages = meta && meta.hasImages;
      if (hasImages) {
        const imgs = points.filter(p => p.image).slice(0, 8);
        previewHtml = `<div class="ds-card-imgs">${imgs.map(p =>
          `<img src="${p.image}" alt="" loading="lazy">`
        ).join("")}</div>`;
      } else {
        const colorCol = meta ? meta.defaultColor : "";
        const cmap = meta ? (meta.colorMaps || {})[colorCol] || {} : {};
        const sample = points.slice(0, 60);
        previewHtml = `<div class="ds-preview-dots">${sample.map((p, j) => {
          const c = colorForVal(p[colorCol], cmap);
          return `<div class="ds-preview-dot" style="background:${c};animation-delay:${j * 15}ms"></div>`;
        }).join("")}</div>`;
      }

      // Meta chips
      const rowCount = meta ? meta.totalRows : "?";
      const embMode = meta ? (meta.embeddingMode === "text-gemini" ? "Gemini" : "Numerical") : "?";
      const colCount = meta ? Object.keys(meta.columns || {}).length : "?";

      // Column tags
      const tooltipCols = meta ? (meta.tooltipColumns || []).slice(0, 6) : [];

      card.innerHTML = `
        <div class="ds-card-preview">${previewHtml}</div>
        <div class="ds-card-body">
          <div class="ds-card-name">${ds.displayName}</div>
          <div class="ds-card-meta">
            <span class="ds-meta-chip"><span class="chip-dot" style="background:#3b82f6"></span>${rowCount} rows</span>
            <span class="ds-meta-chip"><span class="chip-dot" style="background:#8b5cf6"></span>${embMode}</span>
            <span class="ds-meta-chip"><span class="chip-dot" style="background:#10b981"></span>${colCount} columns</span>
          </div>
          <div class="ds-card-cols">${tooltipCols.map(c => `<span class="ds-col-tag">${c}</span>`).join("")}</div>
        </div>
        <div class="ds-card-arrow">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M6 4l4 4-4 4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </div>
      `;

      card.addEventListener("click", () => openDataset(i));
      cardsGrid.appendChild(card);
    }

    pageLoader.classList.add("hidden");
    setTimeout(() => { pageLoader.style.display = "none"; }, 400);
  }

  await buildCards();

  // ── Open a dataset ───────────────────────────────────
  async function openDataset(idx) {
    const ds = datasets[idx];
    let json = ds._cache;
    if (!json) {
      const res = await fetch(ds.path);
      json = await res.json();
    }

    current = { meta: json.meta, points: json.points.map((p, i) => ({ _i: i, ...p })) };

    // Determine all columns (exclude internal ones)
    const skip = new Set(["x", "y", "cluster", "_i", "rangeVal", "label", "image"]);
    const colSet = new Set();
    current.points.forEach(p => {
      Object.keys(p).forEach(k => { if (!skip.has(k)) colSet.add(k); });
    });
    allCols = Array.from(colSet);

    // Default visible: tooltip columns + label
    const tip = current.meta.tooltipColumns || [];
    visCols = tip.length ? [...tip] : allCols.slice(0, 8);

    sortCol = null;
    sortAsc = true;
    page = 0;
    searchInput.value = "";

    applyFilter();
    buildColChips();
    renderViewer();

    // Show viewer, hide cards
    cardsSection.style.display = "none";
    viewer.style.display = "flex";
    viewerTitle.textContent = ds.displayName;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ── Back to cards ────────────────────────────────────
  viewerBack.addEventListener("click", () => {
    viewer.style.display = "none";
    cardsSection.style.display = "block";
    current = null;
  });

  // ── Column chip toggles ──────────────────────────────
  function buildColChips() {
    colChips.innerHTML = "";
    allCols.forEach(col => {
      const chip = document.createElement("button");
      chip.className = "col-chip" + (visCols.includes(col) ? " active" : "");
      chip.textContent = col;
      chip.addEventListener("click", () => {
        if (visCols.includes(col)) {
          visCols = visCols.filter(c => c !== col);
          chip.classList.remove("active");
        } else {
          visCols.push(col);
          chip.classList.add("active");
        }
        renderViewer();
      });
      colChips.appendChild(chip);
    });
  }

  // ── Searching ────────────────────────────────────────
  function applyFilter() {
    const q = searchInput.value.trim().toLowerCase();
    if (!q) {
      filtered = [...current.points];
    } else {
      filtered = current.points.filter(p =>
        allCols.some(c => String(p[c] ?? "").toLowerCase().includes(q)) ||
        String(p.label ?? "").toLowerCase().includes(q)
      );
    }

    // Sort
    if (sortCol) {
      filtered.sort((a, b) => {
        let va = a[sortCol], vb = b[sortCol];
        if (va == null) va = "";
        if (vb == null) vb = "";
        if (isNum(va) && isNum(vb)) return sortAsc ? va - vb : vb - va;
        return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
      });
    }

    page = 0;
    searchCount.textContent = `${filtered.length} / ${current.points.length}`;
  }

  searchInput.addEventListener("input", () => {
    applyFilter();
    renderViewer();
  });

  // ── Sorting ──────────────────────────────────────────
  function setSort(col) {
    if (sortCol === col) {
      sortAsc = !sortAsc;
    } else {
      sortCol = col;
      sortAsc = true;
    }
    applyFilter();
    renderViewer();
  }

  // ── Render ───────────────────────────────────────────
  function renderViewer() {
    if (!current) return;

    const meta = current.meta;
    const hasImg = meta.hasImages;
    const colorCol = meta.defaultColor;
    const cmap = (meta.colorMaps || {})[colorCol] || {};
    const totalPages = Math.ceil(filtered.length / ROWS_PER_PAGE);
    const start = page * ROWS_PER_PAGE;
    const slice = filtered.slice(start, start + ROWS_PER_PAGE);

    // Stats
    const numCols = allCols.length + (hasImg ? 1 : 0);
    viewerStats.innerHTML = `
      <span><span class="vs-num">${current.points.length}</span> rows</span>
      <span><span class="vs-num">${numCols}</span> columns</span>
      <span><span class="vs-num">${filtered.length}</span> matching</span>
    `;

    if (viewMode === "table") {
      tableWrap.style.display = "";
      dataCards.style.display = "none";
      renderTable(slice, hasImg, colorCol, cmap);
    } else {
      tableWrap.style.display = "none";
      dataCards.style.display = "";
      renderCards(slice, hasImg, colorCol, cmap);
    }

    renderPagination(totalPages);
  }

  // ── Table view ───────────────────────────────────────
  function renderTable(slice, hasImg, colorCol, cmap) {
    const arrow = (col) => {
      if (sortCol !== col) return "";
      return `<span class="sort-arrow">${sortAsc ? "\u25B2" : "\u25BC"}</span>`;
    };

    // Header
    let headHtml = "";
    if (hasImg) headHtml += `<th style="width:60px"></th>`;
    headHtml += `<th data-col="_label" class="${sortCol === "_label" ? "sorted" : ""}">Label ${arrow("_label")}</th>`;
    visCols.forEach(col => {
      headHtml += `<th data-col="${col}" class="${sortCol === col ? "sorted" : ""}">${col} ${arrow(col)}</th>`;
    });
    dataHead.innerHTML = headHtml;

    // Rows
    let bodyHtml = "";
    slice.forEach(p => {
      bodyHtml += "<tr>";
      if (hasImg) {
        const src = p.image || "";
        bodyHtml += src
          ? `<td class="img-cell"><img class="tbl-thumb" src="${src}" alt="" loading="lazy" data-label="${(p.label || "").replace(/"/g, "&quot;")}"></td>`
          : `<td class="img-cell"></td>`;
      }

      // Label with colour dot
      const color = colorForVal(p[colorCol], cmap);
      bodyHtml += `<td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:7px;vertical-align:middle"></span>${fmt(p.label)}</td>`;

      visCols.forEach(col => {
        const val = p[col];
        if (isNum(val)) {
          bodyHtml += `<td class="cell-num">${fmt(val)}</td>`;
        } else if (col === colorCol && cmap[val]) {
          bodyHtml += `<td><span class="cell-badge" style="background:${cmap[val]}18;color:${cmap[val]};border:1px solid ${cmap[val]}33">${fmt(val)}</span></td>`;
        } else {
          bodyHtml += `<td>${fmt(val)}</td>`;
        }
      });
      bodyHtml += "</tr>";
    });
    dataBody.innerHTML = bodyHtml;

    // Sort clicks
    dataHead.querySelectorAll("th[data-col]").forEach(th => {
      th.addEventListener("click", () => {
        const col = th.dataset.col;
        if (col === "_label") {
          // sort by label
          if (sortCol === "label") sortAsc = !sortAsc;
          else { sortCol = "label"; sortAsc = true; }
          applyFilter();
          renderViewer();
        } else {
          setSort(col);
        }
      });
    });

    // Thumbnail clicks → lightbox
    dataBody.querySelectorAll(".tbl-thumb").forEach(img => {
      img.addEventListener("click", (e) => {
        e.stopPropagation();
        openLightbox(img.src, img.dataset.label);
      });
    });
  }

  // ── Card view ────────────────────────────────────────
  function renderCards(slice, hasImg, colorCol, cmap) {
    const tipCols = visCols.slice(0, 5);
    let html = "";

    slice.forEach(p => {
      const color = colorForVal(p[colorCol], cmap);
      let imgHtml = "";
      if (hasImg && p.image) {
        imgHtml = `<img class="data-card-img" src="${p.image}" alt="" loading="lazy" data-label="${(p.label || "").replace(/"/g, "&quot;")}">`;
      }

      const fields = tipCols.map(c => {
        const v = p[c];
        return `<div class="data-card-field"><b>${c}:</b> <span>${fmt(v)}</span></div>`;
      }).join("");

      html += `
        <div class="data-card">
          ${imgHtml}
          <div class="data-card-body">
            <div class="data-card-label" style="color:${color}">${fmt(p.label)}</div>
            <div class="data-card-cluster">Cluster ${p.cluster}</div>
            <div class="data-card-fields">${fields}</div>
          </div>
        </div>
      `;
    });

    dataCards.innerHTML = html;

    // Image click → lightbox
    dataCards.querySelectorAll(".data-card-img").forEach(img => {
      img.addEventListener("click", (e) => {
        e.stopPropagation();
        openLightbox(img.src, img.dataset.label);
      });
    });
  }

  // ── Pagination ───────────────────────────────────────
  function renderPagination(totalPages) {
    if (totalPages <= 1) {
      pagination.innerHTML = `<span class="pg-info">Showing all ${filtered.length} rows</span>`;
      return;
    }

    let html = "";

    // Prev
    html += `<button class="pg-btn" data-page="${page - 1}" ${page === 0 ? "disabled" : ""}>&laquo;</button>`;

    // Pages with ellipsis
    const visible = buildPageRange(page, totalPages);
    visible.forEach(p => {
      if (p === "...") {
        html += `<span class="pg-info">&hellip;</span>`;
      } else {
        html += `<button class="pg-btn ${p === page ? "active" : ""}" data-page="${p}">${p + 1}</button>`;
      }
    });

    // Next
    html += `<button class="pg-btn" data-page="${page + 1}" ${page >= totalPages - 1 ? "disabled" : ""}>&raquo;</button>`;

    // Info
    const start = page * ROWS_PER_PAGE + 1;
    const end = Math.min(start + ROWS_PER_PAGE - 1, filtered.length);
    html += `<span class="pg-info">${start}\u2013${end} of ${filtered.length}</span>`;

    pagination.innerHTML = html;

    pagination.querySelectorAll(".pg-btn[data-page]").forEach(btn => {
      btn.addEventListener("click", () => {
        const p = parseInt(btn.dataset.page, 10);
        if (p >= 0 && p < totalPages) {
          page = p;
          renderViewer();
          tableWrap.scrollTo({ top: 0, behavior: "smooth" });
        }
      });
    });
  }

  function buildPageRange(current, total) {
    if (total <= 7) return Array.from({ length: total }, (_, i) => i);
    const pages = [];
    pages.push(0);
    if (current > 3) pages.push("...");
    for (let i = Math.max(1, current - 1); i <= Math.min(total - 2, current + 1); i++) {
      pages.push(i);
    }
    if (current < total - 4) pages.push("...");
    pages.push(total - 1);
    return pages;
  }

  // ── View toggle ──────────────────────────────────────
  document.querySelectorAll(".vt-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      viewMode = btn.dataset.view;
      document.querySelectorAll(".vt-btn").forEach(b => b.classList.toggle("active", b === btn));
      renderViewer();
    });
  });

  // ── Lightbox ─────────────────────────────────────────
  function openLightbox(src, label) {
    lbImg.src = src;
    lbLabel.textContent = label || "";
    lightbox.classList.add("visible");
  }

  lightbox.addEventListener("click", () => {
    lightbox.classList.remove("visible");
    lbImg.src = "";
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightbox.classList.contains("visible")) {
      lightbox.classList.remove("visible");
      lbImg.src = "";
    }
  });

})();

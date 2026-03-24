(async function () {
  // --- Config ---
  const POINT_RADIUS = 5;
  const POINT_RADIUS_HOVER = 8;
  const HIT_RADIUS = 14;
  const SEPARATION_RADIUS = 10;
  const SEPARATION_TICKS = 110;
  const SEPARATION_MAX_OFFSET = 28;
  const PADDING = { top: 24, right: 24, bottom: 34, left: 34 };

  const DOMAIN_PAD = 0.08;

  const PALETTE = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#6366f1",
    "#84cc16", "#e11d48", "#0ea5e9", "#d946ef", "#a3e635",
    "#fb923c", "#2dd4bf", "#818cf8", "#fbbf24", "#34d399",
  ];

  const CLUSTER_PAL = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#e6ab02",
    "#66c2a5", "#fc8d62", "#8da0cb", "#ffd92f",
  ];

  // --- DOM refs ---
  const canvas = document.getElementById("plot");
  const ctx = canvas.getContext("2d");
  const container = document.getElementById("plotContainer");
  const brushSvg = d3.select("#overlay");
  const tip = document.getElementById("tip");
  const loadEl = document.getElementById("loading");
  const datasetSel = document.getElementById("dataset-sel");
  const headerControls = document.querySelector(".header-controls");
  const popup = document.getElementById("popup");
  const popupBackdrop = document.getElementById("popupBackdrop");
  const popupScroll = document.getElementById("popupScroll");
  const popupTitle = document.getElementById("pop-title");
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightboxImg");
  const lightboxLabel = document.getElementById("lightboxLabel");


  // --- State ---
  let DATA = [];
  let META = {};
  let X_DOM, Y_DOM, R_DOM;
  let colorMaps = {};
  let qt, xs, ys;
  let W = 0, H = 0, dpr = 1;
  let playRaf = null, playPrev = null;
  let brushG;
  let hoveredIdx = null;
  let highlightedVal = null;  // clicked legend value
  let hoveredVal = null;      // hovered legend value
  let popupView = "grid";
  let popupSortBy = "label";
  let popupSortAsc = true;
  let popupPosition = null;
  let popupDrag = null;

  let lightboxList = [];
  let lightboxIdx = -1;

  const st = {
    colorBy: "",
    filters: {},
    rMin: 0,
    rMax: 1,
    brushed: null,
    playing: false,
  };

  const slS = document.getElementById("sl-start");
  const slE = document.getElementById("sl-end");
  const rfill = document.getElementById("rfill");
  const lblS = document.getElementById("lbl-start");
  const lblE = document.getElementById("lbl-end");
  const durEl = document.getElementById("range-dur");
  const playBtn = document.getElementById("play-btn");
  const hasSlider = !!(slS && slE && rfill && playBtn);
  const selectedStatus = document.createElement("span");
  selectedStatus.className = "selected-status";
  selectedStatus.textContent = "Selected: 0";
  if (headerControls) headerControls.appendChild(selectedStatus);

  // ── helpers ───────────────────────────────────────────────

  function fmt(v) {
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
    return String(v);
  }

  function rangeLabel(v) {
    const label = (META.rangeColumn && META.rangeColumn !== "_index") ? META.rangeColumn : "Row";
    return `${label}: ${fmt(v)}`;
  }

  function getColor(d) {
    const col = st.colorBy;
    if (!col) return "#3b82f6";
    const map = colorMaps[col];
    if (map) return map[d[col]] || "#6b7280";
    return "#3b82f6";
  }

  function getPointX(d) {
    return Number.isFinite(d._sx) ? d._sx : xs(d.x);
  }

  function getPointY(d) {
    return Number.isFinite(d._sy) ? d._sy : ys(d.y);
  }

  function isFiltered(d) {
    for (const col in st.filters) {
      const val = st.filters[col];
      if (val && String(d[col]) !== val) return true;
    }
    return false;
  }

  // ── legend ──────────────────────────────────────────────────

  function buildLegend() {
    const el = document.getElementById("legend");
    el.innerHTML = "";
    const col = st.colorBy;
    if (!col) return;
    const map = colorMaps[col];
    if (!map) return;

    // Count items per value
    const counts = {};
    DATA.forEach(d => {
      if (!isFiltered(d)) {
        const key = String(d[col]);
        counts[key] = (counts[key] || 0) + 1;
      }
    });

    Object.keys(map).sort().forEach((key) => {
      const count = counts[key] || 0;
      const item = document.createElement("div");
      item.className = "legend-item";
      item.dataset.val = key;
      item.innerHTML = `<span class="legend-dot" style="background:${map[key]}"></span>${key}<span class="legend-count">(${count})</span>`;

      item.addEventListener("click", (e) => {
        e.stopPropagation();
        if (highlightedVal === key) {
          highlightedVal = null;
        } else {
          highlightedVal = key;
        }
        updateLegendStyles();
        render();
      });


      el.appendChild(item);
    });
    updateLegendStyles();
  }

  function updateLegendStyles() {
    document.querySelectorAll(".legend-item").forEach(item => {
      const val = item.dataset.val;
      item.classList.remove("highlighted", "dimmed");
      if (highlightedVal) {
        if (val === highlightedVal) {
          item.classList.add("highlighted");
        } else {
          item.classList.add("dimmed");
        }
      }
    });
  }

  // ── quadtree ──────────────────────────────────────────────

  function buildQT() {
    qt = d3.quadtree()
      .x((d) => getPointX(d))
      .y((d) => getPointY(d))
      .addAll(DATA.filter((d) => !isFiltered(d)));
  }

  function updatePointLayout() {
    if (!xs || !ys) return;

    const minX = PADDING.left + POINT_RADIUS;
    const maxX = W - PADDING.right - POINT_RADIUS;
    const minY = PADDING.top + POINT_RADIUS;
    const maxY = H - PADDING.bottom - POINT_RADIUS;

    DATA.forEach((d) => {
      d._sx = xs(d.x);
      d._sy = ys(d.y);
    });

    const visible = DATA.filter((d) => !isFiltered(d));
    if (visible.length < 2) return;

    const nodes = visible.map((d) => ({
      d,
      x: d._sx,
      y: d._sy,
      anchorX: d._sx,
      anchorY: d._sy,
    }));

    const sim = d3.forceSimulation(nodes)
      .stop()
      .alpha(1)
      .velocityDecay(0.3)
      .force("x", d3.forceX((n) => n.anchorX).strength(0.08))
      .force("y", d3.forceY((n) => n.anchorY).strength(0.08))
      .force("collide", d3.forceCollide(SEPARATION_RADIUS).iterations(2));

    for (let i = 0; i < SEPARATION_TICKS; i++) {
      sim.tick();
      nodes.forEach((n) => {
        const dx = n.x - n.anchorX;
        const dy = n.y - n.anchorY;
        const dist = Math.hypot(dx, dy);
        if (dist > SEPARATION_MAX_OFFSET) {
          const scale = SEPARATION_MAX_OFFSET / dist;
          n.x = n.anchorX + dx * scale;
          n.y = n.anchorY + dy * scale;
        }
        n.x = Math.min(maxX, Math.max(minX, n.x));
        n.y = Math.min(maxY, Math.max(minY, n.y));
      });
    }

    nodes.forEach((n) => {
      n.d._sx = n.x;
      n.d._sy = n.y;
    });
  }

  // ── render ────────────────────────────────────────────────

  function render() {
    if (!W || !H || !xs || !ys) return;
    ctx.clearRect(0, 0, W, H);

    const hasBrush = st.brushed !== null;
    const brushedSet = hasBrush ? new Set(st.brushed.map((d) => d._i)) : null;
    const col = st.colorBy;

    for (let pass = 0; pass < 2; pass++) {
      for (const d of DATA) {
        if (isFiltered(d)) continue;
        const inRange = d.rangeVal >= st.rMin && d.rangeVal <= st.rMax;
        const inBrush = !hasBrush || brushedSet.has(d._i);
        const isHovered = d._i === hoveredIdx;

        const activeVal = highlightedVal || hoveredVal;
        let alpha;
        if (activeVal) {
          alpha = (String(d[col]) === activeVal) ? 1.0 : 0.12;
        } else if (!inRange && (!hasBrush || !inBrush)) {
          alpha = 0.05;
        } else if (!inRange) {
          alpha = 0.1;
        } else if (hasBrush && !inBrush) {
          alpha = 0.08;
        } else if (isHovered) {
          alpha = 1.0;
        } else {
          alpha = 0.85;
        }

        const bright = alpha > 0.5;
        if (pass === 0 && bright) continue;
        if (pass === 1 && !bright) continue;

        const r = isHovered ? POINT_RADIUS_HOVER : POINT_RADIUS;
        ctx.globalAlpha = alpha;
        ctx.fillStyle = getColor(d);
        ctx.beginPath();
        ctx.arc(getPointX(d), getPointY(d), r, 0, Math.PI * 2);
        ctx.fill();

        if (isHovered) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;

  }

  // ── layout ────────────────────────────────────────────────

  function expandDomain(dom) {
    const [lo, hi] = dom;
    const span = hi - lo || 1;
    const pad = span * DOMAIN_PAD;
    return [lo - pad, hi + pad];
  }

  function resize() {
    if (!X_DOM || !Y_DOM) return;
    dpr = devicePixelRatio || 1;
    const r = container.getBoundingClientRect();
    W = r.width; H = r.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    xs = d3.scaleLinear().domain(expandDomain(X_DOM)).range([PADDING.left + POINT_RADIUS, W - PADDING.right - POINT_RADIUS]);
    ys = d3.scaleLinear().domain(expandDomain(Y_DOM)).range([H - PADDING.bottom - POINT_RADIUS, PADDING.top + POINT_RADIUS]);
    brushSvg.attr("width", W).attr("height", H);
    updatePointLayout();
    buildQT();
    render();
  }

  // ── dynamic controls ──────────────────────────────────────

  function buildColorBySelect(colorCols, defaultCol) {
    // Color-by dropdown removed from UI; default color is set automatically
  }

  function buildFilterSelects(filterCols, colMeta) {
    // Filter dropdowns removed from UI
  }

  // ── range slider ──────────────────────────────────────────

  function syncSliderUI() {
    if (!R_DOM || !hasSlider) return;
    const total = R_DOM[1] - R_DOM[0] || 1;
    slS.min = R_DOM[0]; slS.max = R_DOM[1]; slS.step = total / 500;
    slE.min = R_DOM[0]; slE.max = R_DOM[1]; slE.step = total / 500;
    slS.value = st.rMin; slE.value = st.rMax;
    const left = ((st.rMin - R_DOM[0]) / total) * 100;
    const width = ((st.rMax - st.rMin) / total) * 100;
    rfill.style.left = `${left}%`;
    rfill.style.width = `${width}%`;
    lblS.textContent = rangeLabel(Math.round(st.rMin));
    lblE.textContent = rangeLabel(Math.round(st.rMax));
    const count = DATA.filter((d) => !isFiltered(d) && d.rangeVal >= st.rMin && d.rangeVal <= st.rMax).length;
    durEl.textContent = `${count.toLocaleString()} rows`;
    durEl.style.left = `${left + width / 2}%`;
  }

  function buildTicks() {
    if (!R_DOM || !hasSlider) return;
    const n = 8;
    const ticks = [];
    for (let i = 0; i < n; i++) {
      const v = R_DOM[0] + (i / (n - 1)) * (R_DOM[1] - R_DOM[0]);
      ticks.push(`<span>${fmt(Math.round(v))}</span>`);
    }
    document.getElementById("year-ticks").innerHTML = ticks.join("");
  }

  function updateSelectedStatus(count) {
    selectedStatus.textContent = `Selected: ${count.toLocaleString()}`;
    selectedStatus.classList.toggle("active", count > 0);
  }

  function clampPopupPosition(left, top) {
    const rect = popup.getBoundingClientRect();
    const margin = 12;
    const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
    return {
      left: Math.min(Math.max(margin, left), maxLeft),
      top: Math.min(Math.max(margin, top), maxTop),
    };
  }

  function applyPopupPosition(pos) {
    if (!pos) return;
    popup.style.left = `${pos.left}px`;
    popup.style.top = `${pos.top}px`;
    popup.style.transform = "none";
  }

  function positionPopup() {
    if (!popup.classList.contains("visible")) return;
    if (!popupPosition) {
      const rect = popup.getBoundingClientRect();
      popupPosition = clampPopupPosition(window.innerWidth - rect.width - 24, 88);
    } else {
      popupPosition = clampPopupPosition(popupPosition.left, popupPosition.top);
    }
    applyPopupPosition(popupPosition);
  }

  if (hasSlider) {
    slS.addEventListener("input", () => {
      if (!R_DOM) return;
      st.rMin = Math.min(Number(slS.value), st.rMax - (R_DOM[1] - R_DOM[0]) / 500);
      syncSliderUI();
      render();
      if (popup.classList.contains("visible") && st.brushed) renderPopupContent(st.brushed);
    });
    slE.addEventListener("input", () => {
      if (!R_DOM) return;
      st.rMax = Math.max(Number(slE.value), st.rMin + (R_DOM[1] - R_DOM[0]) / 500);
      syncSliderUI();
      render();
      if (popup.classList.contains("visible") && st.brushed) renderPopupContent(st.brushed);
    });

    rfill.addEventListener("mousedown", (e) => {
      if (st.playing || !R_DOM) return;
      e.preventDefault();
      const trackRect = document.getElementById("rwrap").getBoundingClientRect();
      const dragX = e.clientX, dragMin = st.rMin, dragMax = st.rMax;
      rfill.style.cursor = "grabbing";
      function onMove(ev) {
        const dx = ev.clientX - dragX;
        const delta = (dx / trackRect.width) * (R_DOM[1] - R_DOM[0]);
        let lo = dragMin + delta, hi = dragMax + delta;
        const span = dragMax - dragMin;
        if (lo < R_DOM[0]) { lo = R_DOM[0]; hi = R_DOM[0] + span; }
        if (hi > R_DOM[1]) { hi = R_DOM[1]; lo = R_DOM[1] - span; }
        st.rMin = lo; st.rMax = hi;
        syncSliderUI(); render();
      }
      function onUp() {
        rfill.style.cursor = "grab";
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      }
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
  }

  // ── playback ──────────────────────────────────────────────

  function startPlay() {
    if (!R_DOM || !hasSlider) return;
    if (st.rMax >= R_DOM[1]) {
      const span = st.rMax - st.rMin;
      st.rMin = R_DOM[0]; st.rMax = R_DOM[0] + span;
    }
    st.playing = true; playPrev = null;
    playBtn.textContent = "Pause"; playBtn.classList.add("playing");
    playRaf = requestAnimationFrame(playStep);
  }
  function pausePlay() {
    st.playing = false;
    if (playRaf) cancelAnimationFrame(playRaf);
    if (playBtn) { playBtn.textContent = "\u25B6 Play"; playBtn.classList.remove("playing"); }
  }
  function playStep(ts) {
    if (!st.playing || !R_DOM) return;
    if (!playPrev) playPrev = ts;
    const dt = ts - playPrev; playPrev = ts;
    const speed = (R_DOM[1] - R_DOM[0]) / 8;
    const span = st.rMax - st.rMin;
    st.rMax = Math.min(R_DOM[1], st.rMax + dt * speed / 1000);
    st.rMin = Math.max(R_DOM[0], st.rMax - span);
    syncSliderUI(); render();
    if (st.rMax >= R_DOM[1]) { pausePlay(); return; }
    playRaf = requestAnimationFrame(playStep);
  }
  if (playBtn) playBtn.addEventListener("click", () => { st.playing ? pausePlay() : startPlay(); });

  // ── color-by (uses default from dataset metadata) ─────────

  // ── brush + popup ─────────────────────────────────────────

  const brush = d3.brush().on("brush end", onBrushChanged);
  brushG = brushSvg.append("g").attr("class", "brush").call(brush);

  function getBrushedPoints(selection) {
    if (!selection) return [];
    const [[x0, y0], [x1, y1]] = selection;
    return DATA.filter((d) => {
      if (isFiltered(d)) return false;
      if (highlightedVal && String(d[st.colorBy]) !== highlightedVal) return false;
      const sx = getPointX(d);
      const sy = getPointY(d);
      return sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1;
    });
  }

  function clearBrushedSelection() {
    st.brushed = null;
    closePopup();
    updateSelectedStatus(0);
    render();
  }

  function onBrushChanged(event) {
    if (!event.selection) {
      clearBrushedSelection();
      return;
    }

    const pts = getBrushedPoints(event.selection);
    st.brushed = pts;
    updateSelectedStatus(pts.length);
    render();
    showPopup(pts);
  }

  function showPopup(pts) {
    if (!pts || !pts.length) {
      closePopup(); return;
    }
    popupTitle.textContent = `${pts.length} row${pts.length === 1 ? "" : "s"} selected`;
    popup.classList.add("visible");
    popup.classList.remove("hidden");
    tip.style.display = "none";

    // Build badges from color-by column
    const col = st.colorBy;
    const badgesEl = document.getElementById("popupBadges");
    if (col && colorMaps[col]) {
      const selCounts = {};
      pts.forEach(d => {
        const key = String(d[col]);
        selCounts[key] = (selCounts[key] || 0) + 1;
      });
      badgesEl.innerHTML = Object.entries(selCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([val, count]) => {
          const c = colorMaps[col][val] || "#6b7280";
          return `<span class="popup-badge" style="background:${c}18; border-color:${c}33">
            <span class="popup-badge-dot" style="background:${c}"></span>
            ${val}
            <span class="popup-badge-count" style="color:${c}">${count}</span>
          </span>`;
        }).join("");
    } else {
      badgesEl.innerHTML = "";
    }

    try {
      renderPopupContent(pts);
    } catch (err) {
      popupScroll.innerHTML = `<div style="padding:16px;color:#cbd5e1">Selection opened, but the detail view failed to render.</div>`;
      console.error(err);
    }
    requestAnimationFrame(positionPopup);
  }

  function renderPopupContent(pts) {
    const sorted = [...(pts || st.brushed || [])];
    sorted.sort((a, b) => {
      let va = a[popupSortBy], vb = b[popupSortBy];
      if (va == null) va = ""; if (vb == null) vb = "";
      if (typeof va === "number" && typeof vb === "number") return popupSortAsc ? va - vb : vb - va;
      return popupSortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });

    if (popupView === "grid") {
      renderGridView(sorted);
    } else {
      renderTableView(sorted);
    }
  }

  function renderGridView(sorted) {
    const tipCols = META.tooltipColumns || [];
    const hasImg = META.hasImages;
    const col = st.colorBy;

    popupScroll.innerHTML = "";
    const grid = document.createElement("div");
    grid.className = "popup-grid";

    sorted.forEach(d => {
      const item = document.createElement("div");
      item.className = "popup-grid-item";

      let imgHtml = "";
      if (hasImg && d.image) {
        imgHtml = `<img src="${d.image}" alt="${d.label}" loading="lazy">`;
      }

      const color = getColor(d);
      const details = tipCols.slice(0, 3).map(c => {
        const val = d[c];
        return val != null ? `${c}: ${fmt(val)}` : "";
      }).filter(Boolean).join(" · ");

      item.innerHTML = `
        ${imgHtml}
        <div class="popup-grid-label">
          <div class="popup-grid-pub" style="color:${color}">${d.label}</div>
          <div style="font-size:11px;color:#94a3b8;margin-top:2px">${details}</div>
        </div>
      `;

      if (hasImg && d.image) {
        item.addEventListener("click", () => openLightbox(d, sorted));
      }
      grid.appendChild(item);
    });

    popupScroll.appendChild(grid);
  }

  function renderTableView(sorted) {
    const tipCols = META.tooltipColumns || [];
    const hasImg = META.hasImages;
    const allCols = hasImg ? ["image", "label", "cluster", ...tipCols] : ["label", "cluster", ...tipCols];

    const sortArrow = (col) => {
      if (popupSortBy !== col) return "";
      return `<span class="sort-arrow">${popupSortAsc ? "\u25B2" : "\u25BC"}</span>`;
    };

    let html = `<table id="pop-table"><thead><tr>`;
    allCols.forEach(col => {
      const dispName = col === "image" ? "" : col === "label" ? "Label" : col === "cluster" ? "Cluster" : col;
      const cls = col === popupSortBy ? "sorted" : "";
      const w = col === "image" ? ' style="width:60px"' : "";
      html += `<th data-sort="${col}" class="${cls}"${w}>${dispName} ${sortArrow(col)}</th>`;
    });
    html += `</tr></thead><tbody>`;

    sorted.forEach(d => {
      const inRange = d.rangeVal >= st.rMin && d.rangeVal <= st.rMax;
      const cells = allCols.map(col => {
        if (col === "image") {
          const src = d.image || "";
          return src ? `<td><img class="pop-thumb" src="${src}" alt="" loading="lazy"></td>` : `<td></td>`;
        }
        if (col === "cluster") {
          const cc = CLUSTER_PAL[d.cluster % CLUSTER_PAL.length];
          return `<td><span class="clust-sq" style="background:${cc}"></span> ${d.cluster}</td>`;
        }
        const val = d[col];
        return `<td>${val != null ? fmt(val) : ""}</td>`;
      }).join("");
      html += `<tr style="opacity:${inRange ? 1 : .45}">${cells}</tr>`;
    });

    html += `</tbody></table>`;
    popupScroll.innerHTML = html;

    // Sortable headers
    popupScroll.querySelector("thead").addEventListener("click", (e) => {
      const th = e.target.closest("th[data-sort]");
      if (!th || th.dataset.sort === "image") return;
      const col = th.dataset.sort;
      if (col === popupSortBy) popupSortAsc = !popupSortAsc;
      else { popupSortBy = col; popupSortAsc = true; }
      renderPopupContent(st.brushed);
    });

    // Row click opens lightbox if images exist
    if (hasImg) {
      popupScroll.querySelectorAll("tbody tr").forEach((tr, i) => {
        tr.addEventListener("click", () => openLightbox(sorted[i], sorted));
      });
    }
  }

  function closePopup() {
    popup.classList.remove("visible");
    popup.classList.add("hidden");
    popupBackdrop.classList.remove("visible");
  }

  function dismissAll() {
    closePopup();
    st.brushed = null;
    updateSelectedStatus(0);
    brushG.call(brush.move, null);
    render();
  }

  document.getElementById("pop-close").addEventListener("click", dismissAll);
  popupBackdrop.addEventListener("click", dismissAll);

  document.querySelector(".pop-head").addEventListener("mousedown", (e) => {
    if (e.target.closest("button")) return;
    e.preventDefault();
    if (!popup.classList.contains("visible")) return;
    const rect = popup.getBoundingClientRect();
    popupPosition = { left: rect.left, top: rect.top };
    popupDrag = {
      dx: e.clientX - rect.left,
      dy: e.clientY - rect.top,
    };
    popup.classList.add("dragging");
  });

  document.addEventListener("mousemove", (e) => {
    if (!popupDrag) return;
    popupPosition = clampPopupPosition(e.clientX - popupDrag.dx, e.clientY - popupDrag.dy);
    applyPopupPosition(popupPosition);
  });

  document.addEventListener("mouseup", () => {
    if (!popupDrag) return;
    popupDrag = null;
    popup.classList.remove("dragging");
  });

  // View toggle
  document.querySelectorAll(".view-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      popupView = btn.dataset.view;
      document.querySelectorAll(".view-btn").forEach(b => b.classList.toggle("active", b === btn));
      if (st.brushed) renderPopupContent(st.brushed);
    });
  });

  // ── lightbox ──────────────────────────────────────────────

  function openLightbox(d, list) {
    if (!d.image) return;
    lightboxList = (list || []).filter(p => p.image);
    lightboxIdx = lightboxList.findIndex(p => p._i === d._i);
    if (lightboxIdx === -1) lightboxIdx = 0;
    showLightboxImage();
    lightbox.classList.add("visible");
  }

  function showLightboxImage() {
    const d = lightboxList[lightboxIdx];
    if (!d) return;
    lightboxImg.src = d.image;
    const color = getColor(d);
    lightboxLabel.innerHTML = `<strong style="color:${color}">${d.label}</strong><span style="color:#64748b;margin-left:8px">${lightboxIdx + 1} / ${lightboxList.length}</span>`;
  }

  lightbox.addEventListener("click", () => {
    lightbox.classList.remove("visible");
    lightboxImg.src = "";
    lightboxList = [];
    lightboxIdx = -1;
  });

  // ── tooltip ───────────────────────────────────────────────

  document.getElementById("overlay").addEventListener("mousemove", (e) => {
    if (!qt || st.playing) { tip.style.display = "none"; return; }
    if (e.buttons > 0) { tip.style.display = "none"; return; }
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const found = qt.find(mx, my, HIT_RADIUS);

    if (!found) {
      if (hoveredIdx !== null) {
        hoveredIdx = null;
        render();
      }
      tip.style.display = "none";
      return;
    }

    const idx = found._i;
    if (idx !== hoveredIdx) {
      hoveredIdx = idx;
      render();
    }

    const tipCols = META.tooltipColumns || [];
    const rows = tipCols.map((col) => {
      const val = found[col];
      return `<div class="tip-row"><b>${col}:</b> ${val != null ? fmt(val) : "\u2014"}</div>`;
    }).join("");

    const tipImgEl = document.getElementById("tipImg");
    if (META.hasImages && found.image) {
      tipImgEl.src = found.image;
      tipImgEl.style.display = "block";
    } else {
      tipImgEl.style.display = "none";
    }

    document.getElementById("tipTitle").textContent = found.label;
    document.getElementById("tipCluster").textContent = `Cluster ${found.cluster}`;
    document.getElementById("tipRows").innerHTML = rows;

    tip.style.display = "block";
    let tx = e.clientX + 16;
    let ty = e.clientY - 10;
    if (tx + 320 > window.innerWidth) tx = e.clientX - 320 - 16;
    if (ty < 0) ty = 10;
    tip.style.left = `${tx}px`;
    tip.style.top = `${ty}px`;
  });

  document.getElementById("overlay").addEventListener("mouseleave", () => {
    hoveredIdx = null;
    tip.style.display = "none";
    render();
  });

  // ── keyboard ──────────────────────────────────────────────

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const archModal = document.getElementById("archModal");
      if (archModal.classList.contains("visible")) {
        closeArchModal();
      } else if (lightbox.classList.contains("visible")) {
        lightbox.classList.remove("visible");
        lightboxImg.src = "";
      } else if (popup.classList.contains("visible")) {
        dismissAll();
      } else if (highlightedVal) {
        highlightedVal = null;
        updateLegendStyles();
        render();
      }
    }

    if (lightbox.classList.contains("visible")) {
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        lightboxIdx = (lightboxIdx - 1 + lightboxList.length) % lightboxList.length;
        showLightboxImage();
      } else if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        lightboxIdx = (lightboxIdx + 1) % lightboxList.length;
        showLightboxImage();
      }
    }
  });

  // ── architecture modal ────────────────────────────────────

  const ARCH_STEPS = [
    {
      icon: "\uD83D\uDCC4", color: "#3b82f6",
      title: "Load CSV Data",
      desc: "Fetch a real-world CSV dataset (Pokemon, Movies, Art, E-commerce) with rich categorical and numerical columns.",
      tags: [{ text: "pandas", bg: "#1e3a5f", fg: "#60a5fa" }, { text: "CSV", bg: "#1e3a5f", fg: "#60a5fa" }],
    },
    {
      icon: "\uD83E\uDDE0", color: "#f59e0b",
      title: "Gemini Text Embeddings",
      desc: "Convert each row to a descriptive sentence and embed via Google Gemini embedding model into 768-dimensional vectors capturing semantic meaning.",
      tags: [{ text: "gemini-embedding-001", bg: "#451a03", fg: "#fbbf24" }, { text: "768 dims", bg: "#451a03", fg: "#fbbf24" }],
    },
    {
      icon: "\uD83D\uDCCC", color: "#22c55e",
      title: "UMAP Dimensionality Reduction",
      desc: "UMAP projects 768-dim vectors to 2D coordinates preserving local structure \u2014 similar rows land near each other on the scatter plot.",
      tags: [{ text: "n_neighbors=15", bg: "#052e16", fg: "#4ade80" }, { text: "cosine", bg: "#052e16", fg: "#4ade80" }],
    },
    {
      icon: "\uD83D\uDD2E", color: "#8b5cf6",
      title: "KMeans Clustering",
      desc: "Automatic cluster detection groups similar rows together, revealing hidden patterns in the data.",
      tags: [{ text: "scikit-learn", bg: "#2e1065", fg: "#c084fc" }, { text: "KMeans", bg: "#2e1065", fg: "#c084fc" }],
    },
    {
      icon: "\u2728", color: "#06b6d4",
      title: "Interactive Scatter Plot",
      desc: "D3.js + Canvas renders the 2D scatter with brush selection, quadtree hover, tooltips, legend highlighting, range slider, and modal views.",
      tags: [{ text: "D3.js v7", bg: "#083344", fg: "#22d3ee" }, { text: "Canvas", bg: "#083344", fg: "#22d3ee" }],
    },
  ];

  function buildArchCards() {
    const cardsEl = document.getElementById("archCards");
    cardsEl.innerHTML = ARCH_STEPS.map(s => `
      <div class="arch-card">
        <div class="arch-card-icon" style="background:${s.color}22; color:${s.color}">${s.icon}</div>
        <div class="arch-card-body">
          <div class="arch-card-title">${s.title}</div>
          <div class="arch-card-desc">${s.desc}</div>
          <div class="arch-card-tags">${s.tags.map(t =>
            `<span class="arch-tag" style="background:${t.bg};color:${t.fg}">${t.text}</span>`
          ).join("")}</div>
        </div>
      </div>
    `).join("");
  }

  function openArchModal() {
    buildArchCards();
    document.getElementById("archModal").classList.add("visible");
    document.getElementById("archBackdrop").classList.add("visible");
  }

  function closeArchModal() {
    document.getElementById("archModal").classList.remove("visible");
    document.getElementById("archBackdrop").classList.remove("visible");
  }

  document.getElementById("archBtn").addEventListener("click", openArchModal);
  document.getElementById("archClose").addEventListener("click", closeArchModal);
  document.getElementById("archBackdrop").addEventListener("click", closeArchModal);

  // ── resize ────────────────────────────────────────────────

  new ResizeObserver(() => {
    resize();
    positionPopup();
  }).observe(container);

  // ── load a dataset ────────────────────────────────────────

  async function loadDataset(path) {
    if (loadEl) { loadEl.classList.remove("error"); loadEl.querySelector(".loader-text").textContent = "Loading dataset"; loadEl.style.display = "flex"; }
    pausePlay();
    st.brushed = null;
    st.filters = {};
    highlightedVal = null;
    hoveredVal = null;
    hoveredIdx = null;
    popupPosition = null;
    closePopup();

    try {
      const json = await fetch(path).then((r) => {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      });

      META = json.meta;
      colorMaps = META.colorMaps || {};
      X_DOM = json.domains.x;
      Y_DOM = json.domains.y;
      R_DOM = json.domains.range;
      DATA = json.points.map((d, i) => ({ _i: i, ...d }));

      st.colorBy = META.defaultColor || "";
      st.rMin = R_DOM[0];
      st.rMax = R_DOM[1];

      document.title = META.displayName || "CSV Data Explorer";
      document.querySelector("h1").innerHTML = `${META.displayName || "CSV Data"} <span class="accent">Explorer</span>`;
      document.getElementById("statCount").textContent = `${META.totalRows} rows`;
      document.getElementById("statMode").textContent = META.embeddingMode === "text-gemini" ? "Gemini Embeddings" : "Numerical Embeddings";
      updateSelectedStatus(0);

      buildColorBySelect(META.colorColumns || [], META.defaultColor || "");
      buildFilterSelects(META.filterColumns || [], META.columns || {});
      buildLegend();
      buildTicks();
      syncSliderUI();
      if (loadEl) loadEl.style.display = "none";
      resize();
    } catch (err) {
      if (loadEl) { loadEl.classList.add("error"); loadEl.querySelector(".loader-text").textContent = `Failed to load ${path} \u2013 ${err.message}`; loadEl.style.display = "flex"; }
      console.error(err);
    }
  }

  // ── dataset switcher ──────────────────────────────────────

  datasetSel.addEventListener("change", () => {
    const path = datasetSel.value;
    if (path) loadDataset(path);
  });

  // ── init ──────────────────────────────────────────────────

  try {
    const datasets = await fetch("data/datasets.json").then((r) => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    });

    datasetSel.innerHTML = "";
    datasets.forEach((ds) => {
      const opt = document.createElement("option");
      opt.value = ds.path;
      opt.textContent = ds.displayName;
      datasetSel.appendChild(opt);
    });

    if (datasets.length > 0) {
      loadDataset(datasets[0].path);
    }
  } catch (err) {
    if (loadEl) { loadEl.classList.add("error"); loadEl.querySelector(".loader-text").textContent = `No datasets found \u2013 run: python scripts/build_dataset.py --all`; }
    console.error(err);
  }
})();

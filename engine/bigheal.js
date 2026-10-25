"use strict";
// Healing a large scan at full size, one tile at a time. The marks were found on the working copy
// (k times smaller); each is healed here on the full-size pixels, so the fill is the scan's own
// grain at its own scale and nothing is ever resampled.
//
// Marks are grouped by the cell (CELL px square) their box centre falls in. For each cell:
//   1. read the box around its marks plus MARGIN (the farthest a heal looks for a source patch);
//   2. put back what earlier cells already healed there, so those spots are clean sources;
//   3. heal this cell's marks, never copying from marks that still wait for their own cell;
//   4. keep the healed values, and write the box around this cell's marks.
// Only one cell's pixels are in memory at a time, plus the healed pixels themselves.
const R = require("./regions");
const { heal } = require("./heal");

const CELL = 2048;
const MARGIN = 192;

// mask, outside: working-size 0/1 (Wd x Hd). read(x0, y0, x1, y1) -> Promise<Uint16Array>
// (chunky, C components, full size); write(x0, y0, x1, y1, data) -> Promise.
async function healLarge(maskW, outsideW, Wd, Hd, k, W, H, C, io, hooks = {}, opts = {}) {
  const cell = opts.cell || CELL, margin = opts.margin || MARGIN;
  const tick = hooks.tick || (async () => {});
  const progress = hooks.progress || (() => {});
  const { lab, n } = R.label(maskW, Wd, Hd, 8);
  const boxes = R.findObjects(lab, Wd, Hd, n);
  // full-size box of each mark (the last block row/column also covers the odd edge pixels)
  const fx0 = new Int32Array(n), fy0 = new Int32Array(n), fx1 = new Int32Array(n), fy1 = new Int32Array(n);
  const cols = Math.ceil(W / cell);
  const groups = new Map();
  for (let i = 0; i < n; i++) {
    const b = boxes[i];
    fx0[i] = b.x0 * k; fy0[i] = b.y0 * k;
    fx1[i] = b.x1 === Wd ? W : b.x1 * k; fy1[i] = b.y1 === Hd ? H : b.y1 * k;
    const g = Math.floor((fy0[i] + fy1[i]) / 2 / cell) * cols + Math.floor((fx0[i] + fx1[i]) / 2 / cell);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(i);
  }
  const own = (i, x, y) => lab[Math.min(Hd - 1, Math.floor(y / k)) * Wd + Math.min(Wd - 1, Math.floor(x / k))] === i + 1;
  const isOut = (x, y) => outsideW && outsideW[Math.min(Hd - 1, Math.floor(y / k)) * Wd + Math.min(Wd - 1, Math.floor(x / k))];
  const healedBy = new Uint8Array(n);       // 1 once a mark's cell is done
  const store = new Array(n);               // healed full-size values over each mark's box
  const order = [...groups.keys()].sort((a, b) => a - b);
  let marksDone = 0, fallbacks = 0, cells = 0;

  for (const g of order) {
    const mine = groups.get(g);
    let x0 = W, y0 = H, x1 = 0, y1 = 0;
    for (const i of mine) { x0 = Math.min(x0, fx0[i]); y0 = Math.min(y0, fy0[i]); x1 = Math.max(x1, fx1[i]); y1 = Math.max(y1, fy1[i]); }
    const wx0 = x0, wy0 = y0, wx1 = x1, wy1 = y1;   // what this cell writes
    x0 = Math.max(0, x0 - margin); y0 = Math.max(0, y0 - margin); x1 = Math.min(W, x1 + margin); y1 = Math.min(H, y1 + margin);
    const rw = x1 - x0, rh = y1 - y0;
    const tile = await io.read(x0, y0, x1, y1);
    const tmask = new Uint8Array(rw * rh), avoid = new Uint8Array(rw * rh), tout = new Uint8Array(rw * rh);
    const mineSet = new Set(mine);
    // every mark whose box meets this region: healed ones go back in, waiting ones are avoided
    for (let i = 0; i < n; i++) {
      if (fx1[i] <= x0 || fx0[i] >= x1 || fy1[i] <= y0 || fy0[i] >= y1) continue;
      const ax0 = Math.max(x0, fx0[i]), ay0 = Math.max(y0, fy0[i]), ax1 = Math.min(x1, fx1[i]), ay1 = Math.min(y1, fy1[i]);
      const bw = fx1[i] - fx0[i];
      for (let y = ay0; y < ay1; y++) for (let x = ax0; x < ax1; x++) {
        if (!own(i, x, y)) continue;
        const t = (y - y0) * rw + (x - x0);
        if (mineSet.has(i)) tmask[t] = 1;
        else if (healedBy[i]) { const s = ((y - fy0[i]) * bw + (x - fx0[i])) * C; for (let c = 0; c < C; c++) tile[t * C + c] = store[i][s + c]; }
        else avoid[t] = 1;
      }
    }
    if (outsideW) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (isOut(x, y)) tout[(y - y0) * rw + (x - x0)] = 1;
    const r = await heal(tile, rw, rh, C, tmask, outsideW ? tout : null, null, tick, avoid);
    fallbacks += r.fallbacks;
    for (const i of mine) {
      const bw = fx1[i] - fx0[i], bh = fy1[i] - fy0[i];
      const vals = new Uint16Array(bw * bh * C);
      for (let y = 0; y < bh; y++) {
        const src = ((fy0[i] + y - y0) * rw + (fx0[i] - x0)) * C;
        vals.set(tile.subarray(src, src + bw * C), y * bw * C);
      }
      store[i] = vals;
      healedBy[i] = 1;
    }
    // the write box: this cell's marks plus whatever already-healed spots lie inside it
    const ww = wx1 - wx0, wh = wy1 - wy0;
    const out = new Uint16Array(ww * wh * C);
    for (let y = 0; y < wh; y++) {
      const src = ((wy0 + y - y0) * rw + (wx0 - x0)) * C;
      out.set(tile.subarray(src, src + ww * C), y * ww * C);
    }
    await io.write(wx0, wy0, wx1, wy1, out);
    marksDone += mine.length;
    cells++;
    progress(marksDone / Math.max(1, n));
    await tick();
  }
  return { marks: n, cells, fallbacks };
}

module.exports = { healLarge, CELL, MARGIN };

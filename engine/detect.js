"use strict";
// The detector (port of detect.py): strength maps -> pieces -> objects -> verdicts -> fill mask.
// Images: Float32Array 0..1, row-major. Masks: Uint8Array 0/1. The scan itself stays a 16-bit
// interleaved Uint16Array (H*W*C) and is only read through rgbAt().
const F = require("./filters");
const R = require("./regions");
const M = require("./maps");
const K = require("./marks");
const { pyRound } = require("./heal");
const LEARN = require("./learned");   // not "L": find() has its own L (link labels)
const DNET = require("./dustnet");
const NATR = require("./native");   // native ridge maps for big images, bit for bit the JS below

const r1 = (v) => R.pyRoundTo(v, 1);
const r2 = (v) => R.pyRoundTo(v, 2);
const r3 = (v) => R.pyRoundTo(v, 3);
// gaussianBlur's kernels at sigma 1 and 3, and filters.js ellipse half-widths for sizes 1..127 (hwPack back to back,
// hwStart[k] its start), as the native stage-3 steps take them
const GK1 = F.gaussKernel(1.0), GK3 = F.gaussKernel(3.0), GK15 = F.gaussKernel(1.5), GK08 = F.gaussKernel(0.8), GK25 = F.gaussKernel(2.5);
const HW = (() => {
  const start = new Int32Array(128).fill(-1), parts = [];
  let o = 0;
  for (let k = 1; k < 128; k++) { const t = F.ellipseHalfWidths(k); start[k] = o; parts.push(t); o += k; }
  const pack = new Int32Array(o);
  for (let k = 1, q = 0; k < 128; k++) { pack.set(parts[k - 1], q); q += k; }
  return { pack, start };
})();
const SOFT_WHY = "soft debris (out of focus: film base side or scanner glass)";

// ---------- small helpers --------------------------------------------------------------------

// any subject pixel within m px (square window) of (cy, cx)?
function nearSubject(subject, W, H, cy, cx, m) {
  if (!subject) return false;
  const y0 = Math.max(0, cy - m), y1 = Math.min(H, cy + m + 1), x0 = Math.max(0, cx - m), x1 = Math.min(W, cx + m + 1);
  for (let y = y0; y < y1; y++) { const o = y * W; for (let x = x0; x < x1; x++) if (subject[o + x]) return true; }
  return false;
}

function maxOver(src, W, m, w, h, y0, x0) {
  let mx = -Infinity;
  for (let y = 0; y < h; y++) {
    const row = (y0 + y) * W + x0, mr = y * w;
    for (let x = 0; x < w; x++) if (m[mr + x]) { const v = src[row + x]; if (v > mx) mx = v; }
  }
  return mx;
}

function touchesBorder(m, w, h) {
  for (let x = 0; x < w; x++) if (m[x] || m[(h - 1) * w + x]) return true;
  for (let y = 0; y < h; y++) if (m[y * w] || m[y * w + w - 1]) return true;
  return false;
}

function energy(g, W, H) {
  const gx = F.sobel(g, W, H, 1, 0), gy = F.sobel(g, W, H, 0, 1);
  for (let i = 0; i < gx.length; i++) gx[i] = gx[i] * gx[i] + gy[i] * gy[i];
  return gx;
}

function maxDist(m, w, h) {
  const pw = w + 2, ph = h + 2, pad = new Uint8Array(pw * ph);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pad[(y + 1) * pw + x + 1] = m[y * w + x];
  const dt = R.distanceTransform(pad, pw, ph);
  let d = 0;
  for (let i = 0; i < dt.length; i++) if (dt[i] > d) d = dt[i];
  return d;
}

// every pixel within th/2 + 0.5 px of the segment (a thick line with round ends), as detect.py's
// _draw_band: the same arithmetic in the same order, so both engines set the same pixels
function drawLine(mask, W, H, ya, xa, yb, xb, th, onSet) {
  const r = th / 2.0 + 0.5, e = Math.ceil(r);
  const y0 = Math.max(0, Math.min(ya, yb) - e), y1 = Math.min(H - 1, Math.max(ya, yb) + e);
  const x0 = Math.max(0, Math.min(xa, xb) - e), x1 = Math.min(W - 1, Math.max(xa, xb) + e);
  const dy = yb - ya, dx = xb - xa, L2 = dy * dy + dx * dx;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    let t = L2 ? ((y - ya) * dy + (x - xa) * dx) / L2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const qy = (ya + t * dy) - y, qx = (xa + t * dx) - x;
    if (qy * qy + qx * qx <= r * r) { const i = y * W + x; mask[i] = 1; if (onSet) onSet(i); }
  }
}

// ---------- hair tracing ---------------------------------------------------------------------

// Follow a thin bright ridge from `start` along unit `tangent` (dy, dx). Returns accepted points.
function traceRidge(zs, W, H, start, tangent, p, outside, maxLen) {
  const step = 6.0;
  const pts = [];
  let py = start[0], px = start[1];
  const t0y = tangent[0], t0x = tangent[1];
  let ty = t0y, tx = t0x;
  let misses = 0;
  const cosLim = Math.cos((p.trace_turn * Math.PI) / 180);
  const zval = (qy, qx) => {
    const y = pyRound(qy), x = pyRound(qx);
    return y >= 0 && y < H && x >= 0 && x < W ? zs[y * W + x] : -99.0;
  };
  let travelled = 0;
  const limit = Math.min(p.trace_max, maxLen || p.trace_max);
  while (travelled < limit) {
    const ny = -tx, nx = ty;
    let bestS = -99.0, bqy = null, bqx = null;
    for (let k = -3; k <= 3; k++) {
      const qy = py + ty * step + ny * k, qx = px + tx * step + nx * k;
      let sc = 0;
      for (const a of [0.2, 0.4, 0.6, 0.8, 1.0]) sc += zval(py + (qy - py) * a, px + (qx - px) * a);
      sc /= 5;
      if (sc > bestS) { bestS = sc; bqy = qy; bqx = qx; }
    }
    if (bqy === null) break;
    const iy = pyRound(bqy), ix = pyRound(bqx);
    if (!(iy >= 0 && iy < H && ix >= 0 && ix < W) || outside[iy * W + ix]) break;
    const sideA = (zval(bqy + ny * 4, bqx + nx * 4) + zval(bqy + ny * 5, bqx + nx * 5)) / 2;
    const sideB = (zval(bqy - ny * 4, bqx - nx * 4) + zval(bqy - ny * 5, bqx - nx * 5)) / 2;
    const ridge = bestS >= p.trace_min_z && bestS - Math.max(sideA, sideB) >= 1.0;
    if (ridge) {
      let dy = bqy - py, dx = bqx - px;
      const dn = Math.hypot(dy, dx) + 1e-9;
      dy /= dn; dx /= dn;
      let nty = 0.7 * ty + 0.3 * dy, ntx = 0.7 * tx + 0.3 * dx;
      const nn = Math.hypot(nty, ntx) + 1e-9;
      nty /= nn; ntx /= nn;
      if (nty * t0y + ntx * t0x < cosLim) break;
      ty = nty; tx = ntx;
      py = bqy; px = bqx;
      pts.push([py, px]);
      misses = 0;
    } else {
      misses++;
      if (misses >= 4) break;
      py += ty * step; px += tx * step;
    }
    travelled += step;
  }
  return pts;
}

// ---------- blob crowd (determinant of the Hessian) ------------------------------------------

// Scale-normalised bright-blob DoH response at half-octave scales, cached per image and polarity.
// Only the local maxima are kept for the whole image; a blob's own response is recomputed in a
// small window around it.
class BlobField {
  constructor(Y, W, H, sigLevel, sgn) {
    Object.assign(this, { Y, W, H, sigLevel, sgn });
    this.tiles = new Map();
    this.T = 512;
  }
  static scale(s) { return pyRound(2 * Math.log2(Math.min(Math.max(s, 1.0), 16.0))) / 2; }
  static get KEEP() { return 96; }
  response(src, w, h, sl, sq) {
    const g = F.gaussianBlur(src, w, h, sq, null, true);
    const Lxx = F.sobel(g, w, h, 2, 0), Lyy = F.sobel(g, w, h, 0, 2), Lxy = F.sobel(g, w, h, 1, 1);
    const resp = new Float32Array(w * h), s2 = sq * sq, sgn = this.sgn;
    for (let i = 0; i < resp.length; i++) {
      const a = sgn * Lxx[i], b = sgn * Lyy[i];
      if (a < 0 && b < 0) {
        const det = a * b - Lxy[i] * Lxy[i];
        if (det > 0) resp[i] = (Math.sqrt(det) * s2) / Math.max(sl[i], 1e-4);
      }
    }
    return resp;
  }
  static kOf(sq) { return Math.trunc(2 * pyRound(1.5 * sq) + 1); }
  static margin(sq) { return ((Math.round(sq * 8 + 1) | 1) >> 1) + 2 + (BlobField.kOf(sq) >> 1); }
  // local maxima (response >= 1) inside one tile, computed on the tile plus a margin
  tile(q, ty, tx) {
    const key = q + ":" + ty + ":" + tx;
    let t = this.tiles.get(key);
    if (t) return t;
    const t0 = Date.now();
    const { Y, W, H, sigLevel, T } = this;
    const sq = 2 ** q, k = BlobField.kOf(sq), mg = BlobField.margin(sq);
    const y0 = ty * T, x0 = tx * T, y1 = Math.min(H, y0 + T), x1 = Math.min(W, x0 + T);
    const ya = Math.max(0, y0 - mg), yb = Math.min(H, y1 + mg), xa = Math.max(0, x0 - mg), xb = Math.min(W, x1 + mg);
    const w = xb - xa, h = yb - ya;
    const resp = this.response(K.crop(Y, W, ya, yb, xa, xb), w, h, K.crop(sigLevel, W, ya, yb, xa, xb), sq);
    const dil = F.rectMorph(resp, w, h, k, true);
    const ys = [], xs = [], vals = [];
    for (let y = y0; y < y1; y++) {
      const o = (y - ya) * w - xa;
      for (let x = x0; x < x1; x++) { const v = resp[o + x]; if (v >= 1.0 && v === dil[o + x]) { ys.push(y); xs.push(x); vals.push(v); } }
    }
    const inner = new Float32Array((y1 - y0) * (x1 - x0));
    for (let y = y0; y < y1; y++) inner.set(resp.subarray((y - ya) * w + (x0 - xa), (y - ya) * w + (x1 - xa)), (y - y0) * (x1 - x0));
    t = { ys, xs, vals, resp: inner, y0, x0, w: x1 - x0 };
    this.tiles.set(key, t);
    this.keepResp(key);
    this.tileMs = (this.tileMs || 0) + (Date.now() - t0);
    return t;
  }
  // the tiles at scale q in list (ty, tx pairs) not made yet, or whose numbers were let go when needResp, made at once
  // natively in the order tile() would make them (same tiles, same keep order); tile() then finds them. Tiles too
  // small for the native filters (image corners) stay with tile().
  fetch(q, list, needResp) {
    const { W, H, T } = this;
    if (!NATR.available() || !(this.Y instanceof Float32Array) || !(this.sigLevel instanceof Float32Array)) return;
    const sq = 2 ** q, k = BlobField.kOf(sq), mg = BlobField.margin(sq);
    const want = [], keys = [], seen = new Set();
    for (let i = 0; i < list.length; i += 2) {
      const ty = list[i], tx = list[i + 1], key = q + ":" + ty + ":" + tx;
      if (seen.has(key)) continue;
      seen.add(key);
      const t = this.tiles.get(key);
      if (t && (t.resp || !needResp)) continue;
      const y0 = ty * T, x0 = tx * T, y1 = Math.min(H, y0 + T), x1 = Math.min(W, x0 + T);
      if (!NATR.big(Math.min(W, x1 + mg) - Math.max(0, x0 - mg), Math.min(H, y1 + mg) - Math.max(0, y0 - mg))) continue;
      want.push(ty, tx);
      keys.push(key);
    }
    if (!want.length) return;
    const t0 = Date.now();
    const res = NATR.blobTiles(this.Y, this.sigLevel, W, H, T, sq, this.sgn, F.gaussKernel(sq), k, mg, want);
    if (res) res.forEach((t, i) => {
      this.tiles.delete(keys[i]);
      this.tiles.set(keys[i], t);
      this.keepResp(keys[i]);
    });
    this.tileMs = (this.tileMs || 0) + (Date.now() - t0);
  }
  keepResp(key) {
    const lru = this.lru || (this.lru = []);
    const i = lru.indexOf(key);
    if (i >= 0) lru.splice(i, 1);
    lru.push(key);
    // ~1 MB per tile kept: a raster sweep over several blob scales needs about a row of tiles each
    while (lru.length > BlobField.KEEP) { const old = this.tiles.get(lru.shift()); if (old) old.resp = null; }
  }
  // the response at one pixel, from its tile (recomputed if that tile's numbers were let go)
  respAt(q, gy, gx) {
    const T = this.T, ty = Math.floor(gy / T), tx = Math.floor(gx / T), key = q + ":" + ty + ":" + tx;
    let t = this.tile(q, ty, tx);
    if (!t.resp) { this.tiles.delete(key); t = this.tile(q, ty, tx); } else this.keepResp(key);
    return t.resp[(gy - t.y0) * t.w + (gx - t.x0)];
  }
  // own response inside a region (window y0, x0, ww x hh): recomputed locally with a margin
  own(region, y0, x0, ww, hh, sq) {
    const { Y, W, H, sigLevel } = this;
    const mg = BlobField.margin(sq);
    const ya = Math.max(0, y0 - mg), yb = Math.min(H, y0 + hh + mg), xa = Math.max(0, x0 - mg), xb = Math.min(W, x0 + ww + mg);
    const w = xb - xa, h = yb - ya;
    const resp = this.response(K.crop(Y, W, ya, yb, xa, xb), w, h, K.crop(sigLevel, W, ya, yb, xa, xb), sq);
    let own = -Infinity;
    for (let y = 0; y < hh; y++) for (let x = 0; x < ww; x++) {
      if (!region[y * ww + x]) continue;
      const v = resp[(y0 + y - ya) * w + (x0 + x - xa)];
      if (v > own) own = v;
    }
    return own;
  }
  // [similar, stronger, row]: same-size blobs within rng px (Chebyshev), excluding the blob itself
  // and anything already healed; row = two near-identical blobs close by on opposite sides
  // (stitches, beads, sequins)
  crowd(region, y0, x0, ww, hh, healed, s, rng = 300) {
    const q = BlobField.scale(s), sq = 2 ** q, k = BlobField.kOf(sq);
    let n = 0, sy = 0, sx = 0;
    for (let y = 0; y < hh; y++) for (let x = 0; x < ww; x++) if (region[y * ww + x]) { n++; sy += y; sx += x; }
    if (!n) return [0, 0, false];
    const T = this.T;
    {
      // the tiles the region's own response is read from, in the order respAt first reaches them
      const list = [], seen = new Set();
      for (let y = 0; y < hh; y++) for (let x = 0; x < ww; x++) {
        if (!region[y * ww + x]) continue;
        const ty = Math.floor((y0 + y) / T), tx = Math.floor((x0 + x) / T), key = ty * 65536 + tx;
        if (!seen.has(key)) { seen.add(key); list.push(ty, tx); }
      }
      this.fetch(q, list, true);
    }
    let own = -Infinity;
    for (let y = 0; y < hh; y++) for (let x = 0; x < ww; x++) {
      if (!region[y * ww + x]) continue;
      const v = this.respAt(q, y0 + y, x0 + x);
      if (v > own) own = v;
    }
    if (!(own > 0)) return [0, 0, false];
    const cy = Math.trunc(sy / n) + y0, cx = Math.trunc(sx / n) + x0;
    const regD = F.dilateMask(region, ww, hh, k);
    {
      const list = [];
      for (let ty = Math.max(0, Math.floor((cy - rng) / T)); ty <= Math.floor(Math.min(this.H - 1, cy + rng) / T); ty++)
        for (let tx = Math.max(0, Math.floor((cx - rng) / T)); tx <= Math.floor(Math.min(this.W - 1, cx + rng) / T); tx++) list.push(ty, tx);
      this.fetch(q, list, false);
    }
    const d_ = 2.0 * Math.sqrt(n / Math.PI), dLo = Math.max(6.0, 0.6 * d_), dHi = Math.max(24.0, 2.5 * d_);
    let sim = 0, stronger = 0;
    const uy = [], ux = [];
    for (let ty = Math.max(0, Math.floor((cy - rng) / T)); ty <= Math.floor(Math.min(this.H - 1, cy + rng) / T); ty++) {
      for (let tx = Math.max(0, Math.floor((cx - rng) / T)); tx <= Math.floor(Math.min(this.W - 1, cx + rng) / T); tx++) {
        const t = this.tile(q, ty, tx);
        for (let j = 0; j < t.ys.length; j++) {
          const py = t.ys[j], px = t.xs[j];
          if (Math.abs(py - cy) > rng || Math.abs(px - cx) > rng) continue;
          const ry = py - y0, rx = px - x0;
          if (ry >= 0 && ry < hh && rx >= 0 && rx < ww && regD[ry * ww + rx]) continue;
          if (healed(py * this.W + px)) continue;
          const v = t.vals[j];
          if (v >= 0.5 * own && v <= 2.0 * own) sim++;
          if (v >= own) stronger++;
          const dd = Math.hypot(py - cy, px - cx);
          if (v >= 0.6 * own && v <= 1.6 * own && dd >= dLo && dd <= dHi) { uy.push((py - cy) / dd); ux.push((px - cx) / dd); }
        }
      }
    }
    let row = false;
    for (let i = 0; i < uy.length && !row; i++) for (let j = i + 1; j < uy.length; j++) if (uy[i] * uy[j] + ux[i] * ux[j] <= -0.9) { row = true; break; }
    return [sim, stronger, row];
  }
}

// in the film's edge print (maps.js imageBox: outside the box past the print, inside the box as it was): never healed
function printBand(box, x, y) {
  const o = box && box.outer;
  if (!o) return false;
  const inNew = box[0] <= y && y <= box[1] && box[2] <= x && x <= box[3];
  const inOld = o[0] <= y && y <= o[1] && o[2] <= x && x <= o[3];
  return !inNew && inOld;
}

// small scans: "" or why a heal must stay (maps.js res_guard). kind: "disc", "hair", "line" (a soft fibre) or anything
// else (dust); area 0 skips the size test
function resGuard(p, W, H, x0, y0, x1, y1, area, kind, plain = false) {
  if (!p.res_guard) return "";
  const mp = (W * H) / 1e6;
  if (Math.max(mp, p.res_scan_mp || 0) >= p.res_full_mp) return "";
  const k2 = mp / p.res_ref_mp;
  const cap = (kind === "disc" ? p.res_cap_disc : kind === "hair" ? p.res_cap_hair : kind === "line" ? p.res_cap_line : p.res_cap_area) * (plain === true ? p.res_plain_mult : plain || 1);
  if (area > cap * k2) return "too big for dust at this scan's size";
  if (Math.min(x0, y0, W - 1 - x1, H - 1 - y1) < p.res_edge_px) return "at the edge of a small scan (the frame line)";
  return "";
}

// how much picture detail surrounds a box (marks.js detail: gradient energy at sigma 2.5 against the grain, median over
// a ring 4-40 px out), for a soft heal the size guard would refuse: plain ground earns it the plain-ground allowance
function detailAround(ctx, sigLevel, x0, y0, x1, y1) {
  const { W, H, E25, gradCal } = ctx;
  const e = [], s = [];
  for (let y = Math.max(0, y0 - 40); y < Math.min(H, y1 + 41); y += 2) for (let x = Math.max(0, x0 - 40); x < Math.min(W, x1 + 41); x += 2) {
    if (x >= x0 - 4 && x <= x1 + 4 && y >= y0 - 4 && y <= y1 + 4) continue;
    e.push(E25[y * W + x]); s.push(sigLevel[y * W + x]);
  }
  if (e.length < 20) return Infinity;
  return R.median(e) / (gradCal * R.median(s) ** 2);
}

// a soft heal at a small scan's size: past the size limit only on plain ground (twice the limit), or very plain ground
// (sky, backdrop: res_soft_plain_mult times), where a big soft blob is dust far more often than picture
function softGuard(p, ctx, sigLevel, W, H, r, n) {
  const gw = resGuard(p, W, H, r.x0, r.y0, r.x1, r.y1, n, "dust");
  if (!gw.startsWith("too big")) return gw;
  const d = detailAround(ctx, sigLevel, r.x0, r.y0, r.x1, r.y1);
  return d <= p.res_soft_plain_detail ? resGuard(p, W, H, r.x0, r.y0, r.x1, r.y1, n, "dust", p.res_soft_plain_mult)
    : d <= p.res_plain_detail ? resGuard(p, W, H, r.x0, r.y0, r.x1, r.y1, n, "dust", true) : gw;
}

// ---------- the sharp pass for one polarity --------------------------------------------------

async function find(ctx, bright) {
  const { Y, W, H, G, outside, p, gradCal, box, cache, rgbAt, C } = ctx;
  const N = W * H;
  const tick = (f) => ctx.tick(f);
  const tm = { t: Date.now(), parts: [] };
  const lap = (name) => { const now = Date.now(); tm.parts.push(name + " " + ((now - tm.t) / 1000).toFixed(1)); tm.t = now; };
  const maps = await M.residualMaps(Y, W, H, bright, outside, cache, (f) => tick(0.6 * f));
  // the last polarity's maps are made: the grain maps both shared (medians, residual, noise) are done with, so they
  // go before its judging (the maps a result keeps are its own references)
  if (ctx.lastMaps) for (const key of Object.keys(cache)) delete cache[key];
  const zs = maps.z_s, zl = maps.z_l, zx = maps.z_xl, sigLevel = maps.sigma_level;
  const z = new Float32Array(N), seeds = new Uint8Array(N);
  {
    // the parameters read once: per pixel they cost five times the loop
    const sd = p.seed, soft = !!p.soft_dust, s3 = p.seed + 3, s5 = p.seed + 5;
    for (let i = 0; i < N; i++) {
      const a = zs[i], b = zl[i], c = zx[i] * 0.85;
      z[i] = a > b ? (a > c ? a : c) : b > c ? b : c;
      seeds[i] = a >= sd || (soft && (b >= s3 || zx[i] >= s5)) ? 1 : 0;
    }
  }
  const reach = F.dilateMask(seeds, W, H, 2 * p.grow_px + 1, true);
  { const gr = p.grow; for (let i = 0; i < N; i++) reach[i] = reach[i] && z[i] >= gr ? 1 : 0; }
  const { lab, n } = R.label(reach, W, H, 8);
  const seeded = new Uint8Array(n + 1);
  for (let i = 0; i < N; i++) if (seeds[i] && lab[i]) seeded[lab[i]] = 1;
  const objs = R.findObjects(lab, W, H, n);

  // 1. pieces: filled masks; drop noise and soft photo blobs
  const pieces = new Map();
  const fineSet = new Set();
  for (let i = 1; i <= n; i++) {
    if (!seeded[i]) continue;
    const b = objs[i - 1];
    const y0 = Math.max(0, b.y0 - 8), y1 = Math.min(H, b.y1 + 8), x0 = Math.max(0, b.x0 - 8), x1 = Math.min(W, b.x1 + 8);
    const w = x1 - x0, h = y1 - y0;
    let m = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) { const row = (y0 + y) * W + x0; for (let x = 0; x < w; x++) m[y * w + x] = lab[row + x] === i ? 1 : 0; }
    m = R.fillHoles(m, w, h);
    let area = K.count(m);
    let zf = maxOver(zs, W, m, w, h, y0, x0);
    if (area < p.min_area || area > p.max_area) continue;
    if (area < p.tiny_area && zf < p.tiny_seed) continue;
    if (area >= 40 && M.edgeWidth(Y, G, W, m, w, h, y0, x0) > p.max_edge_width) {
      // the large scales can spread a crisp mark into the soft highlight under it: if its
      // fine-scale core alone is crisp, keep just that core
      if (!p.edge_fine) continue;
      let mf = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const k = y * w + x;
        mf[k] = m[k] && zs[(y0 + y) * W + x0 + x] >= p.grow ? 1 : 0;
      }
      mf = R.fillHoles(mf, w, h);
      const af = K.count(mf);
      if (af < p.min_area || af > 0.6 * area || (af < p.tiny_area && zf < p.tiny_seed)) continue;
      if (af >= 40 && M.edgeWidth(Y, G, W, mf, w, h, y0, x0) > p.max_edge_width) continue;
      m = mf; area = af; zf = maxOver(zs, W, m, w, h, y0, x0);
      fineSet.add(i);
    }
    pieces.set(i, { y0, x0, w, h, m, area, zf, zmax: maxOver(z, W, m, w, h, y0, x0) });
  }
  const cand = new Uint8Array(N);
  {
    const keepArr = new Uint8Array(n + 1);
    for (const k of pieces.keys()) keepArr[k] = 1;
    for (let i = 0; i < N; i++) cand[i] = keepArr[lab[i]];
  }
  for (const i of fineSet) {
    const pc = pieces.get(i);
    for (let y = 0; y < pc.h; y++) for (let x = 0; x < pc.w; x++) {
      const gi = (pc.y0 + y) * W + pc.x0 + x;
      if (lab[gi] === i) cand[gi] = 0;
      if (pc.m[y * pc.w + x]) cand[gi] = 1;
    }
  }

  lap("maps+pieces");
  await tick(0.52);
  // 1b. opaque flakes: flat inside (no grain), found at the large scales
  {
    const sigma = maps.sigma;
    const g2 = ctx.g2 || (ctx.g2 = F.gaussianBlur(Y, W, H, 2.0));   // shared by both polarities
    const big = new Uint8Array(N);
    { const s2 = p.seed + 2, s4 = p.seed + 4; for (let i = 0; i < N; i++) big[i] = zl[i] >= s2 || zx[i] >= s4 ? 1 : 0; }
    const B = R.label(big, W, H, 8);
    const bobjs = R.findObjects(B.lab, W, H, B.n);
    const sgn = bright ? 1 : -1;
    for (let j = 1; j <= B.n; j++) {
      const b = bobjs[j - 1];
      if ((b.y1 - b.y0) * (b.x1 - b.x0) < 60) continue;          // filled area can't reach 60
      const y0 = Math.max(0, b.y0 - 10), y1 = Math.min(H, b.y1 + 10), x0 = Math.max(0, b.x0 - 10), x1 = Math.min(W, b.x1 + 10);
      const w = x1 - x0, h = y1 - y0;
      let m = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) { const row = (y0 + y) * W + x0; for (let x = 0; x < w; x++) m[y * w + x] = B.lab[row + x] === j ? 1 : 0; }
      m = R.fillHoles(m, w, h);
      const area = K.count(m);
      if (area < 60 || area > p.max_area) continue;
      let out = false;
      for (let y = 0; y < h && !out; y++) for (let x = 0; x < w; x++) if (m[y * w + x] && outside[(y0 + y) * W + x0 + x]) { out = true; break; }
      if (out) continue;
      const inner = F.erodeMask(m, w, h, 7);
      if (K.count(inner) < 20) continue;
      const ring = K.andNot(F.dilateMask(m, w, h, 17), F.dilateMask(m, w, h, 7));
      const hp = [], sg = [], yi = [], yr = [];
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const k = y * w + x, gi = (y0 + y) * W + x0 + x;
        if (inner[k]) { hp.push(Y[gi] - g2[gi]); sg.push(sigma[gi]); yi.push(Y[gi]); }
        if (ring[k]) yr.push(Y[gi]);
      }
      const flat = R.std(hp) / Math.max(R.median(sg), 1e-5);
      const contrast = sgn * (R.median(yi) - R.median(yr));
      if (flat > p.blob_flat || contrast < p.blob_contrast) continue;
      if (M.edgeWidth(Y, G, W, m, w, h, y0, x0) > p.blob_edge) continue;
      pieces.set(n + j, { y0, x0, w, h, m, area, zf: maxOver(zs, W, m, w, h, y0, x0), zmax: maxOver(z, W, m, w, h, y0, x0) });
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (m[y * w + x]) cand[(y0 + y) * W + x0 + x] = 1;
    }
  }

  lap("flakes");
  await tick(0.56);
  // 2. objects: pieces within 2*link_px of each other are one mark
  const L = R.label(F.dilateMask(cand, W, H, 2 * p.link_px + 1, true), W, H, 4);
  const glab = L.lab;
  const ng = L.n;
  let groups = new Map();
  for (const [i, pc] of pieces) {
    let first = -1;
    for (let k = 0; k < pc.m.length; k++) if (pc.m[k]) { first = k; break; }
    const gi = (pc.y0 + Math.floor(first / pc.w)) * W + pc.x0 + (first % pc.w);
    const g = glab[gi];
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(i);
  }
  const groupMask = (members) => {
    let gy0 = Infinity, gx0 = Infinity, gy1 = -Infinity, gx1 = -Infinity;
    for (const i of members) {
      const pc = pieces.get(i);
      gy0 = Math.min(gy0, pc.y0); gx0 = Math.min(gx0, pc.x0);
      gy1 = Math.max(gy1, pc.y0 + pc.h); gx1 = Math.max(gx1, pc.x0 + pc.w);
    }
    const gw = gx1 - gx0, gh = gy1 - gy0, gm = new Uint8Array(gw * gh);
    for (const i of members) {
      const pc = pieces.get(i);
      for (let y = 0; y < pc.h; y++) for (let x = 0; x < pc.w; x++) if (pc.m[y * pc.w + x]) gm[(pc.y0 - gy0 + y) * gw + pc.x0 - gx0 + x] = 1;
    }
    return { gy0, gx0, gw, gh, gm };
  };
  const groupFeats = (g, members) => {
    const { gy0, gx0, gw, gh, gm } = groupMask(members);
    let ty0 = Infinity, ty1 = -1, tx0 = Infinity, tx1 = -1, sy = 0, sx = 0, area = 0;
    for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
      if (!gm[y * gw + x]) continue;
      if (y < ty0) ty0 = y; if (y > ty1) ty1 = y; if (x < tx0) tx0 = x; if (x > tx1) tx1 = x;
      sy += y; sx += x; area++;
    }
    ty1++; tx1++;
    let zmax = -Infinity, zfine = -Infinity;
    for (const i of members) { const pc = pieces.get(i); zmax = Math.max(zmax, pc.zmax); zfine = Math.max(zfine, pc.zf); }
    const f = {
      id: g, x0: gx0 + tx0, y0: gy0 + ty0, x1: gx0 + tx1, y1: gy0 + ty1,
      cx: pyRound(sx / area) + gx0, cy: pyRound(sy / area) + gy0,
      area, width: r1(2 * maxDist(gm, gw, gh)), length: Math.max(ty1 - ty0, tx1 - tx0),
      z: r1(zmax), z_fine: r1(zfine), pieces: members.length,
    };
    f.kind = M.kindOf(f.area, f.length, f.width);
    if (members.some((i) => fineSet.has(i))) f.fine_core = true;
    return f;
  };
  let feats = new Map();
  for (const [g, mem] of groups) feats.set(g, groupFeats(g, mem));

  // 2b. hair chaining: thin elongated pieces whose ends come within chain_px join up
  const elong = (f) => f.length >= 8 && f.length >= 2.5 * Math.max(f.width, 1.0);
  const parent = new Map();
  for (const g of feats.keys()) parent.set(g, g);
  const root = (a) => {
    while (parent.get(a) !== a) { parent.set(a, parent.get(parent.get(a))); a = parent.get(a); }
    return a;
  };
  const ptsOf = (members) => {
    const ys = [], xs = [];
    for (const i of members) {
      const pc = pieces.get(i);
      for (let y = 0; y < pc.h; y++) for (let x = 0; x < pc.w; x++) if (pc.m[y * pc.w + x]) { ys.push(pc.y0 + y); xs.push(pc.x0 + x); }
    }
    return { ys, xs };
  };
  const links = [], bridges = [];
  const cp = p.chain_px;
  for (const [g, f] of feats) {
    if (!elong(f)) continue;
    const { gy0, gx0, gw, gh, gm } = groupMask(groups.get(g));
    const ya = Math.max(0, gy0 - cp - 2), xa = Math.max(0, gx0 - cp - 2);
    const yb = Math.min(H, gy0 + gh + cp + 2), xb = Math.min(W, gx0 + gw + cp + 2);
    const lw = xb - xa, lh = yb - ya, local = new Uint8Array(lw * lh);
    for (let y = 0; y < Math.min(gh, yb - gy0); y++) for (let x = 0; x < Math.min(gw, xb - gx0); x++) local[(gy0 - ya + y) * lw + gx0 - xa + x] = gm[y * gw + x];
    const near = F.dilateMask(local, lw, lh, 2 * cp + 1, true);
    const others = new Set();
    for (let y = 0; y < lh; y++) for (let x = 0; x < lw; x++) {
      const gi = (ya + y) * W + xa + x;
      if (near[y * lw + x] && cand[gi]) others.add(glab[gi]);
    }
    for (const o of [...others].sort((a, b) => a - b)) {
      if (o === g || !feats.has(o) || !elong(feats.get(o)) || root(g) === root(o)) continue;
      // a real hair is still faintly there across the gap; glints in a row are not
      // the closest pair: o was found inside g's chain_px neighbourhood, so it lies within
      // reach of some point of g; points of g farther than that from all of o cannot win
      const A = ptsOf(groups.get(g)), Bp = ptsOf(groups.get(o));
      const grid = new R.PointGrid(Bp.ys, Bp.xs, 16);
      const reachPx = cp + 2;
      let best = Infinity, ka = -1, kb = -1;
      for (let a = 0; a < A.ys.length; a++) {
        const ay = A.ys[a], ax = A.xs[a];
        for (const j of grid.queryBox(ay, ax, reachPx)) {
          const dy = Bp.ys[j] - ay, dx = Bp.xs[j] - ax, d = Math.sqrt(dy * dy + dx * dx);
          if (d < best || (d === best && a === ka && j < kb)) { best = d; ka = a; kb = j; }
        }
      }
      if (ka < 0) continue;
      const ya_ = A.ys[ka], xa_ = A.xs[ka], yb_ = Bp.ys[kb], xb_ = Bp.xs[kb];
      const nstep = Math.max(Math.abs(yb_ - ya_), Math.abs(xb_ - xa_)) + 1;
      if (nstep > 3) {
        let s = 0, c = 0;
        for (let t = 1; t < nstep - 1; t++) {
          const ly = pyRound(ya_ + ((yb_ - ya_) * t) / (nstep - 1)), lx = pyRound(xa_ + ((xb_ - xa_) * t) / (nstep - 1));
          s += zs[ly * W + lx]; c++;
        }
        if (s / c < 1.5) continue;
      }
      parent.set(root(o), root(g));
      links.push([g, o]);
      const th = Math.trunc(Math.max(2, pyRound(Math.min(Math.max(f.width, feats.get(o).width), 6))));
      bridges.push([g, ya_, xa_, yb_, xb_, th]);
    }
  }
  if (links.length) {
    const merged = new Map();
    for (const g of feats.keys()) {
      const r = root(g);
      if (!merged.has(r)) merged.set(r, []);
      merged.get(r).push(...groups.get(g));
    }
    const lut = new Int32Array(ng + 1);
    for (let i = 0; i <= ng; i++) lut[i] = i;
    for (const g of feats.keys()) lut[g] = root(g);
    for (let i = 0; i < N; i++) glab[i] = lut[glab[i]];
    groups = merged;
    feats = new Map();
    for (const [g, mem] of groups) feats.set(g, groupFeats(g, mem));
    for (const b of bridges) b[0] = root(b[0]);
  }

  const smallScanRules = p.res_guard && Math.max((W * H) / 1e6, p.res_scan_mp || 0) < p.res_full_mp;
  // 3. colour + crispness at each object's brightest point; solid fill; line features; cues
  const [bt, bb, bl, br] = box;
  const zm = p.zone_margin;
  const { G3, E25, E1 } = ctx;
  lap("objects");
  const list = [...feats.entries()], nO = list.length;
  const boxes = new Int32Array(4 * nO), offs = new Float64Array(nO + 1);
  {
    const pad = 45;
    let tot = 0;
    for (let i = 0; i < nO; i++) {
      const f = list[i][1];
      const wy0 = Math.max(0, f.y0 - pad), wy1 = Math.min(H, f.y1 + pad), wx0 = Math.max(0, f.x0 - pad), wx1 = Math.min(W, f.x1 + pad);
      boxes[4 * i] = wy0; boxes[4 * i + 1] = wy1; boxes[4 * i + 2] = wx0; boxes[4 * i + 3] = wx1;
      offs[i] = tot; tot += (wx1 - wx0) * (wy1 - wy0);
    }
    offs[nO] = tot;
  }
  const dk = p.dilate > 0 ? 2 * p.dilate + 1 : 0;
  // each object's solid fill: its raw candidate pixels are its own (no other object's claim reaches them), so every
  // fill can be made before any is claimed; the native module makes them all at once
  const nat = NATR.available() && nO > 0 && p.native_stage3 !== false ? NATR.solidBatch(Y, sigLevel, glab, cand, W, H, nO, boxes, list.map(([g]) => g), bright, GK1, HW, dk, offs) : null;
  const fillOf = async (i, tickAt) => {
    if (nat) return { how: nat.how[i], solid: nat.solid.subarray(offs[i], offs[i + 1]), objd: nat.objd.subarray(offs[i], offs[i + 1]), area: nat.area[i], width: nat.width[i] };
    if (tickAt && (i + 1) % 200 === 0) await tick(tickAt + 0.1 * (i / nO));
    const g = list[i][0], wy0 = boxes[4 * i], wy1 = boxes[4 * i + 1], wx0 = boxes[4 * i + 2], wx1 = boxes[4 * i + 3], ww = wx1 - wx0, wh = wy1 - wy0;
    const raw = new Uint8Array(ww * wh);
    for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) {
      const gi = (wy0 + y) * W + wx0 + x;
      raw[y * ww + x] = glab[gi] === g && cand[gi] ? 1 : 0;
    }
    const sol = K.solidify(K.crop(Y, W, wy0, wy1, wx0, wx1), raw, ww, wh, bright, K.crop(sigLevel, W, wy0, wy1, wx0, wx1));
    const how = sol.how === "grown" ? 1 : sol.how === "filled" ? 2 : 0;
    return { how, solid: sol.mask, objd: dk ? F.dilateMask(sol.mask, ww, wh, dk, true) : sol.mask, area: how ? K.count(sol.mask) : 0, width: how ? 2 * maxDist(sol.mask, ww, wh) : 0 };
  };
  // the claims of fills in object order: a pixel no candidate holds goes to the first fill that covers it
  const claims = [];
  const claimAll = (fills) => {
    for (let i = 0; i < nO; i++) {
      const fl = fills[i];
      if (!fl.how) continue;
      const g = list[i][0], wy0 = boxes[4 * i], wx0 = boxes[4 * i + 2], ww = boxes[4 * i + 3] - wx0, wh = boxes[4 * i + 1] - wy0;
      for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) {
        if (!fl.solid[y * ww + x]) continue;
        const gi = (wy0 + y) * W + wx0 + x;
        if (!cand[gi] && glab[gi] === 0) { cand[gi] = 1; glab[gi] = g; claims.push(gi, i); }
      }
    }
  };
  // auto polarity, dark after bright: dark is used only with ctx.darkNeed strong isolated marks (defects of z 20+ with
  // at most 2 look-alikes around). z and the areas are final once the fills are claimed, so the look-alikes can be
  // counted as the verdicts count them: the marks of z 20+ with at most 2 bound that count. Below the need dark can't
  // win and is judged no further (all the bright result still takes from it, cand for the mixed-polarity partners, is
  // complete): the result is the same as judging it all.
  const darkCantWin = (fills) => {
    const Zs = list.map(([, f]) => f.z), As = list.map(([, f], i) => (fills[i].how ? fills[i].area : f.area));
    const grid = new R.PointGrid(list.map(([, f]) => f.cy), list.map(([, f]) => f.cx), 64);
    let bound = 0;
    for (let i = 0; i < nO; i++) {
      const f = list[i][1];
      if (!(f.z >= 20)) continue;
      const nb = grid.queryBox(f.cy, f.cx, p.field_radius).filter((j) => j !== i && Zs[j] >= p.field_level * f.z && As[j] >= 4);
      if (nb.length <= 2) bound++;
    }
    return bound < ctx.darkNeed ? bound : -1;
  };
  const stopDark = (bound) => {
    lap("solid fill");
    if (ctx.timing) ctx.log(`[dark] seconds: ${tm.parts.join(" · ")}  (maps: ${(maps.times || []).join(" · ")})`);
    ctx.log(`[dark] pieces ${pieces.size}  objects ${feats.size}  isolated marks of z 20+ ${bound}: can't outvote bright (needs ${ctx.darkNeed}), judged no further`);
    return { mask: new Uint8Array(N), marks: [], n: 0, strong: bound, cand, maps, stopped: true };
  };
  if (nat) {
    // native: the peak looks, the fills claimed, line features, then every object's cues at once (each seeing cand as
    // the loop below would have: after the claims of the objects up to it)
    for (let i = 0; i < nO; i++) {
      if ((i + 1) % 200 === 0) await tick(0.6 + 0.1 * (i / nO));
      const [g, f] = list[i];
      const look = K.peakLook(rgbAt, C, Y, W, H, (y, x) => glab[y * W + x] === g, f, bright);
      f.white = look[0]; f.sharp = look[1]; f.bg = look[2];
      f.zone = !!(p.zone_guard && !(bt + zm <= f.cy && f.cy <= bb - zm && bl + zm <= f.cx && f.cx <= br - zm));
      f.print = printBand(box, f.cx, f.cy);
      f.frame = M.beyondFrame(box.frame, W, H, f.cx, f.cy, p.frame_margin);
    }
    const fills = [];
    for (let i = 0; i < nO; i++) fills.push(await fillOf(i));
    claimAll(fills);
    if (!bright && ctx.darkNeed) { const b = darkCantWin(fills); if (b >= 0) return stopDark(b); }
    for (let i = 0; i < nO; i++) {
      const f = list[i][1], fl = fills[i];
      if (fl.how) { f.solid = fl.how === 1 ? "grown" : "filled"; f.area = fl.area; f.width = r1(fl.width); f.kind = M.kindOf(f.area, f.length, f.width); }
      if (p.line_guard && f.length >= 20 && (elong(f) || f.kind === "hair/fibre")) {
        const wy0 = boxes[4 * i], wx0 = boxes[4 * i + 2], ww = boxes[4 * i + 3] - wx0, objd = fl.objd;
        const ly0 = Math.max(0, f.y0 - 20) - wy0, lx0 = Math.max(0, f.x0 - 20) - wx0;
        const ly1 = Math.min(H, f.y1 + 20) - wy0, lx1 = Math.min(W, f.x1 + 20) - wx0;
        const lw = lx1 - lx0, lh = ly1 - ly0, lm = new Uint8Array(lw * lh);
        for (let y = 0; y < lh; y++) for (let x = 0; x < lw; x++) lm[y * lw + x] = objd[(ly0 + y) * ww + lx0 + x];
        f.line = K.lineFeatures(K.crop(Y, W, wy0 + ly0, wy0 + ly1, wx0 + lx0, wx0 + lx1), lm, lw, lh, bright);
      }
    }
    // the claims sorted by pixel, for each object's view of cand
    const nc = claims.length >> 1, order = new Int32Array(nc);
    for (let k = 0; k < nc; k++) order[k] = k;
    order.sort((a, b) => claims[2 * a] - claims[2 * b]);
    const cgi = new Int32Array(nc), cidx = new Int32Array(nc);
    for (let k = 0; k < nc; k++) { cgi[k] = claims[2 * order[k]]; cidx[k] = claims[2 * order[k] + 1]; }
    const cu = NATR.cuesBatch(Y, G3, E25, E1, sigLevel, cand, cgi, cidx, W, H, nO, boxes, offs, nat.objd, bright, gradCal, GK1, GK3, HW);
    for (let i = 0; i < nO; i++) {
      const f = list[i][1], fl = fills[i];
      if (!cu) {
        // the native cues failed (native is off from now on): the same cues in JS, cand as the loop saw it
        const wy0 = boxes[4 * i], wy1 = boxes[4 * i + 1], wx0 = boxes[4 * i + 2], wx1 = boxes[4 * i + 3], ww = wx1 - wx0, wh = wy1 - wy0;
        if (K.count(fl.objd) < 4) continue;
        const allc = K.cropMask(cand, W, wy0, wy1, wx0, wx1);
        for (let k = 0; k < nc; k++) if (cidx[k] > i) { const gy = (cgi[k] / W) | 0, gx = cgi[k] - gy * W; if (gy >= wy0 && gy < wy1 && gx >= wx0 && gx < wx1) allc[(gy - wy0) * ww + gx - wx0] = 0; }
        Object.assign(f, K.markCues(Y, W, G3, E25, E1, sigLevel, cand, fl.objd, ww, wh, wy0, wx0, bright, gradCal, allc));
        continue;
      }
      const o = 10 * i;
      if (Number.isNaN(cu[o + 9])) continue;
      f.crisp = r2(cu[o]); f.crisp_rel = r2(cu[o + 1]); f.core_y = r3(cu[o + 2]); f.bg_y = r3(cu[o + 3]); f.ring_range = r2(cu[o + 4]);
      if (!Number.isNaN(cu[o + 5])) f.flat = r2(cu[o + 5]);
      if (!Number.isNaN(cu[o + 6])) { f.broad = r2(cu[o + 6]); f.detail = r2(cu[o + 7]); if (E1) f.tex = r2(cu[o + 8]); }
    }
  } else {
    // plain JS: the dark early stop needs every fill claimed first; when it does not stop, the claims are undone and
    // the loop below makes them again, the same
    if (!bright && ctx.darkNeed) {
      const fills = [];
      for (let i = 0; i < nO; i++) fills.push(await fillOf(i, 0.6));
      claimAll(fills);
      const b = darkCantWin(fills);
      if (b >= 0) return stopDark(b);
      for (let k = 0; k < claims.length; k += 2) { cand[claims[k]] = 0; glab[claims[k]] = 0; }
      claims.length = 0;
    }
    let nObj = 0;
    for (const [g, f] of feats) {
      if (++nObj % 200 === 0) await tick(0.6 + 0.3 * (nObj / feats.size));
      const look = K.peakLook(rgbAt, C, Y, W, H, (y, x) => glab[y * W + x] === g, f, bright);
      f.white = look[0]; f.sharp = look[1]; f.bg = look[2];
      f.zone = !!(p.zone_guard && !(bt + zm <= f.cy && f.cy <= bb - zm && bl + zm <= f.cx && f.cx <= br - zm));
      f.print = printBand(box, f.cx, f.cy);
      f.frame = M.beyondFrame(box.frame, W, H, f.cx, f.cy, p.frame_margin);
      const pad = 45;
      const wy0 = Math.max(0, f.y0 - pad), wy1 = Math.min(H, f.y1 + pad), wx0 = Math.max(0, f.x0 - pad), wx1 = Math.min(W, f.x1 + pad);
      const ww = wx1 - wx0, wh = wy1 - wy0;
      const raw = new Uint8Array(ww * wh);
      for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) {
        const gi = (wy0 + y) * W + wx0 + x;
        raw[y * ww + x] = glab[gi] === g && cand[gi] ? 1 : 0;
      }
      const sol = K.solidify(K.crop(Y, W, wy0, wy1, wx0, wx1), raw, ww, wh, bright, K.crop(sigLevel, W, wy0, wy1, wx0, wx1));
      const solid = sol.mask;
      if (sol.how) {
        f.solid = sol.how;
        for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) {
          if (!solid[y * ww + x]) continue;
          const gi = (wy0 + y) * W + wx0 + x;
          if (!cand[gi] && glab[gi] === 0) { cand[gi] = 1; glab[gi] = g; }
        }
        f.area = K.count(solid);
        f.width = r1(2 * maxDist(solid, ww, wh));
        f.kind = M.kindOf(f.area, f.length, f.width);
      }
      const objd = p.dilate > 0 ? F.dilateMask(solid, ww, wh, 2 * p.dilate + 1, true) : solid;
      if (p.line_guard && f.length >= 20 && (elong(f) || f.kind === "hair/fibre")) {
        const ly0 = Math.max(0, f.y0 - 20) - wy0, lx0 = Math.max(0, f.x0 - 20) - wx0;
        const ly1 = Math.min(H, f.y1 + 20) - wy0, lx1 = Math.min(W, f.x1 + 20) - wx0;
        const lw = lx1 - lx0, lh = ly1 - ly0, lm = new Uint8Array(lw * lh);
        for (let y = 0; y < lh; y++) for (let x = 0; x < lw; x++) lm[y * lw + x] = objd[(ly0 + y) * ww + lx0 + x];
        f.line = K.lineFeatures(K.crop(Y, W, wy0 + ly0, wy0 + ly1, wx0 + lx0, wx0 + lx1), lm, lw, lh, bright);
      }
      if (K.count(objd) >= 4) Object.assign(f, K.markCues(Y, W, G3, E25, E1, sigLevel, cand, objd, ww, wh, wy0, wx0, bright, gradCal));
    }
  }

  lap("cues");
  await tick(0.9);
  // 3b. the dust network's opinion of every candidate, at its centre
  if (p.dustnet && DNET.available()) {
    const fl = [...feats.values()];
    const lg = DNET.logits(ctx.img, W, H, ctx.stride, fl.map((f) => [f.cy, f.cx]), bright);
    fl.forEach((f, i) => { f.net = lg[i]; });
  }

  // 4. verdicts: fields of look-alikes, busy texture, crispness, detail, strands
  const Rf = p.field_radius;
  const gl = [...feats.keys()];
  const Cy = gl.map((g) => feats.get(g).cy), Cx = gl.map((g) => feats.get(g).cx);
  const Z = gl.map((g) => feats.get(g).z), A = gl.map((g) => feats.get(g).area), Wh = gl.map((g) => feats.get(g).white);
  const ogrid = new R.PointGrid(Cy, Cx, 64);
  const busy = maps.busy;
  const accepted = new Set();
  const keepPhoto = (f, why) => { f.verdict = "keep-photo"; f.why = why; };
  // points a first pass left as skin texture: a later pass never heals over them
  const prot = ctx.protect || [];
  const protectedMark = (f) => prot.some(([px, py]) => px >= f.x0 - 3 && px < f.x1 + 3 && py >= f.y0 - 3 && py < f.y1 + 3);
  for (let gi = 0; gi < gl.length; gi++) {
    const g = gl[gi], f = feats.get(g);
    const T = busy.T[Math.min(busy.hb - 1, Math.floor(f.cy / 64)) * busy.wb + Math.min(busy.wb - 1, Math.floor(f.cx / 64))];
    f.busy = r2(T);
    const around = ogrid.queryBox(f.cy, f.cx, Rf).filter((j) => j !== gi);
    const nb = around.filter((j) => Z[j] >= p.field_level * f.z && A[j] >= 4);
    f.field = nb.length;
    const onSubj = nearSubject(ctx.subject, W, H, f.cy, f.cx, p.subject_margin);
    if (ctx.subject) f.subject = onSubj;
    const bigm = f.area >= 80;
    const longThin = f.length >= 30 && f.width <= 10;
    if ((f.white < 0.3 && f.sharp < 0.75) || (f.white < 0.5 && f.sharp < 0.5 && f.width <= 10)) { keepPhoto(f, "soft, tinted mark (print/fabric/out-of-focus detail)"); continue; }
    const lf = f.line;
    if (lf) {
      // a film fibre blocks light evenly along its length; a chain is links with gaps
      if (lf.length >= 60 && (lf.gaps >= p.chain_gaps || lf.per >= p.chain_per)) { keepPhoto(f, "link texture along its length (chain/jewellery/beads)"); continue; }
      if (lf.one_med >= p.edge_one && lf.one_frac >= p.edge_frac) { keepPhoto(f, "rides along an edge (highlight, rim light or halo)"); continue; }
      if (f.zone && lf.straight <= p.straight_max && lf.length >= p.straight_len) { keepPhoto(f, "ruler-straight line on the film border (film/holder edge)"); continue; }
    }
    if (f.print) { keepPhoto(f, "on the film's edge print (outside the picture)"); continue; }
    if (f.frame && (f.ring_range ?? 0) >= p.frame_ring && (f.area ?? 0) >= (p.frame_area * W * H) / 1e6 / p.res_ref_mp
      && (f.width ?? 0) >= p.frame_width * Math.sqrt((W * H) / 1e6 / p.res_ref_mp)) {
      keepPhoto(f, "on the film border: edge print, a sprocket hole or a frame marker"); continue;
    }
    if (f.zone) {
      if (f.white < p.zone_white) { keepPhoto(f, "tinted mark on the film border (border grain)"); continue; }
      const crowd = around.filter((j) => A[j] >= 4);
      if (crowd.length >= p.field_count && f.z < p.zone_crowd_z * R.median(crowd.map((j) => Z[j]))) { keepPhoto(f, `one of ${crowd.length} grain marks on the film border`); continue; }
    }
    const crisp = f.crisp ?? 1.0, det = f.detail ?? 0.0;
    const tex = f.tex ?? det;
    const ts = onSubj ? p.subject_tex : 1.0;
    const detailed = tex >= p.tex_min * ts && det >= p.detail_max * ts;
    const cy_ = f.core_y ?? (bright ? 0.0 : 1.0), by_ = f.bg_y ?? 0.5;
    const nearWhite = bright ? cy_ >= 0.95 || (cy_ >= p.paper_white && by_ <= 0.6) : cy_ <= 0.05 || (cy_ <= 1.0 - p.paper_white && by_ >= 0.4);
    const strong = f.z >= 2.5 * p.seed && (f.area <= 1500 || f.width <= 8) && !detailed;
    const needCrisp = strong ? p.strong_crisp : p.soft_crisp;
    // a flake on smooth ground: a wide flake can't read as crisp as a thin mark, but against a
    // perfectly sharp mark of its own shape it does (crisp_rel ~1). Only where nothing in the photo
    // could make such a mark (no detail or texture around, neutral, compact, alone, not on skin) and
    // the learned decision leans toward dust
    let flake = false;
    if (crisp < needCrisp && p.learned && f.area >= p.flake_area && f.width >= p.flake_width
      && f.length <= p.flake_elong * f.width && (f.crisp_rel ?? 0.0) >= p.flake_rel
      && f.white >= p.flake_white && nb.length <= p.flake_field && f.z >= p.flake_z * p.seed
      && det <= p.flake_detail && tex <= p.flake_detail + 0.1 && !M.skinLike(f.bg)) {
      flake = LEARN.probability(f) >= p.flake_learned;
      if (flake) f.flake = true;
    }
    if (crisp < needCrisp && !(nearWhite && f.white >= 0.45) && !flake) { keepPhoto(f, "not crisp enough for debris on the film (photo detail)"); continue; }
    // small scans: picture details on a soft or shrunk scan (lights, window bars, grass, nails) pass the strong-mark bar and
    // the near-white exemption; dust there stays crisper (res_crisp). Not on plain ground, where a soft white blob is a
    // flake (q_05's two in a blue sky, q_10's on plain grey). 10-25: 238 hand-labelled big heals on 33 Commons scans,
    // picture healed 175 -> 92 of 179, dust 17 of 17 kept; the lab copies lose 1-8 heals per size, 7-26 new ones go
    // (nor past the film's frame edge: no picture there, and the edge print has its own rule; u_n10's border flake)
    if (smallScanRules && !f.frame && crisp < p.res_crisp && !flake && !((f.detail ?? 9) <= p.res_plain_detail && (f.busy ?? 9) <= p.res_plain_busy)) {
      keepPhoto(f, "soft for dust on a small scan (picture detail)"); continue;
    }
    if ((f.broad ?? 1.0) < p.broad_min && !(f.kind === "speck" && f.z >= p.broad_speck_z)) { keepPhoto(f, "only bright against dark detail beside it (e.g. skin between tattoo lines)"); continue; }
    if (detailed && !(crisp >= p.detail_crisp && f.white >= 0.5 && f.z >= p.seed * p.detail_gain * Math.max(det, 1.0))
      && !(f.kind === "speck" && f.z >= p.detail_speck_z && crisp >= p.detail_speck_crisp)) { keepPhoto(f, "in detailed photo texture (hair/fabric/foliage): left for safety"); continue; }
    if (p.texture_guard && f.kind === "hair/fibre") {
      const sy0 = Math.max(0, f.y0 - 34), sy1 = Math.min(H, f.y1 + 34), sx0 = Math.max(0, f.x0 - 34), sx1 = Math.min(W, f.x1 + 34);
      const sw = sx1 - sx0, sh = sy1 - sy0, local = new Uint8Array(sw * sh);
      for (let y = f.y0; y < f.y1; y++) for (let x = f.x0; x < f.x1; x++) {
        const gi = y * W + x;
        if (glab[gi] === g && cand[gi]) local[(y - sy0) * sw + x - sx0] = 1;
      }
      const ta = K.textureAlignment(Y, W, H, maps.sigma, cand, f, local, gradCal);
      f.texture = ta;
      if (ta[0] >= 0.3 && ta[1] <= 0.2 && ta[2] >= 2.0 && ta[3] <= 25) { keepPhoto(f, "runs with the strand texture around it (brow/hair/fur/weave)"); continue; }
    }
    const inField0 = nb.length >= p.field_count && (T >= p.busy_field || nb.length >= 2 * p.field_count);
    const onSkin = p.skin_guard && nb.length >= p.skin_count && M.skinLike(f.bg)
      && !(f.kind === "speck" && f.z >= p.skin_speck_z && crisp >= p.skin_speck_crisp && f.sharp >= p.skin_speck_sharp);
    if (inField0 || onSkin) {
      const refZ = R.median(nb.map((j) => Z[j])), refArea = R.median(nb.map((j) => A[j]));
      const refLen = R.median(nb.map((j) => feats.get(gl[j]).length)), refWhite = R.median(nb.map((j) => Wh[j]));
      f.field_z = r1(refZ); f.field_area = r1(refArea); f.field_len = r1(refLen);
      const reasons = [];
      const debrisLook = f.white >= 0.4;
      if (f.area >= Math.max(p.stand_min_area, p.stand_area * refArea) && debrisLook && f.kind !== "hair/fibre") reasons.push("much bigger than the marks around it");
      if (f.length >= Math.max(30, p.stand_len * refLen) && f.width <= p.stand_hair_width && debrisLook) reasons.push("much longer than the marks around it");
      if (f.z >= p.stand_z * refZ && f.white >= 0.35) reasons.push("much brighter than the marks around it");
      if (f.white >= refWhite + p.stand_white && f.sharp >= 0.8) reasons.push("whiter and crisper than the marks around it");
      if (reasons.length && f.z_fine < p.min_fine) keepPhoto(f, "too faint for a second pass");
      else if (reasons.length && protectedMark(f)) keepPhoto(f, "left as skin texture in the first pass");
      else if (reasons.length) { f.verdict = "defect"; f.why = reasons.join("; "); accepted.add(g); }
      else keepPhoto(f, onSkin && nb.length < p.field_count ? `on skin with ${nb.length} look-alikes around it (sweat/pores/skin texture)` : `one of ${nb.length} similar marks around it`);
      continue;
    }
    const need = p.seed * Math.min(Math.max(1 + p.busy_gain * (T - p.busy_start), 1.0), p.busy_cap);
    if (f.z_fine < need && !(bigm || longThin)) { keepPhoto(f, `faint mark in busy texture (needs z ${need.toFixed(0)})`); continue; }
    if (f.z_fine < p.min_fine) { keepPhoto(f, "too faint for a second pass"); continue; }
    if (protectedMark(f)) { keepPhoto(f, "left as skin texture in the first pass"); continue; }
    if (f.kind === "hair/fibre" && f.z < p.hair_seed) { keepPhoto(f, `faint thin line: photo texture more often than a film hair (needs z ${p.hair_seed.toFixed(0)})`); continue; }
    f.verdict = "defect"; f.why = "isolated";
    accepted.add(g);
  }

  // 4a. the learned second opinion (trained on Magnus's answers; engine/learned.js): overrules a
  //     verdict only when sure, only for sharp marks, never against the structural photo rules
  if (p.learned) {
    // the trees' P(dust); then, where it could change the outcome, blended with the second network's
    // opinion (patch + context, fine-tuned on his answers): a heal it could turn to photo (P <= 0.9),
    // a left mark it could turn to dust (P >= 0.5). Natural highlights first: on his answers (scans
    // held out) photo heals 20 -> 8 for 9 of 197 dust.
    const cands = [], ptree = new Map();
    for (const g of gl) {
      const f = feats.get(g), v = f.verdict;
      if ((v !== "defect" && v !== "keep-photo") || f.zone || LEARN.VETO.some((w) => f.why.includes(w))) continue;
      const pt = LEARN.probability(f);
      ptree.set(g, pt);
      // not within 80 px of the picture's edge: there the context view sees the film border and the
      // second network reads it as photo (it vetoed real hairs at boy's and Man's film edges). A hair's
      // whole outline must keep that distance: Man's J hair runs from 5 px below the top edge, its
      // centre 84 px in.
      const inner = f.cy >= box[0] + 80 && f.cy < box[1] - 80 && f.cx >= box[2] + 80 && f.cx < box[3] - 80;
      const hair = f.kind === "hair/fibre";
      // and only short hairs: on long fibres (a 230 px dust hair on his answers, 270-290 px fibres on plain
      // sky in unseen scans) it says photo, pulling sure dust to the edge of the cut
      const reach = hair ? p.learned_net2_hairs && (f.length ?? 0) <= p.learned_net2_hair_len && f.y0 >= box[0] + 80 && f.y1 < box[1] - 80 && f.x0 >= box[2] + 80 && f.x1 < box[3] - 80 : inner;
      if (p.dustnet2 && p.learned_net2_weight > 0 && reach && ((v === "defect" && pt <= 0.9) || (v === "keep-photo" && pt >= 0.5))) cands.push(g);
    }
    if (cands.length && DNET.second.available()) {
      const lg2 = DNET.second.logits(ctx.img, W, H, ctx.stride, cands.map((g) => [feats.get(g).cy, feats.get(g).cx]), bright);
      cands.forEach((g, i) => { feats.get(g).net2 = lg2[i]; });
    }
    const w2 = p.learned_net2_weight;
    for (const g of gl) {
      if (!ptree.has(g)) continue;
      const f = feats.get(g), v = f.verdict;
      let pr = ptree.get(g);
      if (f.net2 != null) pr = (1 - w2) * pr + w2 / (1 + Math.exp(-f.net2));
      f.learned = R.pyRoundTo(pr, 3);
      const onSubj = !!f.subject;
      // a flake's crispness is measured against its own shape, which the model can't see
      if (v === "defect" && f.flake) continue;
      // compact marks (specks, dots, blobs) are what glints, catchlights and chrome look like: they need more
      // to be healed (learned_lo_compact) than hairs and other elongated marks (learned_lo; boy's wall fibre)
      // (thin ones: a 96 x 25 px blob is a word of soft print, not a fibre)
      const elongated = f.kind === "hair/fibre" || ((f.length ?? 0) >= 2.5 * Math.max(f.width || 1, 1) && (f.width || 1) <= p.learned_thin_max);
      // on the subject (Select Subject) a compact mark needs subject_lo: catchlights, jewellery, buttons
      // big compact marks (400 px+) are flakes far more often than glints: they keep learned_lo (Man's 825 px flake)
      const lo = elongated ? p.learned_lo : Math.max((f.area ?? 0) >= p.learned_big_area ? p.learned_lo : p.learned_lo_compact ?? p.learned_lo, f.subject ? p.subject_lo : 0);
      if (v === "defect" && pr <= lo) { keepPhoto(f, "photo, by the decision learned from his answers"); accepted.delete(g); }
      // never a mark of 100 px or more tinted against white (a pale window reflection on a boat: white -3.76;
      // his 65 such answers are all >= 0; on tiny specks the measure is noise)
      else if (v === "keep-photo" && pr >= (onSubj ? p.subject_hi : p.learned_hi) && f.z >= (LEARN.FLOOR[f.kind] ?? 99.0) && f.z_fine >= p.min_fine && !protectedMark(f)
        && !((f.area ?? 0) >= p.learned_flip_white_area && (f.white ?? 1) < p.learned_flip_white)) {
        // a line piece the rules call not crisp whose line runs on past its ends is a stem, twig or wire
        if (p.learned_reach && f.why.startsWith("not crisp") && f.length >= p.line_len && f.length <= p.line_max_len && f.length >= 2.5 * Math.max(f.width, 1.0)) {
          f.reach = markReach(ctx, maps.sigma, glab, cand, g, f, bright);
          if (f.reach >= p.line_reach_px) { f.why = "not crisp, and runs on past its ends into the picture (a stem, twig or wire)"; continue; }
        }
        f.verdict = "defect"; f.why = "dust, by the decision learned from his answers"; accepted.add(g);
      }
    }
  }

  // 4b. thin elongated pieces right next to a confirmed hair are the same hair (not on small scans: there a hair's
  // neighbours were picture 35 times in 38 on the lab's shrunk copies, and the growth ran through the shadow noise of a
  // 16 MP Tri-X scan, 1,163 pieces)
  const smallScan = p.res_guard && Math.max((W * H) / 1e6, p.res_scan_mp || 0) < p.res_full_mp;
  for (let it = 0; it < (smallScan ? 0 : 3); it++) {
    let grew = false;
    for (const g of [...accepted].filter((a) => feats.get(a).kind === "hair/fibre")) {
      const f = feats.get(g), e = p.absorb_px + 2;
      const y0 = Math.max(0, f.y0 - e), y1 = Math.min(H, f.y1 + e), x0 = Math.max(0, f.x0 - e), x1 = Math.min(W, f.x1 + e);
      const w = x1 - x0, h = y1 - y0, own = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const gi = (y0 + y) * W + x0 + x; own[y * w + x] = glab[gi] === g && cand[gi] ? 1 : 0; }
      const near = F.dilateMask(own, w, h, 2 * p.absorb_px + 1, true);
      const others = new Set();
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const gi = (y0 + y) * W + x0 + x; if (near[y * w + x] && cand[gi]) others.add(glab[gi]); }
      for (const o of [...others].sort((a, b) => a - b)) {
        if (o === g || accepted.has(o) || !feats.has(o)) continue;
        const fo = feats.get(o);
        if (!elong(fo)) continue;
        if (fo.white < 0.3 && fo.sharp < 0.75) continue;
        if (fo.width > 5) continue;
        fo.verdict = "defect"; fo.why = "continues the hair/fibre next to it"; fo.kind = "hair/fibre";
        accepted.add(o);
        grew = true;
      }
    }
    if (!grew) break;
  }

  // small scans: nothing bigger than dust ever is, nothing on the frame line
  for (const g of [...accepted]) {
    const f = feats.get(g);
    const plain = f.kind !== "hair/fibre" && f.detail != null && f.busy != null && f.detail <= p.res_plain_detail && f.busy <= p.res_plain_busy;
    const why = resGuard(p, W, H, f.x0, f.y0, f.x1, f.y1, f.area, f.kind === "hair/fibre" ? "hair" : "dust", plain);
    if (why) { keepPhoto(f, why); accepted.delete(g); }
  }
  let mask = new Uint8Array(N);
  if (accepted.size) {
    let maxG = 0;
    for (const g of accepted) maxG = Math.max(maxG, g);
    const acc = new Uint8Array(maxG + 1);
    for (const g of accepted) acc[g] = 1;
    for (let i = 0; i < N; i++) { const g = glab[i]; if (g && g <= maxG && acc[g] && cand[i]) mask[i] = 1; }
    mask = R.fillHoles(mask, W, H);
  }
  const windowOf = (f, e) => {
    const y0 = Math.max(0, f.y0 - e), y1 = Math.min(H, f.y1 + e), x0 = Math.max(0, f.x0 - e), x1 = Math.min(W, f.x1 + e);
    return { y0, y1, x0, x1, w: x1 - x0, h: y1 - y0 };
  };
  const ownMask = (wn, g, src) => {
    const m = new Uint8Array(wn.w * wn.h);
    for (let y = 0; y < wn.h; y++) for (let x = 0; x < wn.w; x++) { const gi = (wn.y0 + y) * W + wn.x0 + x; m[y * wn.w + x] = src[gi] && glab[gi] === g ? 1 : 0; }
    return m;
  };
  const orInto = (wn, m) => {
    for (let y = 0; y < wn.h; y++) for (let x = 0; x < wn.w; x++) if (m[y * wn.w + x]) mask[(wn.y0 + y) * W + wn.x0 + x] = 1;
  };
  // close small gaps in the rim of solid specks/flakes so their centre is healed too
  for (const g of accepted) {
    const f = feats.get(g);
    if (f.kind === "hair/fibre") continue;
    const wn = windowOf(f, 6);
    const sub = ownMask(wn, g, mask);
    orInto(wn, R.fillHoles(F.erodeMask(F.dilateMask(sub, wn.w, wn.h, 7, true), wn.w, wn.h, 7, true), wn.w, wn.h));
  }
  // complete solid debris out to halfway between the background and its own brightness
  {
    const sgn = bright ? 1 : -1;
    for (const g of accepted) {
      const f = feats.get(g);
      if (f.kind === "hair/fibre" || f.area < 20 || f.white < 0.45) continue;
      const wn = windowOf(f, 60);
      const own = ownMask(wn, g, mask);
      const nOwn = K.count(own);
      if (nOwn < 10) continue;
      const cw = new Uint8Array(wn.w * wn.h);
      for (let y = 0; y < wn.h; y++) for (let x = 0; x < wn.w; x++) cw[y * wn.w + x] = cand[(wn.y0 + y) * W + wn.x0 + x];
      const ring = K.andNot(F.dilateMask(own, wn.w, wn.h, 15), F.dilateMask(cw, wn.w, wn.h, 5));
      if (K.count(ring) < 20) continue;
      const yy = new Float32Array(wn.w * wn.h);
      for (let y = 0; y < wn.h; y++) for (let x = 0; x < wn.w; x++) yy[y * wn.w + x] = sgn * Y[(wn.y0 + y) * W + wn.x0 + x];
      const core = R.percentile(K.vals(yy, own), 90), bgl = R.median(K.vals(yy, ring));
      if (core - bgl < p.blob_contrast) continue;
      const reg = new Uint8Array(wn.w * wn.h);
      for (let k = 0; k < reg.length; k++) reg[k] = yy[k] >= bgl + 0.5 * (core - bgl) || own[k] ? 1 : 0;
      const { lab: l2 } = R.label(reg, wn.w, wn.h, 8);
      const ids = new Set();
      for (let k = 0; k < own.length; k++) if (own[k] && l2[k]) ids.add(l2[k]);
      const grown = new Uint8Array(wn.w * wn.h);
      for (let k = 0; k < grown.length; k++) grown[k] = ids.has(l2[k]) ? 1 : 0;
      let touches = false;
      for (let x = 0; x < wn.w && !touches; x++) if ((wn.y0 > 0 && grown[x]) || (wn.y1 < H && grown[(wn.h - 1) * wn.w + x])) touches = true;
      for (let y = 0; y < wn.h && !touches; y++) if ((wn.x0 > 0 && grown[y * wn.w]) || (wn.x1 < W && grown[y * wn.w + wn.w - 1])) touches = true;
      if (!touches && K.count(grown) <= 4 * nOwn + 1500) orInto(wn, grown);
    }
  }
  for (const [r, ya_, xa_, yb_, xb_, th] of bridges) {
    if (accepted.has(r)) drawLine(mask, W, H, ya_, xa_, yb_, xb_, th, (i) => { if (glab[i] === 0) glab[i] = r; });
  }
  lap("verdicts+fill");
  await tick(0.95);
  // 4c. trace confirmed hairs through stretches too faint to detect on their own
  if (p.trace_hairs && accepted.size) {
    for (const g of [...accepted]) {
      const f = feats.get(g);
      if (f.length < 30 || f.length < 2.5 * Math.max(f.width, 1.0) || f.zone) continue;
      const wn = windowOf(f, 4);
      const own = ownMask(wn, g, mask);
      const sk = R.skeletonize(own, wn.w, wn.h);
      const ys = [], xs = [];
      for (let k = 0; k < sk.length; k++) if (sk[k]) { ys.push(Math.floor(k / wn.w)); xs.push(k % wn.w); }
      if (ys.length < 10) continue;
      const grid = new R.PointGrid(ys, xs, 16);
      const ends = [];
      for (let k = 0; k < ys.length; k++) if (grid.queryBox(ys[k], xs[k], 1).length - 1 === 1) ends.push(k);
      let traced = 0;
      for (const e of ends.length >= 2 ? ends.slice(0, 2) : ends) {
        const nb = grid.queryBox(ys[e], xs[e], 15).filter((j) => (ys[j] - ys[e]) ** 2 + (xs[j] - xs[e]) ** 2 <= 225);
        if (nb.length < 5) continue;
        const st = R.shapeStats(nb.map((j) => ys[j]), nb.map((j) => xs[j]));
        let tdy = Math.sin(st.angle), tdx = Math.cos(st.angle);
        const my = R.mean(nb.map((j) => ys[j])), mx = R.mean(nb.map((j) => xs[j]));
        if ((ys[e] - my) * tdy + (xs[e] - mx) * tdx < 0) { tdy = -tdy; tdx = -tdx; }
        const start = [ys[e] + wn.y0, xs[e] + wn.x0];
        const path = traceRidge(zs, W, H, start, [tdy, tdx], p, outside, Math.max(200.0, 4.0 * f.length));
        if (path.length < 2) continue;
        let prev = start;
        for (const q of path) {
          drawLine(mask, W, H, Math.trunc(prev[0]), Math.trunc(prev[1]), Math.trunc(q[0]), Math.trunc(q[1]), 5);
          prev = q;
        }
        traced += path.length * 6;
      }
      if (traced) f.traced_px = traced;
    }
  }
  // 5. follow accepted hairs/fibres into their faint tails
  if (p.fibre_extend > 0) {
    const tailZ = p.grow + 0.5;
    for (const g of accepted) {
      const f = feats.get(g);
      if (f.kind !== "hair/fibre") continue;
      const e = p.fibre_extend, wn = windowOf(f, e);
      let cur = ownMask(wn, g, mask);
      const base = K.count(cur);
      const ok = new Uint8Array(wn.w * wn.h);
      for (let y = 0; y < wn.h; y++) for (let x = 0; x < wn.w; x++) ok[y * wn.w + x] = zs[(wn.y0 + y) * W + wn.x0 + x] >= tailZ ? 1 : 0;
      for (let it = 0; it < e; it++) {
        const nxt = F.dilateMask(cur, wn.w, wn.h, 3);
        for (let k = 0; k < nxt.length; k++) nxt[k] = nxt[k] & ok[k];
        const sn = K.count(nxt);
        if (sn === K.count(cur) || sn > 1.8 * base) break;
        cur = nxt;
      }
      orInto(wn, cur);
    }
  }
  lap("trace+tails");
  if (ctx.timing) ctx.log(`[${bright ? "bright" : "dark"}] seconds: ${tm.parts.join(" · ")}  (maps: ${(maps.times || []).join(" · ")})`);
  const marks = [...feats.values()];
  const strong = marks.filter((f) => f.verdict === "defect" && f.z >= 20 && f.field <= 2).length;
  ctx.log(`[${bright ? "bright" : "dark"}] pieces ${pieces.size}  objects ${feats.size}  defects ${accepted.size}  strong-isolated ${strong}`);
  return { mask, marks, n: accepted.size, strong, cand, maps };
}

// ---------- soft (out-of-focus) debris --------------------------------------------------------

// Judge one soft blob as detect.py does on its 320-px window W0, but compute only what each step
// reads: the blurred window grows around the peak on demand (a blur is used only where it equals
// the W0 computation, at least 7 px from a cut side), and the later steps run on a window that
// holds the object plus everything measured around it.
// scratch shared by every soft judgement (one stage at a time, never re-entered)
const floodGen = { n: 0 };
let floodBuf = new Int32Array(0);
function floodStamp(n) {
  if (floodBuf.length < n) { floodBuf = new Int32Array(n); floodGen.n = 0; }
  return floodBuf;
}
const annulusCache = new Map();
function annulusOffsets(rIn, ro) {
  const key = rIn * 1000 + ro;
  let o = annulusCache.get(key);
  if (!o) {
    const v = [];
    for (let dy = -ro; dy <= ro; dy++) for (let dx = -ro; dx <= ro; dx++) {
      const d = Math.hypot(dy, dx);
      if (d >= rIn && d < ro) v.push(dy, dx);
    }
    o = Int16Array.from(v);
    annulusCache.set(key, o);
  }
  return o;
}
let annBuf = new Float64Array(0);
function annulusBuffer(n) {
  if (annBuf.length < n) annBuf = new Float64Array(n);
  return annBuf;
}

function judgeSoft(ctx, maps, bright, blobs, healed, tree, pk, W0, gy, gx, peak) {
  const { Y, W, H, p, rgbAt, C } = ctx;
  const sigLevel = maps.sigma_level;
  const sgn = bright ? 1 : -1;
  let win = null, ww = 0, hh = 0, Yg = null;
  // window = [y0, y1) x [x0, x1) clipped to W0; blur recomputed when it grows
  const setWin = (y0, y1, x0, x1) => {
    y0 = Math.max(W0.y0, y0); y1 = Math.min(W0.y1, y1); x0 = Math.max(W0.x0, x0); x1 = Math.min(W0.x1, x1);
    if (win && y0 >= win.y0 && y1 <= win.y1 && x0 >= win.x0 && x1 <= win.x1) return false;
    if (win) { y0 = Math.min(y0, win.y0); y1 = Math.max(y1, win.y1); x0 = Math.min(x0, win.x0); x1 = Math.max(x1, win.x1); }
    win = { y0, y1, x0, x1 };
    ww = x1 - x0; hh = y1 - y0;
    const Yw = new Float32Array(ww * hh);
    for (let y = 0; y < hh; y++) { const o = (y0 + y) * W + x0; for (let x = 0; x < ww; x++) Yw[y * ww + x] = sgn * Y[o + x]; }
    Yg = F.gaussianBlur(Yw, ww, hh, 1.5);
    return true;
  };
  // is window-local pixel (y, x) at least m px from every cut side (a side that is not W0's)?
  const safe = (y, x, m) => (win.y0 === W0.y0 || y >= m) && (win.y1 === W0.y1 || y < hh - m) && (win.x0 === W0.x0 || x >= m) && (win.x1 === W0.x1 || x < ww - m);
  const atW0 = (y, x) => (win.y0 === W0.y0 && y === 0) || (win.y1 === W0.y1 && y === hh - 1) || (win.x0 === W0.x0 && x === 0) || (win.x1 === W0.x1 && x === ww - 1);
  // 4-connected flood over global pixels; ok(k) / visit(k) take window-local indices.
  // Returns {px (global indices), n, stop} where stop is "cap", "edge", "far" or "".
  const flood = (seedsG, ok, visit, guard) => {
    const stamp = floodStamp(ww * hh), gen = ++floodGen.n;
    const stack = [], px = [];
    const mark = (g) => {
      const yy = (g / W) | 0, k = (yy - win.y0) * ww + (g - yy * W - win.x0);
      if (stamp[k] === gen) return;
      stamp[k] = gen;
      stack.push(g);
    };
    let stop = "";
    for (const g of seedsG) mark(g);
    while (stack.length && !stop) {
      const g = stack.pop();
      const gyy = (g / W) | 0, gxx = g - gyy * W;
      const ly = gyy - win.y0, lx = gxx - win.x0;
      const k = ly * ww + lx;
      if (guard && !safe(ly, lx, 7)) { stop = "edge"; break; }
      if (!ok(k)) continue;
      px.push(g);
      stop = visit(ly, lx, k, px.length);
      if (stop) break;
      if (lx > 0 && stamp[k - 1] !== gen) { stamp[k - 1] = gen; stack.push(g - 1); }
      if (lx < ww - 1 && stamp[k + 1] !== gen) { stamp[k + 1] = gen; stack.push(g + 1); }
      if (ly > 0 && stamp[k - ww] !== gen) { stamp[k - ww] = gen; stack.push(g - W); }
      if (ly < hh - 1 && stamp[k + ww] !== gen) { stamp[k + ww] = gen; stack.push(g + W); }
    }
    return { px, n: px.length, stop };
  };
  const loc = (g) => { const yy = (g / W) | 0; return (yy - win.y0) * ww + (g - yy * W - win.x0); };

  // 1. its own scale: widen the annulus until the half-contrast region fits inside it
  setWin(gy - 62, gy + 63, gx - 62, gx + 63);
  let qy = gy - win.y0, qx = gx - win.x0;
  let top = -Infinity;
  for (let y = Math.max(0, qy - 2); y < Math.min(hh, qy + 3); y++) for (let x = Math.max(0, qx - 2); x < Math.min(ww, qx + 3); x++) top = Math.max(top, Yg[y * ww + x]);
  let Rg = null, bg = 0, con = 0;
  for (const rIn of [15, 25, 40, 60, 85, 115]) {
    const ro = rIn + 30;
    // past the first window, blur the whole W0 once rather than growing it step by step
    if (ro + 7 > 62 && setWin(W0.y0, W0.y1, W0.x0, W0.x1)) { qy = gy - win.y0; qx = gx - win.x0; }
    const offs = annulusOffsets(rIn, ro), buf = annulusBuffer(offs.length >> 1);
    let na = 0;
    for (let t = 0; t < offs.length; t += 2) {
      const y = qy + offs[t], x = qx + offs[t + 1];
      if (y >= 0 && y < hh && x >= 0 && x < ww) buf[na++] = Yg[y * ww + x];
    }
    if (na < 50) break;
    bg = R.percentileOf(buf, na, 50); con = top - bg;
    if (con <= 0) break;
    const thr = bg + 0.5 * con;
    if (!(Yg[qy * ww + qx] >= thr)) break;
    const lim = rIn - 2;
    const fl = flood([gy * W + gx], (k) => Yg[k] >= thr, (ly, lx) => (Math.hypot(ly - qy, lx - qx) >= lim ? "far" : ""));
    if (!fl.stop) { Rg = fl.px; break; }
  }
  const rec = { cx: gx, cy: gy, z: r1(peak), kind: "soft" };
  if (!Rg) {
    rec.verdict = "keep-photo"; rec.why = "soft: part of a bigger bright area";
    return { rec, R: null, fill: null };
  }
  const area = Rg.length;
  // 2. the whole anomalous object (a dust grain can be partly dark): flood, growing the window
  //    when the flood comes near a cut side; past the cap or touching W0's side it is not used
  const cap2 = 3 * area + 200;
  const isR = new Set(Rg);
  const growTo = (pxs, pad) => {
    let y0 = Infinity, y1 = -Infinity, x0 = Infinity, x1 = -Infinity;
    for (const g of pxs) { const yy = (g / W) | 0, xx = g - yy * W; if (yy < y0) y0 = yy; if (yy > y1) y1 = yy; if (xx < x0) x0 = xx; if (xx > x1) x1 = xx; }
    return setWin(y0 - pad, y1 + pad + 1, x0 - pad, x1 + pad + 1);
  };
  growTo(Rg, 48);
  let f2;
  for (;;) {
    let touch = false;
    f2 = flood(Rg, (k) => isR.has(win.y0 * W + win.x0 + ((k / ww) | 0) * W + (k % ww)) || Math.abs(Yg[k] - bg) >= 0.3 * con, (ly, lx, k, n) => {
      if (atW0(ly, lx)) { touch = true; return "cap"; }
      return n > cap2 ? "cap" : "";
    }, true);
    if (f2.stop !== "edge") { f2.touch = touch; break; }
    const before = ww * hh;
    setWin(win.y0 - 64, win.y1 + 64, win.x0 - 64, win.x1 + 64);
    if (ww * hh === before) { f2.touch = touch; break; }       // already W0: cannot happen
  }
  const r2ok = f2.stop === "" && f2.n <= cap2;
  const objG = r2ok ? f2.px : Rg;
  // 3. the window for everything measured around the object (rings up to 40 px, blurs, colour)
  growTo(objG, 48);
  qy = gy - win.y0; qx = gx - win.x0;
  const n = ww * hh;
  const Rm = new Uint8Array(n);
  for (const g of Rg) Rm[loc(g)] = 1;
  const obj = new Uint8Array(n);
  for (const g of objG) obj[loc(g)] = 1;
  const R2 = r2ok ? obj : null;
  const rys = Rg.map((g) => (g / W) | 0), rxs = Rg.map((g) => g % W);
  const elong = rxs.length > 3 ? R.shapeStats(rys, rxs).elong : 1.0;
  let plat = 0;
  for (const g of Rg) if (Yg[loc(g)] >= bg + 0.85 * con) plat++;
  const plateau = plat / area;
  const ring = K.andNot(F.dilateMask(obj, ww, hh, 41), F.dilateMask(obj, ww, hh, 9));
  const nRing = K.count(ring);
  let fine = 99.0;
  if (nRing > 30) {
    const Yraw = K.crop(Y, W, win.y0, win.y1, win.x0, win.x1);
    const g08 = F.gaussianBlur(Yraw, ww, hh, 0.8), g25 = F.gaussianBlur(Yraw, ww, hh, 2.5);
    const d2 = [], sl = [];
    for (let k = 0; k < n; k++) if (ring[k]) { const d = g08[k] - g25[k]; d2.push(d * d); sl.push(sigLevel[(win.y0 + Math.floor(k / ww)) * W + win.x0 + (k % ww)]); }
    fine = R.percentile(d2, 90) / Math.max(R.median(sl) ** 2, 1e-10);
  }
  let useCore = false;
  for (const g of Rg) if (Yg[loc(g)] >= bg + 0.7 * con) { useCore = true; break; }
  const corec = new Array(C).fill(0);
  let nc = 0;
  for (const g of Rg) {
    if (useCore && !(Yg[loc(g)] >= bg + 0.7 * con)) continue;
    const yy = (g / W) | 0, xx = g - yy * W;
    for (let c = 0; c < C; c++) corec[c] += rgbAt(yy, xx, c);
    nc++;
  }
  for (let c = 0; c < C; c++) corec[c] /= nc;
  let bgc = corec;
  if (nRing > 30) {
    bgc = [];
    for (let c = 0; c < C; c++) {
      const v = [];
      for (let k = 0; k < n; k++) if (ring[k]) v.push(rgbAt(win.y0 + Math.floor(k / ww), win.x0 + (k % ww), c));
      bgc.push(R.median(v));
    }
  }
  let dot = 0, nl = 0, nt = 0;
  for (let c = 0; c < C; c++) {
    const lift = corec[c] - bgc[c], tow = bright ? 1.0 - bgc[c] : -bgc[c];
    dot += lift * tow; nl += lift * lift; nt += tow * tow;
  }
  const cosw = dot / (Math.sqrt(nl) * Math.sqrt(nt) + 1e-9);
  const bigr = K.andNot(F.dilateMask(obj, ww, hh, 81), F.dilateMask(obj, ww, hh, 21));
  const broad = K.count(bigr) > 30 ? (top - R.median(K.vals(Yg, bigr))) / con : 0.0;
  const onSubj = nearSubject(ctx.subject, W, H, gy, gx, p.subject_margin);
  let near = 0;
  if (tree) {
    for (const j of tree.queryBox(gy, gx, 300)) {
      const ry = pk.ys[j] - win.y0, rx = pk.xs[j] - win.x0;
      const own = ry >= 0 && ry < hh && rx >= 0 && rx < ww && obj[ry * ww + rx];
      if (pk.vs[j] >= Math.max(6.0, 0.6 * peak) && !own) near++;
    }
  }
  Object.assign(rec, { area, elong: r1(elong), plateau: r2(plateau), fine: r2(fine), cosw: r3(cosw), broad: r2(broad), near });
  if (ctx.subject) rec.subject = onSubj;
  const reasons = [];
  if (onSubj && ctx.groupAt && ctx.groupAt[gy * W + gx]) reasons.push("a highlight among look-alikes on the subject (glasses, jewellery, buttons)");
  if (area > p.soft_area_max) reasons.push(`big soft patch (${area} px)`);
  if (elong > p.soft_elong_max) reasons.push("long soft highlight");
  if (plateau >= p.soft_plateau && elong <= 2.5 && area >= 150) reasons.push("flat-topped disc (bokeh)");
  if (broad > 6.0) reasons.push("far brighter than everything around (a highlight)");
  if (fine > p.soft_fine_max) reasons.push("in-focus detail beside it");
  // a strong, elongated blob with crisp detail inside is a highlight in focus (a button, ring, chrome
  // trim): soft dust is smooth. His soft answers: 15 photo, 6 dust match, none of that dust was healed.
  if (fine >= p.soft_bar_fine && elong >= p.soft_bar_elong && peak >= p.soft_bar_z) reasons.push("a crisp bright bar (a highlight on metal, glass or a button)");
  if (cosw < p.soft_cos_white) reasons.push("coloured glow");
  if (broad < 0.3) reasons.push("only bright against dark detail beside it");
  // a bright blob in a dark socket inside brighter ground is a catchlight (an eye, a glint in a hole): the
  // ring just around it is far darker than the ground further out. His soft dust: at most 0.65.
  if (bright && p.soft_socket > 0) {
    const Rb = Math.max(2.0, Math.sqrt(area / Math.PI)), M = Math.ceil(Rb + 62);
    const inner = [], near = [], far = [];
    for (let y = Math.max(0, gy - M); y <= Math.min(H - 1, gy + M); y++) {
      for (let x = Math.max(0, gx - M); x <= Math.min(W - 1, gx + M); x++) {
        const d = Math.hypot(y - gy, x - gx), v = ctx.Y[y * W + x];
        if (d <= Math.max(1.5, 0.7 * Rb)) inner.push(v);
        else if (d >= Rb + 2 && d <= Rb + 8) near.push(v);
        else if (d >= Rb + 20 && d <= Rb + 60) far.push(v);
      }
    }
    if (inner.length && near.length && far.length) {
      inner.sort((a, b) => a - b);
      const pk = inner[Math.floor(0.9 * (inner.length - 1))], nm = R.median(near), fm = R.median(far);
      rec.socket = R.pyRoundTo((fm - nm) / Math.max(pk - nm, 1e-6), 2);
      if (rec.socket >= p.soft_socket) reasons.push("a highlight in a dark socket (a catchlight, a glint in a hole)");
    }
  }
  if (ctx.surePhoto && ctx.surePhoto.length) {
    let by0 = Infinity, by1 = -1, bx0 = Infinity, bx1 = -1;
    for (const g of Rg) { const yy = (g / W) | 0, xx = g - yy * W; if (yy < by0) by0 = yy; if (yy > by1) by1 = yy; if (xx < bx0) bx0 = xx; if (xx > bx1) bx1 = xx; }
    if (ctx.surePhoto.some((m) => m.cy >= by0 && m.cy <= by1 && m.cx >= bx0 && m.cx <= bx1)) reasons.push("holds a piece the learned decision is sure is picture");
  }
  if (!reasons.length) {
    const sBlob = Math.max(1.0, Math.sqrt(area / Math.PI) / 1.414);
    const [sim, stronger, row] = blobs.crowd(Rm, win.y0, win.x0, ww, hh, healed, sBlob);
    rec.sim = sim; rec.stronger = stronger; rec.row = row;
    const t = Math.min(1.0, Math.max(0.0, (area - 300) / 300.0));
    const simLimit = p.soft_sim_max + t * (p.soft_sim_big - p.soft_sim_max);
    // ... unless the crowd is faint specks scattered over plain ground: many same-size blobs but few strong peaks
    // around (sweat on skin and dot prints are strong and dense) and small. Among the soft blobs this rule left,
    // answered (his rounds 1-5, Claude's soft labels): 15 dust, 1 photo; the learned decision still judges them.
    const scatter = near <= p.soft_scatter_near && area <= p.soft_scatter_area;
    if ((sim > simLimit || stronger > p.soft_stronger_max) && !scatter) reasons.push(`one of many same-size blobs around it (${sim} similar, ${stronger} stronger)`);
    else if (sim >= p.soft_tex_sim && near > p.soft_tex_near) reasons.push(`among many soft highlights (leaves, bark, hair, sheen: ${near} peaks around)`);
    else if (p.soft_row && row) reasons.push("one of a row of look-alikes (stitching, beads, pattern)");
  }
  // the soft step's learned decision (his soft answers): overrules the rules only when sure, never the
  // subject rule or the rare structural reasons his answers did not cover
  let learnedDust = false;
  if (p.learned) {
    if (p.dustnet && DNET.available()) rec.net = ctx.softNet && ctx.softNet.has(gy * W + gx) ? ctx.softNet.get(gy * W + gx) : DNET.logit(ctx.img, W, H, ctx.stride, gy, gx, bright);
    const ps = LEARN.softProbability(rec);
    if (ps !== null) {
      // a heal is turned into photo by the decision also trained on Claude's labels of soft blobs on unseen scans
      // (photo heals on his answers, scans held out: 23 -> 11 of 164 for 4 of 145 dust); a left blob is turned into
      // dust only by the one trained on his answers alone (with Claude's labels it called crowds of jersey dots and
      // sweat on skin dust: those labels' crowds are specks on plain sky, and these measurements cannot tell the two apart)
      const ph = reasons.length ? ps : LEARN.softHealProbability(rec);
      rec.learned = R.pyRoundTo(reasons.length ? ps : ph, 3);
      if (!reasons.length && ph <= p.soft_learned_lo && !(ctx.whiteAt && ctx.whiteAt[gy * W + gx])) reasons.push("photo, by the decision learned from his answers");
      // a big soft patch only when strong and not huge: on unseen scans the decision turned faint patches of
      // 2,500-11,000 px (background bokeh, a cloud, lips, a patterned dress) into dust; his big dust answers
      // are 1,044 px z 25 and 2,164 px z 65; z 26-29 patches on unseen scans were grass, a stem and a word of print
      // blobs over soft_learned_floor_area px need z soft_learned_floor2 (faint mid-size blobs it overruled on
      // unseen scans were bokeh in dark foliage, glints on water, a bokeh field; his one such answer is photo)
      else if (reasons.length && ps >= p.soft_learned_hi && peak >= (area > p.soft_learned_floor_area ? p.soft_learned_floor2 : p.soft_learned_floor)
        && reasons.every((w) => LEARN.SOFT_OVERRULES.some((o) => w.startsWith(o)))
        // ... and smooth inside (a shirt button's holes) and alone (bright foliage has many peaks around)
        && !(reasons.some((w) => w.startsWith("big soft patch")) && (area > p.soft_learned_big_area || peak < p.soft_learned_big_z
          || fine > p.soft_learned_big_fine || near > p.soft_learned_big_near))) {
        reasons.length = 0;
        learnedDust = true;
      }
    }
  }
  if (reasons.length) {
    rec.verdict = "keep-photo"; rec.why = "soft: " + reasons.join(", ");
    return { rec, R: Rm, win, fill: null };
  }
  // 4. heal out to a low fraction of its contrast (no halo), plus any dark part of the same grain
  const cap3 = 4 * Math.max(area, 30);
  const thr3 = bg + p.soft_fill * con;
  let f3, touch3 = false;
  for (;;) {
    touch3 = false;
    f3 = flood(Rg, (k) => Yg[k] >= thr3, (ly, lx, k, cnt) => {
      if (atW0(ly, lx)) { touch3 = true; return "cap"; }
      return cnt > cap3 ? "cap" : "";
    }, true);
    if (f3.stop !== "edge") break;
    const before = ww * hh;
    setWin(win.y0 - 64, win.y1 + 64, win.x0 - 64, win.x1 + 64);
    if (ww * hh === before) break;
  }
  // the window may have grown: rebuild the masks in its frame
  const n2 = ww * hh;
  const Rm2 = new Uint8Array(n2);
  for (const g of Rg) Rm2[loc(g)] = 1;
  let fillM;
  if (f3.stop || f3.n > cap3 || touch3) fillM = F.dilateMask(Rm2, ww, hh, 7);
  else { fillM = new Uint8Array(n2); for (const g of f3.px) fillM[loc(g)] = 1; }
  if (r2ok) {
    const R2b = new Uint8Array(n2);
    for (const g of objG) R2b[loc(g)] = 1;
    const d5 = F.dilateMask(R2b, ww, hh, 5);
    for (let k = 0; k < n2; k++) fillM[k] = fillM[k] | d5[k];
  }
  fillM = R.fillHoles(fillM, ww, hh);
  let fy0 = Infinity, fy1 = -1, fx0 = Infinity, fx1 = -1;
  for (let k = 0; k < n2; k++) if (fillM[k]) {
    const yy = Math.floor(k / ww), xx = k % ww;
    if (yy < fy0) fy0 = yy; if (yy > fy1) fy1 = yy; if (xx < fx0) fx0 = xx; if (xx > fx1) fx1 = xx;
  }
  Object.assign(rec, {
    verdict: "defect", why: learnedDust ? "soft debris, by the decision learned from his answers" : SOFT_WHY, x0: fx0 + win.x0, x1: fx1 + win.x0 + 1, y0: fy0 + win.y0, y1: fy1 + win.y0 + 1,
    length: Math.max(fy1 - fy0, fx1 - fx0) + 1, width: Math.min(fy1 - fy0, fx1 - fx0) + 1, white: 1.0,
  });
  return { rec, R: Rm2, win, fill: fillM };
}

// judgeSoft from soft.cpp's measurements of seed i (the same record and decision; the crowd, which depends on what is
// already healed, and the learned decision are made here as judgeSoft makes them). Returns { rec, Rg, fill } (global
// pixel lists; Rg null when the blob has no scale of its own).
function judgeSoftNative(ctx, meas, i, bright, blobs, healed, tree, pk, gy, gx, peak) {
  const { W, H, p } = ctx;
  const o = 24 * i, F_ = meas.fixed;
  const rec = { cx: gx, cy: gy, z: r1(peak), kind: "soft" };
  if (!F_[o]) {
    rec.verdict = "keep-photo"; rec.why = "soft: part of a bigger bright area";
    return { rec, Rg: null, fill: null };
  }
  const area = F_[o + 1], elong = F_[o + 2], plateau = F_[o + 3], fine = F_[o + 4], cosw = F_[o + 5], broad = F_[o + 6];
  const wy0 = F_[o + 9], wx0 = F_[o + 10], ww = F_[o + 11], hh = F_[o + 12];
  const Rg = meas.list(i, 0), objG = F_[o + 8] ? meas.list(i, 1) : Rg;
  const onSubj = nearSubject(ctx.subject, W, H, gy, gx, p.subject_margin);
  let near = 0;
  if (tree) {
    const own = new Set(objG);
    for (const j of tree.queryBox(gy, gx, 300)) {
      if (pk.vs[j] >= Math.max(6.0, 0.6 * peak) && !own.has(pk.ys[j] * W + pk.xs[j])) near++;
    }
  }
  Object.assign(rec, { area, elong: r1(elong), plateau: r2(plateau), fine: r2(fine), cosw: r3(cosw), broad: r2(broad), near });
  if (ctx.subject) rec.subject = onSubj;
  const reasons = [];
  if (onSubj && ctx.groupAt && ctx.groupAt[gy * W + gx]) reasons.push("a highlight among look-alikes on the subject (glasses, jewellery, buttons)");
  if (area > p.soft_area_max) reasons.push(`big soft patch (${area} px)`);
  if (elong > p.soft_elong_max) reasons.push("long soft highlight");
  if (plateau >= p.soft_plateau && elong <= 2.5 && area >= 150) reasons.push("flat-topped disc (bokeh)");
  if (broad > 6.0) reasons.push("far brighter than everything around (a highlight)");
  if (fine > p.soft_fine_max) reasons.push("in-focus detail beside it");
  if (fine >= p.soft_bar_fine && elong >= p.soft_bar_elong && peak >= p.soft_bar_z) reasons.push("a crisp bright bar (a highlight on metal, glass or a button)");
  if (cosw < p.soft_cos_white) reasons.push("coloured glow");
  if (broad < 0.3) reasons.push("only bright against dark detail beside it");
  if (bright && p.soft_socket > 0 && !Number.isNaN(F_[o + 7])) {
    rec.socket = R.pyRoundTo(F_[o + 7], 2);
    if (rec.socket >= p.soft_socket) reasons.push("a highlight in a dark socket (a catchlight, a glint in a hole)");
  }
  if (ctx.surePhoto && ctx.surePhoto.length) {
    let by0 = Infinity, by1 = -1, bx0 = Infinity, bx1 = -1;
    for (const g of Rg) { const yy = (g / W) | 0, xx = g - yy * W; if (yy < by0) by0 = yy; if (yy > by1) by1 = yy; if (xx < bx0) bx0 = xx; if (xx > bx1) bx1 = xx; }
    if (ctx.surePhoto.some((m) => m.cy >= by0 && m.cy <= by1 && m.cx >= bx0 && m.cx <= bx1)) reasons.push("holds a piece the learned decision is sure is picture");
  }
  if (!reasons.length) {
    const Rm = new Uint8Array(ww * hh);
    for (const g of Rg) { const yy = (g / W) | 0; Rm[(yy - wy0) * ww + (g - yy * W - wx0)] = 1; }
    const sBlob = Math.max(1.0, Math.sqrt(area / Math.PI) / 1.414);
    const [sim, stronger, row] = blobs.crowd(Rm, wy0, wx0, ww, hh, healed, sBlob);
    rec.sim = sim; rec.stronger = stronger; rec.row = row;
    const t = Math.min(1.0, Math.max(0.0, (area - 300) / 300.0));
    const simLimit = p.soft_sim_max + t * (p.soft_sim_big - p.soft_sim_max);
    const scatter = near <= p.soft_scatter_near && area <= p.soft_scatter_area;
    if ((sim > simLimit || stronger > p.soft_stronger_max) && !scatter) reasons.push(`one of many same-size blobs around it (${sim} similar, ${stronger} stronger)`);
    else if (sim >= p.soft_tex_sim && near > p.soft_tex_near) reasons.push(`among many soft highlights (leaves, bark, hair, sheen: ${near} peaks around)`);
    else if (p.soft_row && row) reasons.push("one of a row of look-alikes (stitching, beads, pattern)");
  }
  let learnedDust = false;
  if (p.learned) {
    if (p.dustnet && DNET.available()) rec.net = ctx.softNet && ctx.softNet.has(gy * W + gx) ? ctx.softNet.get(gy * W + gx) : DNET.logit(ctx.img, W, H, ctx.stride, gy, gx, bright);
    const ps = LEARN.softProbability(rec);
    if (ps !== null) {
      const ph = reasons.length ? ps : LEARN.softHealProbability(rec);
      rec.learned = R.pyRoundTo(reasons.length ? ps : ph, 3);
      if (!reasons.length && ph <= p.soft_learned_lo && !(ctx.whiteAt && ctx.whiteAt[gy * W + gx])) reasons.push("photo, by the decision learned from his answers");
      else if (reasons.length && ps >= p.soft_learned_hi && peak >= (area > p.soft_learned_floor_area ? p.soft_learned_floor2 : p.soft_learned_floor)
        && reasons.every((w) => LEARN.SOFT_OVERRULES.some((q) => w.startsWith(q)))
        && !(reasons.some((w) => w.startsWith("big soft patch")) && (area > p.soft_learned_big_area || peak < p.soft_learned_big_z
          || fine > p.soft_learned_big_fine || near > p.soft_learned_big_near))) {
        reasons.length = 0;
        learnedDust = true;
      }
    }
  }
  if (reasons.length) {
    rec.verdict = "keep-photo"; rec.why = "soft: " + reasons.join(", ");
    return { rec, Rg, fill: null };
  }
  Object.assign(rec, {
    verdict: "defect", why: learnedDust ? "soft debris, by the decision learned from his answers" : SOFT_WHY, x0: F_[o + 16], x1: F_[o + 17], y0: F_[o + 18], y1: F_[o + 19],
    length: F_[o + 20], width: F_[o + 21], white: 1.0,
  });
  return { rec, Rg, fill: meas.list(i, 2) };
}

async function softDebris(ctx, maps, zz, bright, seeds, taken) {
  const { W, H, outside, p, box } = ctx;
  const N = W * H;
  const T0 = Date.now();
  const sgn = bright ? 1 : -1;
  const zd = F.rectMorph(zz, W, H, 15, true);
  const pk = { ys: [], xs: [], vs: [] };
  for (let i = 0; i < N; i++) {
    const v = zz[i];
    if (v >= 6.0 && v === zd[i] && !outside[i] && !taken[i]) { pk.ys.push((i / W) | 0); pk.xs.push(i % W); pk.vs.push(v); }
  }
  const tree = pk.ys.length ? new R.PointGrid(pk.ys, pk.xs, 64) : null;
  const out = new Uint8Array(N), done = new Uint8Array(N);
  const recs = [];
  const [bt, bb, bl, br] = box;
  const blobs = new BlobField(ctx.Y, W, H, maps.sigma_level, sgn);
  const healed = (i) => taken[i] || out[i];
  // every blob the loop below can judge has its centre at its seed's peak: their logits in one batch
  ctx.softNet = null;
  if (p.learned && p.dustnet && DNET.available()) {
    const cs = [], seen = new Set();
    for (const [sy, sx] of seeds) {
      const y0 = Math.max(0, sy - 160), x0 = Math.max(0, sx - 160), hh0 = Math.min(H, sy + 160) - y0, ww0 = Math.min(W, sx + 160) - x0;
      const cy = sy - y0, cx = sx - x0;
      let peak = -Infinity, qy = 0, qx = 0;
      for (let y = Math.max(0, cy - 20); y < Math.min(hh0, cy + 21); y++) for (let x = Math.max(0, cx - 20); x < Math.min(ww0, cx + 21); x++) {
        const v = zz[(y0 + y) * W + x0 + x];
        if (v > peak) { peak = v; qy = y; qx = x; }
      }
      if (peak < p.soft_seed) continue;
      const gy = qy + y0, gx = qx + x0, key = gy * W + gx;
      if (!seen.has(key)) { seen.add(key); cs.push([gy, gx]); }
    }
    if (cs.length) {
      const lg = DNET.logits(ctx.img, W, H, ctx.stride, cs, bright);
      ctx.softNet = new Map(cs.map(([gy, gx], i) => [gy * W + gx, lg[i]]));
    }
  }
  // native: every seed the loop below could judge, measured at once (its peak, window and checks are its own; whether
  // it is judged, and its crowd, depend on the blobs before it and stay with the loop)
  let meas = null, measAt = null;
  const mPeak = [];
  if (NATR.available() && p.native_soft !== false && seeds.length) {
    const list = [];
    measAt = new Int32Array(seeds.length).fill(-1);
    seeds.forEach(([sy, sx], si) => {
      const y0 = Math.max(0, sy - 160), y1 = Math.min(H, sy + 160), x0 = Math.max(0, sx - 160), x1 = Math.min(W, sx + 160);
      const cy = sy - y0, cx = sx - x0, hh0 = y1 - y0, ww0 = x1 - x0;
      let peak = -Infinity, qy = 0, qx = 0;
      for (let y = Math.max(0, cy - 20); y < Math.min(hh0, cy + 21); y++) for (let x = Math.max(0, cx - 20); x < Math.min(ww0, cx + 21); x++) {
        const v = zz[(y0 + y) * W + x0 + x];
        if (v > peak) { peak = v; qy = y; qx = x; }
      }
      if (peak < p.soft_seed) return;
      const gy = qy + y0, gx = qx + x0;
      if (!(bt + p.zone_margin <= gy && gy <= bb - p.zone_margin && bl + p.zone_margin <= gx && gx <= br - p.zone_margin)) return;
      if (taken[gy * W + gx]) return;
      measAt[si] = list.length / 7;
      list.push(gy, gx, 0, y0, y1, x0, x1);
      mPeak.push(peak);
    });
    const n = list.length / 7;
    if (n) {
      const sl = Int32Array.from(list);
      meas = NATR.softBatch(ctx.Y, maps.sigma_level, ctx.img, ctx.stride, ctx.C, W, H, bright, n, sl, GK15, GK08, GK25, p.soft_fill, bright && p.soft_socket > 0);
      if (meas) meas.seeds = sl;
    }
  }
  const tPeaks = Date.now() - T0;
  let nSeed = 0, retries = 0, tJudge = 0;
  for (let si = 0; si < seeds.length; si++) {
    const [sy, sx] = seeds[si];
    if (++nSeed % 100 === 0) await ctx.tick(0.1 + 0.9 * (nSeed / seeds.length));
    if (done[sy * W + sx]) continue;
    if (meas) {
      const i = measAt[si];
      if (i < 0) continue;
      const gy = meas.seeds[7 * i], gx = meas.seeds[7 * i + 1], peak = mPeak[i];
      const tj = Date.now();
      const res = judgeSoftNative(ctx, meas, i, bright, blobs, healed, tree, pk, gy, gx, peak);
      tJudge += Date.now() - tj;
      recs.push(res.rec);
      if (!res.Rg) {
        for (let y = Math.max(0, gy - 10); y < Math.min(H, gy + 11); y++) for (let x = Math.max(0, gx - 10); x < Math.min(W, gx + 11); x++) done[y * W + x] = 1;
        continue;
      }
      for (const g of res.Rg) done[g] = 1;
      const gw = res.fill ? softGuard(p, ctx, maps.sigma_level, W, H, res.rec, res.fill.length) : "";
      if (gw) { res.rec.verdict = "keep-photo"; res.rec.why = gw; continue; }
      if (res.fill) for (const g of res.fill) out[g] = 1;
      continue;
    }
    const hw = 160;
    const W0 = { y0: Math.max(0, sy - hw), y1: Math.min(H, sy + hw), x0: Math.max(0, sx - hw), x1: Math.min(W, sx + hw) };
    const cy = sy - W0.y0, cx = sx - W0.x0, hh0 = W0.y1 - W0.y0, ww0 = W0.x1 - W0.x0;
    let peak = -Infinity, qy = 0, qx = 0;
    for (let y = Math.max(0, cy - 20); y < Math.min(hh0, cy + 21); y++) for (let x = Math.max(0, cx - 20); x < Math.min(ww0, cx + 21); x++) {
      const v = zz[(W0.y0 + y) * W + W0.x0 + x];
      if (v > peak) { peak = v; qy = y; qx = x; }
    }
    if (peak < p.soft_seed) continue;
    const gy = qy + W0.y0, gx = qx + W0.x0;
    if (!(bt + p.zone_margin <= gy && gy <= bb - p.zone_margin && bl + p.zone_margin <= gx && gx <= br - p.zone_margin)) continue;
    if (taken[gy * W + gx]) continue;
    const tj = Date.now();
    const res = judgeSoft(ctx, maps, bright, blobs, healed, tree, pk, W0, gy, gx, peak);
    tJudge += Date.now() - tj;
    recs.push(res.rec);
    if (!res.R) {
      for (let y = Math.max(0, gy - 10); y < Math.min(H, gy + 11); y++) for (let x = Math.max(0, gx - 10); x < Math.min(W, gx + 11); x++) done[y * W + x] = 1;
      continue;
    }
    const wn = res.win, ww = wn.x1 - wn.x0, hh = wn.y1 - wn.y0;
    let fillN = 0;
    if (res.fill) for (let k = 0; k < res.fill.length; k++) fillN += res.fill[k] ? 1 : 0;
    const gw = res.fill ? softGuard(p, ctx, maps.sigma_level, W, H, res.rec, fillN) : "";
    if (gw) { res.rec.verdict = "keep-photo"; res.rec.why = gw; }
    for (let y = 0; y < hh; y++) for (let x = 0; x < ww; x++) {
      const k = y * ww + x, gi = (wn.y0 + y) * W + wn.x0 + x;
      if (res.R[k]) done[gi] = 1;
      if (!gw && res.fill && res.fill[k]) out[gi] = 1;
    }
  }
  ctx.softNet = null;
  if (ctx.timing) ctx.log(`[soft] seconds: peaks ${(tPeaks / 1000).toFixed(1)} · judging ${((tJudge - (blobs.tileMs || 0)) / 1000).toFixed(1)} · blob tiles ${((blobs.tileMs || 0) / 1000).toFixed(1)} (${blobs.tiles.size}) · ${recs.length} blobs`);
  return { out, recs };
}

async function softStage(ctx, res, bright) {
  const { W, H, outside, p } = ctx;
  const N = W * H;
  const maps = res.maps;
  const zz = new Float32Array(N);
  for (let i = 0; i < N; i++) zz[i] = Math.max(maps.z_l[i], maps.z_xl[i]);
  const zd = F.rectMorph(zz, W, H, 21, true);
  const seeds = [];
  { const ss = p.soft_seed, rm = res.mask; for (let i = 0; i < N; i++) if (zz[i] >= ss && zz[i] === zd[i] && !outside[i] && !rm[i]) seeds.push([(i / W) | 0, i % W]); }
  const retry = res.marks.filter((f) => f.verdict === "keep-photo" && (f.why.startsWith("not crisp") || f.why.startsWith("in detailed")));
  for (const f of retry) seeds.push([f.cy, f.cx]);
  // sharp marks near paper white (paper black for dark dust): opaque debris, which the learned decision
  // never turns into photo
  ctx.whiteAt = new Uint8Array(N);
  for (const f of res.marks) {
    const c = f.core_y;
    if (c === undefined || c === null) continue;
    if (!(bright ? c >= p.soft_learned_white : c <= 1.0 - p.soft_learned_white)) continue;
    for (let y = Math.max(0, f.y0 - 3); y < Math.min(H, f.y1 + 3); y++) for (let x = Math.max(0, f.x0 - 3); x < Math.min(W, f.x1 + 3); x++) ctx.whiteAt[y * W + x] = 1;
  }
  // on the subject: sharp marks with look-alikes around (glasses hardware, jewellery, buttons)
  ctx.groupAt = null;
  if (ctx.subject) {
    ctx.groupAt = new Uint8Array(N);
    for (const f of res.marks) {
      if ((f.field ?? 0) < p.subject_soft_field || (f.field ?? 0) > p.subject_soft_field_max) continue;
      for (let y = Math.max(0, f.y0 - 3); y < Math.min(H, f.y1 + 3); y++) for (let x = Math.max(0, f.x0 - 3); x < Math.min(W, f.x1 + 3); x++) ctx.groupAt[y * W + x] = 1;
    }
  }
  await ctx.tick(0.05);
  // sharp pieces the learned decision is sure are picture: a soft blob holding one is left (ceramic insulators)
  ctx.surePhoto = p.soft_sure_photo ? res.marks.filter((m) => m.verdict === "keep-photo" && m.kind !== "soft" && m.learned != null && m.learned < p.line_photo_p) : null;
  const { out: smask, recs } = await softDebris(ctx, maps, zz, bright, seeds, res.mask);
  const healedSoft = recs.filter((r) => r.verdict === "defect");
  for (const r of recs) {
    if (r.verdict === "keep-photo" && (r.z || 0) >= 10 && r.x0 === undefined) {
      Object.assign(r, { id: -1, area: r.area || 0, z_fine: 0.0, field: r.sim || 0, pieces: 0, zone: false, x0: r.cx - 3, x1: r.cx + 4, y0: r.cy - 3, y1: r.cy + 4, length: 0.0, width: 0.0, white: 0.0 });
      res.marks.push(r);
    }
  }
  for (const f of retry) if (smask[f.cy * W + f.cx]) { f.verdict = "defect"; f.why = SOFT_WHY; }
  const known = new Set(retry.map((f) => f.cy * W + f.cx));
  for (const r of healedSoft) {
    const at = r.cy * W + r.cx;
    if (!smask[at] || known.has(at)) continue;
    Object.assign(r, { id: -1, area: r.area || 0, z_fine: 0.0, field: r.sim || 0, pieces: 0, zone: false });
    res.marks.push(r);
  }
  for (let i = 0; i < N; i++) if (smask[i]) res.mask[i] = 1;
  ctx.groupAt = null;
  ctx.whiteAt = null;
  ctx.log(`[${bright ? "bright" : "dark"}] soft debris: ${healedSoft.length} healed of ${recs.length} soft blobs judged`);
}


// ---------- soft fibres: pale soft lines alone on plain ground (port of _soft_lines) -----------
//
// Fibres lying out of focus on the film's base side. Every ridge (Hessian across a line, sigma 2, 3
// and 4.5, against the local grain) with a skeleton of line_len+ px is traced; a fibre holds its
// strength along its length, is soft, near neutral, on plain ground, with no other line within
// line_near_px and at most line_alone_max within line_alone_px. Photo lines come in company.
// float32 arithmetic is mirrored step by step (Math.fround) so the maps match numpy's.

const LINE_SCALES = [2.0, 3.0, 4.5];
const f32 = Math.fround;

// median of float32 values the way numpy takes it on a float32 array
function medianF32(vals) {
  const n = vals.length;
  if (!n) return NaN;
  const a = Float32Array.from(vals).sort();
  if (n % 2) return a[(n - 1) >> 1];
  return f32(f32(a[n / 2 - 1] + a[n / 2]) / 2);
}

function ridgeMaps(Y, W, H, outside, bright, sig) {
  if (NATR.big(W, H) && Y instanceof Float32Array && outside instanceof Uint8Array && sig instanceof Float32Array) {
    const r = NATR.ridgeMaps(Y, W, H, outside, bright, sig, LINE_SCALES, LINE_SCALES.map((s) => F.gaussKernel(s)));
    if (r) return r;
  }
  const N = W * H;
  const best = new Float32Array(N), bestS = new Float32Array(N);
  const tiny = f32(1e-5), c1 = f32(1.4826);
  for (const s of LINE_SCALES) {
    const g = F.gaussianBlur(Y, W, H, s, null, true);
    const xx = F.sobel(g, W, H, 2, 0), yy = F.sobel(g, W, H, 0, 2), xy = F.sobel(g, W, H, 1, 1);
    if (!bright) for (let i = 0; i < N; i++) { xx[i] = -xx[i]; yy[i] = -yy[i]; xy[i] = -xy[i]; }
    const s2 = f32(s * s);
    const samp = [];
    for (let y = 0; y < H; y += 5) for (let x = 0; x < W; x += 5) {
      const i = y * W + x;
      if (!outside[i]) samp.push(f32(f32(f32(c1 * Math.abs(xx[i])) * s2) / Math.max(sig[i], tiny)));
    }
    const k = f32(medianF32(samp));
    for (let i = 0; i < N; i++) {
      const tr = f32(xx[i] + yy[i]);
      const h = f32(tr / 2);
      const det = f32(f32(xx[i] * yy[i]) - f32(xy[i] * xy[i]));
      const disc = f32(Math.sqrt(Math.max(f32(f32(h * h) - det), 0)));
      const r = f32(Math.max(f32(-f32(h - disc) - Math.abs(f32(h + disc))), 0) * s2);
      const z = f32(r / f32(k * Math.max(sig[i], tiny)));
      if (z > best[i]) { best[i] = z; bestS[i] = s; }
    }
  }
  for (let i = 0; i < N; i++) if (outside[i]) best[i] = 0;
  return { best, bestS };
}

// How far the ridge runs on past the line's ends (px, the longer of the two): a film fibre ends
// where it ends; a twig, stem, wire or strand of hair carries on past the stretch strong enough to
// be traced. Each end is followed in 2 px steps along its own direction (skeleton points within 8 px),
// looking 2 px to either side, while the ridge holds line_cont_z; three misses in a row end it.
// The same steps as detect.py's _line_reach (plain square roots, Python rounding).
// How far a sharp-stage mark's line runs on past its ends into the picture (px): lineReach on its
// skeleton, over a ridge map of the ground around it (line_cont_max + 24 px on every side, calibrated on
// that ground). The same steps as detect.py's _mark_reach.
function markReach(ctx, sig, glab, cand, g, f, bright) {
  const { Y, W, H, outside, p } = ctx;
  const y0 = Math.max(0, f.y0 - 2), y1 = Math.min(H, f.y1 + 3), x0 = Math.max(0, f.x0 - 2), x1 = Math.min(W, f.x1 + 3);
  const w = x1 - x0, h = y1 - y0, own = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const gi = (y0 + y) * W + x0 + x; own[y * w + x] = glab[gi] === g && cand[gi] ? 1 : 0; }
  const sk = R.skeletonize(own, w, h);
  const ys = [], xs = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (sk[y * w + x]) { ys.push(y); xs.push(x); }
  if (ys.length < 2) return 0;
  const r = Math.trunc(p.line_cont_max) + 24;
  const ya = Math.max(0, y0 - r), yb = Math.min(H, y1 + r), xa = Math.max(0, x0 - r), xb = Math.min(W, x1 + r);
  const cw = xb - xa, ch = yb - ya;
  const Yc = new Float32Array(cw * ch), oc = new Uint8Array(cw * ch), sc = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) { const i = (ya + y) * W + xa + x, k = y * cw + x; Yc[k] = Y[i]; oc[k] = outside[i]; sc[k] = sig[i]; }
  const { best } = ridgeMaps(Yc, cw, ch, oc, bright, sc);
  return R.pyRoundTo(lineReach(best, cw, ch, ys, xs, y0 - ya, x0 - xa, p), 1);
}

function lineReach(best, W, H, ys, xs, y0, x0, p) {
  const L = ys.length, S = new Set();
  for (let k = 0; k < L; k++) S.add(ys[k] * 65536 + xs[k]);
  let ends = [];
  for (let k = 0; k < L; k++) {
    let nb = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if ((dy || dx) && S.has((ys[k] + dy) * 65536 + xs[k] + dx)) nb++;
    if (nb === 1) ends.push([ys[k], xs[k]]);
  }
  if (ends.length < 2) { ends = []; for (let k = 0; k < L; k++) ends.push([ys[k], xs[k]]); }
  let bestD = -1, ea = ends[0], eb = ends[ends.length - 1];
  for (let i = 0; i < ends.length; i++) for (let j = i + 1; j < ends.length; j++) {
    const dy = ends[i][0] - ends[j][0], dx = ends[i][1] - ends[j][1], d = dy * dy + dx * dx;
    if (d > bestD) { bestD = d; ea = ends[i]; eb = ends[j]; }
  }
  let reach = 0.0;
  for (const [ey, ex] of [ea, eb]) {
    let sy = 0, sx = 0, nn = 0;
    for (let k = 0; k < L; k++) if (Math.max(Math.abs(ys[k] - ey), Math.abs(xs[k] - ex)) <= 8) { sy += ys[k]; sx += xs[k]; nn++; }
    let ty = ey - sy / nn, tx = ex - sx / nn;
    const n = Math.sqrt(ty * ty + tx * tx);
    if (n < 1e-6) continue;
    ty /= n; tx /= n;
    let py = ey + y0, px = ex + x0, got = 0.0, travelled = 0.0, miss = 0;
    while (travelled < p.line_cont_max) {
      const qy = py + 2.0 * ty, qx = px + 2.0 * tx, ny = -tx, nx = ty;
      let bv = -1.0, bk = 0;
      for (let k = -2; k <= 2; k++) {
        const yy = pyRound(qy + k * ny), xx = pyRound(qx + k * nx);
        const v = yy >= 0 && yy < H && xx >= 0 && xx < W ? best[yy * W + xx] : 0.0;
        if (v > bv) { bv = v; bk = k; }
      }
      travelled += 2.0;
      if (bv >= p.line_cont_z) {
        const ry = qy + bk * ny, rx = qx + bk * nx, dy = ry - py, dx = rx - px;
        const dn = Math.sqrt(dy * dy + dx * dx);
        ty = 0.7 * ty + 0.3 * dy / dn; tx = 0.7 * tx + 0.3 * dx / dn;
        const tn = Math.sqrt(ty * ty + tx * tx);
        ty /= tn; tx /= tn;
        py = ry; px = rx; got = travelled; miss = 0;
      } else {
        py = qy; px = qx; miss++;
        if (miss >= 3) break;
      }
    }
    reach = Math.max(reach, got);
  }
  return reach;
}

// How far the line bows from the chord between its two farthest points, as a share of that chord
// (0 = ruler-straight); integer arithmetic until the last division, as in detect.py's _line_bow
function lineBow(ys, xs) {
  const n = ys.length;
  let bestD = -1, ia = 0, ib = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const dy = ys[i] - ys[j], dx = xs[i] - xs[j], d = dy * dy + dx * dx;
    if (d > bestD) { bestD = d; ia = i; ib = j; }
  }
  if (bestD <= 0) return 1.0;
  const ay = ys[ia], ax = xs[ia], dy = ys[ib] - ay, dx = xs[ib] - ax;
  let dev = 0;
  for (let k = 0; k < n; k++) dev = Math.max(dev, Math.abs(dy * (xs[k] - ax) - dx * (ys[k] - ay)));
  return dev / bestD;
}

// main axis of the skeleton points near one point: closed form, as in detect.py's _main_axis
function mainAxis(a, b, c) {
  if (b === 0) return a > c ? [1.0, 0.0] : [0.0, 1.0];
  const lam = (a + c) / 2 + Math.sqrt(((a - c) / 2) ** 2 + b * b);
  const vy = b, vx = lam - a, n = Math.hypot(vy, vx);
  return [vy / n, vx / n];
}

// how evenly the line stands above the ground on both sides (fibre ~1, outline rim ~0 or below)
function twoSided(yb, W, H, ys, xs, y0, x0, scale, bright) {
  const d = 2 * scale + 3;
  const vals = [];
  for (let k = 0; k < ys.length; k++) {
    let n = 0, sy = 0, sx = 0;
    const idx = [];
    for (let j = 0; j < ys.length; j++) {
      if (Math.max(Math.abs(ys[j] - ys[k]), Math.abs(xs[j] - xs[k])) <= 5) { idx.push(j); sy += ys[j]; sx += xs[j]; n++; }
    }
    if (n < 3) continue;
    const my = sy / n, mx = sx / n;
    let a = 0, b = 0, c = 0;
    for (const j of idx) { const dy = ys[j] - my, dx = xs[j] - mx; a += dy * dy; b += dy * dx; c += dx * dx; }
    const [ty, tx] = mainAxis(a, b, c);
    const ny = -tx, nx = ty;
    const py = ys[k] + y0, px = xs[k] + x0;
    const ay = pyRound(py + d * ny), ax = pyRound(px + d * nx), by = pyRound(py - d * ny), bx = pyRound(px - d * nx);
    if (!(ay >= 0 && ay < H && ax >= 0 && ax < W && by >= 0 && by < H && bx >= 0 && bx < W)) continue;
    const core = yb[py * W + px];
    let ca = core - yb[ay * W + ax], cb = core - yb[by * W + bx];
    if (!bright) { ca = -ca; cb = -cb; }
    const hi = Math.max(ca, cb);
    vals.push(hi <= 1e-6 ? -1.0 : Math.min(ca, cb) / hi);
  }
  return vals.length ? R.medianOf(Float64Array.from(vals), vals.length) : -1.0;
}

function softLines(ctx, bright, sig, taken, sharp = []) {
  const { W, H, Y, outside, p, box, img, C } = ctx;
  // sharp-stage pieces the learned decision is sure are picture (a printed logo, a lit roof edge)
  const sure = sharp.filter((m) => m.verdict === "keep-photo" && m.kind !== "soft" && m.learned != null && m.learned < p.line_photo_p);
  const out = new Uint8Array(W * H);
  const marks = [], tinted = [];
  const { best, bestS } = ridgeMaps(Y, W, H, outside, bright, sig);
  const lz = f32(p.line_z);
  const on = new Uint8Array(W * H);
  for (let i = 0; i < on.length; i++) on[i] = best[i] >= lz ? 1 : 0;
  const { lab, n } = R.label(on, W, H, 8);
  const boxes = R.findObjects(lab, W, H, n);
  const [bt, bb, bl, br] = box, m_ = p.zone_margin;
  const lines = [];
  // lab only (p.line_debug: boxes [x0, y0, x1, y1]): why each soft line inside one was left
  const dbgBoxes = Array.isArray(p.line_debug) ? p.line_debug : null;
  const inBox = (x, y) => dbgBoxes && dbgBoxes.some(([a, b, c2, d]) => x >= a && x <= c2 && y >= b && y <= d);
  const dbg = (j, why) => { const c = lines[j]; if (inBox(c.cx, c.cy)) ctx.log(`[line] ${c.cx},${c.cy} L${c.L} (${c.x0 + Math.min(...c.xs)}-${c.x0 + Math.max(...c.xs)}, ${c.y0 + Math.min(...c.ys)}-${c.y0 + Math.max(...c.ys)}): left: ${why}`); };
  for (let i = 0; i < n; i++) {
    const o = boxes[i];
    if (o && dbgBoxes && inBox((o.x0 + o.x1) >> 1, (o.y0 + o.y1) >> 1) && Math.max(o.y1 - o.y0, o.x1 - o.x0) < p.line_len) ctx.log(`[line] ${(o.x0 + o.x1) >> 1},${(o.y0 + o.y1) >> 1}: left: shorter than ${p.line_len} px (${o.x1 - o.x0} x ${o.y1 - o.y0})`);
    if (!o || Math.max(o.y1 - o.y0, o.x1 - o.x0) < p.line_len) continue;
    const y0 = Math.max(0, o.y0 - 40), y1 = Math.min(H, o.y1 + 40), x0 = Math.max(0, o.x0 - 40), x1 = Math.min(W, o.x1 + 40);
    const w = x1 - x0, h = y1 - y0;
    const own = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) own[y * w + x] = lab[(y0 + y) * W + x0 + x] === i + 1 ? 1 : 0;
    const sk = R.skeletonize(own, w, h);
    const ys = [], xs = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (sk[y * w + x]) { ys.push(y); xs.push(x); }
    const L = ys.length;
    if (L < p.line_len) { if (inBox((o.x0 + o.x1) >> 1, (o.y0 + o.y1) >> 1)) ctx.log(`[line] ${(o.x0 + o.x1) >> 1},${(o.y0 + o.y1) >> 1}: left: skeleton ${L} px`); continue; }
    let sx = 0, sy = 0;
    for (let k = 0; k < L; k++) { sx += xs[k]; sy += ys[k]; }
    lines.push({ y0, y1, x0, x1, w, h, own, sk, ys, xs, L, cx: Math.trunc(sx / L + x0), cy: Math.trunc(sy / L + y0) });
  }
  if (!lines.length) return { mask: out, marks, tinted };
  const company = (j, Rr) => {
    let c = -1;
    for (const q of lines) { const dy = q.cy - lines[j].cy, dx = q.cx - lines[j].cx; if (dy * dy + dx * dx <= Rr * Rr) c++; }
    return c;
  };
  const yb = F.gaussianBlur(Y, W, H, 1.5);
  const gx = F.sobel(yb, W, H, 1, 0), gy = F.sobel(yb, W, H, 0, 1);
  const taken3 = F.dilateMask(taken, W, H, 7);
  const nC = C === 1 ? 1 : 3;
  const rgb = (y, x, c) => f32(img[(y * W + x) * C + c] / 65535);
  for (let j = 0; j < lines.length; j++) {
    if (company(j, p.line_near_px) > 0 || company(j, p.line_alone_px) > p.line_alone_max) { if (dbgBoxes) dbg(j, `company ${company(j, p.line_near_px)} within ${p.line_near_px} px, ${company(j, p.line_alone_px)} within ${p.line_alone_px}`); continue; }
    const c = lines[j];
    const { y0, x0, w, h, ys, xs, sk, own, L } = c;
    if (L > p.line_max_len) { if (dbgBoxes) dbg(j, "too long"); continue; }
    let ymin = Infinity, ymax = -Infinity, xmin = Infinity, xmax = -Infinity;
    for (let k = 0; k < L; k++) { ymin = Math.min(ymin, ys[k]); ymax = Math.max(ymax, ys[k]); xmin = Math.min(xmin, xs[k]); xmax = Math.max(xmax, xs[k]); }
    if (ymin + y0 < bt + m_ || ymax + y0 > bb - m_ || xmin + x0 < bl + m_ || xmax + x0 > br - m_) { if (dbgBoxes) dbg(j, "at the picture's edge"); continue; }
    const zv = [], sv = [], gv = [];
    for (let k = 0; k < L; k++) { const gi = (ys[k] + y0) * W + xs[k] + x0; zv.push(best[gi]); sv.push(bestS[gi]); gv.push(sig[gi]); }
    const zmed = R.pyRoundTo(medianF32(zv), 1), scale = R.pyRoundTo(medianF32(sv), 1);
    if (zmed < p.line_zmed || scale < p.line_scale) { if (dbgBoxes) dbg(j, `z ${zmed} (needs ${p.line_zmed}) scale ${scale} (needs ${p.line_scale})`); continue; }
    // the ground: a ring beyond the line's own soft halo (2.5 x its scale), 10 px wide
    const din = pyRound(2.5 * scale) + 2;
    const dOut = F.dilateMask(own, w, h, 2 * (din + 10) + 1), dIn = F.dilateMask(own, w, h, 2 * din + 1);
    const ringIdx = [];
    for (let k = 0; k < w * h; k++) if (dOut[k] && !dIn[k]) ringIdx.push(k);
    if (!ringIdx.length) continue;
    let white = 1.0;
    const diff = [], ground = [];
    if (C >= 3) {
      for (let ch = 0; ch < nC; ch++) {
        let m = 0;
        for (let k = 0; k < L; k++) m += rgb(ys[k] + y0, xs[k] + x0, ch);
        m /= L;
        const rv = ringIdx.map((k) => rgb(y0 + Math.floor(k / w), x0 + (k % w), ch));
        ground.push(medianF32(rv));
        diff.push(m - ground[ch]);
      }
      const dd = bright ? diff : diff.map((v) => -v);
      const mx = Math.max(...dd), mn = Math.min(...dd);
      white = mx > 1e-4 ? R.pyRoundTo(mn / mx, 2) : 0.0;
    }
    const sg = medianF32(gv);
    let gm = 0;
    for (const k of ringIdx) { const gi = (y0 + Math.floor(k / w)) * W + x0 + (k % w); gm += f32(Math.hypot(gx[gi], gy[gi])); }
    const busy = R.pyRoundTo(gm / ringIdx.length / Math.max(sg, 1e-5), 2);
    // a tinted line is the picture, unless the tint is the scan's own dust colour (line_colour: the pipeline decides
    // once the discs have given that colour; such a line must pass every other test here)
    const isTinted = white < p.line_white;
    if ((isTinted && !(p.line_colour && bright)) || busy > p.line_busy) { if (dbgBoxes) dbg(j, `white ${white} busy ${busy}`); continue; }
    const ev = ringIdx.map((k) => ctx.E25[(y0 + Math.floor(k / w)) * W + x0 + (k % w)]);
    const detail = R.pyRoundTo(medianF32(ev) / (ctx.gradCal * sg * sg), 2);
    if (detail > p.line_detail_max) { if (dbgBoxes) dbg(j, `detail ${detail}`); continue; }          // on detailed ground: bark, foliage, fabric texture
    // busy and detailed together: blurred pebbles on pavement, grass blades, a gap between two jeans legs
    // (unseen scans, 10-18); his soft fibres lie on plain ground
    if (busy > p.line_busy_joint && detail > p.line_detail_joint) { if (dbgBoxes) dbg(j, `busy ${busy} and detail ${detail}`); continue; }
    let tk = 0;
    for (let k = 0; k < L; k++) if (taken3[(ys[k] + y0) * W + xs[k] + x0]) tk++;
    if (tk / L > p.line_taken_max) { if (dbgBoxes) dbg(j, `taken ${(tk / L).toFixed(2)}`); continue; }
    const two = R.pyRoundTo(twoSided(yb, W, H, ys, xs, y0, x0, scale, bright), 2);
    if (two < p.line_two) { if (dbgBoxes) dbg(j, `two-sided ${two} (busy ${busy} detail ${detail} white ${white} z ${zmed} scale ${scale})`); continue; }
    if (nearSubject(ctx.subject, W, H, c.cy, c.cx, p.subject_margin)) { if (dbgBoxes) dbg(j, "near the subject"); continue; }   // hair, cords, seams on the subject
    const reach = lineReach(best, W, H, ys, xs, y0, x0, p);
    if (reach >= p.line_reach_px) { if (dbgBoxes) dbg(j, `reach ${reach}`); continue; }          // runs on past its end: twig, stem or strand
    const bow = R.pyRoundTo(lineBow(ys, xs), 3);
    if (L >= p.line_straight_len && bow <= p.line_straight_max) { if (dbgBoxes) dbg(j, `straight (bow ${bow})`); continue; }
    // nearly straight on textured ground: grass blades, twigs, a lit edge, a reflection's streak, bokeh (on 70 scans:
    // 40 of 109 soft lines Claude or the eye called picture, none of 56 called dust; also turning down lines that run on
    // past an end took a real fibre with a curled tail on u_1305)
    if (p.line_bent > 0 && bow < p.line_bent && (detail > p.line_bent_detail || busy > p.line_bent_busy)) { if (dbgBoxes) dbg(j, `nearly straight (bow ${bow}) on detail ${detail} busy ${busy} reach ${reach}`); continue; }   // ruler-straight: twig, stem, wire
    if (sure.some((m) => m.cx >= xmin + x0 && m.cx <= xmax + x0 && m.cy >= ymin + y0 && m.cy <= ymax + y0)) { if (dbgBoxes) dbg(j, "holds a sure picture piece"); continue; }   // holds a piece that is surely picture
    const rad = pyRound(2 * scale);
    // fine-scale edges along the line (grain = about 1): a fibre out of focus has none, print and logos do
    const fv = [];
    for (let k = 0; k < L; k++) for (let dy = -rad; dy <= rad; dy += 2) for (let dx = -rad; dx <= rad; dx += 2) {
      const yy = ys[k] + y0 + dy, xx = xs[k] + x0 + dx;
      if (yy >= 0 && yy < H && xx >= 0 && xx < W) fv.push(ctx.E1[yy * W + xx]);
    }
    fv.sort((a, b) => a - b);
    const edge = fv.length ? R.pyRoundTo(fv[Math.floor(0.9 * (fv.length - 1))] / (ctx.gradCal * sg * sg), 2) : 0;
    if (edge > p.line_edge_max) { if (dbgBoxes) dbg(j, `crisp (edge ${edge})`); continue; }   // crisp: in focus, so not on the film's base side (print, a logo, a glint)
    const gres = resGuard(p, W, H, xmin + x0, ymin + y0, xmax + x0, ymax + y0, L * (2 * pyRound(2 * scale) + 1), "line");
    if (gres) { if (dbgBoxes) dbg(j, gres); continue; }
    if (dbgBoxes) dbg(j, `${isTinted ? "TINTED, handed on" : "TAKEN"} (busy ${busy} detail ${detail} bow ${bow} reach ${reach} two ${two} edge ${edge} white ${white})`);
    const band = F.dilateMask(sk, w, h, 2 * rad + 1, true);
    if (isTinted) {
      // healed in the pipeline if its tint is the scan's dust colour: its band, its lift and ground per channel
      let area = 0;
      for (let k = 0; k < band.length; k++) area += band[k];
      tinted.push({ band, x0, y0, w, h, lift: bright ? diff : diff.map((v) => -v), ground,
        mark: { id: -2, x0: xmin + x0, y0: ymin + y0, x1: xmax + x0, y1: ymax + y0, cx: c.cx, cy: c.cy, area, width: 2 * rad + 1,
          length: L, z: zmed, z_fine: 0.0, scale, white, busy, two_sided: two, reach, bow, detail, edge, pieces: 0, field: 0, zone: false, kind: "hair/fibre",
          verdict: "keep-photo", why: "a soft line, tinted" } });
      continue;
    }
    let area = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (band[y * w + x]) { out[(y0 + y) * W + x0 + x] = 1; area++; }
    marks.push({ id: -2, x0: xmin + x0, y0: ymin + y0, x1: xmax + x0, y1: ymax + y0, cx: c.cx, cy: c.cy, area, width: 2 * rad + 1,
      length: L, z: zmed, z_fine: 0.0, scale, white, busy, two_sided: two, reach, bow, detail, edge, pieces: 0, field: 0, zone: false, kind: "hair/fibre", verdict: "defect",
      why: "soft fibre: a pale line alone on plain ground, out of focus on the film" });
  }
  ctx.log(`[${bright ? "bright" : "dark"}] soft fibres: ${marks.length} of ${lines.length} soft lines${tinted.length ? `, ${tinted.length} tinted handed on` : ""}`);
  return { mask: out, marks, tinted };
}

// ---------- the whole detector ----------------------------------------------------------------

// img: Uint16Array (H*W*C, interleaved, 0..65535). Returns { mask (0/1, dilated), core, marks,
// outside, box, polarities }.
async function detect(img, W, H, C, opts = {}, hooks = {}) {
  const p = M.params(opts);
  const log = hooks.log || (() => {});
  // stage(name, fraction) reports progress; tick() lets the host breathe (and throws when cancelled)
  let stageName = "grain";
  const stage = (name) => { stageName = name; if (hooks.stage) hooks.stage(name, 0); };
  const tick = async (f) => { if (hooks.stage) hooks.stage(stageName, f || 0); if (hooks.tick) await hooks.tick(); };
  const N = W * H;
  const rgbAt = C === 1 ? (y, x) => img[y * W + x] / 65535 : (y, x, c) => img[(y * W + x) * C + c] / 65535;
  const nC = C === 1 ? 1 : Math.min(C, 3);
  stage("grain");
  let tg = Date.now();
  const gt = [];
  const glap = (name) => { const now = Date.now(); gt.push(name + " " + ((now - tg) / 1000).toFixed(1)); tg = now; };
  const Y = M.lumaFromU16(img, W, H, C);
  const outside = p.exclude_outside ? M.outsideMask(Y, W, H) : new Uint8Array(N);
  glap("luma+outside");
  await tick(0.15);
  const yb1 = F.gaussianBlur(Y, W, H, 1.0);
  const [gyN, gxN] = F.npGradient(yb1, W, H);
  if (!NATR.hypotInto(gyN, gxN)) for (let i = 0; i < N; i++) gyN[i] = Math.hypot(gxN[i], gyN[i]);
  const G = gyN;
  const E1 = energy(yb1, W, H);
  glap("blur1+gradients");
  await tick(0.3);
  const cache = {};
  cache.m15 = M.median8f(Y, W, H, 15);
  glap("m15");
  await tick(0.5);
  const valid = new Uint8Array(N);
  for (let i = 0; i < N; i++) valid[i] = outside[i] ? 0 : 1;
  cache.resid = new Float32Array(N);
  for (let i = 0; i < N; i++) cache.resid[i] = Y[i] - cache.m15[i];
  cache.sigLocal = M.noiseLocal(cache.resid, valid, W, H);
  // how much Sobel gradient energy plain grain makes per sigma^2 (for the strand-texture test)
  const cal = [];
  for (let y = 0; y < H; y += 7) for (let x = 0; x < W; x += 7) {
    const i = y * W + x;
    if (outside[i]) continue;
    const s = Math.max(cache.sigLocal[i], 1e-4);
    cal.push(E1[i] / (s * s));
  }
  const gradCal = R.median(cal);
  glap("local noise");
  await tick(0.7);
  const E25 = energy(F.gaussianBlur(Y, W, H, 2.5), W, H);
  await tick(0.85);
  const G3 = F.gaussianBlur(Y, W, H, 3.0);
  glap("blur2.5+3");
  const box = p.zone_guard ? M.imageBox(Y, W, H) : [0, H, 0, W];
  if (p.frame_guard) box.frame = M.frameEdges(Y, W, H);
  glap("box");
  if (hooks.timing) log(`[grain] seconds: ${gt.join(" · ")}`);
  // the subject (Photoshop's Select Subject), only when it has the scan's size and something in it
  let subject = opts.subject && opts.subject.length === N ? opts.subject : null;
  if (subject) { let any = false; for (let i = 0; i < N; i++) if (subject[i]) { any = true; break; } if (!any) subject = null; }
  const ctx = { img, C: nC, stride: C, Y, W, H, G, outside, p, gradCal, box, cache, rgbAt, E1, E25, G3, log, tick, timing: !!hooks.timing, protect: opts.protect || null, subject };
  const pols = p.polarity === "bright" ? [true] : p.polarity === "dark" ? [false] : [true, false];
  const results = new Map();
  // auto: dark wins only with need(nb) strong isolated marks, nb being bright's: max(15, 2 nb) on scans of 24 MP and
  // up; on smaller scans fewer strong marks exist at all, so the 15 shrinks with the scan's size (never below 5) and
  // dark must also beat bright by 3 (slides of 6 MP showed 7-25 strong dark marks and 0-4 bright; negatives at most
  // 4 dark with no bright). Judged by the original scan's size, like the small-scan rules.
  const scanMpPol = Math.max((W * H) / 1e6, p.res_scan_mp || 0);
  const darkNeedOf = (nb) => Math.max(Math.max(p.polarity_need_min, p.polarity_need * Math.min(1, scanMpPol / p.polarity_need_mp)), 2 * nb, scanMpPol < p.polarity_need_mp ? nb + p.polarity_lead : 0);
  // too few strong bright marks: dark is judged in full, its count of marks may be needed below
  const weakOf = (nb) => p.polarity_weak > 0 && nb < p.polarity_weak;
  for (const b of pols) {
    stage(b ? "white" : "black");
    // find() stops judging dark early when it can't reach the need
    ctx.darkNeed = !b && p.polarity === "auto" && p.polarity_stop !== false && results.has(true) && !weakOf(results.get(true).strong) ? darkNeedOf(results.get(true).strong) : 0;
    ctx.lastMaps = b === pols[pols.length - 1];
    results.set(b, await find(ctx, b));
    ctx.darkNeed = 0;
  }
  ctx.lastMaps = false;
  let use;
  if (p.polarity === "auto" && results.size === 2) {
    const nb = results.get(true).strong, nd = results.get(false).strong;
    let more = "";
    if (nd >= darkNeedOf(nb)) use = [false];
    else if (!weakOf(nb)) use = [true];
    else {
      // too few strong marks either way: the side whose judging found clearly more marks, or neither
      const hb = results.get(true).n, hd = results.get(false).n;
      const kb = hb >= p.polarity_many ? p.polarity_ratio_many : p.polarity_ratio;
      use = hb >= p.polarity_min && hb + 1 >= kb * (hd + 1) ? [true] : hd >= p.polarity_min && hd + 1 >= p.polarity_ratio * (hb + 1) ? [false] : [];
      more = `; marks found bright=${hb} dark=${hd}`;
    }
    log(`auto polarity: strong isolated bright=${nb} dark${results.get(false).stopped ? " at most" : ""}=${nd}${more} -> ${use.length ? use.map((b) => (b ? "bright" : "dark")) : "can't tell, nothing healed"}`);
  } else use = [...results.keys()];
  for (const b of results.keys()) if (!use.includes(b)) results.get(b).maps = null;
  if (p.soft_path) {
    stage("soft");
    for (const b of use) await softStage(ctx, results.get(b), b);
  }
  const core = new Uint8Array(N);
  for (const b of use) { const m = results.get(b).mask; for (let i = 0; i < N; i++) if (m[i]) core[i] = 1; }
  if (p.partners && use.length === 1 && results.has(!use[0])) {
    const other = results.get(!use[0]).cand;
    let added = 0;
    for (const f of results.get(use[0]).marks) {
      if (f.verdict !== "defect" || (f.z || 0) < 20 || (f.white || 0) < 0.5 || f.zone) continue;
      const e = 30;
      const y0 = Math.max(0, f.y0 - e), y1 = Math.min(H, f.y1 + e), x0 = Math.max(0, f.x0 - e), x1 = Math.min(W, f.x1 + e);
      const w = x1 - x0, h = y1 - y0;
      const own = new Uint8Array(w * h), oth = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const gi = (y0 + y) * W + x0 + x; own[y * w + x] = core[gi]; oth[y * w + x] = other[gi]; }
      const nOwn = K.count(own);
      if (nOwn < 10) continue;
      const touch = F.dilateMask(own, w, h, 9);
      const { lab, n } = R.label(oth, w, h, 4);
      if (!n) continue;
      const ids = new Set();
      for (let k = 0; k < lab.length; k++) if (touch[k] && lab[k]) ids.add(lab[k]);
      for (const i of [...ids].sort((a, b) => a - b)) {
        const comp = new Uint8Array(w * h);
        for (let k = 0; k < lab.length; k++) comp[k] = lab[k] === i ? 1 : 0;
        if (touchesBorder(comp, w, h)) continue;       // runs out of the window: part of the photo
        const nc = K.count(comp);
        if (nc > 1.5 * nOwn + 200) continue;
        const d = F.dilateMask(comp, w, h, 3);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (d[y * w + x]) core[(y0 + y) * W + x0 + x] = 1;
        f.partner_px = (f.partner_px || 0) + nc;
        added++;
      }
    }
    if (added) log(`mixed-polarity grains: ${added} dark parts healed with their bright grain`);
  }
  // fibres out of focus: after everything else (lines already healed are skipped); not in a second
  // pass (healing pass-1 hairs would leave photo lines looking alone)
  const tintedLines = [];   // soft lines left only for their tint (the pipeline judges their colour)
  if (p.soft_lines && !p.second_pass) {
    stage("soft");
    for (const b of use) {
      const sl = softLines(ctx, b, results.get(b).maps.sigma, core, results.get(b).marks);
      for (let i = 0; i < N; i++) if (sl.mask[i]) core[i] = 1;
      for (const f of sl.marks) results.get(b).marks.push(f);
      if (sl.tinted && sl.tinted.length) tintedLines.push(...sl.tinted);
    }
  }
  const marks = [];
  for (const b of use) for (const f of results.get(b).marks) { f.polarity = b ? "bright" : "dark"; marks.push(f); }
  const mask = p.dilate > 0 ? F.dilateMask(core, W, H, 2 * p.dilate + 1, true) : core;
  return { mask, core, marks, outside, box, polarities: use.map((b) => (b ? "bright" : "dark")), tintedLines };
}

module.exports = { detect, find, softStage, softDebris, traceRidge, BlobField, drawLine, resGuard, _ridgeMaps: ridgeMaps };

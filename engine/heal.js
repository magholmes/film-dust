"use strict";
// Grain-matched healing (port of heal.py). Works in place on a 16-bit interleaved image
// (Uint16Array, H*W*C, 0..65535). For each mark, or each 40 px stretch of a long one:
//   1. try a ring of nearby source offsets, keep the clean one whose surroundings match best;
//   2. copy that texture in (real grain from the same frame);
//   3. solve a membrane (Poisson) correction so the copy meets the surroundings exactly.
// Pixels outside the mask are never changed.
const F = require("./filters");
const R = require("./regions");

const CHUNK = 40;
const RING = 4;

function offsets(thick, size) {
  const base = Math.max(thick, 2.0);
  const ds = [base + 5, 1.6 * base + 7, 2.4 * base + 9, Math.min(size, 60) + 6, 3.5 * base + 12, 5 * base + 16];
  const dists = [...new Set(ds.map((d) => pyRound(d)))].sort((a, b) => a - b);
  const seen = new Set(), out = [];
  for (const d of dists) {
    for (let k = 0; k < 24; k++) {
      const a = (2 * Math.PI * k) / 24;
      const o = [pyRound(d * Math.sin(a)), pyRound(d * Math.cos(a))];
      const key = o[0] * 100000 + o[1];
      if (!seen.has(key)) { seen.add(key); out.push(o); }
    }
  }
  return out;
}

// Python's round(): half to even
function pyRound(v) {
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// seeded PRNG (mulberry32) + Box-Muller for the rare smooth fallback
function makeRng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return { normal() { const u = Math.max(next(), 1e-12), v = next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); } };
}

// Membrane solve on `unknown` (local window ww x wh): Laplace(D)=0, D = target-guess on known
// neighbours, zero flux next to frozen pixels. Conjugate gradient; returns values per unknown.
function poisson(known, unknown, guess, target, ww, wh, C) {
  const idx = new Int32Array(ww * wh).fill(-1);
  const pos = [];
  for (let i = 0; i < unknown.length; i++) if (unknown[i]) { idx[i] = pos.length; pos.push(i); }
  const n = pos.length;
  const diag = new Float64Array(n);
  const nbr = new Int32Array(n * 4).fill(-1);
  const rhs = new Float64Array(n * C);
  for (let j = 0; j < n; j++) {
    const i = pos[j], x = i % ww, y = (i - x) / ww;
    const cand = [[y - 1, x], [y + 1, x], [y, x - 1], [y, x + 1]];
    for (let k = 0; k < 4; k++) {
      const [ny, nx] = cand[k];
      if (ny < 0 || ny >= wh || nx < 0 || nx >= ww) continue;
      const q = ny * ww + nx;
      if (unknown[q]) { nbr[j * 4 + k] = idx[q]; diag[j] += 1; }
      else if (known[q]) {
        diag[j] += 1;
        for (let c = 0; c < C; c++) rhs[j * C + c] += target[q * C + c] - guess[q * C + c];
      }
    }
    if (diag[j] === 0) diag[j] = 1;
  }
  const out = new Float64Array(n * C);
  const x = new Float64Array(n), r = new Float64Array(n), p = new Float64Array(n), Ap = new Float64Array(n), z = new Float64Array(n);
  const mul = (v, res) => {
    for (let j = 0; j < n; j++) {
      let s = (diag[j] + 1e-9) * v[j];
      for (let k = 0; k < 4; k++) { const m = nbr[j * 4 + k]; if (m >= 0) s -= v[m]; }
      res[j] = s;
    }
  };
  for (let c = 0; c < C; c++) {
    x.fill(0);
    for (let j = 0; j < n; j++) r[j] = rhs[j * C + c];
    for (let j = 0; j < n; j++) z[j] = r[j] / diag[j];          // Jacobi preconditioner
    p.set(z);
    let rz = 0;
    for (let j = 0; j < n; j++) rz += r[j] * z[j];
    const r0 = Math.sqrt(r.reduce((s, v) => s + v * v, 0)) || 1;
    for (let it = 0; it < 2000; it++) {
      mul(p, Ap);
      let pAp = 0;
      for (let j = 0; j < n; j++) pAp += p[j] * Ap[j];
      if (pAp <= 0) break;
      const alpha = rz / pAp;
      let rr = 0;
      for (let j = 0; j < n; j++) { x[j] += alpha * p[j]; r[j] -= alpha * Ap[j]; rr += r[j] * r[j]; }
      if (Math.sqrt(rr) < 1e-7 * r0 || Math.sqrt(rr) < 1e-10) break;
      let rz2 = 0;
      for (let j = 0; j < n; j++) { z[j] = r[j] / diag[j]; rz2 += r[j] * z[j]; }
      const beta = rz2 / rz;
      rz = rz2;
      for (let j = 0; j < n; j++) p[j] = z[j] + beta * p[j];
    }
    for (let j = 0; j < n; j++) out[j * C + c] = guess[pos[j] * C + c] + x[j];
  }
  return { pos, values: out };
}

// Heal one region (mask `part` of size ph x pw placed at ry0, rx0) in place.
function healRegion(img, W, H, C, validSrc, todo, ry0, rx0, part, ph, pw, thick, rng) {
  const offs = offsets(thick, Math.max(ph, pw));
  let reach = 0;
  for (const [a, b] of offs) reach = Math.max(reach, Math.abs(a), Math.abs(b));
  reach += RING + 2;
  const y0 = Math.max(0, ry0 - reach), y1 = Math.min(H, ry0 + ph + reach);
  const x0 = Math.max(0, rx0 - reach), x1 = Math.min(W, rx0 + pw + reach);
  const wh = y1 - y0, ww = x1 - x0;
  // window copy in float (0..1), interleaved
  const win = new Float32Array(wh * ww * C);
  for (let y = 0; y < wh; y++) {
    const src = ((y0 + y) * W + x0) * C, dst = y * ww * C;
    for (let i = 0; i < ww * C; i++) win[dst + i] = img[src + i] / 65535;
  }
  const reg = new Uint8Array(wh * ww), unfilled = new Uint8Array(wh * ww), vs = new Uint8Array(wh * ww);
  for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) if (part[y * pw + x]) reg[(ry0 - y0 + y) * ww + (rx0 - x0 + x)] = 1;
  for (let y = 0; y < wh; y++) {
    for (let x = 0; x < ww; x++) {
      const gi = (y0 + y) * W + (x0 + x), li = y * ww + x;
      unfilled[li] = todo[gi] && !reg[li] ? 1 : 0;
      vs[li] = validSrc[gi];
    }
  }
  const dil = F.dilateMask(reg, ww, wh, 2 * RING + 1, true);
  let ring = new Uint8Array(wh * ww);
  let rc = 0;
  for (let i = 0; i < ring.length; i++) { ring[i] = dil[i] && !reg[i] && !unfilled[i] ? 1 : 0; rc += ring[i]; }
  if (rc < 4) {
    const d15 = F.dilateMask(reg, ww, wh, 15, false);
    rc = 0;
    for (let i = 0; i < ring.length; i++) { ring[i] = d15[i] && !reg[i] && !unfilled[i] ? 1 : 0; rc += ring[i]; }
  }
  const ty = [], tx = [], gy = [], gx = [];
  for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) {
    const i = y * ww + x;
    if (reg[i]) { ty.push(y); tx.push(x); }
    if (ring[i]) { gy.push(y); gx.push(x); }
  }
  let tyMin = Infinity, tyMax = -Infinity, txMin = Infinity, txMax = -Infinity;
  for (let k = 0; k < ty.length; k++) { tyMin = Math.min(tyMin, ty[k]); tyMax = Math.max(tyMax, ty[k]); txMin = Math.min(txMin, tx[k]); txMax = Math.max(txMax, tx[k]); }
  let gyMin = Infinity, gyMax = -Infinity, gxMin = Infinity, gxMax = -Infinity;
  for (let k = 0; k < gy.length; k++) { gyMin = Math.min(gyMin, gy[k]); gyMax = Math.max(gyMax, gy[k]); gxMin = Math.min(gxMin, gx[k]); gxMax = Math.max(gxMax, gx[k]); }
  let best = null, bestCost = Infinity;
  for (const [dy, dx] of offs) {
    if (tyMin + dy < 0 || txMin + dx < 0 || tyMax + dy >= wh || txMax + dx >= ww) continue;
    let ok = true;
    for (let k = 0; k < ty.length; k++) if (!vs[(ty[k] + dy) * ww + tx[k] + dx]) { ok = false; break; }
    if (!ok) continue;
    if (gy.length && (gyMin + dy < 0 || gxMin + dx < 0 || gyMax + dy >= wh || gxMax + dx >= ww)) continue;
    let cost = 0, allValid = true;
    for (let k = 0; k < gy.length; k++) {
      const a = ((gy[k] + dy) * ww + gx[k] + dx), b = gy[k] * ww + gx[k];
      for (let c = 0; c < C; c++) { const d = win[a * C + c] - win[b * C + c]; cost += d * d; }
      if (!vs[a]) allValid = false;
    }
    cost = (cost / Math.max(1, gy.length * C)) * (1 + 0.004 * Math.hypot(dy, dx));
    if (!allValid) cost *= 1.5;
    if (cost < bestCost) { best = [dy, dx]; bestCost = cost; }
  }
  const known = new Uint8Array(wh * ww);
  for (let i = 0; i < known.length; i++) known[i] = !reg[i] && !unfilled[i] ? 1 : 0;
  const guess = new Float32Array(wh * ww * C);
  let fallback = false;
  if (best === null) {
    fallback = true;
  } else {
    const [dy, dx] = best;
    for (let k = 0; k < ty.length; k++) {
      const a = (ty[k] + dy) * ww + tx[k] + dx, b = ty[k] * ww + tx[k];
      for (let c = 0; c < C; c++) guess[b * C + c] = win[a * C + c];
    }
    const d3 = F.dilateMask(reg, ww, wh, 3, false);
    for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) {
      const i = y * ww + x;
      if (!(d3[i] && known[i])) continue;
      const sy = Math.min(Math.max(y + dy, 0), wh - 1), sx = Math.min(Math.max(x + dx, 0), ww - 1);
      for (let c = 0; c < C; c++) guess[i * C + c] = win[(sy * ww + sx) * C + c];
    }
  }
  const sol = poisson(known, reg, guess, win, ww, wh, C);
  if (fallback) {
    // grain-matched noise on top of the smooth membrane
    const g2 = [];
    for (let c = 0; c < C; c++) {
      const plane = new Float32Array(wh * ww);
      for (let i = 0; i < plane.length; i++) plane[i] = win[i * C + c];
      const bl = F.gaussianBlur(plane, ww, wh, 2.0);
      const vals = [];
      for (let i = 0; i < plane.length; i++) if (ring[i]) vals.push(plane[i] - bl[i]);
      g2.push(vals.length ? R.std(vals) : 0.01);
    }
    for (let j = 0; j < sol.pos.length; j++) for (let c = 0; c < C; c++) sol.values[j * C + c] += rng.normal() * g2[c];
  }
  for (let j = 0; j < sol.pos.length; j++) {
    const li = sol.pos[j], x = li % ww, y = (li - x) / ww;
    const gi = ((y0 + y) * W + (x0 + x)) * C;
    for (let c = 0; c < C; c++) {
      let v = sol.values[j * C + c];
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      img[gi + c] = Math.round(v * 65535);
    }
  }
  return !fallback;
}

// img: Uint16Array interleaved (H*W*C), mask: Uint8Array (H*W, nonzero = heal),
// outside: Uint8Array or null. Heals in place. Returns {regions, fallbacks}.
// avoid (optional): more marks, healed elsewhere, that must not be copied from either; a tile of a
// large scan passes the marks of its neighbours here.
async function heal(img, W, H, C, mask, outside, onProgress, tick = async () => {}, avoid = null) {
  const todo = new Uint8Array(W * H);
  for (let i = 0; i < todo.length; i++) todo[i] = mask[i] ? 1 : 0;
  let shield = todo;
  if (avoid) {
    shield = Uint8Array.from(todo);
    for (let i = 0; i < shield.length; i++) if (avoid[i]) shield[i] = 1;
  }
  const guard = F.dilateMask(shield, W, H, 5, false);
  const validSrc = new Uint8Array(W * H);
  for (let i = 0; i < validSrc.length; i++) validSrc[i] = !guard[i] && !(outside && outside[i]) ? 1 : 0;
  const rng = makeRng(7);
  const { lab, n } = R.label(todo, W, H, 8);
  const boxes = R.findObjects(lab, W, H, n);
  const order = [...Array(n).keys()].sort((a, b) => (boxes[b].y1 - boxes[b].y0) * (boxes[b].x1 - boxes[b].x0) - (boxes[a].y1 - boxes[a].y0) * (boxes[a].x1 - boxes[a].x0));
  let fallbacks = 0, done = 0;
  for (const i of order) {
    const b = boxes[i];
    const rh = b.y1 - b.y0, rw = b.x1 - b.x0;
    const regFull = new Uint8Array(rh * rw);
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) regFull[y * rw + x] = lab[(b.y0 + y) * W + b.x0 + x] === i + 1 ? 1 : 0;
    // thickness from the distance transform of the padded region
    const pw2 = rw + 2, ph2 = rh + 2, padded = new Uint8Array(pw2 * ph2);
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) padded[(y + 1) * pw2 + x + 1] = regFull[y * rw + x];
    const dt = R.distanceTransform(padded, pw2, ph2);
    let dmax = 0;
    for (let k = 0; k < dt.length; k++) if (dt[k] > dmax) dmax = dt[k];
    const thick = 2 * dmax;
    for (let cy = 0; cy < rh; cy += CHUNK) {
      for (let cx = 0; cx < rw; cx += CHUNK) {
        const ph = Math.min(CHUNK, rh - cy), pw = Math.min(CHUNK, rw - cx);
        const part = new Uint8Array(ph * pw);
        let any = false;
        for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
          const v = regFull[(cy + y) * rw + cx + x];
          part[y * pw + x] = v;
          if (v) any = true;
        }
        if (!any) continue;
        const ry0 = b.y0 + cy, rx0 = b.x0 + cx;
        if (!healRegion(img, W, H, C, validSrc, todo, ry0, rx0, part, ph, pw, thick, rng)) fallbacks++;
        for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
          if (!part[y * pw + x]) continue;
          const gi = (ry0 + y) * W + rx0 + x;
          todo[gi] = 0;
          if (!(outside && outside[gi])) validSrc[gi] = 1;
        }
      }
    }
    done++;
    if (done % 25 === 0) { if (onProgress) onProgress(done / n); await tick(done / n); }
  }
  return { regions: n, fallbacks };
}

module.exports = { heal, offsets, poisson, pyRound };

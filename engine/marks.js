"use strict";
// Per-mark measurements (port of detect.py's helpers). Windows are small; clarity over speed.
const F = require("./filters");
const R = require("./regions");
// Python's round(): half to even
function pyRound(v) { const f = Math.floor(v), d = v - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : f % 2 === 0 ? f : f + 1; }

// ---------- small window helpers ------------------------------------------------------------

function crop(src, W, y0, y1, x0, x1) {
  const w = x1 - x0, h = y1 - y0, out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) out.set(src.subarray((y0 + y) * W + x0, (y0 + y) * W + x1), y * w);
  return out;
}
function cropMask(src, W, y0, y1, x0, x1) {
  const w = x1 - x0, h = y1 - y0, out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) out.set(src.subarray((y0 + y) * W + x0, (y0 + y) * W + x1), y * w);
  return out;
}
function count(m) { let n = 0; for (let i = 0; i < m.length; i++) n += m[i] ? 1 : 0; return n; }
function andNot(a, b) { const o = new Uint8Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] && !b[i] ? 1 : 0; return o; }
function or(a, b) { const o = new Uint8Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] || b[i] ? 1 : 0; return o; }
function vals(img, m) { const out = []; for (let i = 0; i < m.length; i++) if (m[i]) out.push(img[i]); return out; }
function scaled(arr, s) { const o = new Float32Array(arr.length); for (let i = 0; i < arr.length; i++) o[i] = arr[i] * s; return o; }

// ---------- line features --------------------------------------------------------------------

function lineFeatures(Yw, m, w, h, bright) {
  const sk = R.skeletonize(m, w, h);
  const ys = [], xs = [];
  for (let i = 0; i < sk.length; i++) if (sk[i]) { ys.push((i / w) | 0); xs.push(i % w); }
  if (ys.length < 6) return null;
  const yb = F.gaussianBlur(Yw, w, h, 1.2);
  const pad = new Uint8Array((w + 2) * (h + 2));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pad[(y + 1) * (w + 2) + x + 1] = m[y * w + x];
  const dtp = R.distanceTransform(pad, w + 2, h + 2);
  const dsk = [];
  for (let k = 0; k < ys.length; k++) dsk.push(dtp[(ys[k] + 1) * (w + 2) + xs[k] + 1]);
  const halfW = R.median(dsk);
  const grid = new R.PointGrid(ys, xs, 16);
  const sgn = bright ? 1 : -1;
  const one = [], resid = [], uy = [], ux = [];
  const samp = (qy, qx) => {
    const yy = pyRound(qy), xx = pyRound(qx);
    if (yy >= 0 && yy < h && xx >= 0 && xx < w && !m[yy * w + xx]) return yb[yy * w + xx];
    return NaN;
  };
  for (let i = 0; i < ys.length; i += 2) {
    const nb = grid.queryBox(ys[i], xs[i], 7).filter((j) => (ys[j] - ys[i]) ** 2 + (xs[j] - xs[i]) ** 2 <= 49);
    if (nb.length < 4) continue;
    const s = R.shapeStats(nb.map((j) => ys[j]), nb.map((j) => xs[j]));
    // tangent (dy, dx) from the major axis angle (angle measured from x axis)
    const ty = Math.sin(s.angle), tx = Math.cos(s.angle);
    const ny = -tx, nx = ty;   // normal [-t1, t0] with t = (ty, tx)
    const a = [samp(ys[i] + ny * (halfW + 4), xs[i] + nx * (halfW + 4)), samp(ys[i] + ny * (halfW + 8), xs[i] + nx * (halfW + 8))];
    const b = [samp(ys[i] - ny * (halfW + 4), xs[i] - nx * (halfW + 4)), samp(ys[i] - ny * (halfW + 8), xs[i] - nx * (halfW + 8))];
    const fa = a.filter((v) => !Number.isNaN(v)), fb = b.filter((v) => !Number.isNaN(v));
    if (!fa.length || !fb.length) continue;
    const s1 = R.mean(fa), s2 = R.mean(fb);
    let mv = bright ? -Infinity : Infinity;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = ys[i] + dy, xx = xs[i] + dx;
      if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
      const v = yb[yy * w + xx];
      mv = bright ? Math.max(mv, v) : Math.min(mv, v);
    }
    const cval = Math.max(sgn * (mv - 0.5 * (s1 + s2)), 1e-4);
    one.push(Math.abs(s1 - s2) / cval);
    resid.push(cval);
    uy.push(ys[i]); ux.push(xs[i]);
  }
  if (one.length < 4) return null;
  // order residuals along the main axis of the used points
  const su = R.shapeStats(uy, ux);
  const ay = Math.sin(su.angle), ax = Math.cos(su.angle);
  const my = R.mean(uy), mx = R.mean(ux);
  const ordIdx = uy.map((_, k) => k).sort((p, q) => ((uy[p] - my) * ay + (ux[p] - mx) * ax) - ((uy[q] - my) * ay + (ux[q] - mx) * ax));
  const r = ordIdx.map((k) => resid[k]);
  const k = Math.max(3, Math.floor(r.length / 8));
  const sm = convSame(r, k);
  let core;
  if (r.length > 2 * k + 2) core = r.map((v, i) => v - sm[i]).slice(k, r.length - k);
  else { const mr = R.mean(r); core = r.map((v) => v - mr); }
  const medr = R.median(r);
  const hf = R.std(core) / Math.max(medr, 1e-4);
  const dips = r.filter((v) => v < 0.5 * medr).length / r.length;
  const sall = R.shapeStats(ys, xs);
  const py = R.mean(ys), px = R.mean(xs);
  const uay = Math.sin(sall.angle), uax = Math.cos(sall.angle);
  let mn = Infinity, mxv = -Infinity, rss = 0;
  for (let i = 0; i < ys.length; i++) {
    const along = (ys[i] - py) * uay + (xs[i] - px) * uax;
    const across = -(ys[i] - py) * uax + (xs[i] - px) * uay;
    mn = Math.min(mn, along); mxv = Math.max(mxv, along); rss += across * across;
  }
  const length = mxv - mn, rms = Math.sqrt(rss / ys.length);
  // read the mark's own pixels along the skeleton's main path
  const order = mainPath(ys, xs, grid);
  const bg6 = F.gaussianBlur(Yw, w, h, 6, null, true);
  const ctr = order.map((j) => sgn * Yw[ys[j] * w + xs[j]]);
  const lift = order.map((j, i) => ctr[i] - sgn * bg6[ys[j] * w + xs[j]]);
  const medl = R.median(lift);
  const gaps = medl > 0 ? lift.filter((v) => v < 0.25 * medl).length / lift.length : 1;
  let per = 0;
  if (ctr.length >= 20) {
    const sm9 = convSame(ctr, 9);
    let x = ctr.map((v, i) => v - sm9[i]).slice(4, ctr.length - 4);
    const mxx = R.mean(x); x = x.map((v) => v - mxx);
    let den = 0; for (const v of x) den += v * v;
    const ac = [];
    for (let kk = 3; kk < 11; kk++) {
      if (x.length > kk + 6 && den > 1e-12) { let s = 0; for (let i = 0; i + kk < x.length; i++) s += x[i] * x[i + kk]; ac.push(s / den); }
    }
    per = ac.length ? Math.max(...ac) : 0;
  }
  const r2 = R.pyRoundTo;
  return {
    one_med: r2(R.median(one), 2), one_frac: r2(one.filter((v) => v > 0.8).length / one.length, 2),
    hf: r2(hf, 2), dips: r2(dips, 2), straight: r2(rms / Math.max(length, 1), 4), length: pyRound(length),
    half_w: r2(halfW, 1), per: r2(per, 2), gaps: r2(gaps, 2), path: order.length,
  };
}

// The skeleton's longest route, walked in ~1 px steps: shortest paths (Dijkstra; steps up to 3 px,
// a step longer than 1.5 px costs 4 px extra) from an end to the farthest point, then from there
// to the farthest point. Same order and ties as detect.py, so both engines pick the same path.
function mainPath(ys, xs, grid) {
  const n = ys.length;
  const nbJ = new Array(n), nbW = new Array(n), deg = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const cand = grid.queryBox(ys[i], xs[i], 3).filter((j) => j !== i).sort((p, q) => p - q);
    const J = [], Wt = [];
    for (const j of cand) {
      const dy = ys[j] - ys[i], dx = xs[j] - xs[i];
      const d = Math.sqrt(dy * dy + dx * dx);
      if (d > 3.0) continue;
      J.push(j); Wt.push(d <= 1.5 ? d : d + 4.0);
      if (d <= 1.5) deg[i]++;
    }
    nbJ[i] = J; nbW[i] = Wt;
  }
  const sweep = (src) => {
    const cost = new Float64Array(n).fill(Infinity), prev = new Int32Array(n).fill(-1);
    cost[src] = 0;
    // binary heap of (cost, index), smallest first - the same pop order as Python's heapq on tuples
    const hc = [], hi = [];
    const less = (i, j) => hc[i] < hc[j] || (hc[i] === hc[j] && hi[i] < hi[j]);
    const swap = (i, j) => { const c = hc[i]; hc[i] = hc[j]; hc[j] = c; const t = hi[i]; hi[i] = hi[j]; hi[j] = t; };
    const push = (c, v) => {
      hc.push(c); hi.push(v);
      let k = hc.length - 1;
      while (k > 0) { const p = (k - 1) >> 1; if (less(k, p)) { swap(k, p); k = p; } else break; }
    };
    const pop = () => {
      const c = hc[0], v = hi[0], last = hc.length - 1;
      hc[0] = hc[last]; hi[0] = hi[last]; hc.pop(); hi.pop();
      let k = 0;
      for (;;) {
        const l = 2 * k + 1, r = l + 1;
        let m = k;
        if (l < hc.length && less(l, m)) m = l;
        if (r < hc.length && less(r, m)) m = r;
        if (m === k) break;
        swap(k, m); k = m;
      }
      return [c, v];
    };
    push(0, src);
    while (hc.length) {
      const [c, u] = pop();
      if (c > cost[u]) continue;
      const J = nbJ[u], Wt = nbW[u];
      for (let t = 0; t < J.length; t++) {
        const v = J[t], nc = c + Wt[t];
        if (nc < cost[v]) { cost[v] = nc; prev[v] = u; push(nc, v); }
      }
    }
    let far = -1;
    for (let i = 0; i < n; i++) if (cost[i] < Infinity && (far < 0 || cost[i] > cost[far])) far = i;
    return [far, prev];
  };
  let start = 0;
  for (let i = 1; i < n; i++) if (deg[i] < deg[start]) start = i;
  const [a0] = sweep(start);
  const [b0, prev] = sweep(a0);
  const path = [];
  for (let u = b0; u !== -1; u = prev[u]) path.push(u);
  return path.reverse();
}

// np.convolve(a, ones(k)/k, mode="same")
function convSame(a, k) {
  const n = a.length, out = new Array(n).fill(0);
  const off = Math.floor((k - 1) / 2);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < k; j++) { const idx = i + off - (k - 1) + j; if (idx >= 0 && idx < n) s += a[idx]; }
    out[i] = s / k;
  }
  return out;
}

// ---------- solid flakes ---------------------------------------------------------------------

function growFlake(Yw, raw, w, h, sgn, sig) {
  if (!sig || count(raw) < 20) return null;
  const yy = scaled(Yw, sgn);
  const d7 = F.dilateMask(raw, w, h, 7), d25 = F.dilateMask(raw, w, h, 25), d15 = F.dilateMask(raw, w, h, 15);
  const near = andNot(d7, raw), far = andNot(d25, d15);
  if (count(near) < 8 || count(far) < 8) return null;
  const rim = R.percentile(vals(yy, raw), 75), bgl = R.percentile(vals(yy, far), 30);
  if (rim - bgl <= 0) return null;
  const reg = new Uint8Array(w * h);
  for (let i = 0; i < reg.length; i++) reg[i] = yy[i] >= bgl + 0.6 * (rim - bgl) || raw[i] ? 1 : 0;
  const { lab } = R.label(reg, w, h, 8);
  const ids = new Set(); for (let i = 0; i < raw.length; i++) if (raw[i] && lab[i]) ids.add(lab[i]);
  const region = new Uint8Array(w * h);
  for (let i = 0; i < region.length; i++) region[i] = ids.has(lab[i]) ? 1 : 0;
  const extra = count(andNot(region, raw)), rawN = count(raw), regN = count(region);
  if (extra < 0.25 * rawN) return null;
  for (let x = 0; x < w; x++) if (region[x] || region[(h - 1) * w + x]) return null;
  for (let y = 0; y < h; y++) if (region[y * w] || region[y * w + w - 1]) return null;
  if (regN > 4 * rawN + 1500) return null;
  const inner = F.erodeMask(region, w, h, 7, true);
  if (count(inner) < 15) return null;
  const g1 = F.gaussianBlur(Yw, w, h, 1.0);
  const hp = [], sg = [];
  for (let i = 0; i < inner.length; i++) if (inner[i]) { hp.push(Yw[i] - g1[i]); sg.push(sig[i]); }
  const flat = R.std(hp) / Math.max(R.median(sg), 1e-5);
  return flat <= 0.6 ? R.fillHoles(region, w, h) : null;
}

function solidify(Yw, raw, w, h, bright, sig) {
  const sgn = bright ? 1 : -1;
  const grown = growFlake(Yw, raw, w, h, sgn, sig);
  if (grown) return { mask: grown, how: "grown" };
  const closed = F.erodeMask(F.dilateMask(raw, w, h, 5, true), w, h, 5, true);
  const filled = R.fillHoles(closed, w, h);
  const hole = andNot(filled, raw);
  if (count(hole) < 6) return { mask: raw, how: false };
  const outside = andNot(F.dilateMask(filled, w, h, 9), F.dilateMask(filled, w, h, 3));
  if (count(outside) < 8) return { mask: raw, how: false };
  const yy = scaled(Yw, sgn);
  const rim = R.percentile(vals(yy, raw), 75), out = R.median(vals(yy, outside)), hv = R.median(vals(yy, hole));
  if (rim - out > 0 && hv >= out + 0.5 * (rim - out)) return { mask: filled, how: "filled" };
  return { mask: raw, how: false };
}

// ---------- crispness, flatness, detail, texture --------------------------------------------

// allc: cand in the window as the caller's loop saw it (default: cand cropped)
function markCues(Y, W, G3, E25, E1, sigLevel, cand, obj, ww, wh, y0, x0, bright, gradCal, allc0) {
  if (count(obj) < 4) return {};
  const sgn = bright ? 1 : -1;
  const allc = allc0 || cropMask(cand, W, y0, y0 + wh, x0, x0 + ww);
  const Yw = crop(Y, W, y0, y0 + wh, x0, x0 + ww);
  const yy = scaled(Yw, sgn);
  const excl = or(F.dilateMask(allc, ww, wh, 7), obj);
  const ring = andNot(F.dilateMask(obj, ww, wh, 21), excl);
  if (count(ring) < 10) return {};
  const core = R.percentile(vals(yy, obj), 85), bgm = R.median(vals(yy, ring));
  const con = Math.max(core - bgm, 1e-4);
  const out = {};
  const g3 = crop(G3, W, y0, y0 + wh, x0, x0 + ww);
  const ad = []; for (let i = 0; i < obj.length; i++) if (obj[i]) ad.push(Math.abs(Yw[i] - g3[i]));
  const r2 = R.pyRoundTo;
  const crisp = R.percentile(ad, 95) / con;
  out.crisp = r2(crisp, 2);
  // a perfectly sharp mark of exactly this shape: a 3 px blur only changes the rim of a wide mark,
  // so a wide flake can never read as crisp as a thin one. crisp_rel = crispness against that
  const of = new Float32Array(obj.length);
  for (let i = 0; i < obj.length; i++) of[i] = obj[i] ? 1 : 0;
  const gb = F.gaussianBlur(of, ww, wh, 3.0);
  const ad2 = [];
  for (let i = 0; i < obj.length; i++) if (obj[i]) ad2.push(Math.fround(Math.abs(of[i] - gb[i])));
  out.crisp_rel = r2(crisp / Math.max(R.percentile(ad2, 95), 1e-3), 2);
  out.core_y = r2(R.percentile(vals(Yw, obj), bright ? 85 : 15), 3);
  out.bg_y = r2(R.median(vals(Yw, ring)), 3);
  const rv = vals(yy, ring);
  out.ring_range = r2((R.percentile(rv, 90) - R.percentile(rv, 10)) / con, 2);
  const sl = crop(sigLevel, W, y0, y0 + wh, x0, x0 + ww);
  const inner = F.erodeMask(obj, ww, wh, 7, true);
  if (count(inner) >= 15) {
    const g1 = F.gaussianBlur(Yw, ww, wh, 1.0);
    const hp = [], sg = [];
    for (let i = 0; i < inner.length; i++) if (inner[i]) { hp.push(Yw[i] - g1[i]); sg.push(sl[i]); }
    out.flat = r2(R.std(hp) / Math.max(R.median(sg), 1e-5), 2);
  }
  const big = andNot(F.dilateMask(obj, ww, wh, 81), F.dilateMask(allc, ww, wh, 9));
  if (count(big) >= 50) {
    out.broad = r2((core - R.median(vals(yy, big))) / con, 2);
    const s2 = gradCal * R.median(vals(sl, big)) ** 2;
    const e25 = R.median(vals(crop(E25, W, y0, y0 + wh, x0, x0 + ww), big)) / s2;
    out.detail = r2(e25, 2);
    if (E1) {
      const e1 = R.median(vals(crop(E1, W, y0, y0 + wh, x0, x0 + ww), big)) / s2;
      out.tex = r2(e1 - 1 - e25, 2);
    }
  }
  return out;
}

// (whiteness, crispness, background colour) at the object's strongest pixel.
// rgb: interleaved Float32 or a getter; here an accessor rgbAt(y, x, c).
function peakLook(rgbAt, C, Y, W, H, glabEq, f, bright) {
  const r = 12;
  let best = -Infinity, py = f.y0, px = f.x0;
  for (let y = f.y0; y < f.y1; y++) for (let x = f.x0; x < f.x1; x++) {
    if (!glabEq(y, x)) continue;
    const v = bright ? Y[y * W + x] : -Y[y * W + x];
    if (v > best) { best = v; py = y; px = x; }
  }
  const ya = Math.max(0, py - r), yb = Math.min(H, py + r + 1), xa = Math.max(0, px - r), xb = Math.min(W, px + r + 1);
  const edgeVals = Array.from({ length: C }, () => []), lumEdge = [], ring3 = [];
  for (let y = ya; y < yb; y++) for (let x = xa; x < xb; x++) {
    const d = Math.max(Math.abs(y - py), Math.abs(x - px));
    if (d >= r - 2) { for (let c = 0; c < C; c++) edgeVals[c].push(rgbAt(y, x, c)); lumEdge.push(Y[y * W + x]); }
    if (d === 3) ring3.push(Y[y * W + x]);
  }
  if (lumEdge.length < 10) return [1, 1, []];
  const bg = edgeVals.map((v) => R.median(v));
  const pk = new Array(C).fill(0);
  let n = 0;
  for (let y = Math.max(0, py - 1); y < Math.min(H, py + 2); y++) for (let x = Math.max(0, px - 1); x < Math.min(W, px + 2); x++) {
    for (let c = 0; c < C; c++) pk[c] += rgbAt(y, x, c);
    n++;
  }
  for (let c = 0; c < C; c++) pk[c] /= n;
  const d = pk.map((v, c) => (bright ? v - bg[c] : bg[c] - v));
  const dmax = Math.max(...d), dmin = Math.min(...d);
  const white = dmax > 1e-4 ? dmin / dmax : 1;
  const lc = Y[py * W + px], me = R.median(lumEdge), m3 = R.median(ring3);
  const num = bright ? lc - m3 : m3 - lc;
  const den = bright ? lc - me : me - lc;
  const sharp = den > 1e-4 ? num / den : 1;
  const r2 = R.pyRoundTo;
  return [r2(white, 2), r2(sharp, 2), bg.map((v) => r2(v, 3))];
}

function textureAlignment(Y, W, H, sigma, cand, f, owner, cal) {
  const pad = 34;
  const y0 = Math.max(0, f.y0 - pad), y1 = Math.min(H, f.y1 + pad), x0 = Math.max(0, f.x0 - pad), x1 = Math.min(W, f.x1 + pad);
  const w = x1 - x0, h = y1 - y0;
  const yb = F.gaussianBlur(crop(Y, W, y0, y1, x0, x1), w, h, 1.0);
  const gx = F.sobel(yb, w, h, 1, 0), gy = F.sobel(yb, w, h, 0, 1);
  const obj = owner; // local mask w x h
  if (count(obj) < 4) return [0, 1, 0, 90];
  const allc = cropMask(cand, W, y0, y1, x0, x1);
  const ring = andNot(F.dilateMask(obj, w, h, 61, true), F.dilateMask(allc, w, h, 9));
  if (count(ring) < 50) return [0, 1, 0, 90];
  let jxx = 0, jyy = 0, jxy = 0, en = 0, nn = 0;
  const sg = [];
  const sw = crop(sigma, W, y0, y1, x0, x1);
  for (let i = 0; i < ring.length; i++) if (ring[i]) { jxx += gx[i] * gx[i]; jyy += gy[i] * gy[i]; jxy += gx[i] * gy[i]; en += gx[i] * gx[i] + gy[i] * gy[i]; nn++; sg.push(sw[i]); }
  const coh = Math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / Math.max(jxx + jyy, 1e-12);
  const th = 0.5 * Math.atan2(2 * jxy, jxx - jyy);
  let ps = 0, pa = 0;
  for (let i = 0; i < ring.length; i++) if (ring[i]) { const pr = gx[i] * Math.cos(th) + gy[i] * Math.sin(th); ps += pr; pa += Math.abs(pr); }
  const edge = Math.abs(ps) / Math.max(pa, 1e-12);
  const energy = en / nn / (cal * R.median(sg) ** 2);
  const oy = [], ox = [];
  for (let i = 0; i < obj.length; i++) if (obj[i]) { oy.push((i / w) | 0); ox.push(i % w); }
  const st = R.shapeStats(oy, ox);
  const od = ((st.angle * 180) / Math.PI + 180) % 180;
  const strand = ((th * 180) / Math.PI + 90 + 180) % 180;
  const d = Math.abs(od - strand);
  const r2 = R.pyRoundTo;
  return [r2(coh, 2), r2(edge, 2), r2(energy, 2), r2(Math.min(d, 180 - d), 1)];
}

module.exports = { crop, cropMask, count, andNot, or, vals, lineFeatures, growFlake, solidify, markCues, peakLook, textureAlignment, convSame, mainPath };

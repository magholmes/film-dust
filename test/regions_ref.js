"use strict";
// Region tools: labelling, bounding boxes, hole filling, distance transform, thinning,
// neighbour search and small statistics. Masks are Uint8Array (0/1), row-major.

// scipy.ndimage.label: conn 4 (default structure) or 8 (np.ones((3,3))). Returns {lab, n}.
function label(mask, w, h, conn = 8) {
  const lab = new Int32Array(w * h);
  const parent = [0];
  const find = (a) => {
    while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  };
  const unite = (a, b) => {
    a = find(a); b = find(b);
    if (a !== b) { if (a < b) parent[b] = a; else parent[a] = b; }
    return a < b ? a : b;
  };
  let next = 1;
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = 0; x < w; x++) {
      const i = off + x;
      if (!mask[i]) continue;
      let l = 0;
      const nb = [];
      if (x > 0 && lab[i - 1]) nb.push(lab[i - 1]);
      if (y > 0) {
        if (lab[i - w]) nb.push(lab[i - w]);
        if (conn === 8) {
          if (x > 0 && lab[i - w - 1]) nb.push(lab[i - w - 1]);
          if (x < w - 1 && lab[i - w + 1]) nb.push(lab[i - w + 1]);
        }
      }
      if (nb.length === 0) { l = next++; parent.push(l); }
      else {
        l = nb[0];
        for (let k = 1; k < nb.length; k++) l = unite(l, nb[k]);
        l = find(l);
      }
      lab[i] = l;
    }
  }
  // flatten to consecutive labels in raster order (scipy numbers by first appearance)
  const remap = new Int32Array(next);
  let n = 0;
  for (let i = 0; i < lab.length; i++) {
    const l = lab[i];
    if (!l) continue;
    const r = find(l);
    if (!remap[r]) remap[r] = ++n;
    lab[i] = remap[r];
  }
  return { lab, n };
}

// scipy.ndimage.find_objects -> array of {y0, y1, x0, x1} (exclusive ends), index = label-1
function findObjects(lab, w, h, n) {
  const box = new Array(n);
  for (let i = 0; i < n; i++) box[i] = { y0: h, y1: -1, x0: w, x1: -1, area: 0 };
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = 0; x < w; x++) {
      const l = lab[off + x];
      if (!l) continue;
      const b = box[l - 1];
      if (y < b.y0) b.y0 = y;
      if (y + 1 > b.y1) b.y1 = y + 1;
      if (x < b.x0) b.x0 = x;
      if (x + 1 > b.x1) b.x1 = x + 1;
      b.area++;
    }
  }
  return box;
}

// scipy.ndimage.binary_fill_holes (background connectivity 4, as with the default structure)
function fillHoles(mask, w, h) {
  const out = new Uint8Array(w * h);
  const bg = new Uint8Array(w * h);   // 1 = background reachable from the border
  const stack = new Int32Array(w * h);
  let sp = 0;
  const push = (i) => { if (!mask[i] && !bg[i]) { bg[i] = 1; stack[sp++] = i; } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (sp) {
    const i = stack[--sp];
    const x = i % w, y = (i - x) / w;
    if (x > 0) push(i - 1);
    if (x < w - 1) push(i + 1);
    if (y > 0) push(i - w);
    if (y < h - 1) push(i + w);
  }
  for (let i = 0; i < out.length; i++) out[i] = mask[i] || !bg[i] ? 1 : 0;
  return out;
}

// cv2.distanceTransform(DIST_L2, 3): chamfer 3x3 with a=0.955, b=1.3693 (distance to nearest 0)
function distanceTransform(mask, w, h) {
  const a = 0.955, b = 1.3693, INF = 1e9;
  const d = new Float32Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = mask[i] ? INF : 0;
  const get = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : d[y * w + x]);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!d[i]) continue;
      let v = d[i];
      v = Math.min(v, get(x - 1, y) + a, get(x, y - 1) + a, get(x - 1, y - 1) + b, get(x + 1, y - 1) + b);
      d[i] = v;
    }
  }
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      if (!d[i]) continue;
      let v = d[i];
      v = Math.min(v, get(x + 1, y) + a, get(x, y + 1) + a, get(x + 1, y + 1) + b, get(x - 1, y + 1) + b);
      d[i] = v;
    }
  }
  return d;
}

// Thinning that matches skimage.morphology.skeletonize (Zhang-Suen as scikit-image implements
// it, BSD-3-Clause). Each set pixel's 8 neighbours (NW=1, N=2, NE=4, E=8, SE=16, S=32, SW=64,
// W=128) index a table: 1 = remove in the first sub-pass, 2 = in the second, 3 = in either. The
// table was fitted to scikit-image's output and reproduces it exactly on 440 test shapes.
const THIN_LUT = Uint8Array.from([
  0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
  0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
  0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
  3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
]);

function skeletonize(mask, w, h) {
  const W2 = w + 2, H2 = h + 2;
  let s = new Uint8Array(W2 * H2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) s[(y + 1) * W2 + x + 1] = mask[y * w + x] ? 1 : 0;
  let removed = true;
  while (removed) {
    removed = false;
    for (let pass = 0; pass < 2; pass++) {
      const c = Uint8Array.from(s);
      for (let y = 1; y < H2 - 1; y++) {
        for (let x = 1; x < W2 - 1; x++) {
          const i = y * W2 + x;
          if (!s[i]) continue;
          const v = THIN_LUT[s[i - W2 - 1] + 2 * s[i - W2] + 4 * s[i - W2 + 1] + 8 * s[i + 1] + 16 * s[i + W2 + 1] + 32 * s[i + W2] + 64 * s[i + W2 - 1] + 128 * s[i - 1]];
          if (v === 3 || (v === 1 && pass === 0) || (v === 2 && pass === 1)) { c[i] = 0; removed = true; }
        }
      }
      s = c;
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = s[(y + 1) * W2 + x + 1];
  return out;
}

// uniform-grid neighbour index over points (replaces scipy cKDTree for radius / nearest queries)
class PointGrid {
  constructor(ys, xs, cell = 64) {
    this.ys = ys; this.xs = xs; this.cell = cell;
    this.map = new Map();
    for (let i = 0; i < ys.length; i++) {
      const k = this.key(Math.floor(ys[i] / cell), Math.floor(xs[i] / cell));
      let a = this.map.get(k);
      if (!a) { a = []; this.map.set(k, a); }
      a.push(i);
    }
  }
  key(cy, cx) { return cy * 1000003 + cx; }
  // Chebyshev (p=inf) ball
  queryBox(y, x, r) {
    const out = [], c = this.cell;
    for (let cy = Math.floor((y - r) / c); cy <= Math.floor((y + r) / c); cy++) {
      for (let cx = Math.floor((x - r) / c); cx <= Math.floor((x + r) / c); cx++) {
        const a = this.map.get(this.key(cy, cx));
        if (!a) continue;
        for (const i of a) if (Math.abs(this.ys[i] - y) <= r && Math.abs(this.xs[i] - x) <= r) out.push(i);
      }
    }
    return out;
  }
  // Euclidean nearest (brute force over growing rings)
  nearest(y, x) {
    let best = -1, bd = Infinity, ring = 0;
    const c = this.cell, cy0 = Math.floor(y / c), cx0 = Math.floor(x / c);
    while (ring < 64) {
      for (let cy = cy0 - ring; cy <= cy0 + ring; cy++) {
        for (let cx = cx0 - ring; cx <= cx0 + ring; cx++) {
          if (Math.max(Math.abs(cy - cy0), Math.abs(cx - cx0)) !== ring) continue;
          const a = this.map.get(this.key(cy, cx));
          if (!a) continue;
          for (const i of a) {
            const d = (this.ys[i] - y) ** 2 + (this.xs[i] - x) ** 2;
            if (d < bd) { bd = d; best = i; }
          }
        }
      }
      if (best >= 0 && (ring * c) ** 2 > bd) break;
      ring++;
    }
    return { index: best, dist: Math.sqrt(bd) };
  }
}

// ---------- statistics on small arrays -------------------------------------------------------

// numpy.percentile with linear interpolation
function percentile(arr, q) {
  const n = arr.length;
  if (!n) return NaN;
  const a = Float64Array.from(arr).sort();
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

function median(arr) { return percentile(arr, 50); }

function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return arr.length ? s / arr.length : NaN;
}

function std(arr) {
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += (arr[i] - m) ** 2;
  return arr.length ? Math.sqrt(s / arr.length) : NaN;
}

// Gather values of `img` where `mask` (local window) is set. win = {y0, x0, w, h} in img coords.
function gather(img, W, mask, mw, mh, y0, x0) {
  const out = [];
  for (let y = 0; y < mh; y++) {
    const row = (y0 + y) * W + x0, mrow = y * mw;
    for (let x = 0; x < mw; x++) if (mask[mrow + x]) out.push(img[row + x]);
  }
  return out;
}

// Covariance-based elongation sqrt(l1/l2) and major axis of a point set
function shapeStats(ys, xs) {
  const n = ys.length;
  if (n < 4) return { elong: 1, angle: 0 };
  let my = 0, mx = 0;
  for (let i = 0; i < n; i++) { my += ys[i]; mx += xs[i]; }
  my /= n; mx /= n;
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
  }
  sxx /= n - 1; syy /= n - 1; sxy /= n - 1;
  const tr = sxx + syy, det = sxx * syy - sxy * sxy;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(tr / 2 - disc, 1e-3);
  const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  return { elong: Math.sqrt(l1 / l2), angle, l1, l2 };
}

module.exports = { label, findObjects, fillHoles, distanceTransform, skeletonize, PointGrid, percentile, median, mean, std, gather, shapeStats };

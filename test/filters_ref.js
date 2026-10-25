"use strict";
// Image primitives on typed arrays (row-major, single channel unless noted).
// They mirror the OpenCV / SciPy calls the Python engine uses, so both engines agree:
// borders follow OpenCV's defaults (REFLECT_101 for linear filters, REPLICATE for medians,
// "outside does not count" for morphology).

// ---------- helpers -------------------------------------------------------------------------

function reflect101(i, n) {
  if (n === 1) return 0;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i;
    if (i >= n) i = 2 * n - 2 - i;
  }
  return i;
}

function clampIdx(i, n) {
  return i < 0 ? 0 : i >= n ? n - 1 : i;
}

// ---------- separable convolution (Gaussian, box) -------------------------------------------

function gaussKernel(sigma) {
  // cv::getGaussianKernel size rule for float images: ksize = round(sigma*4*2+1) | 1
  let k = Math.round(sigma * 8 + 1) | 1;
  if (k < 3) k = 3;
  const r = k >> 1;
  const ker = new Float32Array(k);
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const x = i - r;
    const v = Math.exp(-(x * x) / (2 * sigma * sigma));
    ker[i] = v;
    sum += v;
  }
  for (let i = 0; i < k; i++) ker[i] /= sum;
  return ker;
}

function convolveRows(src, w, h, ker, dst) {
  const r = ker.length >> 1;
  const row = new Float32Array(w + 2 * r);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = -r; x < w + r; x++) row[x + r] = src[off + reflect101(x, w)];
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = 0; k < ker.length; k++) s += row[x + k] * ker[k];
      dst[off + x] = s;
    }
  }
  return dst;
}

function convolveCols(src, w, h, ker, dst) {
  const r = ker.length >> 1;
  const col = new Float32Array(h + 2 * r);
  const out = new Float32Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = -r; y < h + r; y++) col[y + r] = src[reflect101(y, h) * w + x];
    for (let y = 0; y < h; y++) {
      let s = 0;
      for (let k = 0; k < ker.length; k++) s += col[y + k] * ker[k];
      out[y] = s;
    }
    for (let y = 0; y < h; y++) dst[y * w + x] = out[y];
  }
  return dst;
}

// Gaussian blur. Exact FIR for sigma < 4 (matches cv2.GaussianBlur); a 3-pass box
// approximation above that (only used for wide-scale, robust measures).
function gaussianBlur(src, w, h, sigma, dst, exact = false) {
  dst = dst || new Float32Array(w * h);
  if (sigma < 4 || exact) {
    const ker = gaussKernel(sigma);
    const tmp = new Float32Array(w * h);
    convolveRows(src, w, h, ker, tmp);
    convolveCols(tmp, w, h, ker, dst);
    return dst;
  }
  // three box passes approximate a Gaussian (Wells 1986); radius from sigma
  const n = 3;
  const wIdeal = Math.sqrt((12 * sigma * sigma) / n + 1);
  let wl = Math.floor(wIdeal);
  if (wl % 2 === 0) wl--;
  const wu = wl + 2;
  const m = Math.round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
  let a = Float32Array.from(src);
  let b = new Float32Array(w * h);
  for (let i = 0; i < n; i++) {
    const r = ((i < m ? wl : wu) - 1) >> 1;
    boxBlurH(a, b, w, h, r);
    boxBlurV(b, a, w, h, r);
  }
  dst.set(a);
  return dst;
}

function boxBlurH(src, dst, w, h, r) {
  const inv = 1 / (2 * r + 1);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    let acc = 0;
    for (let x = -r; x <= r; x++) acc += src[off + reflect101(x, w)];
    for (let x = 0; x < w; x++) {
      dst[off + x] = acc * inv;
      acc += src[off + reflect101(x + r + 1, w)] - src[off + reflect101(x - r, w)];
    }
  }
}

function boxBlurV(src, dst, w, h, r) {
  const inv = 1 / (2 * r + 1);
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -r; y <= r; y++) acc += src[reflect101(y, h) * w + x];
    for (let y = 0; y < h; y++) {
      dst[y * w + x] = acc * inv;
      acc += src[reflect101(y + r + 1, h) * w + x] - src[reflect101(y - r, h) * w + x];
    }
  }
}

// cv2.blur(src, (k, k)) - normalised box, REFLECT_101
function boxBlur(src, w, h, k, dst) {
  dst = dst || new Float32Array(w * h);
  const tmp = new Float32Array(w * h);
  boxBlurH(src, tmp, w, h, k >> 1);
  boxBlurV(tmp, dst, w, h, k >> 1);
  return dst;
}

// cv2.boxFilter(..., normalize=False, borderType=BORDER_CONSTANT): window sums with zeros outside
function boxSumConst(src, w, h, k, dst) {
  dst = dst || new Float32Array(w * h);
  const r = k >> 1;
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    let acc = 0;
    for (let x = 0; x <= Math.min(r, w - 1); x++) acc += src[off + x];
    for (let x = 0; x < w; x++) {
      tmp[off + x] = acc;
      const add = x + r + 1, sub = x - r;
      if (add < w) acc += src[off + add];
      if (sub >= 0) acc -= src[off + sub];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = 0; y <= Math.min(r, h - 1); y++) acc += tmp[y * w + x];
    for (let y = 0; y < h; y++) {
      dst[y * w + x] = acc;
      const add = y + r + 1, sub = y - r;
      if (add < h) acc += tmp[add * w + x];
      if (sub >= 0) acc -= tmp[sub * w + x];
    }
  }
  return dst;
}

// ---------- derivatives (cv2.Sobel ksize=3, REFLECT_101) ------------------------------------

function sobel(src, w, h, dx, dy, dst) {
  dst = dst || new Float32Array(w * h);
  // separable: derivative kernel [-1,0,1] (1st) or [1,-2,1] (2nd); smoothing [1,2,1]
  const kd1 = [-1, 0, 1], kd2 = [1, -2, 1], ks = [1, 2, 1];
  const kx = dx === 1 ? kd1 : dx === 2 ? kd2 : ks;
  const ky = dy === 1 ? kd1 : dy === 2 ? kd2 : ks;
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = 0; x < w; x++) {
      tmp[off + x] = kx[0] * src[off + reflect101(x - 1, w)] + kx[1] * src[off + x] + kx[2] * src[off + reflect101(x + 1, w)];
    }
  }
  for (let y = 0; y < h; y++) {
    const up = reflect101(y - 1, h) * w, mid = y * w, dn = reflect101(y + 1, h) * w;
    for (let x = 0; x < w; x++) dst[mid + x] = ky[0] * tmp[up + x] + ky[1] * tmp[mid + x] + ky[2] * tmp[dn + x];
  }
  return dst;
}

// np.gradient (central differences inside, one-sided at the edges): returns [gy, gx]
function npGradient(src, w, h) {
  const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = 0; x < w; x++) {
      if (x === 0) gx[off] = src[off + 1] - src[off];
      else if (x === w - 1) gx[off + x] = src[off + x] - src[off + x - 1];
      else gx[off + x] = 0.5 * (src[off + x + 1] - src[off + x - 1]);
      if (y === 0) gy[off + x] = src[off + w + x] - src[off + x];
      else if (y === h - 1) gy[off + x] = src[off + x] - src[off - w + x];
      else gy[off + x] = 0.5 * (src[off + w + x] - src[off - w + x]);
    }
  }
  return [gy, gx];
}

// ---------- median on 8-bit (cv2.medianBlur, BORDER_REPLICATE) ------------------------------

// Huang's sliding-histogram median. src: Uint8Array. Returns Uint8Array.
function median8(src, w, h, k, dst) {
  dst = dst || new Uint8Array(w * h);
  const r = k >> 1;
  const half = (k * k) >> 1;   // median rank (0-based) of k*k values
  const hist = new Int32Array(256);
  for (let y = 0; y < h; y++) {
    hist.fill(0);
    // initial window at x = 0
    for (let dy = -r; dy <= r; dy++) {
      const row = clampIdx(y + dy, h) * w;
      for (let dx = -r; dx <= r; dx++) hist[src[row + clampIdx(dx, w)]]++;
    }
    let med = 0, below = 0;   // below = count of values < med
    {
      let c = 0;
      while (c + hist[med] <= half) { c += hist[med]; med++; }
      below = c;
    }
    dst[y * w] = med;
    for (let x = 1; x < w; x++) {
      const xo = clampIdx(x - r - 1, w), xi = clampIdx(x + r, w);
      for (let dy = -r; dy <= r; dy++) {
        const row = clampIdx(y + dy, h) * w;
        const vo = src[row + xo], vi = src[row + xi];
        hist[vo]--; if (vo < med) below--;
        hist[vi]++; if (vi < med) below++;
      }
      // move med so that below <= half < below + hist[med]
      while (below > half) { med--; below -= hist[med]; }
      while (below + hist[med] <= half) { below += hist[med]; med++; }
      dst[y * w + x] = med;
    }
  }
  return dst;
}

// ---------- morphology -----------------------------------------------------------------------

// van Herk / Gil-Werman running min or max over a 1-D window of size k (outside ignored)
function runMinMax1D(src, n, k, isMax, out, stride, offset, buf1, buf2) {
  const r = k >> 1;
  // pad conceptually with +/-inf so outside never wins (matches OpenCV's default morphology border)
  const pad = isMax ? -Infinity : Infinity;
  const len = n + 2 * r;
  const blocks = Math.ceil(len / k) * k;
  const f = buf1, g = buf2;
  for (let i = 0; i < blocks; i++) {
    const j = i - r;
    const v = j >= 0 && j < n ? src[offset + j * stride] : pad;
    if (i % k === 0) f[i] = v;
    else f[i] = isMax ? (v > f[i - 1] ? v : f[i - 1]) : (v < f[i - 1] ? v : f[i - 1]);
  }
  for (let i = blocks - 1; i >= 0; i--) {
    const j = i - r;
    const v = j >= 0 && j < n ? src[offset + j * stride] : pad;
    if (i % k === k - 1 || i === blocks - 1) g[i] = v;
    else g[i] = isMax ? (v > g[i + 1] ? v : g[i + 1]) : (v < g[i + 1] ? v : g[i + 1]);
  }
  for (let i = 0; i < n; i++) {
    // window [i, i + k - 1] in padded coordinates
    const a = g[i], b = f[i + k - 1];
    out[i] = isMax ? (a > b ? a : b) : (a < b ? a : b);
  }
}

// rectangular (square) erode / dilate, cv2.erode/dilate(np.ones((k, k)))
function rectMorph(src, w, h, k, isMax, dst) {
  dst = dst || new Float32Array(w * h);
  const tmp = new Float32Array(w * h);
  const L = Math.max(w, h) + 2 * k + k;
  const b1 = new Float32Array(L), b2 = new Float32Array(L), out = new Float32Array(Math.max(w, h));
  for (let y = 0; y < h; y++) {
    runMinMax1D(src, w, k, isMax, out, 1, y * w, b1, b2);
    tmp.set(out.subarray(0, w), y * w);
  }
  for (let x = 0; x < w; x++) {
    runMinMax1D(tmp, h, k, isMax, out, w, x, b1, b2);
    for (let y = 0; y < h; y++) dst[y * w + x] = out[y];
  }
  return dst;
}

// half-widths of cv2.getStructuringElement(MORPH_ELLIPSE, (k, k)) per row offset
function ellipseHalfWidths(k) {
  const r = k >> 1;
  const hw = new Int32Array(k);
  // OpenCV: for each row i, dy = i - r; x extent from inv_r2 * dy^2
  const inv_r2 = r ? 1 / (r * r) : 0;
  for (let i = 0; i < k; i++) {
    const dy = i - r;
    let j1 = 0;
    if (dy * dy <= r * r) {
      const dx = Math.round(r * Math.sqrt(Math.max(0, 1 - dy * dy * inv_r2)));
      j1 = dx;
    } else j1 = -1;
    hw[i] = j1;
  }
  return hw;
}

// erode / dilate with an elliptical structuring element (exact, span decomposition)
function ellipseMorph(src, w, h, k, isMax, dst) {
  dst = dst || new Float32Array(w * h);
  const r = k >> 1;
  const hw = ellipseHalfWidths(k);
  const widths = [...new Set(Array.from(hw).filter((v) => v >= 0))];
  // horizontally filtered copies per distinct width, computed row by row on demand
  const L = w + 3 * k + 2;
  const b1 = new Float32Array(L), b2 = new Float32Array(L), out = new Float32Array(w);
  const cache = new Map();   // key row*1000+width -> Float32Array(w)
  const pad = isMax ? -Infinity : Infinity;
  const getRow = (yy, wdt) => {
    const key = yy * 1024 + wdt;
    let a = cache.get(key);
    if (!a) {
      a = new Float32Array(w);
      if (wdt === 0) a.set(src.subarray(yy * w, yy * w + w));
      else {
        runMinMax1D(src, w, 2 * wdt + 1, isMax, out, 1, yy * w, b1, b2);
        a.set(out);
      }
      cache.set(key, a);
    }
    return a;
  };
  const acc = new Float32Array(w);
  for (let y = 0; y < h; y++) {
    acc.fill(pad);
    for (let i = 0; i < k; i++) {
      if (hw[i] < 0) continue;
      const yy = y + i - r;
      if (yy < 0 || yy >= h) continue;
      const row = getRow(yy, hw[i]);
      if (isMax) { for (let x = 0; x < w; x++) if (row[x] > acc[x]) acc[x] = row[x]; }
      else { for (let x = 0; x < w; x++) if (row[x] < acc[x]) acc[x] = row[x]; }
    }
    dst.set(acc, y * w);
    // drop cached rows that can no longer be used
    const dead = y - r;
    if (dead >= 0) for (const wd of widths) cache.delete(dead * 1024 + wd);
  }
  return dst;
}

// cv2.morphologyEx(MORPH_OPEN / MORPH_CLOSE) with an ellipse
function ellipseOpenClose(src, w, h, k, open) {
  const a = ellipseMorph(src, w, h, k, !open);
  return ellipseMorph(a, w, h, k, open);
}

// binary versions on Uint8Array masks (0/1)
function dilateMask(mask, w, h, k, ellipse = false) {
  const f = new Float32Array(w * h);
  for (let i = 0; i < f.length; i++) f[i] = mask[i];
  const d = ellipse ? ellipseMorph(f, w, h, k, true) : rectMorph(f, w, h, k, true);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = d[i] > 0.5 ? 1 : 0;
  return out;
}

function erodeMask(mask, w, h, k, ellipse = false) {
  const f = new Float32Array(w * h);
  for (let i = 0; i < f.length; i++) f[i] = mask[i];
  const d = ellipse ? ellipseMorph(f, w, h, k, false) : rectMorph(f, w, h, k, false);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = d[i] > 0.5 ? 1 : 0;
  return out;
}

// ---------- resize (cv2.INTER_AREA for exact 2x down, INTER_LINEAR up) ------------------------

function downscaleArea2(src, w, h) {
  const w2 = Math.floor(w / 2), h2 = Math.floor(h / 2);
  const dst = new Float32Array(w2 * h2);
  for (let y = 0; y < h2; y++) {
    const r0 = 2 * y * w, r1 = r0 + w;
    for (let x = 0; x < w2; x++) {
      const i = 2 * x;
      dst[y * w2 + x] = 0.25 * (src[r0 + i] + src[r0 + i + 1] + src[r1 + i] + src[r1 + i + 1]);
    }
  }
  return { data: dst, w: w2, h: h2 };
}

// cv2.resize(INTER_LINEAR) from (sw, sh) to (dw, dh)
function resizeLinear(src, sw, sh, dw, dh) {
  const dst = new Float32Array(dw * dh);
  const fx = sw / dw, fy = sh / dh;
  for (let y = 0; y < dh; y++) {
    let sy = (y + 0.5) * fy - 0.5;
    if (sy < 0) sy = 0;
    let y0 = Math.floor(sy); if (y0 >= sh - 1) y0 = sh - 1;
    const y1 = Math.min(y0 + 1, sh - 1);
    const ay = sy - y0;
    for (let x = 0; x < dw; x++) {
      let sx = (x + 0.5) * fx - 0.5;
      if (sx < 0) sx = 0;
      let x0 = Math.floor(sx); if (x0 >= sw - 1) x0 = sw - 1;
      const x1 = Math.min(x0 + 1, sw - 1);
      const ax = sx - x0;
      const a = src[y0 * sw + x0] * (1 - ax) + src[y0 * sw + x1] * ax;
      const b = src[y1 * sw + x0] * (1 - ax) + src[y1 * sw + x1] * ax;
      dst[y * dw + x] = a * (1 - ay) + b * ay;
    }
  }
  return dst;
}

module.exports = {
  reflect101, clampIdx, gaussKernel, gaussianBlur, boxBlur, boxSumConst, sobel, npGradient,
  median8, rectMorph, ellipseMorph, ellipseOpenClose, ellipseHalfWidths, dilateMask, erodeMask,
  downscaleArea2, resizeLinear, runMinMax1D,
};

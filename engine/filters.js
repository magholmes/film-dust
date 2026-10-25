"use strict";
// Image primitives on typed arrays (row-major, single channel unless noted).
// They mirror the OpenCV / SciPy calls the Python engine uses, so both engines agree:
// borders follow OpenCV's defaults (REFLECT_101 for linear filters, REPLICATE for medians,
// "outside does not count" for morphology).

const NAT = require("./native");   // bit-identical C++ versions of the big filters, when available
const natOk = (src, w, h, dst) => NAT.big(w, h) && src instanceof Float32Array && (!dst || dst instanceof Float32Array);

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

// symmetric kernels: each pair of taps is added before it is weighted (half the multiplies);
// sums stay in double, so the float32 results match the plain loop
function convolveRows(src, w, h, ker, dst) {
  const r = ker.length >> 1, kc = ker[r];
  const row = new Float32Array(w + 2 * r);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = -r; x < w + r; x++) row[x + r] = src[off + reflect101(x, w)];
    for (let x = 0; x < w; x++) {
      const c = x + r;
      let acc = row[c] * kc;
      for (let j = 1; j <= r; j++) acc += ker[r - j] * (row[c - j] + row[c + j]);
      dst[off + x] = acc;
    }
  }
  return dst;
}

function convolveCols(src, w, h, ker, dst) {
  const r = ker.length >> 1, kc = ker[r];
  const acc = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    const oc = y * w;
    for (let x = 0; x < w; x++) acc[x] = src[oc + x] * kc;
    for (let j = 1; j <= r; j++) {
      const ra = reflect101(y - j, h) * w, rb = reflect101(y + j, h) * w, kv = ker[r - j];
      for (let x = 0; x < w; x++) acc[x] += kv * (src[ra + x] + src[rb + x]);
    }
    for (let x = 0; x < w; x++) dst[oc + x] = acc[x];
  }
  return dst;
}

// Gaussian blur. Exact FIR for sigma < 4 (matches cv2.GaussianBlur); a 3-pass box
// approximation above that (only used for wide-scale, robust measures).
function gaussianBlur(src, w, h, sigma, dst, exact = false) {
  if (natOk(src, w, h, dst) && (sigma < 4 || exact)) { const r = NAT.sepConv(src, w, h, gaussKernel(sigma), dst); if (r) return r; }
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
  if (natOk(src, w, h, dst)) { const r = NAT.box3(src, w, h, [0, 1, 2].map((i) => ((i < m ? wl : wu) - 1) >> 1), dst); if (r) return r; }
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
  const acc = new Float64Array(w);
  for (let y = -r; y <= r; y++) { const o = reflect101(y, h) * w; for (let x = 0; x < w; x++) acc[x] += src[o + x]; }
  for (let y = 0; y < h; y++) {
    const d = y * w, a = reflect101(y + r + 1, h) * w, b = reflect101(y - r, h) * w;
    for (let x = 0; x < w; x++) { dst[d + x] = acc[x] * inv; acc[x] += src[a + x] - src[b + x]; }
  }
}

// cv2.blur(src, (k, k)) - normalised box, REFLECT_101
function boxBlur(src, w, h, k, dst) {
  if (natOk(src, w, h, dst)) { const r = NAT.boxBlur(src, w, h, k, dst); if (r) return r; }
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
  if (natOk(src, w, h, dst)) { const r = NAT.sobel(src, w, h, dx, dy, dst); if (r) return r; }
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
  if (natOk(src, w, h) && w > 1 && h > 1) { const r = NAT.npGradient(src, w, h); if (r) return r; }
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
  const rows = new Int32Array(k);
  for (let y = 0; y < h; y++) {
    for (let i = 0; i < k; i++) rows[i] = clampIdx(y + i - r, h) * w;
    hist.fill(0);
    for (let i = 0; i < k; i++) { const ro = rows[i]; for (let dx = -r; dx <= r; dx++) hist[src[ro + clampIdx(dx, w)]]++; }
    let med = 0, below = 0;   // below = count of values < med
    while (below + hist[med] <= half) { below += hist[med]; med++; }
    dst[y * w] = med;
    for (let x = 1; x < w; x++) {
      const xo = clampIdx(x - r - 1, w), xi = clampIdx(x + r, w);
      if (xo !== xi) {
        for (let i = 0; i < k; i++) {
          const ro = rows[i];
          const vo = src[ro + xo], vi = src[ro + xi];
          if (vo === vi) continue;
          hist[vo]--; if (vo < med) below--;
          hist[vi]++; if (vi < med) below++;
        }
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

// van Herk / Gil-Werman running min or max over windows of 2r+1 on a padded line P
// (length n + 2r, pads already +/-inf so the outside never wins). out gets n values.
function vhLine(P, n, r, isMax, out, f, g) {
  const k = 2 * r + 1, len = n + 2 * r;
  if (isMax) {
    for (let b0 = 0; b0 < len; b0 += k) {
      const b1 = b0 + k < len ? b0 + k : len;
      let m = -Infinity;
      for (let i = b0; i < b1; i++) { const v = P[i]; if (v > m) m = v; f[i] = m; }
      m = -Infinity;
      for (let i = b1 - 1; i >= b0; i--) { const v = P[i]; if (v > m) m = v; g[i] = m; }
    }
    for (let i = 0; i < n; i++) { const a = g[i], b = f[i + k - 1]; out[i] = a > b ? a : b; }
  } else {
    for (let b0 = 0; b0 < len; b0 += k) {
      const b1 = b0 + k < len ? b0 + k : len;
      let m = Infinity;
      for (let i = b0; i < b1; i++) { const v = P[i]; if (v < m) m = v; f[i] = m; }
      m = Infinity;
      for (let i = b1 - 1; i >= b0; i--) { const v = P[i]; if (v < m) m = v; g[i] = m; }
    }
    for (let i = 0; i < n; i++) { const a = g[i], b = f[i + k - 1]; out[i] = a < b ? a : b; }
  }
}

// 1-D running min/max with a stride (old signature, kept for callers)
function runMinMax1D(src, n, k, isMax, out, stride, offset) {
  const r = k >> 1, pad = isMax ? -Infinity : Infinity;
  const P = new Float32Array(n + 2 * r), f = new Float32Array(n + 4 * r + 1), g = new Float32Array(n + 4 * r + 1);
  P.fill(pad);
  for (let j = 0; j < n; j++) P[r + j] = src[offset + j * stride];
  vhLine(P, n, r, isMax, out, f, g);
}

function morphRows(src, w, h, r, isMax, dst) {
  const pad = isMax ? -Infinity : Infinity;
  const P = new Float32Array(w + 2 * r), f = new Float32Array(w + 4 * r + 1), g = new Float32Array(w + 4 * r + 1), out = new Float32Array(w);
  P.fill(pad);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) P[r + x] = src[o + x];
    vhLine(P, w, r, isMax, out, f, g);
    dst.set(out, o);
  }
  return dst;
}

// vertical van Herk over strips of columns, so every inner loop runs along a row
function morphCols(src, w, h, r, isMax, dst) {
  const k = 2 * r + 1, len = h + 2 * r, S = Math.min(w, 512);
  const F = new Float32Array((len + k) * S), G = new Float32Array((len + k) * S);
  const pad = isMax ? -Infinity : Infinity;
  for (let x0 = 0; x0 < w; x0 += S) {
    const sw = Math.min(S, w - x0);
    for (let b0 = 0; b0 < len; b0 += k) {
      const b1 = b0 + k < len ? b0 + k : len;
      for (let i = b0; i < b1; i++) {
        const y = i - r, fo = i * S, inside = y >= 0 && y < h, so = y * w + x0;
        if (i === b0) {
          if (inside) for (let x = 0; x < sw; x++) F[fo + x] = src[so + x];
          else for (let x = 0; x < sw; x++) F[fo + x] = pad;
        } else if (!inside) {
          for (let x = 0; x < sw; x++) F[fo + x] = F[fo - S + x];
        } else if (isMax) {
          for (let x = 0; x < sw; x++) { const v = src[so + x], q = F[fo - S + x]; F[fo + x] = v > q ? v : q; }
        } else {
          for (let x = 0; x < sw; x++) { const v = src[so + x], q = F[fo - S + x]; F[fo + x] = v < q ? v : q; }
        }
      }
      for (let i = b1 - 1; i >= b0; i--) {
        const y = i - r, go = i * S, inside = y >= 0 && y < h, so = y * w + x0;
        if (i === b1 - 1) {
          if (inside) for (let x = 0; x < sw; x++) G[go + x] = src[so + x];
          else for (let x = 0; x < sw; x++) G[go + x] = pad;
        } else if (!inside) {
          for (let x = 0; x < sw; x++) G[go + x] = G[go + S + x];
        } else if (isMax) {
          for (let x = 0; x < sw; x++) { const v = src[so + x], q = G[go + S + x]; G[go + x] = v > q ? v : q; }
        } else {
          for (let x = 0; x < sw; x++) { const v = src[so + x], q = G[go + S + x]; G[go + x] = v < q ? v : q; }
        }
      }
    }
    for (let y = 0; y < h; y++) {
      const ga = y * S, fb = (y + k - 1) * S, d = y * w + x0;
      if (isMax) for (let x = 0; x < sw; x++) { const a = G[ga + x], b = F[fb + x]; dst[d + x] = a > b ? a : b; }
      else for (let x = 0; x < sw; x++) { const a = G[ga + x], b = F[fb + x]; dst[d + x] = a < b ? a : b; }
    }
  }
  return dst;
}

// rectangular (square) erode / dilate, cv2.erode/dilate(np.ones((k, k)))
function rectMorph(src, w, h, k, isMax, dst) {
  if (natOk(src, w, h, dst)) { const r = NAT.rectMorph(src, w, h, k, isMax, dst); if (r) return r; }
  dst = dst || new Float32Array(w * h);
  const r = k >> 1;
  const tmp = morphRows(src, w, h, r, isMax, new Float32Array(w * h));
  return morphCols(tmp, w, h, r, isMax, dst);
}

// half-widths of cv2.getStructuringElement(MORPH_ELLIPSE, (k, k)) per row offset
function ellipseHalfWidths(k) {
  const r = k >> 1;
  const hw = new Int32Array(k);
  const inv_r2 = r ? 1 / (r * r) : 0;
  for (let i = 0; i < k; i++) {
    const dy = i - r;
    hw[i] = dy * dy <= r * r ? Math.round(r * Math.sqrt(Math.max(0, 1 - dy * dy * inv_r2))) : -1;
  }
  return hw;
}

// erode / dilate with an elliptical structuring element (exact). Each source row is filtered
// once per distinct half-width (each width grown from the previous one) and kept in a ring of
// k rows; an output row is the min/max of the k rows, each at its own width.
function ellipseMorph(src, w, h, k, isMax, dst) {
  if (natOk(src, w, h, dst)) { const r = NAT.ellipse(src, w, h, k, ellipseHalfWidths(k), isMax ? 1 : 0, dst); if (r) return r; }
  dst = dst || new Float32Array(w * h);
  const r = k >> 1;
  const hw = ellipseHalfWidths(k);
  const widths = [...new Set(Array.from(hw).filter((v) => v >= 0))].sort((a, b) => a - b);
  const m = widths.length;
  const widx = new Int32Array(k).fill(-1);
  for (let i = 0; i < k; i++) if (hw[i] >= 0) widx[i] = widths.indexOf(hw[i]);
  const ring = new Float32Array(k * m * w), ringRow = new Int32Array(k).fill(-1);
  const maxR = widths[m - 1];
  const P = new Float32Array(w + 2 * maxR), f = new Float32Array(w + 4 * maxR + 1), g = new Float32Array(w + 4 * maxR + 1), out = new Float32Array(w);
  const pad = isMax ? -Infinity : Infinity;
  const fill = (yy) => {
    const slot = yy % k, base = slot * m * w, so = yy * w;
    for (let j = 0; j < m; j++) {
      const Wj = widths[j], o = base + j * w;
      if (Wj === 0) { for (let x = 0; x < w; x++) ring[o + x] = src[so + x]; continue; }
      const prev = j > 0 ? widths[j - 1] : -1, d = Wj - prev;
      if (j > 0 && d <= prev) {
        const po = base + (j - 1) * w;
        // the clamps only matter within d of either end: keep them out of the long middle run
        const lo = Math.min(d, w), hi = Math.max(lo, w - d);
        if (isMax) {
          for (let x = 0; x < lo; x++) { const a = ring[po], b = ring[po + (x + d >= w ? w - 1 : x + d)]; ring[o + x] = a > b ? a : b; }
          for (let x = lo; x < hi; x++) { const a = ring[po + x - d], b = ring[po + x + d]; ring[o + x] = a > b ? a : b; }
          for (let x = hi; x < w; x++) { const a = ring[po + (x - d < 0 ? 0 : x - d)], b = ring[po + w - 1]; ring[o + x] = a > b ? a : b; }
        } else {
          for (let x = 0; x < lo; x++) { const a = ring[po], b = ring[po + (x + d >= w ? w - 1 : x + d)]; ring[o + x] = a < b ? a : b; }
          for (let x = lo; x < hi; x++) { const a = ring[po + x - d], b = ring[po + x + d]; ring[o + x] = a < b ? a : b; }
          for (let x = hi; x < w; x++) { const a = ring[po + (x - d < 0 ? 0 : x - d)], b = ring[po + w - 1]; ring[o + x] = a < b ? a : b; }
        }
      } else {
        P.fill(pad, 0, Wj);
        for (let x = 0; x < w; x++) P[Wj + x] = src[so + x];
        P.fill(pad, Wj + w, w + 2 * Wj);
        vhLine(P, w, Wj, isMax, out, f, g);
        ring.set(out, o);
      }
    }
    ringRow[slot] = yy;
  };
  const acc = new Float32Array(w);
  for (let y = 0; y < h; y++) {
    acc.fill(pad);
    for (let i = 0; i < k; i++) {
      const j = widx[i];
      if (j < 0) continue;
      const yy = y + i - r;
      if (yy < 0 || yy >= h) continue;
      const slot = yy % k;
      if (ringRow[slot] !== yy) fill(yy);
      const o = slot * m * w + j * w;
      if (isMax) { for (let x = 0; x < w; x++) { const v = ring[o + x]; if (v > acc[x]) acc[x] = v; } }
      else { for (let x = 0; x < w; x++) { const v = ring[o + x]; if (v < acc[x]) acc[x] = v; } }
    }
    dst.set(acc, y * w);
  }
  return dst;
}

// cv2.morphologyEx(MORPH_OPEN / MORPH_CLOSE) with an ellipse
function ellipseOpenClose(src, w, h, k, open) {
  if (natOk(src, w, h)) { const r = NAT.ellipse(src, w, h, k, ellipseHalfWidths(k), open ? 2 : 3); if (r) return r; }
  const a = ellipseMorph(src, w, h, k, !open);
  return ellipseMorph(a, w, h, k, open);
}

// binary rectangle erode/dilate on 0/1 masks with running counts (outside never counts)
function binRect(mask, w, h, k, dilate) {
  const r = k >> 1;
  const hz = new Uint8Array(w * h), out = new Uint8Array(w * h);
  const Pc = new Int32Array(w + 1);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) Pc[x + 1] = Pc[x] + (dilate ? (mask[o + x] ? 1 : 0) : (mask[o + x] ? 0 : 1));
    for (let x = 0; x < w; x++) {
      const c = Pc[x + r + 1 > w ? w : x + r + 1] - Pc[x - r < 0 ? 0 : x - r];
      hz[o + x] = dilate ? (c > 0 ? 1 : 0) : (c === 0 ? 1 : 0);
    }
  }
  const cnt = new Int32Array(w);
  const top = Math.min(r, h - 1);
  for (let yy = 0; yy <= top; yy++) {
    const o = yy * w;
    if (dilate) for (let x = 0; x < w; x++) cnt[x] += hz[o + x];
    else for (let x = 0; x < w; x++) cnt[x] += 1 - hz[o + x];
  }
  for (let y = 0; y < h; y++) {
    const o = y * w;
    if (dilate) for (let x = 0; x < w; x++) out[o + x] = cnt[x] > 0 ? 1 : 0;
    else for (let x = 0; x < w; x++) out[o + x] = cnt[x] === 0 ? 1 : 0;
    const add = y + r + 1, sub = y - r;
    if (add < h) { const a = add * w; if (dilate) for (let x = 0; x < w; x++) cnt[x] += hz[a + x]; else for (let x = 0; x < w; x++) cnt[x] += 1 - hz[a + x]; }
    if (sub >= 0) { const b = sub * w; if (dilate) for (let x = 0; x < w; x++) cnt[x] -= hz[b + x]; else for (let x = 0; x < w; x++) cnt[x] -= 1 - hz[b + x]; }
  }
  return out;
}

// binary versions on Uint8Array masks (0/1)
function dilateMask(mask, w, h, k, ellipse = false) {
  if (NAT.big(w, h) && mask instanceof Uint8Array) { const r = NAT.binMorph(mask, w, h, k, true, ellipse ? ellipseHalfWidths(k) : null); if (r) return r; }
  if (!ellipse) return binRect(mask, w, h, k, true);
  return ellipseMorph(mask, w, h, k, true, new Uint8Array(w * h));
}

function erodeMask(mask, w, h, k, ellipse = false) {
  if (NAT.big(w, h) && mask instanceof Uint8Array) { const r = NAT.binMorph(mask, w, h, k, false, ellipse ? ellipseHalfWidths(k) : null); if (r) return r; }
  if (!ellipse) return binRect(mask, w, h, k, false);
  return ellipseMorph(mask, w, h, k, false, new Uint8Array(w * h));
}

// ---------- resize (cv2.INTER_AREA for exact 2x down, INTER_LINEAR up) ------------------------

function downscaleArea2(src, w, h) {
  const w2 = Math.floor(w / 2), h2 = Math.floor(h / 2);
  return { data: resizeArea(src, w, h, w2, h2), w: w2, h: h2 };
}

// cv2.resize(INTER_AREA) for any down-scale (OpenCV's fractional-weight table; float32 sums)
function areaTab(ssize, dsize) {
  const scale = ssize / dsize;
  const si = [], di = [], al = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale, fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) { di.push(dx); si.push(sx1 - 1); al.push(Math.fround((sx1 - fsx1) / cellWidth)); }
    for (let sx = sx1; sx < sx2; sx++) { di.push(dx); si.push(sx); al.push(Math.fround(1.0 / cellWidth)); }
    if (fsx2 - sx2 > 1e-3) { di.push(dx); si.push(sx2); al.push(Math.fround(Math.min(Math.min(fsx2 - sx2, 1.0), cellWidth) / cellWidth)); }
  }
  return { si: Int32Array.from(si), di: Int32Array.from(di), al: Float32Array.from(al) };
}

function resizeArea(src, sw, sh, dw, dh) {
  const sx = sw / dw, sy = sh / dh;
  const isx = Math.round(sx), isy = Math.round(sy);
  const dst = new Float32Array(dw * dh);
  if (Math.abs(sx - isx) < 2.220446049250313e-16 && Math.abs(sy - isy) < 2.220446049250313e-16) {
    // integer factor: plain block mean (OpenCV's fast path)
    const inv = 1 / (isx * isy);
    for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
      let acc = 0;
      for (let j = 0; j < isy; j++) { const o = (y * isy + j) * sw + x * isx; for (let i = 0; i < isx; i++) acc += src[o + i]; }
      dst[y * dw + x] = acc * inv;
    }
    return dst;
  }
  const xt = areaTab(sw, dw), yt = areaTab(sh, dh);
  const buf = new Float32Array(dw), sum = new Float32Array(dw);
  let prev = -1;
  for (let k = 0; k < yt.si.length; k++) {
    const dy = yt.di[k], syi = yt.si[k], beta = yt.al[k];
    buf.fill(0);
    const o = syi * sw;
    for (let t = 0; t < xt.si.length; t++) buf[xt.di[t]] += src[o + xt.si[t]] * xt.al[t];
    if (dy !== prev) {
      if (prev >= 0) dst.set(sum, prev * dw);
      for (let x = 0; x < dw; x++) sum[x] = beta * buf[x];
      prev = dy;
    } else {
      for (let x = 0; x < dw; x++) sum[x] += beta * buf[x];
    }
  }
  if (prev >= 0) dst.set(sum, prev * dw);
  return dst;
}

// cv2.resize(INTER_LINEAR) from (sw, sh) to (dw, dh)
function resizeLinear(src, sw, sh, dw, dh) {
  if (NAT.big(dw, dh) && src instanceof Float32Array) { const r = NAT.resizeLinear(src, sw, sh, dw, dh); if (r) return r; }
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
  downscaleArea2, resizeArea, resizeLinear, runMinMax1D, vhLine, binRect,
};

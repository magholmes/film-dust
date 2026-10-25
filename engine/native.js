"use strict";
// Native filters (native/src, built by native/build.sh): multi-threaded C++ ports of the heaviest
// whole-image filters, bit-identical to their JS versions in filters.js / maps.js.
//   In Photoshop: the hybrid addon win/x64/filmdust.uxpaddon (manifest "addon"), when installed.
//   In node (tests): native/build/filmdust.node, unless FILMDUST_NATIVE=0.
// When neither loads, `available` is false and the engine runs its JS filters unchanged.
// Tiny images stay in JS: below about 64 x 64 handing them over costs more than the work (calls are
// cheap otherwise: the native side keeps its worker threads, and small jobs run on the calling thread).
const MIN_PX = 1 << 12;

function load() {
  let uxp = null;
  try { uxp = require("uxp"); } catch (e) { uxp = null; }
  if (uxp) {
    // a missing or refused addon leaves the JS filters running; say why in Photoshop's UXP log (Logs/UXPLogs_*.log)
    try { return require("filmdust.uxpaddon"); } catch (e) {
      try { console.error("film dust: native addon not loaded:", String((e && e.message) || e)); } catch (e2) { /* no console */ }
      return null;
    }
  }
  try {
    if (typeof process !== "undefined" && process.env && process.env.FILMDUST_NATIVE === "0") return null;
    return require(require("path").join(__dirname, "..", "native", "build", "filmdust.node"));
  } catch (e) {
    return null;
  }
}

let N = null;
try { N = load(); } catch (e) { N = null; }
if (N) {
  try { if (N.version() < 1) N = null; } catch (e) { N = null; }
}

const big = (w, h) => !!N && w * h >= MIN_PX;

// Any failure (an addon that loads but misbehaves inside Photoshop) switches native off for the rest of
// the session: the wrapper returns null and the caller runs its JS version, so a run never fails here.
function call(fn) {
  if (!N) return null;
  try {
    return fn();
  } catch (e) {
    try { console.error("film dust: native filters off for this session", e); } catch (e2) { /* no console */ }
    N = null;
    return null;
  }
}
const asF32 = (a) => (a instanceof Float32Array ? a : Float32Array.from(a));
const out = (ab, dst) => {
  const r = new Float32Array(ab);
  if (!dst) return r;
  dst.set(r);
  return dst;
};

function median8f(Y, w, h, k) {
  return call(() => {
    const y = asF32(Y);
    return new Float32Array(N.median8f(y.buffer, y.byteOffset, w, h, k));
  });
}

// mode: 0 erode, 1 dilate, 2 open, 3 close; hw from filters.ellipseHalfWidths
function ellipse(src, w, h, k, hw, mode, dst) {
  return call(() => {
    const s = asF32(src), H = Int32Array.from(hw);
    return out(N.ellipse(s.buffer, s.byteOffset, w, h, k, H.buffer, 0, mode), dst);
  });
}

function sepConv(src, w, h, ker, dst) {
  return call(() => {
    const s = asF32(src), K = Float32Array.from(ker);
    return out(N.sepConv(s.buffer, s.byteOffset, w, h, K.buffer, 0, K.length), dst);
  });
}

function box3(src, w, h, radii, dst) {
  return call(() => {
    const s = asF32(src), R = Int32Array.from(radii);
    return out(N.box3(s.buffer, s.byteOffset, w, h, R.buffer, 0, R.length), dst);
  });
}

function boxBlur(src, w, h, k, dst) {
  return call(() => {
    const s = asF32(src);
    return out(N.boxBlur(s.buffer, s.byteOffset, w, h, k), dst);
  });
}

function sobel(src, w, h, dx, dy, dst) {
  return call(() => {
    const s = asF32(src);
    return out(N.sobel(s.buffer, s.byteOffset, w, h, dx, dy), dst);
  });
}

// the dust network: weights once (flat, in native/src/dustnet.cpp's order), then a batch of inputs
function netLoad(flat, towers, hidden, slot = 0) {
  return call(() => {
    return N.netLoad(flat.buffer, flat.byteOffset, flat.length, towers, hidden, slot) === 1;
  });
}

function netForward(inputs, n, slot = 0) {
  return call(() => {
    return new Float64Array(N.netForward(inputs.buffer, inputs.byteOffset, n, slot));
  });
}

// the logits of marks at centres (Int32Array of cy, cx pairs), their inputs made natively from the 16-bit image
function netForwardAt(img, W, H, C, centres, n, bright, slot = 0) {
  return call(() => {
    if (!N.netForwardAt) return null;
    return new Float64Array(N.netForwardAt(img.buffer, img.byteOffset, W, H, C, centres.buffer, centres.byteOffset, n, bright ? 1 : 0, slot));
  });
}

// gy[i] = Math.hypot(gx[i], gy[i]) in place, as V8 computes it, on every thread (false: not done)
function hypotInto(gy, gx) {
  if (!big(gy.length, 1) || !(gy instanceof Float32Array) || !(gx instanceof Float32Array) || gx.length !== gy.length) return false;
  return !!call(() => {
    if (!N.hypotInto) return null;
    N.hypotInto(gy.buffer, gy.byteOffset, gx.buffer, gx.byteOffset, gy.length);
    return true;
  });
}

// detect.js BlobField tiles at scale sq, several at once (list: ty, tx pairs): [{ ys, xs, vals, resp, y0, x0, w }]
// in the order asked, each in its own arrays (so a tile whose response is let go frees it); null: not done
function blobTiles(Y, sl, W, H, T, sq, sgn, ker, k, mg, list) {
  return call(() => {
    if (!N.blobTiles) return null;
    const n = list.length / 2, tl = Int32Array.from(list), K = Float32Array.from(ker);
    const buf = N.blobTiles(Y.buffer, Y.byteOffset, sl.buffer, sl.byteOffset, W, H, T, sq, sgn, K.buffer, 0, K.length, k, mg, tl.buffer, 0, n);
    const counts = new Int32Array(buf, 0, n);
    let o = 4 * n;
    const res = [];
    for (let i = 0; i < n; i++) {
      const y0 = list[2 * i] * T, x0 = list[2 * i + 1] * T, y1 = Math.min(H, y0 + T), x1 = Math.min(W, x0 + T), c = counts[i];
      const ys = new Int32Array(buf, o, c).slice(); o += 4 * c;
      const xs = new Int32Array(buf, o, c).slice(); o += 4 * c;
      const vals = new Float32Array(buf, o, c).slice(); o += 4 * c;
      const resp = new Float32Array(buf, o, (y1 - y0) * (x1 - x0)).slice(); o += 4 * (y1 - y0) * (x1 - x0);
      res.push({ ys, xs, vals, resp, y0, x0, w: x1 - x0 });
    }
    return res;
  });
}

// regions.js skeletonize (Uint8Array 0/1) -> Uint8Array, or null
function skeletonize(mask, w, h) {
  return call(() => (N.skeletonize ? new Uint8Array(N.skeletonize(mask.buffer, mask.byteOffset, w, h)) : null));
}

// maps.js: per-block statistic (mode 0 noiseLocal, 1 busyness), grain by brightness, the z loop
function blockStats(resid, valid, W, H, block, mode, bright) {
  return call(() => {
    return new Float32Array(N.blockStats(resid.buffer, resid.byteOffset, valid.buffer, valid.byteOffset, W, H, block, mode, bright ? 1 : 0));
  });
}

function noiseByLevel(bg, resid, valid, W, H, bins) {
  return call(() => {
    return new Float32Array(N.noiseByLevel(bg.buffer, bg.byteOffset, resid.buffer, resid.byteOffset, valid.buffer, valid.byteOffset, W, H, bins));
  });
}

function zMaps(Y, m15, o15, m41, o41, bxl, sigma, outside, bright) {
  return call(() => {
    const n = Y.length, a = [Y, m15, o15, m41, o41, bxl, sigma].map(asF32), args = [];
    for (const v of a) args.push(v.buffer, v.byteOffset);
    const ab = N.zMaps(...args, outside.buffer, outside.byteOffset, n, bright ? 1 : 0);
    return [new Float32Array(ab, 0, n), new Float32Array(ab, 4 * n, n), new Float32Array(ab, 8 * n, n)];
  });
}

// filters.js: [gy, gx] and cv2.resize INTER_LINEAR
function npGradient(src, w, h) {
  return call(() => {
    const s = asF32(src), n = w * h, ab = N.npGradient(s.buffer, s.byteOffset, w, h);
    return [new Float32Array(ab, 0, n), new Float32Array(ab, 4 * n, n)];
  });
}

function resizeLinear(src, sw, sh, dw, dh) {
  return call(() => {
    const s = asF32(src);
    return new Float32Array(N.resizeLinear(s.buffer, s.byteOffset, sw, sh, dw, dh));
  });
}

// regions.js label: { lab: Int32Array, n } (the count comes back after the labels)
function label(mask, w, h, conn) {
  return call(() => {
    const n = w * h, ab = N.label(mask.buffer, mask.byteOffset, w, h, conn);
    return { lab: new Int32Array(ab, 0, n), n: new Int32Array(ab, 4 * n, 1)[0] };
  });
}

// filters.js binary erode / dilate on 0/1 masks: square, or the ellipse with half-widths hw
function binMorph(mask, w, h, k, dilate, hw) {
  return call(() => {
    const H = hw ? Int32Array.from(hw) : new Int32Array(1);
    return new Uint8Array(N.binMorph(mask.buffer, mask.byteOffset, w, h, k, H.buffer, 0, (hw ? 2 : 0) + (dilate ? 1 : 0)));
  });
}

// maps.js outsideMask
function outsideMask(Y, W, H) {
  return call(() => {
    const y = asF32(Y);
    return new Uint8Array(N.outsideMask(y.buffer, y.byteOffset, W, H));
  });
}

// detect.js ridgeMaps: { best, bestS } over the given scales (Gaussian kernels from filters.gaussKernel)
function ridgeMaps(Y, W, H, outside, bright, sig, scales, kernels) {
  return call(() => {
    const y = asF32(Y), s = asF32(sig), sc = Float64Array.from(scales), lens = Int32Array.from(kernels.map((k) => k.length));
    const ker = new Float32Array(lens.reduce((a, b) => a + b, 0));
    let o = 0;
    for (const k of kernels) { ker.set(k, o); o += k.length; }
    const n = W * H;
    const ab = N.ridgeMaps(y.buffer, y.byteOffset, outside.buffer, outside.byteOffset, s.buffer, s.byteOffset, W, H, bright ? 1 : 0,
      sc.buffer, 0, ker.buffer, 0, lens.buffer, 0, scales.length);
    return { best: new Float32Array(ab, 0, n), bestS: new Float32Array(ab, 4 * n, n) };
  });
}

// detect.js find() stage 3 over all objects (cues.cpp). solidBatch: each object's solid fill and its dilation (objd,
// ellipse dk; the fill itself when dk is 0), how (0 kept, 1 grown, 2 filled), area, 2 x maxDist; masks back to back at
// offs. cuesBatch: markCues per object as the JS loop saw cand (the claims sorted by pixel cgi, the claiming object's
// index cidx); n x 10 doubles, NaN where markCues set nothing.
function solidBatch(Y, sig, glab, cand, W, H, n, boxes, ids, bright, k1, hw, dk, offs) {
  return call(() => {
    const y = asF32(Y), s = asF32(sig), I = Int32Array.from(ids);
    const ab = N.solidBatch(y.buffer, y.byteOffset, s.buffer, s.byteOffset, glab.buffer, glab.byteOffset, cand.buffer, cand.byteOffset, W, H, n,
      boxes.buffer, boxes.byteOffset, I.buffer, 0, bright ? 1 : 0, k1.buffer, k1.byteOffset, k1.length, hw.pack.buffer, 0, hw.start.buffer, 0, dk,
      offs.buffer, offs.byteOffset);
    const tot = offs[n], pad = (8 - ((2 * tot) % 8)) % 8, b = 2 * tot + pad;
    return { solid: new Uint8Array(ab, 0, tot), objd: new Uint8Array(ab, tot, tot), how: new Int32Array(ab, b, n), area: new Int32Array(ab, b + 4 * n, n), width: new Float64Array(ab, b + 8 * n, n) };
  });
}

function cuesBatch(Y, G3, E25, E1, sig, cand, cgi, cidx, W, H, n, boxes, offs, objd, bright, gradCal, k1, k3, hw) {
  return call(() => {
    const y = asF32(Y), g3 = asF32(G3), e25 = asF32(E25), e1 = E1 ? asF32(E1) : new Float32Array(1), s = asF32(sig);
    const gi = cgi.length ? cgi : new Int32Array(1), ix = cidx.length ? cidx : new Int32Array(1);
    return new Float64Array(N.cuesBatch(y.buffer, y.byteOffset, g3.buffer, g3.byteOffset, e25.buffer, e25.byteOffset, e1.buffer, e1.byteOffset, E1 ? 1 : 0,
      s.buffer, s.byteOffset, cand.buffer, cand.byteOffset, gi.buffer, gi.byteOffset, ix.buffer, ix.byteOffset, cgi.length, W, H, n,
      boxes.buffer, boxes.byteOffset, offs.buffer, offs.byteOffset, objd.buffer, objd.byteOffset, bright ? 1 : 0, gradCal,
      k1.buffer, k1.byteOffset, k1.length, k3.buffer, k3.byteOffset, k3.length, hw.pack.buffer, 0, hw.start.buffer, 0));
  });
}

// detect.js judgeSoft's measurements for every seed (soft.cpp). seeds: Int32Array n x 7 (gy, gx, 0, W0 y0, y1, x0, x1).
// Returns { fixed: Float64Array n x 24, list(i, j): Int32Array (j 0 own region, 1 whole object, 2 fill) }.
function softBatch(Y, sig, img, C, Ci, W, H, bright, n, seeds, k15, k08, k25, softFill, socketOn) {
  return call(() => {
    const y = asF32(Y), s = asF32(sig);
    const ab = N.softBatch(y.buffer, y.byteOffset, s.buffer, s.byteOffset, img.buffer, img.byteOffset, C, Ci, W, H, bright ? 1 : 0, n,
      seeds.buffer, seeds.byteOffset, k15.buffer, k15.byteOffset, k15.length, k08.buffer, k08.byteOffset, k08.length, k25.buffer, k25.byteOffset, k25.length,
      softFill, socketOn ? 1 : 0);
    const fixed = new Float64Array(ab, 0, n * 24), no = 3 * n + 1, offBytes = Math.ceil((no * 4) / 8) * 8;
    const offs = new Int32Array(ab, n * 24 * 8, no), data = new Int32Array(ab, n * 24 * 8 + offBytes, offs[no - 1]);
    return { fixed, list: (i, j) => data.subarray(offs[3 * i + j], offs[3 * i + j + 1]) };
  });
}

// regions.js fillHoles
function fillHoles(mask, w, h) {
  return call(() => new Uint8Array(N.fillHoles(mask.buffer, mask.byteOffset, w, h)));
}

// filters.js rectMorph (float square max / min) and scale.js averageStrip (fills dst, returns rows)
function rectMorph(src, w, h, k, isMax, dst) {
  return call(() => {
    const s = asF32(src);
    return out(N.rectMorph(s.buffer, s.byteOffset, w, h, k, isMax ? 1 : 0), dst);
  });
}

function averageStrip(src, srcW, rows, C, k, dst, Wd, yd0, eight) {
  return call(() => {
    return N.averageStrip(src.buffer, src.byteOffset, eight ? 1 : 0, srcW, rows, C, k, dst.buffer, dst.byteOffset, Wd, yd0);
  });
}

module.exports = {
  available: () => !!N, big, threads: (n) => (N ? N.threads(n || 0) : 0), ridgeMaps, rectMorph, averageStrip, fillHoles, solidBatch, cuesBatch, netForwardAt, hypotInto, blobTiles, skeletonize, softBatch,
  median8f, ellipse, sepConv, box3, boxBlur, sobel, netLoad, netForward,
  blockStats, noiseByLevel, zMaps, npGradient, resizeLinear, label, binMorph, outsideMask,
};

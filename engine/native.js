"use strict";
// Native filters (native/src, built by native/build.sh): multi-threaded C++ ports of the heaviest
// whole-image filters, bit-identical to their JS versions in filters.js / maps.js.
//   In Photoshop: the hybrid addon win/x64/filmdust.uxpaddon (manifest "addon"), when installed.
//   In node (tests): native/build/filmdust.node, unless FILMDUST_NATIVE=0.
// When neither loads, `available` is false and the engine runs its JS filters unchanged.
// Tiny images stay in JS: below about 64 x 64 handing them over costs more than the work (calls are
// cheap otherwise: the native side keeps its worker threads, and small jobs run on the calling thread).
const MIN_PX = 1 << 12;

let loadError = null;   // why the hybrid addon didn't load in Photoshop (the panel writes it to PluginData/native.txt)

// an error as text, even one with no message (some host errors carry none)
function why(e) {
  try {
    if (e == null) return String(e);
    const stack = e.stack ? String(e.stack).split(/\r?\n/).slice(0, 3).join(" / ") : "";
    const t = [e.name, e.message, e.code, stack].filter((x) => x != null && String(x) !== "").map(String);
    return (t.length ? t.join(" · ") : String(e)) || `${typeof e} with no text`;
  } catch (e2) {
    return "an error that can't be read";
  }
}
// what require handed back, for the log
function shape(m) {
  try {
    if (m == null) return String(m);
    const keys = Object.getOwnPropertyNames(m).slice(0, 12).join(",");
    return `${typeof m}${typeof m.then === "function" ? " (a promise)" : ""} [${keys}] version: ${typeof m.version}`;
  } catch (e) {
    return `${typeof m} (unreadable: ${why(e)})`;
  }
}

let inUxp = false;
function load() {
  let uxp = null;
  try { uxp = require("uxp"); } catch (e) { uxp = null; }
  if (uxp) {
    inUxp = true;
    // a missing or refused addon leaves the JS filters running; the panel writes why to PluginData/native.txt
    try { return require("filmdust.uxpaddon"); } catch (e) {
      loadError = `require failed: ${why(e)}`;
      try { console.error("film dust: native addon not loaded:", loadError); } catch (e2) { /* no console */ }
      return null;
    }
  }
  try {
    if (typeof process !== "undefined" && process.env && process.env.FILMDUST_NATIVE === "0") return null;
    // FILMDUST_NATIVE_PATH: another build of filmdust.node (tests that compare builds)
    return require(process.env.FILMDUST_NATIVE_PATH || require("path").join(__dirname, "..", "native", "build", "filmdust.node"));
  } catch (e) {
    return null;
  }
}

// the addon as loaded, checked: version() must answer 1 or more
function checked(m) {
  if (!m) {
    if (inUxp && !loadError) loadError = "require returned nothing";
    return null;
  }
  try {
    const v = m.version();
    if (v >= 1) {
      named = namedFields(m);
      return m;
    }
    if (inUxp) loadError = `addon loaded, but version() gave ${typeof v} ${String(v)}; exports ${shape(m)}`;
  } catch (e) {
    if (inUxp) loadError = `addon loaded, but version() failed: ${why(e)}; exports ${shape(m)}`;
  }
  return null;
}

// Photoshop 2026 (UXP 9) hands a hybrid addon over as a promise: native is on once it resolves (ready()); runs and
// the panel wait for that first, so nothing reads N before it is known
let N = null;
let readyP = Promise.resolve();
// blobTiles, ridgeMaps, solidBatch, cuesBatch and softBatch take one object of named fields, which needs two more
// calls of the host's addon API (an object's field, an array's element): checked once at load. Without them only
// those five steps run in JS (the panel's native.txt says so).
let named = false;
function namedFields(m) {
  try {
    return !!m.selfTest && m.selfTest(4, { n: 2, v: [new Float64Array([3]).buffer, 0] }) === 5;
  } catch (e) {
    return false;
  }
}
try {
  const m = load();
  if (m && typeof m.then === "function") {
    readyP = Promise.resolve(m).then((mod) => { N = checked(mod); }, (e) => { loadError = `the addon's promise failed: ${why(e)}`; N = null; });
  } else N = checked(m);
} catch (e) { N = null; if (inUxp) loadError = `load threw: ${why(e)}`; }

const big = (w, h) => !!N && w * h >= MIN_PX;

// Any failure (an addon that loads but misbehaves inside Photoshop) switches native off for the rest of
// the session: the wrapper returns null and the caller runs its JS version, so a run never fails here.
function call(fn) {
  if (!N) return null;
  try {
    return fn();
  } catch (e) {
    try { console.error("film dust: native filters off for this session", e); } catch (e2) { /* no console */ }
    loadError = `loaded, then switched off mid-run: ${why(e)}`;
    N = null;
    return null;
  }
}
const asF32 = (a) => (a instanceof Float32Array ? a : Float32Array.from(a));
// a typed array as the [ArrayBuffer, byteOffset] pair the addon reads (its API has no typed-array accessor)
const ab = (a) => [a.buffer, a.byteOffset];
const out = (buf, dst) => {
  const r = new Float32Array(buf);
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
  if (!named) return null;
  return call(() => {
    if (!N.blobTiles) return null;
    const n = list.length / 2, tl = Int32Array.from(list), K = Float32Array.from(ker);
    const buf = N.blobTiles({ Y: ab(Y), sl: ab(sl), W, H, T, sq, sgn, ker: ab(K), klen: K.length, k, mg, tiles: ab(tl), n });
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
    const res = N.zMaps(...args, outside.buffer, outside.byteOffset, n, bright ? 1 : 0);
    return [new Float32Array(res, 0, n), new Float32Array(res, 4 * n, n), new Float32Array(res, 8 * n, n)];
  });
}

// the same z loop written over o15, o41 and bxl, which the caller then no longer has: [z_s, z_l, z_xl] are those
// three arrays (each pixel's inputs are read before its results are stored), so no new maps are made
function zMapsInto(Y, m15, o15, m41, o41, bxl, sigma, outside, bright) {
  const maps = [Y, m15, o15, m41, o41, bxl, sigma];
  if (!maps.every((a) => a instanceof Float32Array && a.length === Y.length)) return null;
  return call(() => {
    if (!N.zMapsInto) return null;
    const args = [];
    for (const v of maps) args.push(v.buffer, v.byteOffset);
    N.zMapsInto(...args, outside.buffer, outside.byteOffset, Y.length, bright ? 1 : 0);
    return [o15, o41, bxl];
  });
}

// filters.js: [gy, gx] and cv2.resize INTER_LINEAR
function npGradient(src, w, h) {
  return call(() => {
    const s = asF32(src), n = w * h, res = N.npGradient(s.buffer, s.byteOffset, w, h);
    return [new Float32Array(res, 0, n), new Float32Array(res, 4 * n, n)];
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
    const n = w * h, res = N.label(mask.buffer, mask.byteOffset, w, h, conn);
    return { lab: new Int32Array(res, 0, n), n: new Int32Array(res, 4 * n, 1)[0] };
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
  if (!named) return null;
  return call(() => {
    const y = asF32(Y), s = asF32(sig), sc = Float64Array.from(scales), lens = Int32Array.from(kernels.map((k) => k.length));
    const ker = new Float32Array(lens.reduce((a, b) => a + b, 0));
    let o = 0;
    for (const k of kernels) { ker.set(k, o); o += k.length; }
    const n = W * H;
    const res = N.ridgeMaps({ Y: ab(y), outside: ab(outside), sig: ab(s), W, H, bright: bright ? 1 : 0, scales: ab(sc), kers: ab(ker), klens: ab(lens), ns: scales.length });
    return { best: new Float32Array(res, 0, n), bestS: new Float32Array(res, 4 * n, n) };
  });
}

// detect.js find() stage 3 over all objects (cues.cpp). solidBatch: each object's solid fill and its dilation (objd,
// ellipse dk; the fill itself when dk is 0), how (0 kept, 1 grown, 2 filled), area, 2 x maxDist; masks back to back at
// offs. cuesBatch: markCues per object as the JS loop saw cand (the claims sorted by pixel cgi, the claiming object's
// index cidx); n x 10 doubles, NaN where markCues set nothing.
function solidBatch(Y, sig, glab, cand, W, H, n, boxes, ids, bright, k1, hw, dk, offs) {
  if (!named) return null;
  return call(() => {
    const y = asF32(Y), s = asF32(sig), I = Int32Array.from(ids);
    const res = N.solidBatch({ Y: ab(y), sig: ab(s), glab: ab(glab), cand: ab(cand), W, H, n, boxes: ab(boxes), ids: ab(I), bright: bright ? 1 : 0,
      k1: ab(k1), k1n: k1.length, hwPack: ab(hw.pack), hwStart: ab(hw.start), dk, offs: ab(offs) });
    const tot = offs[n], pad = (8 - ((2 * tot) % 8)) % 8, b = 2 * tot + pad;
    return { solid: new Uint8Array(res, 0, tot), objd: new Uint8Array(res, tot, tot), how: new Int32Array(res, b, n), area: new Int32Array(res, b + 4 * n, n), width: new Float64Array(res, b + 8 * n, n) };
  });
}

function cuesBatch(Y, G3, E25, E1, sig, cand, cgi, cidx, W, H, n, boxes, offs, objd, bright, gradCal, k1, k3, hw) {
  if (!named) return null;
  return call(() => {
    const y = asF32(Y), g3 = asF32(G3), e25 = asF32(E25), s = asF32(sig);
    return new Float64Array(N.cuesBatch({ Y: ab(y), G3: ab(g3), E25: ab(e25), E1: E1 ? ab(asF32(E1)) : null, sig: ab(s), cand: ab(cand),
      cgi: ab(cgi), cidx: ab(cidx), nc: cgi.length, W, H, n, boxes: ab(boxes), offs: ab(offs), objd: ab(objd), bright: bright ? 1 : 0, gradCal,
      k1: ab(k1), k1n: k1.length, k3: ab(k3), k3n: k3.length, hwPack: ab(hw.pack), hwStart: ab(hw.start) }));
  });
}

// detect.js judgeSoft's measurements for every seed (soft.cpp). seeds: Int32Array n x 7 (gy, gx, 0, W0 y0, y1, x0, x1).
// Returns { fixed: Float64Array n x 24, list(i, j): Int32Array (j 0 own region, 1 whole object, 2 fill) }.
function softBatch(Y, sig, img, C, Ci, W, H, bright, n, seeds, k15, k08, k25, softFill, socketOn) {
  if (!named) return null;
  return call(() => {
    const y = asF32(Y), s = asF32(sig);
    const res = N.softBatch({ Y: ab(y), sig: ab(s), img: ab(img), C, Ci, W, H, bright: bright ? 1 : 0, n, seeds: ab(seeds),
      k15: ab(k15), k15n: k15.length, k08: ab(k08), k08n: k08.length, k25: ab(k25), k25n: k25.length, softFill, socketOn: socketOn ? 1 : 0 });
    const fixed = new Float64Array(res, 0, n * 24), no = 3 * n + 1, offBytes = Math.ceil((no * 4) / 8) * 8;
    const offs = new Int32Array(res, n * 24 * 8, no), data = new Int32Array(res, n * 24 * 8 + offBytes, offs[no - 1]);
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
  ready: () => readyP, available: () => !!N, loadError: () => loadError,
  note: () => (N && !named ? "named-field calls unavailable here: five steps run in JS" : ""), big, threads: (n) => (N ? N.threads(n || 0) : 0), ridgeMaps, rectMorph, averageStrip, fillHoles, solidBatch, cuesBatch, netForwardAt, hypotInto, blobTiles, skeletonize, softBatch,
  median8f, ellipse, sepConv, box3, boxBlur, sobel, netLoad, netForward,
  blockStats, noiseByLevel, zMaps, zMapsInto, npGradient, resizeLinear, label, binMorph, outsideMask,
};

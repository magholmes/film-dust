"use strict";
// The dust network (port of film-scan-cleanup/scripts/dustnet.py): a small convolutional net that looks
// at the patch around a candidate mark and says how much it looks like dust on the film (a logit;
// > 0 = dust). Trained on fake dust laid onto his own photographs (tests/dustnet/ in the skill).
//   version 1: one view, the 40 x 40 patch around the mark
//   version 2: two views, that patch and the 160 x 160 neighbourhood averaged 4 x 4 down to 40 x 40,
//              so it sees whether a line runs on into a stem or a speck sits among many glints
// Float64 throughout, logit rounded to 3 decimals before anything uses it, as the skill does (its
// matrix products and these loops differ only around the 13th decimal). With the native module
// (engine/native.js) the same loops run in C++ over all marks at once, bit for bit the same.
const R = require("./regions");
const NAT = require("./native");
let NET = null;
try { NET = require("./dustnet_weights"); } catch (e) { NET = null; }

const S = 40, HALF = 20, CTX = 160, CH = CTX / 2;

// the median by selection on a scratch copy (the same value as sorting: the middle one, or the mean of the two)
let scratch = new Float64Array(S * S);
function median(v) {
  const n = v.length;
  if (scratch.length < n) scratch = new Float64Array(n);
  for (let i = 0; i < n; i++) scratch[i] = v[i];
  return R.medianOf(scratch, n);
}

// minus each channel's median, / the patch's own grain (1.0483 x the median |difference| of
// horizontal neighbours in its luminance), sign-flipped for dark dust, clipped to +-12, x 0.25
function normalise(vals, bright) {
  const sgn = bright ? 1.0 : -1.0;
  const d = new Float64Array(S * (S - 1));
  let nd = 0;
  for (let r = 0; r < S; r++) for (let q = 0; q < S - 1; q++) {
    const k = r * S + q;
    const a = 0.2126 * vals[0][k] + 0.7152 * vals[1][k] + 0.0722 * vals[2][k];
    const b = 0.2126 * vals[0][k + 1] + 0.7152 * vals[1][k + 1] + 0.0722 * vals[2][k + 1];
    d[nd++] = Math.abs(b - a);
  }
  const s = Math.max(median(d) * 1.0483, 1e-4);
  const X = [new Float64Array(S * S), new Float64Array(S * S), new Float64Array(S * S)];
  for (let c = 0; c < 3; c++) {
    const m = median(vals[c]);
    const v = vals[c], x = X[c];
    for (let k = 0; k < S * S; k++) x[k] = Math.min(12.0, Math.max(-12.0, sgn * (v[k] - m) / s)) * 0.25;
  }
  return X;
}

// the v1 input for a mark at (cy, cx): the patch's RGB as the detector holds it (float32 of
// value / 65535). Rows and columns past the image repeat its edge.
function patchInput(img, W, H, C, cy, cx, bright) {
  const vals = [new Float64Array(S * S), new Float64Array(S * S), new Float64Array(S * S)];
  for (let r = 0; r < S; r++) {
    const y = Math.min(H - 1, Math.max(0, cy - HALF + r));
    for (let q = 0; q < S; q++) {
      const x = Math.min(W - 1, Math.max(0, cx - HALF + q));
      const o = (y * W + x) * C;
      for (let c = 0; c < 3; c++) vals[c][r * S + q] = Math.fround(img[o + (C === 1 ? 0 : c)] / 65535);
    }
  }
  return normalise(vals, bright);
}

// the v2 context view: the 160 x 160 neighbourhood (rows cy-80 .. cy+79, edges repeated), each 4 x 4
// block summed in row order and divided by 16 (as dustnet.py's block_mean4)
function contextInput(img, W, H, C, cy, cx, bright) {
  const vals = [new Float64Array(S * S), new Float64Array(S * S), new Float64Array(S * S)];
  const xs = new Int32Array(CTX);
  for (let q = 0; q < CTX; q++) xs[q] = Math.min(W - 1, Math.max(0, cx - CH + q));
  const acc = new Float64Array(S * S * 3);
  for (let by = 0; by < S; by++) {
    for (let dy = 0; dy < 4; dy++) {
      const y = Math.min(H - 1, Math.max(0, cy - CH + 4 * by + dy));
      const ro = y * W;
      for (let bx = 0; bx < S; bx++) {
        const k = by * S + bx;
        for (let dx = 0; dx < 4; dx++) {
          const o = (ro + xs[4 * bx + dx]) * C;
          for (let c = 0; c < 3; c++) acc[c * S * S + k] += Math.fround(img[o + (C === 1 ? 0 : c)] / 65535);
        }
      }
    }
  }
  for (let c = 0; c < 3; c++) for (let k = 0; k < S * S; k++) vals[c][k] = acc[c * S * S + k] / 16.0;
  return normalise(vals, bright);
}

// 3 x 3 'same' convolution: out[o] = b[o] + sum over input channel, ky, kx in that order
function conv(x, n, w, b) {
  const ci = x.length, co = b.length;
  const out = [];
  for (let o = 0; o < co; o++) {
    const acc = new Float64Array(n * n).fill(b[o]);
    for (let i = 0; i < ci; i++) {
      const xi = x[i];
      for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
        const wv = w[o][i][ky][kx];
        for (let y = 0; y < n; y++) {
          const sy = y + ky - 1;
          if (sy < 0 || sy >= n) {
            // zero padding: the term is wv * 0
            for (let xx = 0; xx < n; xx++) acc[y * n + xx] += wv * 0.0;
            continue;
          }
          for (let xx = 0; xx < n; xx++) {
            const sx = xx + kx - 1;
            acc[y * n + xx] += wv * (sx < 0 || sx >= n ? 0.0 : xi[sy * n + sx]);
          }
        }
      }
    }
    for (let k = 0; k < n * n; k++) if (!(acc[k] > 0)) acc[k] = 0.0;   // ReLU
    out.push(acc);
  }
  return out;
}

function pool(x, n) {
  const m = n >> 1;
  return x.map((a) => {
    const p = new Float64Array(m * m);
    for (let y = 0; y < m; y++) for (let xx = 0; xx < m; xx++) {
      const i = 2 * y * n + 2 * xx;
      p[y * m + xx] = Math.max(a[i], a[i + 1], a[i + n], a[i + n + 1]);
    }
    return p;
  });
}

// one tower: 5 convolutions (2 x 2 pooling after the 2nd, 3rd, 4th) -> mean, max, centre (48)
function tower(X, L, pre) {
  let a = conv(X, 40, L[pre + "c1w"], L[pre + "c1b"]);
  a = conv(a, 40, L[pre + "c2w"], L[pre + "c2b"]);
  a = pool(a, 40);
  a = conv(a, 20, L[pre + "c3w"], L[pre + "c3b"]);
  a = pool(a, 20);
  a = conv(a, 10, L[pre + "c4w"], L[pre + "c4b"]);
  a = pool(a, 10);
  a = conv(a, 5, L[pre + "c5w"], L[pre + "c5b"]);
  const feats = [];
  for (const ch of a) { let t = 0.0; for (let k = 0; k < ch.length; k++) t += ch[k]; feats.push(t / ch.length); }
  for (const ch of a) { let m = -Infinity; for (let k = 0; k < ch.length; k++) if (ch[k] > m) m = ch[k]; feats.push(m); }
  for (const ch of a) feats.push(ch[2 * 5 + 2]);
  return feats;
}

// One network: its weights, its forward pass, its slot in the native module. make() builds one;
// the module's own functions are the shipped v1 network, `second` the v2 network (patch + context).
function make(NETW, slot) {
  const V2n = !!NETW && (NETW.version || 1) >= 2;
  // the logit (unrounded) for one mark's views [local] (v1) or [local, context] (v2)
  function forward(views) {
    const L = NETW.layers;
    const feats = V2n ? tower(views[0], L, "l_").concat(tower(views[1], L, "x_")) : tower(views[0], L, "");
    const h = [];
    for (let o = 0; o < L.d1b.length; o++) {
      let t = L.d1b[o];
      for (let i = 0; i < feats.length; i++) t += L.d1w[o][i] * feats[i];
      h.push(t > 0 ? t : 0.0);
    }
    let t = L.d2b[0];
    for (let i = 0; i < h.length; i++) t += L.d2w[0][i] * h[i];
    return t;
  }

  function views(img, W, H, C, cy, cx, bright) {
    const v = [patchInput(img, W, H, C, cy, cx, bright)];
    if (V2n) v.push(contextInput(img, W, H, C, cy, cx, bright));
    return v;
  }

  // native: the weights flattened once, in the order native/src/dustnet.cpp reads them
  let natReady = null;
  function nativeNet() {
    if (natReady !== null) return natReady;
    natReady = false;
    if (!NAT.available() || !NAT.netLoad) return false;
    const L = NETW.layers, flat = [];
    const push = (v) => { if (Array.isArray(v)) for (const e of v) push(e); else flat.push(v); };
    for (const pre of V2n ? ["l_", "x_"] : [""]) for (let i = 1; i <= 5; i++) { push(L[pre + `c${i}w`]); push(L[pre + `c${i}b`]); }
    push(L.d1w); push(L.d1b); push(L.d2w); push(L.d2b);
    natReady = NAT.netLoad(Float64Array.from(flat), V2n ? 2 : 1, L.d1b.length, slot) === true;
    return natReady;
  }

  // the network's opinions (logits, 3 decimals) of marks at centres [[cy, cx], ...]
  function logits(img, W, H, C, centres, bright) {
    if (!NETW) return centres.map(() => null);
    // nothing to judge: the addon refuses an empty buffer, and any native error switches native off for the session
    if (!centres.length) return [];
    const T = V2n ? 2 : 1;
    // native, from the image itself: the inputs made on every thread, no buffer of them
    // (FD_NO_NETAT=1 in node: the inputs made in JS, for checking that both give the same)
    if (nativeNet() && NAT.available() && img instanceof Uint16Array && NAT.netForwardAt && !(typeof process !== "undefined" && process.env && process.env.FD_NO_NETAT)) {
      const cen = new Int32Array(2 * centres.length);
      centres.forEach(([cy, cx], i) => { cen[2 * i] = cy; cen[2 * i + 1] = cx; });
      const z = NAT.netForwardAt(img, W, H, C, cen, centres.length, bright, slot);
      if (z) return Array.from(z, (v) => R.pyRoundTo(v, 3));
    }
    if (nativeNet() && NAT.available()) {
      const per = T * 3 * S * S, buf = new Float64Array(centres.length * per);
      centres.forEach(([cy, cx], i) => {
        const v = views(img, W, H, C, cy, cx, bright);
        for (let t = 0; t < T; t++) for (let c = 0; c < 3; c++) buf.set(v[t][c], i * per + (t * 3 + c) * S * S);
      });
      const z = NAT.netForward(buf, centres.length, slot);
      if (z) return Array.from(z, (v) => R.pyRoundTo(v, 3));
    }
    return centres.map(([cy, cx]) => R.pyRoundTo(forward(views(img, W, H, C, cy, cx, bright)), 3));
  }

  // the network's opinion (logit, 3 decimals) of the mark at (cy, cx), or null without a trained network
  function logit(img, W, H, C, cy, cx, bright) {
    if (!NETW) return null;
    return logits(img, W, H, C, [[cy, cx]], bright)[0];
  }

  return { logit, logits, forward, views, version: () => (NETW ? NETW.version || 1 : 0), available: () => !!NETW };
}

const first = make(NET, 0);
let NET2 = null;
try { NET2 = require("./dustnet2_weights"); } catch (e) { NET2 = null; }
const second = make(NET2, 1);

module.exports = Object.assign({}, first, { patchInput, contextInput, normalise, make, second });

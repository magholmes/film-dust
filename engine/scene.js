"use strict";
// The scene model (training/scene): what the whole picture says about each mark. A MobileNetV3-small (ImageNet
// weights, batch norms folded, float16) reads the scan's thumbnail (1024 px on its long side); its features after
// blocks 3, 8 and 11 (strides 8, 16, 32) are read at each mark's place through a 1 x 1 projection each, joined with
// the engine's own numbers for the mark, and a small MLP says how likely the mark is dust (a logit, > 0 = dust).
// Float32 throughout, in the order training/scene/export_scene.py's ref_forward uses (test/scene_parity.js).
let PACK = null;
try {
  // tests may point at other weights (FILMDUST_SCENE_WEIGHTS, a path)
  const alt = typeof process !== "undefined" && process.env && process.env.FILMDUST_SCENE_WEIGHTS;
  PACK = require(alt || "./scene_weights");
} catch (e) { PACK = null; }

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

// ---- weights ------------------------------------------------------------------------------------------------
function b64bytes(s) {
  const T = new Uint8Array(128);
  const A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (let i = 0; i < 64; i++) T[A.charCodeAt(i)] = i;
  let n = s.length;
  while (n > 0 && s[n - 1] === "=") n--;
  const out = new Uint8Array((n * 3) >> 2);
  let o = 0, i = 0;
  for (; i + 4 <= n; i += 4) {
    const v = (T[s.charCodeAt(i)] << 18) | (T[s.charCodeAt(i + 1)] << 12) | (T[s.charCodeAt(i + 2)] << 6) | T[s.charCodeAt(i + 3)];
    out[o++] = v >> 16; out[o++] = (v >> 8) & 255; out[o++] = v & 255;
  }
  if (n - i === 2) out[o++] = ((T[s.charCodeAt(i)] << 18) | (T[s.charCodeAt(i + 1)] << 12)) >> 16;
  else if (n - i === 3) {
    const v = (T[s.charCodeAt(i)] << 18) | (T[s.charCodeAt(i + 1)] << 12) | (T[s.charCodeAt(i + 2)] << 6);
    out[o++] = v >> 16; out[o++] = (v >> 8) & 255;
  }
  return out;
}

function half(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  if (e === 0) return s * f * 5.960464477539063e-8;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * Math.pow(2, e - 15);
}

let NET = null;
function net() {
  if (NET || !PACK) return NET;
  const b16 = b64bytes(PACK.DATA16), b32 = b64bytes(PACK.DATA32);
  const u16 = new Uint16Array(b16.buffer, 0, b16.length >> 1);
  const f32 = new Float32Array(b32.buffer, 0, b32.length >> 2);
  const lut = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) lut[i] = half(i);
  const get16 = ([o, n]) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = lut[u16[o + i]]; return a; };
  const get32 = ([o, n]) => f32.slice(o, o + n);
  const L = PACK.LAYOUT;
  NET = {
    long: L.long, fnames: L.fnames,
    layers: L.layers.map((l) => {
      const r = Object.assign({}, l);
      for (const k of ["w", "b", "w1", "b1", "w2", "b2"]) if (Array.isArray(l[k])) r[k] = get16(l[k]);
      return r;
    }),
    proj: L.proj.map((p) => ({ w: get32(p.w), b: get32(p.b), cin: p.cin, cout: p.cout })),
    mlp: L.mlp.map((m) => ({ w: get32(m.w), b: get32(m.b), nin: m.in, nout: m.out })),
    med: get32(L.med), iqr: get32(L.iqr),
  };
  return NET;
}

// ---- thumbnail: 8-bit (value >> 8), area-averaged to `long` px on the long side (never enlarged), as cv2's INTER_AREA
function areaTaps(n, m) {
  // for each output index: [first source index, weights...] covering [j * n / m, (j + 1) * n / m)
  const s = n / m, out = [];
  for (let j = 0; j < m; j++) {
    const a = j * s, b = Math.min(n, (j + 1) * s);
    const i0 = Math.floor(a), i1 = Math.min(n - 1, Math.ceil(b) - 1);
    const w = [];
    for (let i = i0; i <= i1; i++) w.push((Math.min(b, i + 1) - Math.max(a, i)) / s);
    out.push([i0, w]);
  }
  return out;
}

function thumbnail(img, W, H, C, long) {
  const s = Math.min(1, long / Math.max(W, H));
  const w = Math.max(1, Math.round(W * s)), h = Math.max(1, Math.round(H * s));
  const tx = areaTaps(W, w), ty = areaTaps(H, h);
  const rows = new Float64Array(w * 3);
  const out = new Float32Array(3 * h * w);
  for (let j = 0; j < h; j++) {
    rows.fill(0);
    const [y0, wy] = ty[j];
    for (let k = 0; k < wy.length; k++) {
      const ro = (y0 + k) * W * C, fy = wy[k];
      for (let q = 0; q < w; q++) {
        const [x0, wx] = tx[q];
        let r = 0, g = 0, b = 0;
        for (let t = 0; t < wx.length; t++) {
          const o = ro + (x0 + t) * C, f = wx[t];
          if (C === 1) { const v = img[o] >> 8; r += f * v; g += f * v; b += f * v; }
          else { r += f * (img[o] >> 8); g += f * (img[o + 1] >> 8); b += f * (img[o + 2] >> 8); }
        }
        rows[q * 3] += fy * r; rows[q * 3 + 1] += fy * g; rows[q * 3 + 2] += fy * b;
      }
    }
    for (let q = 0; q < w; q++) for (let c = 0; c < 3; c++) out[c * h * w + j * w + q] = Math.min(255, Math.max(0, Math.round(rows[q * 3 + c]))) / 255;
  }
  return { data: out, w, h };
}

// ---- the backbone ----------------------------------------------------------------------------------------------
function conv(x, C, H, Wd, l) {
  const k = l.k, s = l.s, p = (k - 1) >> 1, g = l.g, cout = l.cout;
  const Ho = Math.floor((H + 2 * p - k) / s) + 1, Wo = Math.floor((Wd + 2 * p - k) / s) + 1;
  const P = Ho * Wo, out = new Float32Array(cout * P), w = l.w, b = l.b;
  if (g === 1 && k === 1) {
    // pointwise: out[o] = b[o] + sum_i w[o, i] x[i]
    for (let o = 0; o < cout; o++) {
      const oo = o * P, bo = b[o];
      for (let q = 0; q < P; q++) out[oo + q] = bo;
      for (let i = 0; i < C; i++) {
        const wv = w[o * C + i];
        if (wv === 0) continue;
        const xi = i * P;
        for (let q = 0; q < P; q++) out[oo + q] += wv * x[xi + q];
      }
    }
  } else if (g === 1) {
    // dense k x k (the first layer: 3 -> 16)
    for (let o = 0; o < cout; o++) {
      const oo = o * P, bo = b[o];
      for (let q = 0; q < P; q++) out[oo + q] = bo;
      for (let i = 0; i < C; i++) {
        for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
          const wv = w[((o * C + i) * k + ky) * k + kx];
          for (let oy = 0; oy < Ho; oy++) {
            const iy = oy * s + ky - p;
            if (iy < 0 || iy >= H) continue;
            const xr = (i * H + iy) * Wd, orow = oo + oy * Wo;
            for (let ox = 0; ox < Wo; ox++) {
              const ix = ox * s + kx - p;
              if (ix >= 0 && ix < Wd) out[orow + ox] += wv * x[xr + ix];
            }
          }
        }
      }
    }
  } else {
    // depthwise k x k
    for (let c = 0; c < C; c++) {
      const oo = c * P, bo = b[c], xc = c * H * Wd;
      for (let q = 0; q < P; q++) out[oo + q] = bo;
      for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
        const wv = w[(c * k + ky) * k + kx];
        for (let oy = 0; oy < Ho; oy++) {
          const iy = oy * s + ky - p;
          if (iy < 0 || iy >= H) continue;
          const xr = xc + iy * Wd, orow = oo + oy * Wo;
          const ox0 = Math.max(0, Math.ceil((p - kx) / s)), ox1 = Math.min(Wo - 1, Math.floor((Wd - 1 + p - kx) / s));
          for (let ox = ox0; ox <= ox1; ox++) out[orow + ox] += wv * x[xr + ox * s + kx - p];
        }
      }
    }
  }
  if (l.act === "hs") for (let i = 0; i < out.length; i++) { const v = out[i]; out[i] = v * Math.min(6, Math.max(0, v + 3)) / 6; }
  else if (l.act === "re") for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
  return { x: out, C: cout, H: Ho, W: Wo };
}

function se(t, l) {
  const { x, C, H, W } = t, P = H * W, S = l.b1.length;
  const m = new Float64Array(C);
  for (let c = 0; c < C; c++) { let a = 0; for (let q = 0; q < P; q++) a += x[c * P + q]; m[c] = a / P; }
  const z = new Float64Array(S);
  for (let j = 0; j < S; j++) { let a = l.b1[j]; for (let c = 0; c < C; c++) a += l.w1[j * C + c] * m[c]; z[j] = Math.max(0, a); }
  for (let c = 0; c < C; c++) {
    let a = l.b2[c];
    for (let j = 0; j < S; j++) a += l.w2[c * S + j] * z[j];
    const f = Math.fround(Math.min(6, Math.max(0, a + 3)) / 6);
    for (let q = 0; q < P; q++) x[c * P + q] *= f;
  }
  return t;
}

async function backbone(th, tick) {
  const N = net();
  let t = { x: new Float32Array(th.data.length), C: 3, H: th.h, W: th.w };
  const P = th.h * th.w;
  for (let c = 0; c < 3; c++) for (let q = 0; q < P; q++) t.x[c * P + q] = Math.fround((th.data[c * P + q] - MEAN[c]) / STD[c]);
  const taps = [], stack = [];
  for (const l of N.layers) {
    if (l.op === "conv") t = conv(t.x, t.C, t.H, t.W, l);
    else if (l.op === "begin") stack.push(t);
    else if (l.op === "se") t = se(t, l);
    else if (l.op === "end") {
      const t0 = stack.pop();
      if (l.res) for (let i = 0; i < t.x.length; i++) t.x[i] += t0.x[i];
      if (l.tap) taps.push(t);
      if (tick) await tick();
    }
  }
  return taps;
}

// ---- the engine's numbers for a mark (training/scene/scene_data.py's features) ----------------------------------
const NUM = ["z", "z_fine", "area", "width", "length", "pieces", "white", "sharp", "crisp", "core_y", "bg_y",
  "ring_range", "flat", "broad", "detail", "tex", "busy", "field", "elong", "plateau", "fine", "cosw", "near", "sim", "stronger", "net"];
const LINE = ["one_med", "one_frac", "hf", "dips", "straight", "per", "gaps", "half_w"];
const KINDS = ["hair/fibre", "dust", "speck", "soft"];
const SOFT = ["elong", "plateau", "fine", "cosw", "socket", "crisp_rel"];
const num = (v) => (typeof v === "number" ? v : NaN);

function features(m, W, H) {
  const f = NUM.map((k) => num(m[k]));
  const ln = m.line || {};
  for (const k of LINE) f.push(num(ln[k]));
  for (const k of KINDS) f.push(m.kind === k ? 1 : 0);
  const bg = m.bg;
  f.push(Array.isArray(bg) && bg.length ? bg.reduce((a, v) => a + v, 0) / bg.length : NaN);
  f.push(typeof m.area === "number" ? Math.log1p(m.area) : NaN);
  for (const k of SOFT) f.push(num(m[k]));
  const edge = Math.min(m.cx, W - 1 - m.cx, m.cy, H - 1 - m.cy) / Math.max(W, H);
  f.push(m.verdict === "defect" ? 1 : 0, num(m.learned), (m.polarity || "bright") === "bright" ? 1 : 0, Math.log((W * H) / 1e6), edge,
    m.kind === "soft" || "plateau" in m ? 1 : 0, m.frame ? 1 : 0, m.zone ? 1 : 0);
  return f;
}
const ALL_NAMES = [...NUM, ...LINE.map((k) => "line." + k), ...KINDS.map((k) => "kind=" + k), "bg_mean", "log_area", ...SOFT,
  "verdict", "learned", "bright", "log_mp", "edge", "soft_record", "frame", "zone"];

// ---- the head --------------------------------------------------------------------------------------------------
// rows: [{ x, y, f }] - the mark's place in thumbnail px and its numbers in the model's feature order
function rowsFor(marks, W, H) {
  const N = net();
  const cols = [], used = new Map();
  for (const n of N.fnames) { const j = ALL_NAMES.indexOf(n, used.get(n) || 0); cols.push(j); used.set(n, j + 1); }
  const s = Math.min(1, N.long / Math.max(W, H));
  return marks.map((m) => {
    const F = features(m, W, H);
    return { x: (m.cx + 0.5) * s - 0.5, y: (m.cy + 0.5) * s - 0.5, f: cols.map((j) => F[j]) };
  });
}

function headLogits(taps, th, rows) {
  const N = net();
  const proj = N.proj.map((p, k) => {
    const t = taps[k], P = t.H * t.W, out = new Float32Array(p.cout * P);
    for (let o = 0; o < p.cout; o++) {
      const bo = p.b[o];
      for (let q = 0; q < P; q++) out[o * P + q] = bo;
      for (let i = 0; i < p.cin; i++) {
        const wv = p.w[o * p.cin + i], xi = i * P;
        for (let q = 0; q < P; q++) out[o * P + q] += wv * t.x[xi + q];
      }
    }
    return { x: out, C: p.cout, H: t.H, W: t.W };
  });
  const logits = new Float64Array(rows.length);
  for (let ri = 0; ri < rows.length; ri++) {
    const { x: tx, y: ty, f: F } = rows[ri], nf = F.length;
    const x = [];
    for (let j = 0; j < nf; j++) {
      const v = F[j];
      x.push(v === null || Number.isNaN(v) ? 0 : Math.fround(Math.min(6, Math.max(-6, (v - N.med[j]) / N.iqr[j]))));
    }
    for (let j = 0; j < nf; j++) x.push(F[j] === null || Number.isNaN(F[j]) ? 1 : 0);
    for (const t of proj) {
      const fx = Math.min(t.W - 1, Math.max(0, (tx + 0.5) / th.w * t.W - 0.5)), fy = Math.min(t.H - 1, Math.max(0, (ty + 0.5) / th.h * t.H - 0.5));
      const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(x0 + 1, t.W - 1), y1 = Math.min(y0 + 1, t.H - 1);
      const ax = fx - x0, ay = fy - y0, P = t.H * t.W;
      for (let c = 0; c < t.C; c++) {
        const o = c * P;
        x.push((t.x[o + y0 * t.W + x0] * (1 - ax) + t.x[o + y0 * t.W + x1] * ax) * (1 - ay) + (t.x[o + y1 * t.W + x0] * (1 - ax) + t.x[o + y1 * t.W + x1] * ax) * ay);
      }
    }
    let h = x;
    for (let li = 0; li < N.mlp.length; li++) {
      const L = N.mlp[li], o = new Float64Array(L.nout);
      for (let r = 0; r < L.nout; r++) {
        let a = L.b[r];
        for (let c = 0; c < L.nin; c++) a += L.w[r * L.nin + c] * h[c];
        o[r] = li < N.mlp.length - 1 ? Math.max(0, a) : a;
      }
      h = o;
    }
    logits[ri] = Math.round(h[0] * 1000) / 1000;
  }
  return logits;
}

// the logits for marks on a scan (img: Uint16Array, W x H x C), or null without weights; `keep` (an object) receives
// the thumbnail and the backbone's features, so later marks on the same scan can be scored without running it again
async function score(img, W, H, C, marks, tick, keep) {
  if (!net() || !marks.length) return null;
  const th = thumbnail(img, W, H, C, NET.long);
  const taps = await backbone(th, tick);
  if (keep) { keep.th = th; keep.taps = taps; }
  return headLogits(taps, th, rowsFor(marks, W, H));
}

// ---- the second opinion (pipeline.js, on the first pass's marks, before anything is healed) ---------------------
// A heal the model is sure is picture (logit <= scene_veto) is left; a compact mark it is sure is dust (logit >=
// scene_add, not on the frame, the print or the zone) is healed. r1's core and mask are edited in place, the way
// Claude's look did in 1.8.38. Returns the spots left as picture (the second pass leaves them alone too).
const addable = (m) => m.kind !== "hair/fibre" && (m.area || 0) <= (m.kind === "soft" ? 2500 : 400);

async function apply(img, W, H, C, r1, P, log, tick) {
  const out = { vetoed: 0, added: 0, protect: [], scored: 0, ms: 0 };
  if (!P.scene || !net()) return out;
  const t0 = Date.now();
  const marks = r1.marks.filter((m) => m.verdict === "defect" || m.verdict === "keep-photo");
  out.keep = {};
  const lg = await score(img, W, H, C, marks, tick, out.keep);
  if (!lg) return out;
  out.scored = marks.length;
  const vetoes = [], adds = [];
  for (let i = 0; i < marks.length; i++) {
    const m = marks[i];
    m.scene = lg[i];
    if (m.verdict === "defect" && lg[i] <= P.scene_veto) vetoes.push(m);
    else if (m.verdict === "keep-photo" && lg[i] >= P.scene_add && addable(m) && !m.frame && !m.print && !m.zone) adds.push(m);
  }
  const F = require("./filters");
  const { core, mask, outside } = r1;
  const boxes = [];
  for (const m of vetoes) { m.verdict = "keep-photo"; m.why = "part of the picture, by the scene model"; out.protect.push([m.cx, m.cy]); }
  const heals = r1.marks.filter((m) => m.verdict === "defect");
  for (const m of vetoes) {
    const x0 = Math.max(0, m.x0 - 1), y0 = Math.max(0, m.y0 - 1), x1 = Math.min(W - 1, m.x1 + 1), y1 = Math.min(H - 1, m.y1 + 1);
    const near = heals.filter((o) => o.x0 - 1 <= x1 && o.x1 + 1 >= x0 && o.y0 - 1 <= y1 && o.y1 + 1 >= y0);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        if (!near.some((o) => x >= o.x0 - 1 && x <= o.x1 + 1 && y >= o.y0 - 1 && y <= o.y1 + 1)) core[y * W + x] = 0;
      }
    }
    boxes.push([x0, y0, x1, y1]);
  }
  for (const m of adds) {
    const r = Math.ceil(Math.max(m.x1 - m.x0 + 1, m.y1 - m.y0 + 1) / 2) + 1, cx = Math.round(m.cx), cy = Math.round(m.cy);
    const x0 = Math.max(0, cx - r), y0 = Math.max(0, cy - r), x1 = Math.min(W - 1, cx + r), y1 = Math.min(H - 1, cy + r);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * W + x;
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r && !(outside && outside[i])) core[i] = 1;
      }
    }
    m.verdict = "defect"; m.why = "dust, by the scene model";
    boxes.push([x0, y0, x1, y1]);
  }
  if (mask !== core) {
    const dilate = P.dilate, k = 2 * dilate + 1;
    for (const [bx0, by0, bx1, by1] of boxes) {
      // the mask round the box, from the core one dilation further out
      const ox0 = Math.max(0, bx0 - dilate), oy0 = Math.max(0, by0 - dilate), ox1 = Math.min(W - 1, bx1 + dilate), oy1 = Math.min(H - 1, by1 + dilate);
      const cx0 = Math.max(0, ox0 - dilate), cy0 = Math.max(0, oy0 - dilate), cx1 = Math.min(W - 1, ox1 + dilate), cy1 = Math.min(H - 1, oy1 + dilate);
      const cw = cx1 - cx0 + 1, ch = cy1 - cy0 + 1, crop = new Uint8Array(cw * ch);
      for (let y = 0; y < ch; y++) crop.set(core.subarray((cy0 + y) * W + cx0, (cy0 + y) * W + cx1 + 1), y * cw);
      const d = F.dilateMask(crop, cw, ch, k, true);
      for (let y = oy0; y <= oy1; y++) for (let x = ox0; x <= ox1; x++) mask[y * W + x] = d[(y - cy0) * cw + (x - cx0)] ? 1 : 0;
    }
  }
  out.vetoed = vetoes.length; out.added = adds.length; out.ms = Date.now() - t0;
  if (log) log(`[scene] ${marks.length} marks scored in ${(out.ms / 1000).toFixed(1)} s: ${vetoes.length} heals left as picture, ${adds.length} marks healed as dust`);
  return out;
}

// the second pass's new heals (detect on the healed image): those the model is sure are picture are dropped from r2,
// scored with the first pass's picture (the heals change it too little to matter)
function vetoLater(state, r2, W, H, P, log) {
  if (!state || !state.keep || !state.keep.taps || !P.scene) return 0;
  const fresh = r2.marks.filter((m) => m.verdict === "defect");
  if (!fresh.length) return 0;
  const lg = headLogits(state.keep.taps, state.keep.th, rowsFor(fresh, W, H));
  const F = require("./filters");
  let n = 0;
  const heals = fresh.filter((m, i) => lg[i] > P.scene_veto);
  for (let i = 0; i < fresh.length; i++) {
    const m = fresh[i];
    m.scene = lg[i];
    if (lg[i] > P.scene_veto) continue;
    m.verdict = "keep-photo"; m.why = "part of the picture, by the scene model";
    n++;
    const x0 = Math.max(0, m.x0 - 1 - P.dilate), y0 = Math.max(0, m.y0 - 1 - P.dilate), x1 = Math.min(W - 1, m.x1 + 1 + P.dilate), y1 = Math.min(H - 1, m.y1 + 1 + P.dilate);
    const near = heals.filter((o) => o.x0 - 2 - P.dilate <= x1 && o.x1 + 2 + P.dilate >= x0 && o.y0 - 2 - P.dilate <= y1 && o.y1 + 2 + P.dilate >= y0);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (near.some((o) => x >= o.x0 - 2 - P.dilate && x <= o.x1 + 2 + P.dilate && y >= o.y0 - 2 - P.dilate && y <= o.y1 + 2 + P.dilate)) continue;
      r2.mask[y * W + x] = 0;
      if (r2.core) r2.core[y * W + x] = 0;
    }
  }
  if (n && log) log(`[scene] pass 2: ${n} of ${fresh.length} new heals left as picture`);
  return n;
}

module.exports = { available: () => !!PACK, score, apply, vetoLater, addable, thumbnail, features, rowsFor, ALL_NAMES, backbone, headLogits, net };

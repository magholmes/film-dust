"use strict";
// Working scale. The detector was tuned on 60-70 MP scans, where a speck is a few pixels across and
// the grain model fits. A 270 MP scan of the same frame shows every speck four times larger and its
// grain finer than the model expects, so marks are judged on a copy averaged down by a whole factor
// k, and the result is drawn back at full size.
//   k = 1 up to ~100 MP; above that, the whole factor that brings the copy nearest 65 MP.
// Averaging is a plain k x k box mean rounded half up: for k = 2 it is exactly the half-size copies
// the Python skill was checked on (cv2 INTER_AREA at 0.5, + 0.5, floor).

const TARGET_MP = 65;
const FULL_SIZE_MAX_MP = 100;

const NAT = require("./native");   // the same averaging in C++, when available

function workingFactor(W, H) {
  const mp = (W * H) / 1e6;
  if (mp <= FULL_SIZE_MAX_MP) return 1;
  return Math.max(2, Math.round(Math.sqrt(mp / TARGET_MP)));
}

// Average k x k blocks of a strip into the working image.
//   src: chunky Uint8Array or Uint16Array, `rows` rows of `srcW` pixels, C components
//   dst: Uint16Array, Wd x Hd x C (Wd = floor(W / k)); the strip fills working rows yd0 ..
//   eight: the source is 8-bit (values scale by 257 into 16-bit, as the rest of the engine expects)
// Only whole blocks are used: a last partial row or column of blocks is dropped, as in the skill.
function averageStrip(src, srcW, rows, C, k, dst, Wd, yd0, eight) {
  if (NAT.available() && NAT.averageStrip && dst instanceof Uint16Array && (eight ? src instanceof Uint8Array : src instanceof Uint16Array)) {
    const r = NAT.averageStrip(src, srcW, rows, C, k, dst, Wd, yd0, eight);
    if (r !== null) return r;
  }
  const kk = k * k, half = kk >> 1, scale = eight ? 257 : 1;
  const rowsOut = Math.floor(rows / k);
  const acc = new Float64Array(Wd * C);
  for (let r = 0; r < rowsOut; r++) {
    acc.fill(0);
    for (let dy = 0; dy < k; dy++) {
      let s = (r * k + dy) * srcW * C;
      for (let xd = 0; xd < Wd; xd++) {
        const o = xd * C;
        for (let dx = 0; dx < k; dx++) {
          for (let c = 0; c < C; c++) acc[o + c] += src[s + c];
          s += C;
        }
      }
    }
    const d = (yd0 + r) * Wd * C;
    for (let i = 0; i < Wd * C; i++) dst[d + i] = Math.floor((acc[i] * scale + half) / kk);
  }
  return rowsOut;
}

// Draw a working-size 0/1 mask at full size as 0/255 bytes, ready for a Photoshop selection.
// Pixels past the last whole block (an odd edge row or column) copy their neighbour.
function enlargeMask(mask, Wd, Hd, k, W, H) {
  const out = new Uint8Array(W * H);
  const row = new Uint8Array(W);
  for (let yd = 0; yd < Hd; yd++) {
    const m = yd * Wd;
    let any = false;
    for (let x = 0; x < W; x++) {
      const v = mask[m + Math.min(Wd - 1, Math.floor(x / k))] ? 255 : 0;
      row[x] = v;
      if (v) any = true;
    }
    if (!any) continue;
    const y1 = yd === Hd - 1 ? H : (yd + 1) * k;
    for (let y = yd * k; y < y1; y++) out.set(row, y * W);
  }
  return out;
}

// Marks found on the working copy, in full-size pixel coordinates (centre of the block they sit in).
function enlargeMarks(marks, k) {
  if (k === 1) return marks;
  const at = (v) => (typeof v === "number" ? v * k + (k - 1) / 2 : v);
  const span = (v) => (typeof v === "number" ? v * k : v);
  return marks.map((m) => Object.assign({}, m, {
    cx: at(m.cx), cy: at(m.cy),
    x0: span(m.x0), y0: span(m.y0), x1: m.x1 === undefined ? m.x1 : m.x1 * k + k - 1, y1: m.y1 === undefined ? m.y1 : m.y1 * k + k - 1,
    width: span(m.width), length: span(m.length), area: typeof m.area === "number" ? m.area * k * k : m.area,
  }));
}

module.exports = { workingFactor, averageStrip, enlargeMask, enlargeMarks, TARGET_MP, FULL_SIZE_MAX_MP };

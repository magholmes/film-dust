"use strict";
// Checks the working-scale helpers:
//   node test/scale_check.js [<full.npy> <half.npy>]
// Without arguments: random arrays against a direct reference. With a full-size crop and the
// skill's half-size copy of the same area: the strip-wise average must match it exactly.
const assert = require("assert");
const { readNpy } = require("./npy");
const S = require("../engine/scale");

function reference(src, W, H, C, k, eight) {
  const Wd = Math.floor(W / k), Hd = Math.floor(H / k), out = new Uint16Array(Wd * Hd * C);
  for (let y = 0; y < Hd; y++) for (let x = 0; x < Wd; x++) for (let c = 0; c < C; c++) {
    let s = 0;
    for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) s += src[((y * k + dy) * W + x * k + dx) * C + c] * (eight ? 257 : 1);
    out[(y * Wd + x) * C + c] = Math.floor(s / (k * k) + 0.5);
  }
  return out;
}

function stripwise(src, W, H, C, k, rowsPerStrip, eight) {
  const Wd = Math.floor(W / k), Hd = Math.floor(H / k), out = new Uint16Array(Wd * Hd * C);
  let yd = 0;
  for (let y0 = 0; y0 < Hd * k; y0 += rowsPerStrip) {
    const rows = Math.min(rowsPerStrip, Hd * k - y0);
    const strip = src.subarray(y0 * W * C, (y0 + rows) * W * C);
    yd += S.averageStrip(strip, W, rows, C, k, out, Wd, yd, eight);
  }
  assert.strictEqual(yd, Hd);
  return out;
}

let n = 0;
for (const [W, H, C, k, eight] of [[37, 29, 3, 2, false], [41, 30, 1, 3, false], [64, 48, 3, 2, true], [15, 15, 3, 1, false]]) {
  const src = eight ? new Uint8Array(W * H * C) : new Uint16Array(W * H * C);
  for (let i = 0; i < src.length; i++) src[i] = Math.floor(Math.random() * (eight ? 256 : 65536));
  const a = reference(src, W, H, C, k, eight);
  for (const rps of [k, 2 * k, 7 * k, H]) assert.deepStrictEqual(stripwise(src, W, H, C, k, rps, eight), a, `average ${W}x${H} k${k} strips ${rps}`);
  n++;
}

// enlarge: every full-size pixel takes its block's value; edge rows/columns copy the last block
{
  const Wd = 5, Hd = 4, k = 2, W = 11, H = 9;
  const m = new Uint8Array(Wd * Hd);
  m[0] = 1; m[Wd * Hd - 1] = 1; m[2 * Wd + 2] = 1;
  const big = S.enlargeMask(m, Wd, Hd, k, W, H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = m[Math.min(Hd - 1, Math.floor(y / k)) * Wd + Math.min(Wd - 1, Math.floor(x / k))] ? 255 : 0;
    assert.strictEqual(big[y * W + x], v, `enlarge at ${x},${y}`);
  }
  n++;
}
assert.strictEqual(S.workingFactor(7000, 8600), 1);
assert.strictEqual(S.workingFactor(15059, 17948), 2);
assert.strictEqual(S.workingFactor(22000, 26000), 3);
n++;

if (process.argv[3]) {
  const full = readNpy(process.argv[2]), half = readNpy(process.argv[3]);
  const [H, W, C] = full.shape;
  const got = stripwise(full.data, W, H, C, 2, 708, false);
  const want = half.data;
  assert.strictEqual(got.length, want.length, "half-size shapes differ");
  let diff = 0;
  for (let i = 0; i < got.length; i++) if (got[i] !== want[i]) diff++;
  assert.strictEqual(diff, 0, `${diff} values differ from the skill's half-size copy`);
  console.log(`matches the skill's half-size copy: ${W}x${H} -> ${half.shape[1]}x${half.shape[0]}`);
  n++;
}
console.log(`scale checks passed: ${n}`);

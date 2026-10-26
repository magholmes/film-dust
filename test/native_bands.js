"use strict";
// The native filters that work in bands of rows (Gaussian, Sobel) against the JS on heights around the band
// sizes (one band, exactly one, one more row, several) and kernels up to 73 wide: identical, bit for bit.
//   node test/native_bands.js
const path = require("path");
const load = (off) => {
  for (const k of Object.keys(require.cache)) if (k.includes(path.join("engine", ""))) delete require.cache[k];
  if (off) process.env.FILMDUST_NATIVE = "0"; else delete process.env.FILMDUST_NATIVE;
  return { F: require("../engine/filters"), M: require("../engine/maps"), NAT: require("../engine/native") };
};
const js = load(true), nat = load(false);
const N = nat.NAT;
if (!N.available()) { console.log("native module not built"); process.exit(1); }
let seed = 5;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const bits = (a) => new Uint32Array(a.buffer, a.byteOffset, a.length);
let fails = 0, checks = 0;
const check = (what, a, b) => {
  checks++;
  let same = !!a && !!b && a.length === b.length;
  if (same) { const x = a instanceof Float32Array ? bits(a) : a, y = b instanceof Float32Array ? bits(b) : b; for (let i = 0; same && i < x.length; i++) if (x[i] !== y[i]) same = false; }
  if (!same) { fails++; if (fails <= 12) console.log("DIFF", what); }
};
const t0 = Date.now();
for (const h of [1, 2, 3, 255, 256, 257, 300, 511, 512, 513, 590, 1025, 1300]) {
  const w = 1 + Math.floor(rnd() * 90);
  const Y = new Float32Array(w * h).map(() => Math.fround(rnd()));
  for (const s of [0.8, 1.0, 2.5, 3.0, 6.0, 12.0]) check(`gauss ${s} ${w}x${h}`, js.F.gaussianBlur(Y, w, h, s, null, true), N.sepConv(Y, w, h, js.F.gaussKernel(s)));
  for (const [dx, dy] of [[1, 0], [0, 1], [2, 0], [0, 2], [1, 1]]) check(`sobel ${dx}${dy} ${w}x${h}`, js.F.sobel(Y, w, h, dx, dy), N.sobel(Y, w, h, dx, dy));
}
// the outside mask (clipped, grain-free areas touching the frame) on frames with flat borders, flat patches inside
// and grainy picture, at sizes from tiny to several bands
for (const [w, h] of [[64, 64], [90, 300], [300, 90], [257, 513], [700, 1100]]) {
  for (let v = 0; v < 3; v++) {
    const Y = new Float32Array(w * h).map(() => Math.fround(0.3 + 0.2 * rnd()));
    const bw = 3 + Math.floor(rnd() * 20), level = v === 1 ? 0.01 : 0.92;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const border = x < bw || (v !== 2 && y < bw) || (v === 0 && x >= w - bw);
      const inner = Math.abs(x - w / 2) < w / 8 && Math.abs(y - h / 2) < h / 8;
      if (border || inner) Y[y * w + x] = Math.fround(level);
    }
    check(`outside ${w}x${h} v${v}`, js.M.outsideMask(Y, w, h), N.outsideMask(Y, w, h));
  }
}
console.log(`bands: ${checks - fails}/${checks} identical (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
if (!N.available()) { console.log("native switched off during the test:", N.loadError()); fails++; }
process.exitCode = fails ? 1 : 0;

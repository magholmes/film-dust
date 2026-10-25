"use strict";
// Native vs JS on small and odd sizes with random data (edges, kernels larger than the image):
// every function must match bit for bit. Calls the native wrappers directly (no size threshold).
const path = require("path");
const load = (off) => {
  for (const k of Object.keys(require.cache)) if (k.includes(path.join("engine", ""))) delete require.cache[k];
  if (off) process.env.FILMDUST_NATIVE = "0"; else delete process.env.FILMDUST_NATIVE;
  return { F: require("../engine/filters"), M: require("../engine/maps"), R: require("../engine/regions"), S: require("../engine/scale"), NAT: require("../engine/native") };
};
const js = load(true), nat = load(false);
const N = nat.NAT;
let seed = 11;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const bits = (a) => new Uint32Array(a.buffer, a.byteOffset, a.length);
const same = (a, b) => { if (!a || !b || a.length !== b.length) return false; const x = a instanceof Float32Array ? bits(a) : a, y = b instanceof Float32Array ? bits(b) : b; for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false; return true; };
let fails = 0, checks = 0;
const check = (what, a, b) => { checks++; if (!same(a, b)) { fails++; if (fails <= 12) console.log("DIFF", what); } };
const sizes = [[1, 1], [1, 7], [7, 1], [3, 5], [17, 4], [9, 23], [40, 40], [63, 31], [100, 37], [130, 96]];
for (const [w, h] of sizes) {
  const Y = new Float32Array(w * h).map(() => Math.fround(rnd()));
  const m = new Uint8Array(w * h).map(() => (rnd() < 0.3 ? 1 : 0));
  for (const k of [3, 5, 15, 41]) {
    check(`median ${k} ${w}x${h}`, js.M.median8f(Y, w, h, k), N.median8f(Y, w, h, k));
    for (const mode of [0, 1, 2, 3]) {
      const ref = mode === 0 ? js.F.ellipseMorph(Y, w, h, k, false) : mode === 1 ? js.F.ellipseMorph(Y, w, h, k, true) : js.F.ellipseOpenClose(Y, w, h, k, mode === 2);
      check(`ellipse ${mode} ${k} ${w}x${h}`, ref, N.ellipse(Y, w, h, k, js.F.ellipseHalfWidths(k), mode));
    }
    for (const dil of [true, false]) {
      check(`binRect ${dil} ${k} ${w}x${h}`, dil ? js.F.dilateMask(m, w, h, k) : js.F.erodeMask(m, w, h, k), N.binMorph(m, w, h, k, dil, null));
      check(`binEllipse ${dil} ${k} ${w}x${h}`, dil ? js.F.dilateMask(m, w, h, k, true) : js.F.erodeMask(m, w, h, k, true), N.binMorph(m, w, h, k, dil, js.F.ellipseHalfWidths(k)));
      check(`rectMorph ${dil} ${k} ${w}x${h}`, js.F.rectMorph(Y, w, h, k, dil), N.rectMorph(Y, w, h, k, dil));
    }
    check(`boxBlur ${k} ${w}x${h}`, js.F.boxBlur(Y, w, h, k), N.boxBlur(Y, w, h, k));
  }
  for (const s of [0.8, 1.0, 2.5, 3.0, 6.0]) check(`gauss ${s} ${w}x${h}`, js.F.gaussianBlur(Y, w, h, s, null, true), N.sepConv(Y, w, h, js.F.gaussKernel(s)));
  for (const [dx, dy] of [[1, 0], [0, 1], [2, 0], [0, 2], [1, 1]]) check(`sobel ${dx}${dy} ${w}x${h}`, js.F.sobel(Y, w, h, dx, dy), N.sobel(Y, w, h, dx, dy));
  if (w > 1 && h > 1) { const a = js.F.npGradient(Y, w, h), b = N.npGradient(Y, w, h); check(`grad ${w}x${h}`, a[0], b[0]); check(`grad x ${w}x${h}`, a[1], b[1]); }
  for (const conn of [4, 8]) { const a = js.R.label(m, w, h, conn), b = N.label(m, w, h, conn); check(`label ${conn} ${w}x${h}`, a.lab, b.lab); if (a.n !== b.n) { fails++; console.log("DIFF label n", w, h); } }
  check(`fillHoles ${w}x${h}`, js.R.fillHoles(m, w, h), N.fillHoles(m, w, h));
  check(`resize ${w}x${h}`, js.F.resizeLinear(Y, w, h, w * 2 + 1, h * 2 + 3), N.resizeLinear(Y, w, h, w * 2 + 1, h * 2 + 3));
}
console.log(`fuzz: ${checks - fails}/${checks} identical`);
process.exitCode = fails ? 1 : 0;

"use strict";
// The native float ellipse morphology (cache-sized tiles for plain images, the JS's row order otherwise) against
// the JS: erode, dilate, open and close for odd and even sizes up to 81, images narrower and wider than a tile,
// plain values, and images holding NaN, -0 and infinities. Every result must be identical, bit for bit.
//   node test/native_ellipse.js [rounds]
const path = require("path");
const load = (off) => {
  for (const k of Object.keys(require.cache)) if (k.includes(path.join("engine", ""))) delete require.cache[k];
  if (off) process.env.FILMDUST_NATIVE = "0"; else delete process.env.FILMDUST_NATIVE;
  return { F: require("../engine/filters"), NAT: require("../engine/native") };
};
const js = load(true), nat = load(false);
const N = nat.NAT;
if (!N.available()) { console.log("native module not built"); process.exit(1); }
let seed = 3;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
let fails = 0, checks = 0;
const bits = (a) => new Uint32Array(a.buffer, a.byteOffset, a.length);
const check = (what, a, b) => {
  checks++;
  let same = !!a && !!b && a.length === b.length;
  if (same) { const x = bits(a), y = bits(b); for (let i = 0; same && i < x.length; i++) if (x[i] !== y[i]) same = false; }
  if (!same) { fails++; if (fails <= 12) console.log("DIFF", what); }
};
const rounds = Number(process.argv[2] || 24);
const t0 = Date.now();
for (let round = 0; round < rounds; round++) {
  // every fifth image is tall and narrow: several bands of the open / close
  const tall = round % 5 === 4;
  const w = 1 + Math.floor(rnd() * (tall ? 30 : round % 3 === 0 ? 40 : 700)), h = 1 + Math.floor(rnd() * (tall ? 1600 : 300));
  const special = round % 4 === 3;
  const Y = new Float32Array(w * h).map(() => Math.fround(rnd() < 0.2 ? Math.floor(rnd() * 4) / 4 : rnd()));
  if (special) for (let i = 0; i < Y.length; i++) { const u = rnd(); if (u < 0.01) Y[i] = NaN; else if (u < 0.03) Y[i] = -0; else if (u < 0.04) Y[i] = u < 0.035 ? Infinity : -Infinity; }
  for (const k of [1, 2, 3, 5, 8, 15, 21, 41, 60, 61, 81]) {
    const hw = js.F.ellipseHalfWidths(k);
    for (const mode of [0, 1, 2, 3]) {
      const ref = mode === 0 ? js.F.ellipseMorph(Y, w, h, k, false) : mode === 1 ? js.F.ellipseMorph(Y, w, h, k, true) : js.F.ellipseOpenClose(Y, w, h, k, mode === 2);
      check(`mode ${mode} k${k} ${w}x${h}${special ? " special" : ""}`, ref, N.ellipse(Y, w, h, k, hw, mode));
    }
  }
}
console.log(`ellipse: ${checks - fails}/${checks} identical (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
if (!N.available()) { console.log("native switched off during the test:", N.loadError()); fails++; }
process.exitCode = fails ? 1 : 0;

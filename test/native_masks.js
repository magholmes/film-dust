"use strict";
// The native mask morphology (ellipse and square erode / dilate, hole filling, labelling) against the JS, on many
// random masks: sparse to dense, blobs with holes, odd and even ellipse sizes up to 81, sizes from 1 x 1 to
// 400 x 300, and masks holding other values than 0 and 1. Every result must be identical.
//   node test/native_masks.js [rounds]
const path = require("path");
const load = (off) => {
  for (const k of Object.keys(require.cache)) if (k.includes(path.join("engine", ""))) delete require.cache[k];
  if (off) process.env.FILMDUST_NATIVE = "0"; else delete process.env.FILMDUST_NATIVE;
  return { F: require("../engine/filters"), R: require("../engine/regions"), NAT: require("../engine/native") };
};
const js = load(true), nat = load(false);
const N = nat.NAT;
if (!N.available()) { console.log("native module not built"); process.exit(1); }
let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
let fails = 0, checks = 0;
const check = (what, a, b) => {
  checks++;
  let same = !!a && !!b && a.length === b.length;
  for (let i = 0; same && i < a.length; i++) if (a[i] !== b[i]) same = false;
  if (!same) { fails++; if (fails <= 12) console.log("DIFF", what); }
};
// random blobs (filled discs and rings, so there are holes) on a random speckle
function mask(w, h, density, value) {
  const m = new Uint8Array(w * h);
  for (let i = 0; i < m.length; i++) m[i] = rnd() < density ? value : 0;
  const blobs = Math.floor(rnd() * 6);
  for (let b = 0; b < blobs; b++) {
    const cx = rnd() * w, cy = rnd() * h, R = 2 + rnd() * Math.min(w, h) / 3, ring = rnd() < 0.5;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const d = Math.hypot(x - cx, y - cy);
      if (d <= R && (!ring || d >= R * 0.6)) m[y * w + x] = value;
    }
  }
  return m;
}
const rounds = Number(process.argv[2] || 60);
const t0 = Date.now();
for (let round = 0; round < rounds; round++) {
  const w = 1 + Math.floor(rnd() * (round < 20 ? 40 : 400)), h = 1 + Math.floor(rnd() * (round < 20 ? 40 : 300));
  const value = round % 10 === 9 ? 255 : 1;
  const m = mask(w, h, [0.01, 0.1, 0.3, 0.6, 0.95][round % 5], value);
  for (const k of [1, 2, 3, 4, 5, 7, 9, 15, 21, 25, 41, 60, 61, 81]) {
    const hw = js.F.ellipseHalfWidths(k);
    for (const dil of [true, false]) {
      check(`ellipse ${dil ? "dilate" : "erode"} k${k} ${w}x${h} v${value}`, dil ? js.F.dilateMask(m, w, h, k, true) : js.F.erodeMask(m, w, h, k, true), N.binMorph(m, w, h, k, dil, hw));
      check(`square ${dil ? "dilate" : "erode"} k${k} ${w}x${h} v${value}`, dil ? js.F.dilateMask(m, w, h, k) : js.F.erodeMask(m, w, h, k), N.binMorph(m, w, h, k, dil, null));
    }
  }
  check(`fillHoles ${w}x${h} v${value}`, js.R.fillHoles(m, w, h), N.fillHoles(m, w, h));
  for (const conn of [4, 8]) {
    const a = js.R.label(m, w, h, conn), b = N.label(m, w, h, conn);
    check(`label ${conn} ${w}x${h}`, a.lab, b && b.lab);
    if (!b || a.n !== b.n) { fails++; console.log("DIFF label count", w, h); }
  }
}
console.log(`masks: ${checks - fails}/${checks} identical (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
if (!N.available()) { console.log("native switched off during the test:", N.loadError()); fails++; }
process.exitCode = fails ? 1 : 0;

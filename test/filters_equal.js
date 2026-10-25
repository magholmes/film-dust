"use strict";
// New primitives must give exactly what the reference versions give.
const N = require("../engine/filters");
const O = require("./filters_ref");
let seed = 5;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
const img = (w, h) => Float32Array.from({ length: w * h }, rnd);
const msk = (w, h, p) => Uint8Array.from({ length: w * h }, () => (rnd() < p ? 1 : 0));
const same = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++; return d; };
let bad = 0;
const check = (name, a, b) => { const d = same(a, b); if (d) { bad++; console.log("DIFF", name, d); } };
for (const [w, h] of [[97, 61], [5, 40], [300, 7], [1, 1], [2, 3]]) {
  const f = img(w, h);
  for (const k of [3, 5, 7, 15, 21, 41, 61]) {
    for (const mx of [true, false]) {
      check(`rect ${w}x${h} k${k} ${mx}`, N.rectMorph(f, w, h, k, mx), O.rectMorph(f, w, h, k, mx));
      check(`ellipse ${w}x${h} k${k} ${mx}`, N.ellipseMorph(f, w, h, k, mx), O.ellipseMorph(f, w, h, k, mx));
    }
    for (const p of [0.02, 0.3, 0.9]) {
      const m = msk(w, h, p);
      check(`dilate rect k${k}`, N.dilateMask(m, w, h, k), O.dilateMask(m, w, h, k));
      check(`erode rect k${k}`, N.erodeMask(m, w, h, k), O.erodeMask(m, w, h, k));
      check(`dilate ell k${k}`, N.dilateMask(m, w, h, k, true), O.dilateMask(m, w, h, k, true));
      check(`erode ell k${k}`, N.erodeMask(m, w, h, k, true), O.erodeMask(m, w, h, k, true));
    }
  }
  for (const s of [0.8, 1.0, 1.5, 2.5, 3.0, 8.0]) if (w > 3 && h > 3) check(`gauss ${s}`, N.gaussianBlur(f, w, h, s), O.gaussianBlur(f, w, h, s));
  if (w > 3 && h > 3) check("box 7", N.boxBlur(f, w, h, 7), O.boxBlur(f, w, h, 7));
}
console.log(bad ? `${bad} mismatches` : "all identical");

"use strict";
// The native thread pool under alternating small and large jobs (different thread counts back to back),
// the pattern of the soft stage's many small tiles. A pool race once deadlocked here: it must finish,
// and every result must equal a single-size run's result.
//   node test/native_pool_stress.js [iterations]
const NAT = require("../engine/native");
if (!NAT.available()) { console.log("no native module"); process.exit(0); }
const N = Number(process.argv[2] || 4000);
function img(W, H, seed) {
  const a = new Float32Array(W * H);
  let s = seed;
  for (let i = 0; i < a.length; i++) { s = (s * 1103515245 + 12345) >>> 0; a[i] = (s >>> 8) / 16777216; }
  return a;
}
const sizes = [[64, 64], [96, 64], [512, 384], [64, 2048], [1024, 96], [80, 72]];
const imgs = sizes.map(([w, h], i) => img(w, h, i + 1));
const ref = sizes.map(([w, h], i) => NAT.median8f(imgs[i], w, h, 5));
const t0 = Date.now();
let bad = 0;
for (let it = 0; it < N; it++) {
  const i = (it * 7 + (it >> 3)) % sizes.length;
  const [w, h] = sizes[i];
  const out = NAT.median8f(imgs[i], w, h, 5);
  for (let k = 0; k < out.length; k += 97) if (out[k] !== ref[i][k]) { bad++; break; }
}
console.log(`${N} alternating calls in ${((Date.now() - t0) / 1000).toFixed(1)} s · results differing ${bad}`);
console.log(bad ? "FAIL" : "PASS");
process.exit(bad ? 1 : 0);

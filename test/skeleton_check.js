// Native skeletonize against regions.js's JS thinning, bit for bit: node test/skeleton_check.js
// Random blobs, lines, rings and noise masks of many sizes (the JS loop below is the one in regions.js).
const NAT = require("../engine/native");
const R = require("../engine/regions");

function jsSkeleton(mask, w, h) {
  // regions.js skeletonize with the native shortcut skipped: a copy of the mask that is not a Uint8Array
  return R.skeletonize(Array.from(mask), w, h);
}
let s = 12345;
const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296);
function mask(w, h, kind) {
  const m = new Uint8Array(w * h);
  if (kind === "noise") { for (let i = 0; i < m.length; i++) m[i] = rnd() < 0.45 ? 1 : 0; return m; }
  const n = 1 + Math.floor(rnd() * 6);
  for (let k = 0; k < n; k++) {
    const cy = rnd() * h, cx = rnd() * w, r = 2 + rnd() * Math.min(w, h) / 3, th = 1 + rnd() * 8, ang = rnd() * Math.PI;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const dy = y - cy, dx = x - cx, d = Math.hypot(dy, dx);
      if (kind === "blob" && d <= r) m[y * w + x] = 1;
      if (kind === "ring" && Math.abs(d - r) <= th) m[y * w + x] = 1;
      if (kind === "line" && Math.abs(dy * Math.cos(ang) - dx * Math.sin(ang)) <= th && d <= 2 * r) m[y * w + x] = 1;
    }
  }
  return m;
}
let bad = 0, n = 0, tJ = 0, tN = 0;
for (const kind of ["blob", "ring", "line", "noise"]) for (let t = 0; t < 40; t++) {
  const w = 64 + Math.floor(rnd() * 400), h = 64 + Math.floor(rnd() * 400), m = mask(w, h, kind);
  let t0 = Date.now();
  const a = jsSkeleton(m, w, h);
  tJ += Date.now() - t0;
  t0 = Date.now();
  const b = NAT.skeletonize(m, w, h);
  tN += Date.now() - t0;
  if (!b) { console.log("native skeletonize not available"); process.exit(1); }
  n++;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { bad++; console.log(`${kind} ${w}x${h}: DIFFERENT at ${i}`); break; }
}
console.log(`${bad ? bad + " DIFFERENT" : "IDENTICAL"} · ${n} masks · JS ${tJ} ms, native ${tN} ms`);
process.exit(bad ? 1 : 0);

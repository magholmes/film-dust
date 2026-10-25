"use strict";
// rectMorph (float) and averageStrip: native against JS, bit for bit
const path = require("path");
const { readNpy } = require("./npy");
const src = readNpy(process.argv[2]);
const [H0, W0, C] = src.shape;
const load = (off) => {
  for (const k of Object.keys(require.cache)) if (k.includes(path.join("engine", ""))) delete require.cache[k];
  if (off) process.env.FILMDUST_NATIVE = "0"; else delete process.env.FILMDUST_NATIVE;
  return { F: require("../engine/filters"), M: require("../engine/maps"), S: require("../engine/scale") };
};
const js = load(true), nat = load(false);
const W = Math.min(W0, 3000), H = Math.min(H0, 4000);
const Yf = js.M.lumaFromU16(src.data, W0, H0, C), Y = new Float32Array(W * H);
for (let y = 0; y < H; y++) Y.set(Yf.subarray(y * W0, y * W0 + W), y * W);
const same = (a, b) => { const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length); let n = 0; for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) n++; return n ? `${n} differ` : "identical"; };
for (const [k, mx] of [[15, true], [21, true], [5, false], [9, true]]) {
  let t = Date.now(); const a = js.F.rectMorph(Y, W, H, k, mx); const tj = Date.now() - t;
  t = Date.now(); const b = nat.F.rectMorph(Y, W, H, k, mx); const tn = Date.now() - t;
  console.log(`rectMorph ${mx ? "max" : "min"} ${k}`.padEnd(22), same(a, b).padEnd(14), `js ${(tj / 1000).toFixed(2)} s · native ${(tn / 1000).toFixed(2)} s`);
}
for (const k of [2, 3]) {
  const rows = 600 - (600 % k), Wd = Math.floor(W0 / k);
  const strip = src.data.subarray(0, rows * W0 * C);
  const a = new Uint16Array(Wd * (rows / k) * C), b = new Uint16Array(Wd * (rows / k) * C);
  let t = Date.now(); const ra = js.S.averageStrip(strip, W0, rows, C, k, a, Wd, 0, false); const tj = Date.now() - t;
  t = Date.now(); const rb = nat.S.averageStrip(strip, W0, rows, C, k, b, Wd, 0, false); const tn = Date.now() - t;
  let d = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
  console.log(`averageStrip k ${k}`.padEnd(22), (d ? `${d} differ` : "identical").padEnd(14), `rows ${ra}/${rb} · js ${(tj / 1000).toFixed(2)} s · native ${(tn / 1000).toFixed(2)} s`);
  const s8 = new Uint8Array(strip.length); for (let i = 0; i < s8.length; i++) s8[i] = strip[i] >> 8;
  const a8 = new Uint16Array(a.length), b8 = new Uint16Array(a.length);
  js.S.averageStrip(s8, W0, rows, C, k, a8, Wd, 0, true); nat.S.averageStrip(s8, W0, rows, C, k, b8, Wd, 0, true);
  let d8 = 0; for (let i = 0; i < a8.length; i++) if (a8[i] !== b8[i]) d8++;
  console.log(`averageStrip k ${k} 8-bit`.padEnd(22), d8 ? `${d8} differ` : "identical");
}

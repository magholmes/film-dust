"use strict";
// detect.js ridgeMaps: native against JS on a real scan's luma (bit for bit)
const path = require("path");
const { readNpy } = require("./npy");
const src = readNpy(process.argv[2]);
const [H0, W0, C] = src.shape;
const load = (off) => {
  for (const k of Object.keys(require.cache)) if (k.includes(path.join("engine", ""))) delete require.cache[k];
  if (off) process.env.FILMDUST_NATIVE = "0"; else delete process.env.FILMDUST_NATIVE;
  return { D: require("../engine/detect"), M: require("../engine/maps") };
};
(async () => {
  const js = load(true), nat = load(false);
  const W = Math.min(W0, 3000), H = Math.min(H0, 4000);
  const Yf = js.M.lumaFromU16(src.data, W0, H0, C), Y = new Float32Array(W * H);
  for (let y = 0; y < H; y++) Y.set(Yf.subarray(y * W0, y * W0 + W), y * W);
  const outside = new Uint8Array(W * H); for (let i = 0; i < W * 30; i++) outside[i] = 1;
  const maps = await js.M.residualMaps(Y, W, H, true, outside, {});
  for (const b of [true, false]) {
    let t = Date.now(); const a = js.D._ridgeMaps(Y, W, H, outside, b, maps.sigma); const tj = Date.now() - t;
    t = Date.now(); const c = nat.D._ridgeMaps(Y, W, H, outside, b, maps.sigma); const tn = Date.now() - t;
    const diff = (u, v) => { const x = new Uint32Array(u.buffer, u.byteOffset, u.length), y = new Uint32Array(v.buffer, v.byteOffset, v.length); let n = 0; for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) n++; return n; };
    console.log(`ridgeMaps ${b ? "bright" : "dark"}: best differ ${diff(a.best, c.best)} · scale differ ${diff(a.bestS, c.bestS)} · js ${(tj / 1000).toFixed(2)} s · native ${(tn / 1000).toFixed(2)} s`);
  }
})();

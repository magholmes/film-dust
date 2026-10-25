"use strict";
// Every native filter against its JS version on a real scan's luma: must be bit-identical.
//   node test/native_parity.js <scan.npy>
const { readNpy } = require("./npy");
const NAT = require("../engine/native");
if (!NAT.available()) { console.log("native module not built (native/build.sh)"); process.exit(1); }
process.env.FILMDUST_NATIVE = "0";
// a second copy of the engine modules with native switched off
const path = require("path");
for (const k of Object.keys(require.cache)) if (k.includes(path.join("engine", ""))) delete require.cache[k];
const Fjs = require("../engine/filters"), Mjs = require("../engine/maps");
const NATjs = require("../engine/native");
if (NATjs.available()) { console.log("could not switch native off"); process.exit(1); }
for (const k of Object.keys(require.cache)) if (k.includes(path.join("engine", ""))) delete require.cache[k];
delete process.env.FILMDUST_NATIVE;
const Fn = require("../engine/filters"), Mn = require("../engine/maps");

const src = readNpy(process.argv[2]);
const [H0, W0, C] = src.shape;
// a 4000 x 3000 window keeps the JS side quick; edges included
const W = Math.min(W0, 3000), H = Math.min(H0, 4000);
const Yfull = Mjs.lumaFromU16(src.data, W0, H0, C);
const Y = new Float32Array(W * H);
for (let y = 0; y < H; y++) Y.set(Yfull.subarray(y * W0, y * W0 + W), y * W);
const same = (a, b) => {
  if (a.length !== b.length) return `length ${a.length} vs ${b.length}`;
  const ua = new Uint32Array(a.buffer, a.byteOffset, a.length), ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let n = 0, first = -1;
  for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) { n++; if (first < 0) first = i; }
  return n ? `${n} differ (first at ${first}: ${a[first]} vs ${b[first]})` : "identical";
};
const run = (label, fjs, fn) => {
  let t = Date.now(); const a = fjs(); const tj = Date.now() - t;
  t = Date.now(); const b = fn(); const tn = Date.now() - t;
  console.log(`${label.padEnd(26)} ${same(a, b).padEnd(40)} js ${(tj / 1000).toFixed(2)} s · native ${(tn / 1000).toFixed(2)} s`);
};
console.log(`window ${W} x ${H}, native threads ${NAT.threads()}`);
for (const k of [15, 41]) run(`median ${k}`, () => Mjs.median8f(Y, W, H, k), () => Mn.median8f(Y, W, H, k));
for (const k of [15, 41]) for (const open of [true, false]) run(`ellipse ${open ? "open" : "close"} ${k}`, () => Fjs.ellipseOpenClose(Y, W, H, k, open), () => Fn.ellipseOpenClose(Y, W, H, k, open));
run("ellipse dilate 21", () => Fjs.ellipseMorph(Y, W, H, 21, true), () => Fn.ellipseMorph(Y, W, H, 21, true));
for (const s of [0.8, 1.0, 1.5, 2.5, 3.0]) run(`gauss ${s}`, () => Fjs.gaussianBlur(Y, W, H, s), () => Fn.gaussianBlur(Y, W, H, s));
run("gauss 6 exact", () => Fjs.gaussianBlur(Y, W, H, 6, null, true), () => Fn.gaussianBlur(Y, W, H, 6, null, true));
run("gauss 8 (boxes)", () => Fjs.gaussianBlur(Y, W, H, 8), () => Fn.gaussianBlur(Y, W, H, 8));
run("box 7", () => Fjs.boxBlur(Y, W, H, 7), () => Fn.boxBlur(Y, W, H, 7));
for (const [dx, dy] of [[1, 0], [0, 1], [2, 0], [0, 2], [1, 1]]) run(`sobel ${dx},${dy}`, () => Fjs.sobel(Y, W, H, dx, dy), () => Fn.sobel(Y, W, H, dx, dy));
// odd sizes and a 1-pixel-wide edge case
const w2 = 517, h2 = 263, Ys = Y.subarray(0, w2 * h2);
run("median 41 (517x263)", () => Mjs.median8f(Ys, w2, h2, 41), () => NAT.median8f(Ys, w2, h2, 41));
run("open 41 (517x263)", () => Fjs.ellipseOpenClose(Ys, w2, h2, 41, true), () => NAT.ellipse(Ys, w2, h2, 41, Fjs.ellipseHalfWidths(41), 2));
run("gauss 2.5 (517x263)", () => Fjs.gaussianBlur(Ys, w2, h2, 2.5), () => NAT.sepConv(Ys, w2, h2, Fjs.gaussKernel(2.5)));
// maps.js statistics and loops
const valid = new Uint8Array(W * H).fill(1);
for (let i = 0; i < valid.length; i += 37) valid[i] = 0;
const m15 = Mjs.median8f(Y, W, H, 15), resid = new Float32Array(W * H);
for (let i = 0; i < resid.length; i++) resid[i] = Y[i] - m15[i];
run("noiseLocal", () => Mjs.noiseLocal(resid, valid, W, H), () => Mn.noiseLocal(resid, valid, W, H));
for (const b of [true, false]) run(`busyness ${b ? "bright" : "dark"}`, () => Mjs.busyness(resid, valid, W, H, b).T, () => Mn.busyness(resid, valid, W, H, b).T);
const m41 = Mjs.median8f(Y, W, H, 41);
run("noiseByLevel", () => Mjs.noiseByLevel(m41, resid, valid, W, H), () => Mn.noiseByLevel(m41, resid, valid, W, H));
run("npGradient gy", () => Fjs.npGradient(Y, W, H)[0], () => Fn.npGradient(Y, W, H)[0]);
run("npGradient gx", () => Fjs.npGradient(Y, W, H)[1], () => Fn.npGradient(Y, W, H)[1]);
const half = Fjs.downscaleArea2(Y, W, H);
run("resizeLinear up", () => Fjs.resizeLinear(half.data, half.w, half.h, W, H), () => Fn.resizeLinear(half.data, half.w, half.h, W, H));
(async () => {
  const outside = new Uint8Array(W * H);
  for (let i = 0; i < W * 50; i++) outside[i] = 1;
  for (const b of [true, false]) {
    const a = await Mjs.residualMaps(Y, W, H, b, outside, {}), c = await Mn.residualMaps(Y, W, H, b, outside, {});
    for (const k of ["sigma", "sigma_level", "z_s", "z_l", "z_xl"]) console.log(`residualMaps ${b ? "bright" : "dark"} ${k.padEnd(12)} ${same(a[k], c[k])}`);
    console.log(`residualMaps ${b ? "bright" : "dark"} busy         ${same(a.busy.T, c.busy.T)}`);
  }
})();
// regions + binary morphology + outside mask
(async () => {
  const R = require("../engine/regions");
  for (const k of Object.keys(require.cache)) if (k.includes(require("path").join("engine", ""))) delete require.cache[k];
  process.env.FILMDUST_NATIVE = "0";
  const Rjs = require("../engine/regions"), Fj = require("../engine/filters"), Mj = require("../engine/maps");
  for (const k of Object.keys(require.cache)) if (k.includes(require("path").join("engine", ""))) delete require.cache[k];
  delete process.env.FILMDUST_NATIVE;
  const Rn = require("../engine/regions"), Fnn = require("../engine/filters"), Mnn = require("../engine/maps");
  void R;
  const m = new Uint8Array(W * H);
  for (let i = 0; i < m.length; i++) m[i] = Y[i] > 0.42 && Y[i] < 0.47 ? 1 : 0;
  const same8 = (a, b) => { if (a.length !== b.length) return "length"; let n = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++; return n ? `${n} differ` : "identical"; };
  for (const conn of [4, 8]) {
    let t = Date.now(); const a = Rjs.label(m, W, H, conn); const tj = Date.now() - t;
    t = Date.now(); const b = Rn.label(m, W, H, conn); const tn = Date.now() - t;
    console.log(`label ${conn}`.padEnd(26), `${same8(a.lab, b.lab)} (n ${a.n} vs ${b.n})`.padEnd(40), `js ${(tj / 1000).toFixed(2)} s · native ${(tn / 1000).toFixed(2)} s`);
  }
  for (const [k, el] of [[7, true], [13, true], [5, true], [9, false], [17, false]]) for (const dil of [true, false]) {
    let t = Date.now(); const a = dil ? Fj.dilateMask(m, W, H, k, el) : Fj.erodeMask(m, W, H, k, el); const tj = Date.now() - t;
    t = Date.now(); const b = dil ? Fnn.dilateMask(m, W, H, k, el) : Fnn.erodeMask(m, W, H, k, el); const tn = Date.now() - t;
    console.log(`${dil ? "dilate" : "erode"} ${el ? "ellipse" : "square"} ${k}`.padEnd(26), same8(a, b).padEnd(40), `js ${(tj / 1000).toFixed(2)} s · native ${(tn / 1000).toFixed(2)} s`);
  }
  const Y2 = new Float32Array(Y); for (let y = 0; y < 300; y++) for (let x = 0; x < W; x++) Y2[y * W + x] = 0.95;   // a clipped band touching the top
  let t = Date.now(); const oa = Mj.outsideMask(Y2, W, H); const tj = Date.now() - t;
  t = Date.now(); const ob = Mnn.outsideMask(Y2, W, H); const tn = Date.now() - t;
  console.log("outsideMask".padEnd(26), `${same8(oa, ob)} (${oa.reduce((s, v) => s + v, 0)} px)`.padEnd(40), `js ${(tj / 1000).toFixed(2)} s · native ${(tn / 1000).toFixed(2)} s`);
})();

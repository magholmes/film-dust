"use strict";
// The native network forward against the JS loops, on real marks: logits must be bit-identical,
// for both networks (v1 in slot 0, v2 = patch + context in slot 1).
//   node test/native_net.js <scan.npy> <report.json> [n]
const { readNpy } = require("./npy");
const DN = require("../engine/dustnet");
const NAT = require("../engine/native");
const src = readNpy(process.argv[2]);
const [H, W, C] = src.shape;
const marks = require(require("path").resolve(process.argv[3])).marks.slice(0, Number(process.argv[4] || 600));
const centres = marks.map((m) => [m.cy, m.cx]);
let ok = true;
for (const [net, slot] of [[DN, 0], [DN.second, 1]]) {
  if (!net.available()) { console.log(`slot ${slot}: no network`); continue; }
  let t = Date.now();
  const js = centres.map(([cy, cx]) => net.forward(net.views(src.data, W, H, C, cy, cx, true)));
  const tj = Date.now() - t;
  t = Date.now();
  const nat = net.logits(src.data, W, H, C, centres, true);
  const tn = Date.now() - t;
  // compare unrounded: run the native batch directly
  const T = net.version() >= 2 ? 2 : 1, per = T * 4800, buf = new Float64Array(centres.length * per);
  centres.forEach(([cy, cx], i) => { const v = net.views(src.data, W, H, C, cy, cx, true); for (let k = 0; k < T; k++) for (let c = 0; c < 3; c++) buf.set(v[k][c], i * per + (k * 3 + c) * 1600); });
  const raw = NAT.netForward(buf, centres.length, slot);
  let diff = 0;
  for (let i = 0; i < js.length; i++) if (!raw || Object.is(js[i], raw[i]) === false) diff++;
  const rounded = nat.every((v, i) => v === Math.round(js[i] * 1000) / 1000 || Math.abs(v - js[i]) < 0.0006);
  if (diff || !rounded) ok = false;
  console.log(`slot ${slot} network v${net.version()}: ${centres.length} marks · unrounded logits differing ${diff} · rounded match ${rounded} · js ${(tj / 1000).toFixed(2)} s · native (incl. inputs) ${(tn / 1000).toFixed(2)} s`);
}
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);

"use strict";
// The faint-disc finder on a scan exported as .npy, for checking by eye:
//   node test/disc_find.js <scan.npy> <report.json from run_scan.js> <out.json> [--params '{"disc_z": 5}']
// -> <out.json>: the candidates (working-size pixels) and the quarter-size noise floor
const fs = require("fs");
const path = require("path");
const { readNpy } = require("./npy");
const M = require("../engine/maps");
const D = require("../engine/discs");

const args = process.argv.slice(2);
const [src, repPath, out] = args;
const p = args.includes("--params") ? JSON.parse(args[args.indexOf("--params") + 1]) : {};
const { data, shape } = readNpy(src);
const [H, W, C] = shape.length === 3 ? shape : [shape[0], shape[1], 1];
const rep = JSON.parse(fs.readFileSync(repPath, "utf8"));
const t0 = Date.now();
const outside = M.outsideMask(M.lumaFromU16(data, W, H, C), W, H);
const polarity = (rep.summary && rep.summary.polarity) || ["bright"];
const core = fs.existsSync(repPath.replace(/_report.json$/, "_marks.npy")) ? Uint8Array.from(readNpy(repPath.replace(/_report.json$/, "_marks.npy")).data, (v) => (v ? 1 : 0)) : null;
const res = D.find(data, W, H, C, { outside, marks: rep.marks, core }, p, polarity[0] !== "dark");
for (const c of res.cands) c.why = D.rules(c, p);
const sec = (Date.now() - t0) / 1000;
fs.writeFileSync(out, JSON.stringify({ scan: path.basename(src), W, H, seconds: sec, floor: res.q.floor, cands: res.cands }));
const sm = res.cands.filter((c) => c.smooth), ok = res.cands.filter((c) => !c.why);
console.log(`${path.basename(src)}: ${res.cands.length} candidates (${sm.length} on smooth ground, ${ok.length} pass the rules) in ${sec.toFixed(1)} s`);

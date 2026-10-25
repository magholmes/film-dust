"use strict";
// The subtractive heal on the faint-disc finder's candidates (from test/disc_find.js), for checking by eye:
//   node test/disc_heal.js <scan.npy> <cands.json> <report.json> <out dir> [x0,y0,x1,y1]
// -> <out>/before.npy and after.npy (the crop, default the whole frame), bumps.json (every candidate's fit)
const fs = require("fs");
const path = require("path");
const { readNpy, writeNpy } = require("./npy");
const M = require("../engine/maps");
const D = require("../engine/discs");

const [src, candPath, repPath, outDir, box] = process.argv.slice(2);
const { data, shape } = readNpy(src);
const [H, W, C] = shape.length === 3 ? shape : [shape[0], shape[1], 1];
const cands = JSON.parse(fs.readFileSync(candPath, "utf8")).cands.filter((c) => !c.why);
const marksPath = repPath.replace(/_report\.json$/, "_marks.npy");
const core = fs.existsSync(marksPath) ? Uint8Array.from(readNpy(marksPath).data, (v) => (v ? 1 : 0)) : null;
const outside = M.outsideMask(M.lumaFromU16(data, W, H, C), W, H);
const [x0, y0, x1, y1] = box ? box.split(",").map(Number) : [0, 0, W, H];
const crop = (img) => {
  const out = new Uint16Array((x1 - x0) * (y1 - y0) * C);
  for (let y = y0; y < y1; y++) out.set(img.subarray((y * W + x0) * C, (y * W + x1) * C), (y - y0) * (x1 - x0) * C);
  return out;
};
fs.mkdirSync(outDir, { recursive: true });
writeNpy(path.join(outDir, "before.npy"), crop(data), [y1 - y0, x1 - x0, C]);
const t0 = Date.now();
const bumps = cands.map((c) => Object.assign(D.bump(data, W, H, C, c, core, outside), { cand: c }));
const ok = bumps.filter((b) => b.ok);
const mask = new Uint8Array(W * H);
D.subtract(data, W, H, C, ok, mask);
const sec = (Date.now() - t0) / 1000;
writeNpy(path.join(outDir, "after.npy"), crop(data), [y1 - y0, x1 - x0, C]);
let px = 0;
for (const v of mask) px += v;
const why = {};
for (const b of bumps) if (!b.ok) why[b.why] = (why[b.why] || 0) + 1;
fs.writeFileSync(path.join(outDir, "bumps.json"), JSON.stringify(bumps.map((b) => ({ cx: b.cand.cx, cy: b.cand.cy, r: b.cand.r, z: b.cand.z, ok: b.ok, why: b.why, peak: b.peak, ring: b.ring }))));
console.log(`${path.basename(src)}: ${ok.length} of ${cands.length} discs taken away (${px} px changed) in ${sec.toFixed(1)} s; left: ${JSON.stringify(why)}`);

"use strict";
// The large-scan path on a real crop: find marks on the half-size copy, heal the full-size crop in
// small tiles, and check that
//   - only pixels inside the enlarged mask change,
//   - every mark is healed (its pixels move off the dust),
//   - the tiled heal lands close to healing the whole crop at once.
//   node test/bigheal_check.js <full.npy> <half.npy> <out dir>
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const { readNpy, writeNpy } = require("./npy");
const { cleanScan } = require("../engine/pipeline");
const { heal } = require("../engine/heal");
const { healLarge } = require("../engine/bigheal");
const S = require("../engine/scale");

(async () => {
  const [fullPath, halfPath, outDir] = process.argv.slice(2);
  const full = readNpy(fullPath), half = readNpy(halfPath);
  const [H, W, C] = full.shape, [Hd, Wd] = half.shape;
  const k = 2;
  const work = Uint16Array.from(half.data);
  const res = await cleanScan(work, Wd, Hd, C, { heal: true }, { tick: () => new Promise((r) => setImmediate(r)) });
  console.log(`working copy ${Wd}x${Hd}: ${res.stats.healed} marks (${JSON.stringify(res.stats.by_kind)})`);
  assert(res.stats.healed > 0, "no marks found in the test crop");

  const big = S.enlargeMask(res.mask, Wd, Hd, k, W, H);
  const out = Uint16Array.from(full.data);
  let reads = 0, writes = 0;
  const io = {
    read: async (x0, y0, x1, y1) => {
      reads++;
      const t = new Uint16Array((x1 - x0) * (y1 - y0) * C);
      for (let y = y0; y < y1; y++) t.set(full.data.subarray((y * W + x0) * C, (y * W + x1) * C), (y - y0) * (x1 - x0) * C);
      return t;
    },
    write: async (x0, y0, x1, y1, data) => {
      writes++;
      for (let y = y0; y < y1; y++) out.set(data.subarray((y - y0) * (x1 - x0) * C, (y - y0 + 1) * (x1 - x0) * C), (y * W + x0) * C);
    },
  };
  // the layer is shown only through the mask: compose what Photoshop would show
  const t0 = Date.now();
  const r = await healLarge(res.mask, res.outside, Wd, Hd, k, W, H, C, io, {}, { cell: 256, margin: 192 });
  const shown = Uint16Array.from(full.data);
  for (let i = 0; i < W * H; i++) if (big[i]) for (let c = 0; c < C; c++) shown[i * C + c] = out[i * C + c];
  console.log(`tiled heal: ${r.marks} marks in ${r.cells} cells, ${reads} reads, ${writes} writes, ${r.fallbacks} fallbacks, ${Date.now() - t0} ms`);

  // whole-crop heal for comparison
  const whole = Uint16Array.from(full.data);
  const b01 = new Uint8Array(W * H);
  for (let i = 0; i < b01.length; i++) b01[i] = big[i] ? 1 : 0;
  const outsideFull = res.outside ? S.enlargeMask(res.outside, Wd, Hd, k, W, H) : null;
  await heal(whole, W, H, C, b01, outsideFull);

  let changedOutside = 0, inMask = 0, moved = 0, diffSum = 0;
  for (let i = 0; i < W * H; i++) {
    let d = 0, dw = 0;
    for (let c = 0; c < C; c++) { d += Math.abs(shown[i * C + c] - full.data[i * C + c]); dw += Math.abs(shown[i * C + c] - whole[i * C + c]); }
    if (!big[i]) { if (d) changedOutside++; continue; }
    inMask++;
    if (d) moved++;
    diffSum += dw / C;
  }
  console.log(`mask ${inMask} px, ${moved} changed; mean |tiled - whole| inside the mask ${(diffSum / inMask / 655.35).toFixed(2)}% of full scale`);
  assert.strictEqual(changedOutside, 0, "pixels outside the mask changed");
  assert(moved / inMask > 0.9, "most mask pixels should change");
  fs.mkdirSync(outDir, { recursive: true });
  writeNpy(path.join(outDir, "tiled.npy"), shown, [H, W, C]);
  writeNpy(path.join(outDir, "whole.npy"), whole, [H, W, C]);
  writeNpy(path.join(outDir, "mask.npy"), big, [H, W]);
  console.log("bigheal check passed");
})().catch((e) => { console.error(e); process.exit(1); });

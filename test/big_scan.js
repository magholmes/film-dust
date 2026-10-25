"use strict";
// The panel's big-scan path end to end in node, timed: average the full-size scan k x k into the working
// copy (as readWorking does strip by strip), find and heal on it (cleanScan), then heal the full-size
// pixels tile by tile (healLarge, the panel's healOntoLayer without Photoshop).
//   node --max-old-space-size=24000 test/big_scan.js <full.npy> [out dir]
const path = require("path");
const { readNpy, writeNpy } = require("./npy");
const { cleanScan } = require("../engine/pipeline");
const { healLarge } = require("../engine/bigheal");
const S = require("../engine/scale");
const NAT = require("../engine/native");
const t0 = Date.now();
const lap = (() => { let t = Date.now(); return (what) => { const now = Date.now(); console.log(`${what.padEnd(34)} ${((now - t) / 1000).toFixed(1)} s`); t = now; }; })();
(async () => {
  const full = readNpy(process.argv[2]);
  const [H, W, C] = full.shape;
  lap(`read ${W} x ${H} (${((W * H) / 1e6).toFixed(0)} MP), native ${NAT.available() ? "on" : "off"}`);
  const k = S.workingFactor(W, H), Wd = Math.floor(W / k), Hd = Math.floor(H / k);
  const work = new Uint16Array(Wd * Hd * C);
  const rowsPer = Math.max(k, Math.floor(48e6 / (W * 3 * 2) / k) * k);
  let yd = 0;
  for (let y0 = 0; y0 < Hd * k; y0 += rowsPer) {
    const y1 = Math.min(Hd * k, y0 + rowsPer);
    const strip = new Uint16Array((y1 - y0) * Wd * k * C);
    for (let y = y0; y < y1; y++) strip.set(full.data.subarray((y * W) * C, (y * W + Wd * k) * C), (y - y0) * Wd * k * C);
    yd += S.averageStrip(strip, Wd * k, y1 - y0, C, k, work, Wd, yd, false);
  }
  lap(`average to ${Wd} x ${Hd} (k ${k})`);
  const res = await cleanScan(work, Wd, Hd, C, { heal: true, scanMp: (W * H) / 1e6 }, { tick: () => new Promise((r) => setImmediate(r)) });
  lap(`find + heal the working copy: ${res.stats.healed} marks`);
  let reads = 0, writes = 0;
  const io = {
    read: async (x0, y0, x1, y1) => {
      reads++;
      const w = x1 - x0, out = new Uint16Array((y1 - y0) * w * C);
      for (let y = y0; y < y1; y++) out.set(full.data.subarray((y * W + x0) * C, (y * W + x1) * C), (y - y0) * w * C);
      return out;
    },
    write: async () => { writes++; },
  };
  await healLarge(res.mask, res.outside, Wd, Hd, k, W, H, C, io, { tick: () => new Promise((r) => setImmediate(r)) });
  lap(`heal at full size (${reads} tiles read, ${writes} written)`);
  console.log(`total ${((Date.now() - t0) / 1000).toFixed(1)} s`);
})().catch((e) => { console.error(e); process.exit(1); });

"use strict";
// Every faint disc (and soft fibre of the scan's own dust colour) the engine takes away on a scan, for checking by
// eye: the whole engine (no Claude), then for each one healed its crop before and after (2.5 radii around it, 16-bit
// .npy pairs) and a summary.
//   node test/disc_survey.js <scan.npy> <out dir> [--params '{...}'] [--max 60]
// -> <out>/discs.json (stats, the discs with their measures), <out>/d_NN_before.npy and d_NN_after.npy
const fs = require("fs");
const path = require("path");
const { readNpy, writeNpy } = require("./npy");
const { cleanScan } = require("../engine/pipeline");

const args = process.argv.slice(2);
const [src, outDir] = args;
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const { data, shape } = readNpy(src);
const [H, W, C] = shape.length === 3 ? shape : [shape[0], shape[1], 1];
const orig = Uint16Array.from(data);
(async () => {
  // --subject-box x0,y0,x1,y1 stands in for Select Subject (as in run_scan.js)
  const subject = args.includes("--subject-box") ? (() => { const [a, b, c, d] = opt("--subject-box", "").split(",").map(Number); const m = new Uint8Array(W * H); for (let y = b; y < Math.min(H, d); y++) m.fill(1, y * W + a, y * W + Math.min(W, c)); return m; })() : undefined;
  const res = await cleanScan(data, W, H, C, { heal: true, subject, params: args.includes("--params") ? JSON.parse(opt("--params", "{}")) : undefined }, { tick: () => new Promise((r) => setImmediate(r)) });
  fs.mkdirSync(outDir, { recursive: true });
  // the fibres the detector left as soft and the scan's dust colour took (healed like a speck) count with the discs
  const discOf = (m) => m.disc || { r: Math.max(m.width || 8, (m.length || 16) / 2), z: m.z };
  const discs = res.marks.filter((m) => m.disc || m.fibre);
  const max = Number(opt("--max", 60));
  const crop = (img, m) => {
    const R = Math.round(Math.max(12, discOf(m).r) * 2.5), x0 = Math.max(0, m.cx - R), y0 = Math.max(0, m.cy - R), x1 = Math.min(W, m.cx + R), y1 = Math.min(H, m.cy + R);
    const out = new Uint16Array((x1 - x0) * (y1 - y0) * C);
    for (let y = y0; y < y1; y++) out.set(img.subarray((y * W + x0) * C, (y * W + x1) * C), (y - y0) * (x1 - x0) * C);
    return { out, shape: [y1 - y0, x1 - x0, C] };
  };
  // --near: only the small discs and those beside an edge (disc_rescue_near)
  // --near / --fibre-discs: only the small discs and those beside an edge (disc_rescue_near) / the soft fibres among the
  // candidates (disc_rescue_fibre)
  const pick = discs.filter((m) => m.verdict === "defect" && (!args.includes("--near") || m.nearRescue) && (!args.includes("--fibre-discs") || m.fibreDisc)).sort((a, b) => discOf(b).z - discOf(a).z).slice(0, max);
  pick.forEach((m, i) => {
    const b = crop(orig, m), a = crop(data, m), tag = String(i + 1).padStart(2, "0");
    writeNpy(path.join(outDir, `d_${tag}_before.npy`), b.out, b.shape);
    writeNpy(path.join(outDir, `d_${tag}_after.npy`), a.out, a.shape);
  });
  fs.writeFileSync(path.join(outDir, "discs.json"), JSON.stringify({
    scan: path.basename(src), W, H, stats: res.stats,
    discs: discs.map((m) => Object.assign({ verdict: m.verdict, cx: m.cx, cy: m.cy, fibre: m.fibre, near: m.nearRescue, fibreDisc: m.fibreDisc }, discOf(m))), shown: pick.map((m) => [m.cx, m.cy]),
  }));
  console.log(`${path.basename(src)}: ${discs.filter((m) => m.verdict === "defect").length} discs taken away · healed ${res.stats.healed} · ${res.stats.seconds} s`);
})().catch((e) => { console.error(e); process.exit(1); });

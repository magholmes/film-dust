"use strict";
// Run the plugin engine on a scan exported as .npy (H x W x C uint16) and write what the Python
// skill writes, so its regression check can read it:
//   node test/run_scan.js <scan.npy> <out dir> <stem> [--detect-only] [--passes N] [--polarity P]
// -> <stem>_report.json, <stem>_marks.npy (0/255), <stem>_clean.npy (only without --detect-only)
const fs = require("fs");
const path = require("path");
const { readNpy, writeNpy } = require("./npy");
const { cleanScan } = require("../engine/pipeline");

// an 8-bit grayscale PNG (zlib from node; filter 0 on every row)
function grayPng(px, w, h) {
  const zlib = require("zlib");
  const crcT = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
  const crc = (b) => { let c = -1; for (let i = 0; i < b.length; i++) c = crcT[(c ^ b[i]) & 0xff] ^ (c >>> 8); return (c ^ -1) >>> 0; };
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type, "ascii"), data]);
    const c = Buffer.alloc(4); c.writeUInt32BE(crc(td));
    return Buffer.concat([len, td, c]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 0; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const raw = Buffer.alloc((w + 1) * h);
  for (let y = 0; y < h; y++) { raw[y * (w + 1)] = 0; Buffer.from(px.buffer, px.byteOffset + y * w, w).copy(raw, y * (w + 1) + 1); }
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", zlib.deflateSync(raw, { level: 6 })), chunk("IEND", Buffer.alloc(0))]);
}

const args = process.argv.slice(2);
const [src, outDir, stem] = args;
const flag = (k) => args.includes(k);
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const t0 = Date.now();
const { data, shape } = readNpy(src);
const [H, W, C] = shape.length === 3 ? shape : [shape[0], shape[1], 1];
console.log(`${path.basename(src)}: ${W}x${H}, ${C} ch, loaded in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
let lastWords = "";
(async () => {
const res = await cleanScan(data, W, H, C, {
  polarity: opt("--polarity", "auto"), passes: Number(opt("--passes", 2)), heal: !flag("--detect-only"),
  params: flag("--params") ? JSON.parse(opt("--params", "{}")) : undefined,
  // --subject-box x0,y0,x1,y1: a stand-in for Photoshop's Select Subject (tests the subject rules)
  subject: flag("--subject-box") ? (() => { const [a, b, c, d] = opt("--subject-box", "").split(",").map(Number); const m = new Uint8Array(W * H); for (let y = b; y < d; y++) m.fill(1, y * W + a, y * W + c); return m; })() : undefined,
}, {
  log: (s) => console.log(s), timing: true,
  tick: () => new Promise((r) => setImmediate(r)),
  progress: (f, words) => {
    if (words === lastWords) return;
    console.log(`  ${((Date.now() - t0) / 1000).toFixed(0).padStart(4)} s  ${(100 * f).toFixed(0).padStart(3)}%  ${words}`);
    lastWords = words;
  },
});
fs.mkdirSync(outDir, { recursive: true });
const m255 = new Uint8Array(res.mask.length);
for (let i = 0; i < m255.length; i++) m255[i] = res.mask[i] ? 255 : 0;
writeNpy(path.join(outDir, `${stem}_marks.npy`), m255, [H, W]);
fs.writeFileSync(path.join(outDir, `${stem}_marks.png`), grayPng(m255, W, H));   // what regress.py reads
if (!flag("--detect-only")) writeNpy(path.join(outDir, `${stem}_clean.npy`), data, shape);
fs.writeFileSync(path.join(outDir, `${stem}_report.json`), JSON.stringify({ summary: res.stats, marks: res.marks }));
console.log("RESULT " + JSON.stringify(res.stats));
const mem = process.memoryUsage();
console.log(`peak-ish memory: rss ${(mem.rss / 2 ** 30).toFixed(1)} GB, array buffers ${(mem.arrayBuffers / 2 ** 30).toFixed(1)} GB`);
})().catch((e) => { console.error(e); process.exit(1); });

"use strict";
// For speed work that must not change a single result: the whole engine on a scan, reduced to fingerprints of
// what it produced (the healed image and the mask, SHA-1) and every mark's verdict, then a comparison of two runs.
//   node test/exact_check.js <scan.npy> <out.json> [--subject-box x0,y0,x1,y1] [--params '{...}']
//   node test/exact_check.js --compare <a.json> <b.json>
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const args = process.argv.slice(2);
if (args[0] === "--compare") {
  const [a, b] = args.slice(1, 3).map((f) => JSON.parse(fs.readFileSync(f, "utf8")));
  const diffs = [];
  for (const k of ["clean_sha1", "mask_sha1", "feat_sha1"]) if (a[k] && b[k] && a[k] !== b[k]) diffs.push(`${k} differs`);
  const key = (m) => `${m[0]}|${m[1]}|${m[2]}`;
  const A = new Map(a.marks.map((m) => [key(m), m])), B = new Map(b.marks.map((m) => [key(m), m]));
  let shown = 0;
  for (const [k, m] of A) {
    const o = B.get(k);
    if (!o) { if (shown++ < 10) diffs.push(`only in a: ${m.join(" ")}`); continue; }
    if (o[3] !== m[3] || o[4] !== m[4]) { if (shown++ < 10) diffs.push(`verdict: ${m.join(" ")}  ->  ${o[3]} ${o[4]}`); }
  }
  for (const [k, m] of B) if (!A.has(k) && shown++ < 10) diffs.push(`only in b: ${m.join(" ")}`);
  const same = !diffs.length && a.marks.length === b.marks.length;
  console.log(`${a.scan}: ${same ? "IDENTICAL" : "DIFFERENT"} · ${a.marks.length} / ${b.marks.length} marks · ${a.seconds} s -> ${b.seconds} s${diffs.length ? "\n  " + diffs.join("\n  ") : ""}`);
  process.exit(same ? 0 : 1);
}

const { readNpy } = require("./npy");
const { cleanScan } = require("../engine/pipeline");
const [src, out] = args;
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const { data, shape } = readNpy(src);
const [H, W, C] = shape.length === 3 ? shape : [shape[0], shape[1], 1];
(async () => {
  const subject = args.includes("--subject-box") ? (() => { const [a, b, c, d] = opt("--subject-box", "").split(",").map(Number); const m = new Uint8Array(W * H); for (let y = b; y < Math.min(H, d); y++) m.fill(1, y * W + a, y * W + Math.min(W, c)); return m; })() : undefined;
  const t0 = Date.now();
  const res = await cleanScan(data, W, H, C, { heal: true, subject, params: args.includes("--params") ? JSON.parse(opt("--params", "{}")) : undefined }, { tick: () => new Promise((r) => setImmediate(r)) });
  const seconds = Math.round((Date.now() - t0) / 100) / 10;
  const sha = (a) => crypto.createHash("sha1").update(Buffer.from(a.buffer, a.byteOffset, a.byteLength)).digest("hex");
  const marks = res.marks.map((m) => [m.kind, m.cx, m.cy, m.verdict, String(m.why || "")]).sort((p, q) => p[1] - q[1] || p[2] - q[2] || (p[0] < q[0] ? -1 : 1));
  // every mark with every measure, in position order (a stricter fingerprint than the verdicts)
  const all = res.marks.slice().sort((p, q) => p.cx - q.cx || p.cy - q.cy || (p.kind < q.kind ? -1 : p.kind > q.kind ? 1 : 0));
  const feat_sha1 = crypto.createHash("sha1").update(JSON.stringify(all, (k, v) => (v instanceof Uint8Array ? undefined : v))).digest("hex");
  fs.writeFileSync(out, JSON.stringify({ scan: path.basename(src), seconds, stats: res.stats, clean_sha1: sha(data), mask_sha1: sha(res.mask), feat_sha1, marks }));
  console.log(`${path.basename(src)}: ${seconds} s · ${res.stats.healed} healed · clean ${sha(data).slice(0, 10)} · mask ${sha(res.mask).slice(0, 10)}`);
})().catch((e) => { console.error(e); process.exit(1); });

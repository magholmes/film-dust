"use strict";
// Regression check: known marks on a reference scan must keep their verdicts.
//   node test/regress.js <report.json> <known marks.json>
// Reads the healed mask written next to the report by run_scan.js (<stem>_marks.npy): "defect" needs
// healed pixels within 20 px of the point (a hair's centre can fall between its strands), "keep-photo"
// none within 4 px. The report's nearest entry is shown for context. Exit code 1 if anything regressed.
const fs = require("fs"), path = require("path");
const { readNpy } = require("./npy");
const repPath = process.argv[2];
const report = JSON.parse(fs.readFileSync(repPath, "utf8"));
const known = JSON.parse(fs.readFileSync(process.argv[3], "utf8")).marks;
const stem = path.basename(repPath).replace(/_report\.json$/, "");
const maskPath = path.join(path.dirname(repPath), `${stem}_marks.npy`);
const mask = fs.existsSync(maskPath) ? readNpy(maskPath) : null;
let bad = 0;
for (const k of known) {
  const best = report.marks.reduce((b, m) => (!b || Math.abs(m.cx - k.x) + Math.abs(m.cy - k.y) < Math.abs(b.cx - k.x) + Math.abs(b.cy - k.y) ? m : b), null);
  const near = best && Math.abs(best.cx - k.x) + Math.abs(best.cy - k.y) <= 40;
  let ok, got;
  if (mask) {
    const [H, W] = mask.shape, r = k.expect === "defect" ? 20 : 4;
    let n = 0;
    for (let y = Math.max(0, k.y - r); y <= Math.min(H - 1, k.y + r); y++) for (let x = Math.max(0, k.x - r); x <= Math.min(W - 1, k.x + r); x++) if (mask.data[y * W + x]) n++;
    ok = k.expect === "defect" ? n > 0 : n === 0;
    got = `healed ${n} px`;
  } else {
    got = near ? best.verdict : "not found";
    ok = got === k.expect || (k.expect === "keep-photo" && got === "not found");
  }
  if (!ok) bad++;
  console.log(`${ok ? "ok " : "BAD"}  ${k.what.padEnd(44)} expect ${k.expect.padEnd(10)} ${got.padEnd(11)} ${near ? best.why.slice(0, 48) : ""}`);
}
console.log(`\n${known.length - bad}/${known.length} as expected`);
process.exitCode = bad ? 1 : 0;

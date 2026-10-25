// The native gradient length against Math.hypot, bit for bit: node test/hypot_check.js [scan.npy ...]
// (random numbers over many magnitudes, then each scan's real gradients as detect.js makes them)
const { readNpy } = require("./npy");
const NAT = require("../engine/native");
const F = require("../engine/filters");
const M = require("../engine/maps");

function compare(name, gx, gy) {
  const js = new Float32Array(gy.length);
  let t = Date.now();
  for (let i = 0; i < gy.length; i++) js[i] = Math.hypot(gx[i], gy[i]);
  const tJs = Date.now() - t;
  const nat = Float32Array.from(gy);
  t = Date.now();
  if (!NAT.hypotInto(nat, gx)) throw new Error("native hypot not available");
  const tNat = Date.now() - t;
  const a = new Uint32Array(js.buffer), b = new Uint32Array(nat.buffer);
  let diff = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
  console.log(`${name}: ${diff === 0 ? "IDENTICAL" : diff + " DIFFERENT"} · ${gy.length} values · JS ${tJs} ms, native ${tNat} ms`);
  return diff === 0;
}

let ok = true;
{
  const n = 1 << 22, gx = new Float32Array(n), gy = new Float32Array(n);
  let s = 7;
  const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 4294967296);
  for (let i = 0; i < n; i++) {
    const ex = Math.floor(rnd() * 40) - 30;
    gx[i] = (rnd() - 0.5) * 2 ** ex;
    gy[i] = i % 7 === 0 ? 0 : (rnd() - 0.5) * 2 ** (ex + Math.floor(rnd() * 6) - 3);
  }
  ok = compare("random", gx, gy) && ok;
}
for (const f of process.argv.slice(2)) {
  const { data, shape } = readNpy(f);
  const [H, W, C] = shape.length === 3 ? shape : [shape[0], shape[1], 1];
  const Y = M.lumaFromU16(data, W, H, C);
  const yb1 = F.gaussianBlur(Y, W, H, 1.0);
  const [gy, gx] = F.npGradient(yb1, W, H);
  ok = compare(f.split(/[\\/]/).pop(), gx, gy) && ok;
}
process.exit(ok ? 0 : 1);

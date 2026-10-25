// Native blob tiles (BlobField.fetch) against tile() one at a time, bit for bit: node test/blob_tiles_check.js <scan.npy>
// Every scale, tiles in the middle and on every edge and corner, both polarities.
const { readNpy } = require("./npy");
const M = require("../engine/maps");
const { BlobField } = require("../engine/detect");

const { data, shape } = readNpy(process.argv[2]);
const [H, W, C] = shape.length === 3 ? shape : [shape[0], shape[1], 1];
const Y = M.lumaFromU16(data, W, H, C);
const sl = new Float32Array(W * H);
for (let i = 0; i < sl.length; i++) sl[i] = 0.002 + 0.01 * Y[i] * Y[i];
const T = 512, nty = Math.ceil(H / T), ntx = Math.ceil(W / T);
const spots = [];
for (const ty of [0, 1, (nty >> 1), nty - 2, nty - 1]) for (const tx of [0, 1, (ntx >> 1), ntx - 2, ntx - 1]) spots.push(ty, tx);
const bits = (a) => new Uint32Array(a.buffer, a.byteOffset, a.length);
let bad = 0, tiles = 0, peaks = 0, tA = 0, tB = 0;
for (const sgn of [1, -1]) for (const q of [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]) {
  const a = new BlobField(Y, W, H, sl, sgn), b = new BlobField(Y, W, H, sl, sgn);
  let t = Date.now();
  for (let i = 0; i < spots.length; i += 2) a.tile(q, spots[i], spots[i + 1]);
  tA += Date.now() - t;
  t = Date.now();
  b.fetch(q, spots, true);
  tB += Date.now() - t;
  for (let i = 0; i < spots.length; i += 2) {
    const key = q + ":" + spots[i] + ":" + spots[i + 1];
    const ta = a.tiles.get(key), tb = b.tiles.get(key);
    tiles++;
    if (!tb) { bad++; console.log(`${key} sgn ${sgn}: not fetched`); continue; }
    const same = ta.ys.length === tb.ys.length && ta.ys.every((v, j) => v === tb.ys[j] && ta.xs[j] === tb.xs[j] && ta.vals[j] === tb.vals[j]) &&
      ta.y0 === tb.y0 && ta.x0 === tb.x0 && ta.w === tb.w && ta.resp.length === tb.resp.length && bits(ta.resp).every((v, j) => v === bits(tb.resp)[j]);
    peaks += ta.ys.length;
    if (!same) { bad++; console.log(`${key} sgn ${sgn}: DIFFERENT (${ta.ys.length} vs ${tb.ys.length} maxima)`); }
  }
}
console.log(`${bad ? bad + " DIFFERENT" : "IDENTICAL"} · ${tiles} tiles · ${peaks} maxima · one at a time ${tA} ms, native batch ${tB} ms`);
process.exit(bad ? 1 : 0);

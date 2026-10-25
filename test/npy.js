"use strict";
// Minimal .npy reader/writer for tests (little-endian, C order; f4, u1, u2, b1).
const fs = require("fs");

function readNpy(path) {
  const buf = fs.readFileSync(path);
  if (buf.toString("latin1", 1, 6) !== "NUMPY") throw new Error("not a .npy file: " + path);
  const major = buf[6];
  const hlen = major === 1 ? buf.readUInt16LE(8) : buf.readUInt32LE(8);
  const hoff = major === 1 ? 10 : 12;
  const header = buf.toString("latin1", hoff, hoff + hlen);
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  if (/'fortran_order':\s*True/.test(header)) throw new Error("fortran order not supported");
  const data = buf.subarray(hoff + hlen);
  const ab = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  let arr;
  if (descr === "<f4") arr = new Float32Array(ab);
  else if (descr === "|u1" || descr === "|b1") arr = new Uint8Array(ab);
  else if (descr === "<u2") arr = new Uint16Array(ab);
  else if (descr === "<i4") arr = new Int32Array(ab);
  else if (descr === "<f8") arr = new Float64Array(ab);
  else throw new Error("dtype not supported: " + descr);
  return { data: arr, shape };
}

function writeNpy(path, arr, shape) {
  const descr = arr instanceof Float32Array ? "<f4" : arr instanceof Uint8Array ? "|u1" : arr instanceof Uint16Array ? "<u2" : arr instanceof Int32Array ? "<i4" : "<f8";
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`;
  const pad = 64 - ((10 + header.length + 1) % 64);
  header += " ".repeat(pad) + "\n";
  const h = Buffer.alloc(10);
  h.write("\x93NUMPY", 0, "latin1");
  h[6] = 1; h[7] = 0;
  h.writeUInt16LE(header.length, 8);
  fs.writeFileSync(path, Buffer.concat([h, Buffer.from(header, "latin1"), Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength)]));
}

module.exports = { readNpy, writeNpy };

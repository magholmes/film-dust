"use strict";
// A failure inside the native code must come back to JS as an error (never take the host down), and the
// addon must keep working afterwards: memory running out on the calling thread and inside a worker thread,
// a plain exception, and arguments that don't fit (a buffer too small, a missing array buffer).
//   node test/native_errors.js
const assert = require("assert");
const path = require("path");
const N = require(path.join(__dirname, "..", "native", "build", "filmdust.node"));

const expectError = (label, fn, pattern) => {
  let err = null;
  try { fn(); } catch (e) { err = e; }
  assert(err, `${label}: no error`);
  assert(pattern.test(String(err.message)), `${label}: unexpected message ${err.message}`);
  console.log(`${label}: ${err.message}`);
};
const blurOnce = () => {
  const w = 300, h = 200, img = new Float32Array(w * h).map((_, i) => (i % 97) / 97);
  return new Float32Array(N.boxBlur(img.buffer, 0, w, h, 5));
};
const ref = blurOnce();
for (let round = 0; round < 3; round++) {
  expectError("out of memory, calling thread", () => N.selfTest(1), /out of memory/);
  expectError("out of memory, worker thread", () => N.selfTest(2), /out of memory/);
  expectError("plain exception", () => N.selfTest(3), /self test/);
  expectError("buffer too small", () => N.boxBlur(new Float32Array(10).buffer, 0, 300, 200, 5), /too small/);
  expectError("not an array buffer", () => N.boxBlur(42, 0, 300, 200, 5), /bad image/);
  expectError("empty image", () => N.boxBlur(new Float32Array(10).buffer, 0, 0, 10, 5), /empty image/);
  // named fields (the functions that take one object of them depend on this)
assert.strictEqual(N.selfTest(4, { n: 2, v: [new Float64Array([3]).buffer, 0] }), 5, "named fields not read");
// still healthy: the same result as before the failures, on the worker threads
  assert.deepStrictEqual(blurOnce(), ref, "results changed after the failures");
}
// the engine's wrapper: a native error switches native off and the caller gets null (it then runs its JS version)
const NAT = require("../engine/native");
assert(NAT.available(), "native not loaded");
const r = NAT.boxBlur({ buffer: 42, byteOffset: 0, length: 1 }, 300, 200, 5);
assert.strictEqual(r, null, "the wrapper should hand back null");
assert(!NAT.available() && /switched off mid-run/.test(NAT.loadError()), "native should now be off");
console.log("native errors: all came back as JS errors; the pool keeps working; the engine falls back to JS");

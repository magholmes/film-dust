"use strict";
// The panel's strip-wise mask writer must draw exactly the enlarged mask:
//   node test/mask_strips_check.js
const assert = require("assert");
const Module = require("module");
const S = require("../engine/scale");

const load = Module._load;
Module._load = function (req, ...rest) {
  if (req === "photoshop") return { app: {}, core: {}, action: { addNotificationListener() {} }, imaging: { createImageDataFromBuffer: async (buf, o) => ({ buf, o, dispose() {} }) } };
  return load.call(this, req, ...rest);
};
global.document = {
  getElementById: () => ({ addEventListener() {}, setAttribute() {}, style: {} }), querySelectorAll: () => [],
  documentElement: { style: { setProperty() {} } }, theme: { getCurrent: () => "darkest", onUpdated: { addListener() {} } },
};
const { putMaskStrips } = require("../panel.js");

(async () => {
  for (const [Wd, Hd, k, W, H, density] of [[300, 220, 2, 601, 441, 0.002], [97, 61, 3, 293, 185, 0.01], [120, 90, 1, 120, 90, 0.005]]) {
    const m = new Uint8Array(Wd * Hd);
    for (let i = 0; i < m.length; i++) if (Math.random() < density) m[i] = 1;
    m[Wd * Hd - 1] = 1;   // a mark in the last block: the odd edge must copy it
    const src = { fullW: W, fullH: H, k, W: Wd, H: Hd };
    const got = new Uint8Array(W * H);
    let strips = 0;
    await putMaskStrips(m, src, async (id, top) => { strips++; got.set(id.buf, top * W); assert.strictEqual(id.o.width, W); }, async () => {}, W * 7 * k);
    const want = k > 1 ? S.enlargeMask(m, Wd, Hd, k, W, H) : Uint8Array.from(m, (v) => (v ? 255 : 0));
    assert.deepStrictEqual(got, want, `strips differ for ${W}x${H} k${k}`);
    console.log(`${W}x${H} k${k}: ${strips} strip(s) match the enlarged mask`);
  }
  console.log("mask strip checks passed");
})().catch((e) => { console.error(e); process.exit(1); });

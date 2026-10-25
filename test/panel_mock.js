"use strict";
// Runs panel.js end to end in node against a mock of Photoshop's UXP API (photoshop + uxp modules and
// a tiny DOM): Remove dust on a synthetic 16-bit scan with dust, then "Learn from my fixes" after
// painting two heals back on the mask. Catches code errors in the panel's flow; it can't prove that
// Photoshop accepts each call (only Photoshop can).
//   node test/panel_mock.js      (MOCK_K=2: the big-scan path · MOCK_SMALL=1: judged as the small scan it is)
const fs = require("fs"), os = require("os"), path = require("path"), Module = require("module");

// ---------- a synthetic scan: grain, a gradient, specks, a hair --------------------------------
// MOCK_K: twice the size, so the working copy (half) is as big as the plain test (the faint-disc finder needs room)
const W = process.env.MOCK_K ? 1800 : 900, H = process.env.MOCK_K ? 1400 : 700, C = 3;
const img = new Uint16Array(W * H * C);
let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  let v = 0.35 + 0.15 * Math.sin(x / 140) * Math.cos(y / 170) + (rnd() - 0.5) * 0.025;
  for (const [sx, sy, r] of [[150, 120, 3], [600, 300, 2], [420, 560, 4], [800, 90, 2.5]]) if ((x - sx) ** 2 + (y - sy) ** 2 < r * r) v = 0.92;
  if (Math.abs((y - 400) - 0.5 * (x - 300)) < 1.3 && x > 250 && x < 420) v = 0.88;
  // a patch of plain sky with a faint soft disc on it (out-of-focus dust, ~2% over the sky): the faint-disc finder's
  if (x >= 620 && y >= 430) {
    v = 0.45 + (rnd() - 0.5) * 0.025;
    const [dx0, dy0] = process.env.MOCK_K ? [1300, 1000] : [760, 565];   // well inside the plain patch
    v += 0.012 * Math.exp(-((x - dx0) ** 2 + (y - dy0) ** 2) / (2 * 34 * 34));   // soft, like real ones (big enough to be found at half size too)
  }
  for (let c = 0; c < C; c++) img[(y * W + x) * C + c] = Math.round(Math.min(1, v * (c === 1 ? 1.0 : 0.95)) * 65535);
}

// ---------- the mock host ---------------------------------------------------------------------
const calls = [];
let layerSeq = 10;
const masks = new Map();   // layerID -> Uint8Array (full size) user mask
const pixels = new Map();  // layerID -> Uint16Array (full size) what was written on it
const doc = { id: 1, title: "mock scan.tif", width: W, height: H, bitsPerChannel: "bitDepth16", mode: "RGBColorMode", layers: [], activeLayers: [] };
const newLayer = (name) => {
  const l = { id: ++layerSeq, name, visible: true };
  doc.layers.unshift(l);
  doc.activeLayers = [l];
  return l;
};
newLayer("Background");
const imageData = (data, width, height, components) => ({
  width, height, components, colorSpace: components === 1 ? "Grayscale" : "RGB", colorProfile: "sRGB IEC61966-2.1",
  getData: async () => data, dispose() {},
});
const photoshop = {
  app: { get activeDocument() { return doc; } },
  core: {
    executeAsModal: async (fn) => fn({ isCancelled: false, reportProgress() {}, hostControl: { suspendHistory: async () => 1, resumeHistory: async () => {} } }),
  },
  action: {
    batchPlay: async (descs) => {
      for (const d of descs) {
        calls.push(d._obj);
        if (d._obj === "make" && d.using && d.using._obj === "layer") newLayer(d.using.name);
        if (d._obj === "make" && d.new && d.new._class === "channel") masks.set(doc.activeLayers[0].id, new Uint8Array(W * H));
        if (d._obj === "select" && d._target[0]._ref === "layer") doc.activeLayers = doc.layers.filter((l) => l.id === d._target[0]._id);
      }
      return [];
    },
    addNotificationListener() {},
  },
  imaging: {
    getPixels: async ({ sourceBounds: b }) => {
      const w = b.right - b.left, h = b.bottom - b.top, out = new Uint16Array(w * h * C);
      for (let y = 0; y < h; y++) out.set(img.subarray(((b.top + y) * W + b.left) * C, ((b.top + y) * W + b.right) * C), y * w * C);
      // MOCK_PROFILE=<name>: the scan reports a profile Photoshop then refuses when handed it back (-26120)
      return { imageData: Object.assign(imageData(out, w, h, C), process.env.MOCK_PROFILE ? { colorProfile: process.env.MOCK_PROFILE } : {}), sourceBounds: b };
    },
    getSelection: async ({ sourceBounds: b }) => ({ imageData: imageData(new Uint8Array((b.right - b.left) * (b.bottom - b.top)), b.right - b.left, b.bottom - b.top, 1), sourceBounds: b }),
    createImageDataFromBuffer: async (buf, o) => {
      if (process.env.MOCK_PROFILE && o.colorProfile === process.env.MOCK_PROFILE) throw new Error("Photoshop Error. Code: -26120. Message: Unknown color profile");
      return imageData(buf, o.width, o.height, o.components);
    },
    putPixels: async (o) => {
      calls.push("putPixels");
      if (!o.layerID) throw new Error("putPixels without a layer");
      // keep what lands on each layer (16-bit, full size) so the test can look at it
      if (!pixels.has(o.layerID)) pixels.set(o.layerID, new Uint16Array(W * H * C));
      const L = pixels.get(o.layerID), id = o.imageData, d = await id.getData(), t = o.targetBounds || { left: 0, top: 0 };
      for (let y = 0; y < id.height; y++) for (let x = 0; x < id.width * C; x++) L[((t.top + y) * W + t.left) * C + x] = d[y * id.width * C + x];
    },
    putLayerMask: async ({ layerID, imageData: id, targetBounds: t }) => {
      calls.push("putLayerMask");
      const m = masks.get(layerID);
      if (!m) throw new Error("no mask on layer " + layerID);
      const d = await id.getData();
      for (let y = 0; y < id.height; y++) for (let x = 0; x < id.width; x++) m[(t.top + y) * W + t.left + x] = d[y * id.width + x];
    },
    getLayerMask: async ({ layerID, sourceBounds: b }) => {
      const m = masks.get(layerID), w = b.right - b.left, h = b.bottom - b.top, out = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) out.set(m.subarray((b.top + y) * W + b.left, (b.top + y) * W + b.right), y * w);
      return { imageData: imageData(out, w, h, 1), sourceBounds: b };
    },
  },
};
const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "filmdust-mock-"));
// the mock plugin data folder goes when the test ends, however it ends
process.on("exit", () => { try { fs.rmSync(dataDir, { recursive: true, force: true }); } catch (e) { /* already gone */ } });
const uxp = {
  storage: {
    localFileSystem: {
      getDataFolder: async () => ({
        getEntry: async (n) => { const p = path.join(dataDir, n); if (!fs.existsSync(p)) throw new Error("missing"); return { read: async () => fs.readFileSync(p, "utf8"), delete: async () => fs.unlinkSync(p), nativePath: p }; },
        createFile: async (n) => { const p = path.join(dataDir, n); return { write: async (t) => fs.writeFileSync(p, typeof t === "string" ? t : Buffer.from(t)), nativePath: p }; },
      }),
      getPluginFolder: async () => ({ getEntry: async (n) => ({ read: async () => fs.readFileSync(path.join(ROOT, n), "utf8") }) }),
    },
    formats: { binary: "binary", utf8: "utf8" },
  },
};
// the panel never goes on the network: any request fails the test
global.fetch = async (url) => { throw new Error(`network request to ${url}`); };
const elements = {};
global.document = {
  getElementById: (id) => elements[id] || (elements[id] = { id, textContent: id === "version" ? "1.7.0" : "", hidden: true, disabled: false, className: "", value: "", placeholder: "", attrs: {}, handlers: {},
    addEventListener(ev, fn) { this.handlers[ev] = fn; }, setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return this.attrs[k]; }, style: { setProperty() {} } }),
  documentElement: { style: { setProperty() {} } },
  theme: { getCurrent: () => "darkest", onUpdated: { addListener() {} } },
};
const origLoad = Module._load;
// PANEL_ROOT=dist/film-dust-native tests a built package; MOCK_ADDON=1 resolves the hybrid addon the way
// Photoshop does (require("filmdust.uxpaddon")), backed by the Node build of the same native code
const ROOT = path.resolve(__dirname, "..", process.env.PANEL_ROOT || ".");
Module._load = function (req, parent, isMain) {
  if (req === "photoshop") return photoshop;
  if (req === "uxp") return uxp;
  if (req === "filmdust.uxpaddon") {
    if (process.env.MOCK_ADDON === "1") return origLoad.call(this, path.join(__dirname, "..", "native", "build", "filmdust.node"), parent, isMain);
    throw new Error("Cannot find module 'filmdust.uxpaddon'");
  }
  return origLoad.apply(this, arguments);
};

// ---------- the run ---------------------------------------------------------------------------
(async () => {
  // a synthetic scan gives the learned decision nothing to doubt: mark the first healed spots doubtful
  // so the "doubtful spots" layer is exercised too
  const P = require(path.join(ROOT, "engine", "pipeline"));
  const clean = P.cleanScan;
  P.cleanScan = async (...a) => {
    const r = await clean(...a);
    if (process.env.MOCK_DOUBT !== "0") {
      r.doubtful = r.marks.filter((m) => m.verdict === "defect").slice(0, 3);
      r.stats.doubtful = r.doubtful.length;
    }
    return r;
  };
  // MOCK_K=2: pretend the scan is big enough to be found at half size (the full-size healing path)
  if (process.env.MOCK_K) { const Sc = require(path.join(ROOT, "engine", "scale")); Sc.workingFactor = () => Number(process.env.MOCK_K); }
  // the mock's image is tiny: it stands in for a full-size scan unless MOCK_SMALL=1 (the small-scan guard and the
  // panel's note about it)
  if (process.env.MOCK_SMALL !== "1") require(path.join(ROOT, "engine", "maps")).DEFAULTS.res_full_mp = 0;
  require(path.join(ROOT, "panel.js"));
  console.log("engine:", require(path.join(ROOT, "engine", "native")).available() ? "native" : "plain JS", "· version line:", global.document.getElementById("version").textContent);
  const $ = (id) => global.document.getElementById(id);
  await $("remove").handlers.click();
  const dust = doc.layers.find((l) => l.name === "dust removed"), doubt = doc.layers.find((l) => l.name === "doubtful spots");
  console.log("status:", $("status").textContent);
  console.log("results:", ["r-healed", "r-hairs", "r-soft", "r-discs", "r-kept", "r-doubt", "r-subject", "r-scale"].map((k) => `${k.slice(2)} ${$(k).textContent}`).join(" · "));
  console.log("layers:", doc.layers.map((l) => `${l.name}${l.visible ? "" : " (hidden)"}`).join(", "), "· active:", doc.activeLayers.map((l) => l.name).join(", "));
  const m = masks.get(dust.id);
  let white = 0;
  for (const v of m) white += v > 127;
  // the faint disc's middle (20 px around it): the scan, and the "dust removed" layer there
  {
    const [dx0, dy0] = process.env.MOCK_K ? [1300, 1000] : [760, 565], L = pixels.get(dust.id);
    let a = 0, b = 0, n = 0;
    for (let y = dy0 - 20; y <= dy0 + 20; y++) for (let x = dx0 - 20; x <= dx0 + 20; x++) { if ((x - dx0) ** 2 + (y - dy0) ** 2 > 400) continue; const o = (y * W + x) * C + 1; a += img[o]; b += L ? L[o] : 0; n++; }
    console.log(`faint disc middle (green, mean): scan ${Math.round(a / n)}, layer ${Math.round(b / n)}, mask ${m[dy0 * W + dx0] > 127 ? "on" : "off"}`);
  }
  console.log("dust mask white px:", white, "· doubtful layer:", doubt ? `mask px ${masks.get(doubt.id).reduce((a, v) => a + (v > 127), 0)}` : "none");
  // he paints two heals back: black over the first two healed marks' boxes on the mask
  let painted = 0;
  for (let i = 0; i < m.length && painted < 2; i++) {
    if (m[i] > 127) {
      const y = Math.floor(i / W), x = i % W;
      for (let yy = Math.max(0, y - 12); yy < Math.min(H, y + 12); yy++) for (let xx = Math.max(0, x - 12); xx < Math.min(W, x + 12); xx++) m[yy * W + xx] = 0;
      painted++;
    }
  }
  await $("learn").handlers.click();
  console.log("learn status:", $("status").textContent);
  const lf = path.join(dataDir, "labels.jsonl");
  if (fs.existsSync(lf)) {
    const rows = fs.readFileSync(lf, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    console.log("labels saved:", rows.length, rows.map((r) => `${r.label} ${r.mark.kind} (${r.mark.cx},${r.mark.cy})`).join(" · "));
  } else console.log("labels saved: none");
  console.log("host calls:", [...new Set(calls)].join(", "));

  // check for updates (the one thing that goes online, and only when clicked): GitHub answers with a newer release,
  // its file for this build is downloaded and handed to the installer; then a release equal to this one is up to date
  const opened = [], fetched = [];
  uxp.shell = { openPath: async (p) => { opened.push(p); return ""; } };
  const man = JSON.parse(fs.readFileSync(path.join(ROOT, "manifest.json"), "utf8"));
  const want = man.addon ? "film-dust-windows.ccx" : "film-dust-mac.ccx";
  global.fetch = async (url) => {
    fetched.push(url);
    if (url.startsWith("https://api.github.com/")) return { ok: true, status: 200, json: async () => ({ tag_name: "v9.9.9", assets: ["film-dust-windows.ccx", "film-dust-mac.ccx"].map((n) => ({ name: n, browser_download_url: `https://github.com/magholmes/film-dust/releases/download/v9.9.9/${n}` })) }) };
    if (url.endsWith(want)) { const b = new Uint8Array(150000); b[0] = 0x50; b[1] = 0x4b; return { ok: true, status: 200, arrayBuffer: async () => b.buffer }; }
    throw new Error(`unexpected request to ${url}`);
  };
  await $("update").handlers.click();
  console.log("update check:", $("update").textContent, "·", $("status").textContent);
  await $("update").handlers.click();
  console.log("update install:", $("status").textContent, "· opened", opened.map((p) => path.basename(p)).join(", ") || "nothing");
  const okUpdate = opened.length === 1 && opened[0].endsWith("film-dust-9.9.9.ccx") && fs.existsSync(opened[0]) && fetched.length === 2 && fetched[1].endsWith(want);
  const release = man.addon ? man.version : man.version.replace(/(\d+)$/, (d) => String(Number(d) + 1));
  global.fetch = async () => ({ ok: true, status: 200, json: async () => ({ tag_name: "v" + release, assets: [{ name: want, browser_download_url: "https://github.com/x" }] }) });
  await $("update").handlers.click();
  console.log("update when current:", $("status").textContent);
  if (!okUpdate || !$("status").textContent.startsWith("up to date")) { console.log("UPDATE CHECK FAILED"); process.exitCode = 1; }
})().catch((e) => { console.error("PANEL ERROR", e); process.exit(1); });

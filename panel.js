"use strict";
// film dust — panel logic and the Photoshop glue. The engine (./engine) is plain JavaScript
// on typed arrays; it never touches Photoshop, so the same code runs under Node for testing.
// One button: find every mark on the film (white and black dust, hairs, out-of-focus dust and
// fibres), heal each with the scan's own grain at full size, on a new masked layer.
const { app, core, imaging, action } = require("photoshop");
const { cleanScan } = require("./engine/pipeline");
const S = require("./engine/scale");
const { healLarge } = require("./engine/bigheal");
const { subtractFull } = require("./engine/discs");

const $ = (id) => document.getElementById(id);
// what the detector was tuned and taught at: auto polarity, normal sensitivity, soft dust on
const SETTINGS = { polarity: "auto", sensitivity: 1.0, softPath: true };
let busy = false;

// ---------- theme: Photoshop's panel ground, our ink at alphas ------------------------------

const GROUNDS = { darkest: "#323232", dark: "#535353", medium: "#b8b8b8", light: "#d6d6d6", lightest: "#f0f0f0" };

function applyTheme(name) {
  const dark = name === "darkest" || name === "dark";
  const ink = dark ? "255, 255, 255" : "0, 0, 0";
  const a = dark ? { ink: 0.92, ink2: 0.66, mute: 0.48, mute2: 0.36, hair: 0.2, bg2: 0.06 }
                 : { ink: 0.85, ink2: 0.58, mute: 0.4, mute2: 0.32, hair: 0.14, bg2: 0.05 };
  const root = document.documentElement.style;
  root.setProperty("--bg", GROUNDS[name] || GROUNDS.darkest);
  for (const k of Object.keys(a)) root.setProperty(`--${k}`, `rgba(${ink}, ${a[k]})`);
  root.setProperty("--error", dark ? "#cb76a9" : "#b3453a");
}

try {
  applyTheme(document.theme.getCurrent());
  document.theme.onUpdated.addListener((t) => applyTheme(t));
} catch (e) {
  applyTheme("darkest");
}

// the version line says when the native filters (hybrid addon) are running; a native failure switches
// them off for the session (the engine carries on in JS) and the line follows
const versionText = $("version") ? $("version").textContent : "";
function showEngine() {
  try { if ($("version")) $("version").textContent = versionText + (require("./engine/native").available() ? " · native" : ""); } catch (e) { /* plain JS engine */ }
}
showEngine();

// whether the native filters loaded here, and if not why, in PluginData/native.txt (Photoshop keeps no plugin console
// log): the hybrid addon showed no "native" in Photoshop 2026 and nothing said why. Written when the panel opens and
// again after each run, so a switch-off mid-run shows too.
async function recordEngine() {
  try {
    const NATV = require("./engine/native"), uxp = require("uxp");
    const folder = await uxp.storage.localFileSystem.getDataFolder();
    const file = await folder.createFile("native.txt", { overwrite: true });
    const said = NATV.available() ? "native loaded" : `native not loaded: ${NATV.loadError() || "no addon (plain build)"}`;
    let host = "";
    try { host = `uxp ${uxp.versions.uxp} · photoshop ${uxp.host.version} · ${uxp.host.os || ""}`; } catch (e) { host = "uxp ?"; }
    await file.write(`${new Date().toISOString()} · ${versionText} · ${said} · ${host}\n`);
  } catch (e) { /* no data folder: nothing to record */ }
}
recordEngine();

// ---------- status line --------------------------------------------------------------------

let statusTimer = null;
function status(text, kind = "") {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (kind ? " is-" + kind : "");
  clearTimeout(statusTimer);
  if (kind === "done" || kind === "error") statusTimer = setTimeout(idleStatus, kind === "error" ? 12000 : 30000);
}

const docSize = (doc) => ({ W: Math.round(Number(doc.width)), H: Math.round(Number(doc.height)) });
const scaleWords = (k) => (k === 1 ? "full size" : k === 2 ? "half size" : `1/${k} size`);

function idleStatus() {
  const doc = app.activeDocument;
  if (!doc) return status("open a scan to begin");
  const depth = String(doc.bitsPerChannel || "").replace(/\D/g, "") || "?";
  const { W, H } = docSize(doc);
  const k = S.workingFactor(W, H);
  status(`ready · ${W} × ${H} · ${depth}-bit` + (k > 1 ? ` · finds dust at ${scaleWords(k)}` : ""));
}

const clock = (ms) => {
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
const thousands = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

$("remove").addEventListener("click", run);

// ---------- Photoshop I/O -------------------------------------------------------------------

// RGB or grayscale at 8 or 16 bits; anything else gets a plain answer instead of a wrong result
function unsupported(doc) {
  const depth = String(doc.bitsPerChannel || "").replace(/\D/g, "");
  const mode = String(doc.mode || "").toLowerCase();
  if (depth === "32") return "32-bit documents are not supported · convert to 16-bit first";
  if (mode && !/rgb|gray/.test(mode)) return "works on RGB or grayscale scans · convert the document first";
  return "";
}

// The composite, averaged down k x k (k = 1: as it is), read in strips of about 48 MB so a
// 270 MP scan never sits in memory at full size. 8-bit scans are carried as 16-bit (x 257).
async function readWorking(doc, k, onFrac, tick) {
  const depth = String(doc.bitsPerChannel).replace(/\D/g, "");
  const componentSize = depth === "8" ? 8 : 16;
  const { W, H } = docSize(doc);
  const Wd = Math.floor(W / k), Hd = Math.floor(H / k);
  const rowsPer = Math.max(k, Math.floor(48e6 / (W * 3 * (componentSize / 8)) / k) * k);
  const out = { W: Wd, H: Hd, k, fullW: W, fullH: H, componentSize, img: null, C: 0, colorSpace: null, colorProfile: null };
  let yd = 0;
  for (let y0 = 0; y0 < Hd * k; y0 += rowsPer) {
    const y1 = Math.min(Hd * k, y0 + rowsPer);
    const res = await imaging.getPixels({ documentID: doc.id, componentSize, applyAlpha: true, sourceBounds: { left: 0, top: y0, right: Wd * k, bottom: y1 } });
    const id = res.imageData;
    if (id.width !== Wd * k || id.height !== y1 - y0) {
      id.dispose();
      throw new Error(`the scan's pixels came back at ${id.width} × ${id.height} instead of ${Wd * k} × ${y1 - y0}`);
    }
    if (!out.img) {
      out.C = id.components;
      out.colorSpace = id.colorSpace;
      out.colorProfile = id.colorProfile;
      try { out.docProfile = doc.colorProfileName; } catch (e) { out.docProfile = undefined; }   // untagged: no name
      out.img = new Uint16Array(Wd * Hd * out.C);
    }
    const data = await id.getData({ chunky: true, fullRange: true });
    yd += S.averageStrip(data, id.width, id.height, out.C, k, out.img, Wd, yd, componentSize === 8);
    id.dispose();
    onFrac(y1 / (Hd * k));
    await tick();
  }
  return out;
}

// Photoshop's Select Subject on the composite, read back at the working size (k x k blocks, on at
// half grey) and then undone, so the document's own selection and history stay exactly as they
// were. Around the subject the engine is extra careful with highlights and texture. null when
// Photoshop finds no subject or can't select one: the scan is then cleaned without it.
async function readSubject(ctx, doc, src, tick) {
  const { k, W: Wd, H: Hd } = src;
  let suspension = null;
  try {
    suspension = await ctx.hostControl.suspendHistory({ documentID: doc.id, name: "Find the subject" });
    await action.batchPlay([{ _obj: "autoCutout", sampleAllLayers: true, _options: { dialogOptions: "dontDisplay" } }], {});
    const mask = new Uint8Array(Wd * Hd);
    const rowsPer = Math.max(k, Math.floor(48e6 / (Wd * k) / k) * k);
    let found = 0;
    for (let y0 = 0; y0 < Hd * k; y0 += rowsPer) {
      const y1 = Math.min(Hd * k, y0 + rowsPer);
      const res = await imaging.getSelection({ documentID: doc.id, sourceBounds: { left: 0, top: y0, right: Wd * k, bottom: y1 } });
      const id = res.imageData;
      const sb = res.sourceBounds || { left: 0, top: y0, right: Wd * k, bottom: y1 };
      const data = await id.getData({ chunky: true });
      const w = id.width, h = id.height;
      id.dispose();
      // sums per working pixel of this strip (the strip starts on a block boundary)
      const rows = (y1 - y0) / k, acc = new Uint32Array(rows * Wd);
      for (let r = 0; r < h; r++) {
        const yd = Math.floor((sb.top + r) / k) - y0 / k;
        if (yd < 0 || yd >= rows) continue;
        for (let c = 0; c < w; c++) {
          const xd = Math.floor((sb.left + c) / k);
          if (xd < Wd) acc[yd * Wd + xd] += data[r * w + c];
        }
      }
      const half = 128 * k * k;
      for (let i = 0; i < acc.length; i++) if (acc[i] >= half) { mask[(y0 / k) * Wd + i] = 1; found++; }
      await tick();
    }
    return found ? mask : null;
  } catch (e) {
    if (String(e && e.message).includes("cancelled")) throw e;
    return null;
  } finally {
    if (suspension !== null) {
      try { await ctx.hostControl.resumeHistory(suspension, false); } catch (e) { /* nothing to undo */ }
    }
  }
}

// Full-size pixels of one box, as 16-bit.
async function readBox(doc, src, x0, y0, x1, y1) {
  const res = await imaging.getPixels({ documentID: doc.id, componentSize: src.componentSize, applyAlpha: true, sourceBounds: { left: x0, top: y0, right: x1, bottom: y1 } });
  const id = res.imageData;
  const data = await id.getData({ chunky: true, fullRange: true });
  id.dispose();
  if (src.componentSize === 16) return data;
  const u16 = new Uint16Array(data.length);
  for (let i = 0; i < data.length; i++) u16[i] = data[i] * 257;
  return u16;
}

// The working-size 0/1 mask, drawn at full size in strips of about 32 MB: each k x k block takes
// its working pixel, odd edge rows and columns copy their neighbour. Empty strips are skipped, so
// only the rows with dust are sent. put(imageData, top) places one strip.
async function putMaskStrips(maskW, src, put, tick, budget = 32e6) {
  const { fullW: W, fullH: H, k, W: Wd, H: Hd } = src;
  const rowsPer = Math.max(k, Math.floor(budget / W / k) * k);
  let sent = 0;
  for (let y0 = 0; y0 < H; y0 += rowsPer) {
    const y1 = Math.min(H, y0 + rowsPer);
    const strip = new Uint8Array(W * (y1 - y0));
    let any = false;
    for (let y = y0; y < y1; y++) {
      const m = Math.min(Hd - 1, Math.floor(y / k)) * Wd, o = (y - y0) * W;
      for (let x = 0; x < W; x++) if (maskW[m + Math.min(Wd - 1, Math.floor(x / k))]) { strip[o + x] = 255; any = true; }
    }
    if (!any) continue;
    const id = await maskImageData(strip, W, y1 - y0);
    await put(id, y0);
    id.dispose();
    sent++;
    await tick();
  }
  return sent;
}

async function maskImageData(m8, W, H) {
  const opts = { width: W, height: H, components: 1, chunky: true, colorSpace: "Grayscale" };
  if (maskImageData.plain) return imaging.createImageDataFromBuffer(m8, opts);
  try { return await imaging.createImageDataFromBuffer(m8, Object.assign({ colorProfile: "Gray Gamma 2.2" }, opts)); } catch (e) {
    if (!profileError(e)) throw e;
    maskImageData.plain = true;   // a host without that profile: masks need none
    return imaging.createImageDataFromBuffer(m8, opts);
  }
}

// Photoshop reports a scan's colour profile when it reads the pixels but can refuse that name when it is handed back
// (-26120 "Unknown color profile": a scanner's or an old file's embedded profile, an untagged document). The scan's
// pixels go back with the profile they came with, else the document's own, else none named, else the standard one for
// the colour space; the first that works is kept for the rest of the run.
const profileError = (e) => /profile|-26120/i.test(String((e && e.message) || e));
const STANDARD_PROFILE = { RGB: "sRGB IEC61966-2.1", Grayscale: "Gray Gamma 2.2" };
async function putScanPixels(src, buf, width, height, put) {
  const base = { width, height, components: src.C, chunky: true, colorSpace: src.colorSpace, fullRange: true };
  const tries = src.profileChoice !== undefined ? [src.profileChoice]
    : [src.colorProfile, src.docProfile, null, STANDARD_PROFILE[src.colorSpace]].filter((p, i, a) => p !== undefined && p !== "" && a.indexOf(p) === i);
  let last = null;
  for (const p of tries) {
    let pix = null;
    try {
      pix = await imaging.createImageDataFromBuffer(buf, p === null ? base : Object.assign({ colorProfile: p }, base));
      await put(pix);
      src.profileChoice = p;
      return;
    } catch (e) {
      if (!profileError(e)) throw e;
      last = e;
    } finally {
      if (pix) pix.dispose();
    }
  }
  throw last;
}

function layerBuffer(src, data) {
  if (src.componentSize === 16) return data;
  const buf = new Uint8Array(data.length);
  for (let i = 0; i < buf.length; i++) buf[i] = Math.round(data[i] / 257);
  return buf;
}

async function newMaskedLayer(doc) {
  await action.batchPlay([{ _obj: "make", _target: [{ _ref: "layer" }], using: { _obj: "layer", name: "dust removed" } }], {});
  const layer = doc.activeLayers[0];
  // hidden until the healed spots are known: the composite stays the scan while tiles are read
  await action.batchPlay([{ _obj: "make", new: { _class: "channel" }, at: { _ref: "channel", _enum: "channel", _value: "mask" }, using: { _enum: "userMaskEnabled", _value: "hideAll" } }], {});
  return layer;
}

// k = 1: the healed working image is the whole scan; write it in one piece
async function writeLayer(doc, src, mask, tick) {
  const { W, H, C } = src;
  const layer = await newMaskedLayer(doc);
  writeLayer.last = layer;
  await putScanPixels(src, layerBuffer(src, src.img), W, H, (pix) => imaging.putPixels({ layerID: layer.id, imageData: pix, replace: true, commandName: "dust removed" }));
  await putMaskStrips(mask, src, (id, top) => imaging.putLayerMask({ layerID: layer.id, imageData: id, replace: false, targetBounds: { left: 0, top }, commandName: "healed marks" }), tick);
}

// k > 1: heal at full size, a tile at a time, straight onto the new layer. Every tile read has the faint discs'
// veils taken away first (engine/discs.js), so their windows are written as they are and the patch heals read
// clean ground around them; the patch heals use the patch-heal mask, the layer's mask shows both.
async function healOntoLayer(doc, src, result, hooks) {
  const layer = await newMaskedLayer(doc);
  writeLayer.last = layer;
  const discs = result.discBumps || [];
  const io = {
    read: async (x0, y0, x1, y1) => {
      const tile = await readBox(doc, src, x0, y0, x1, y1);
      if (discs.length) subtractFull(tile, x0, y0, x1, y1, src.C, src.k, discs);
      return tile;
    },
    write: async (x0, y0, x1, y1, data) => {
      await putScanPixels(src, layerBuffer(src, data), x1 - x0, y1 - y0,
        (pix) => imaging.putPixels({ layerID: layer.id, imageData: pix, targetBounds: { left: x0, top: y0 }, replace: false, commandName: "dust removed" }));
    },
  };
  for (const b of discs) {
    const x0 = b.x0 * src.k, y0 = b.y0 * src.k, x1 = Math.min(src.fullW, (b.x0 + b.w) * src.k), y1 = Math.min(src.fullH, (b.y0 + b.h) * src.k);
    await io.write(x0, y0, x1, y1, await io.read(x0, y0, x1, y1));
    await hooks.tick();
  }
  await healLarge(result.healMask || result.mask, result.outside, src.W, src.H, src.k, src.fullW, src.fullH, src.C, io, hooks);
  await putMaskStrips(result.mask, src, (id, top) => imaging.putLayerMask({ layerID: layer.id, imageData: id, replace: false, targetBounds: { left: 0, top }, commandName: "healed marks" }), hooks.tick);
}

// ---------- doubtful spots ----------------------------------------------------------------------

const ringRadius = (d) => Math.max(8, Math.ceil(0.75 * Math.max(d.x1 - d.x0, d.y1 - d.y0)) + 4);

// rings around the doubtful spots (working size): radius from the mark's size, a sixth as thick
function ringMask(doubt, W, H) {
  const m = new Uint8Array(W * H);
  for (const d of doubt) {
    const R = ringRadius(d), t = Math.max(2, Math.round(R / 6));
    for (let y = Math.max(0, d.cy - R); y <= Math.min(H - 1, d.cy + R); y++) {
      for (let x = Math.max(0, d.cx - R); x <= Math.min(W - 1, d.cx + R); x++) {
        const r = Math.hypot(y - d.cy, x - d.cx);
        if (r <= R && r >= R - t) m[y * W + x] = 1;
      }
    }
  }
  return m;
}

// a hidden layer "doubtful spots": magenta shown only through rings around the spots the learned
// decision was unsure of. The same calls as the "dust removed" layer: a masked layer, colour written
// only around the rings, the rings as its mask.
async function writeDoubtLayer(doc, src, doubt, tick) {
  await action.batchPlay([{ _obj: "make", _target: [{ _ref: "layer" }], using: { _obj: "layer", name: "doubtful spots" } }], {});
  const layer = doc.activeLayers[0];
  await action.batchPlay([{ _obj: "make", new: { _class: "channel" }, at: { _ref: "channel", _enum: "channel", _value: "mask" }, using: { _enum: "userMaskEnabled", _value: "hideAll" } }], {});
  const { k, fullW, fullH, C } = src;
  const colour = C === 1 ? [52000] : [65535, 9000, 52000];
  for (const d of doubt) {
    const R = ringRadius(d) + 1;
    const x0 = Math.max(0, (d.cx - R) * k), y0 = Math.max(0, (d.cy - R) * k);
    const x1 = Math.min(fullW, (d.cx + R + 1) * k), y1 = Math.min(fullH, (d.cy + R + 1) * k);
    if (x1 <= x0 || y1 <= y0) continue;
    const buf = new Uint16Array((x1 - x0) * (y1 - y0) * C);
    for (let i = 0; i < buf.length; i += C) for (let c = 0; c < C; c++) buf[i + c] = colour[Math.min(c, colour.length - 1)];
    await putScanPixels(src, layerBuffer(src, buf), x1 - x0, y1 - y0,
      (pix) => imaging.putPixels({ layerID: layer.id, imageData: pix, targetBounds: { left: x0, top: y0 }, replace: false, commandName: "doubtful spots" }));
    await tick();
  }
  await putMaskStrips(ringMask(doubt, src.W, src.H), src, (id, top) => imaging.putLayerMask({ layerID: layer.id, imageData: id, replace: false, targetBounds: { left: 0, top }, commandName: "doubtful spots" }), tick);
  layer.visible = false;
  return layer;
}

// ---------- updates ----------------------------------------------------------------------------

// Only when asked: "check for updates" is the one thing that goes online. Each release on GitHub carries two files
// under fixed names, film-dust-windows.ccx (native; its version is the release's tag) and film-dust-mac.ccx (plain JS,
// one patch behind); an update is downloaded to the plugin's data folder and handed to Creative Cloud's installer.
const RELEASES = "https://api.github.com/repos/magholmes/film-dust/releases/latest";
const parseVersion = (s) => String(s).replace(/^v/, "").split(".").map((n) => parseInt(n, 10) || 0);
const newer = (a, b) => { for (let i = 0; i < 3; i++) if ((a[i] || 0) !== (b[i] || 0)) return (a[i] || 0) > (b[i] || 0); return false; };
let pendingUpdate = null;

// this build: its version, the release it belongs to, and the file that updates it
async function thisBuild() {
  const folder = await require("uxp").storage.localFileSystem.getPluginFolder();
  const man = JSON.parse(await (await folder.getEntry("manifest.json")).read());
  const native = !!man.addon, v = parseVersion(man.version);
  return { version: man.version, release: native ? v : [v[0], v[1], (v[2] || 0) + 1], file: native ? "film-dust-windows.ccx" : "film-dust-mac.ccx" };
}

function updateButton(text, ready = false) {
  $("update").textContent = text;
  $("update").className = "link" + (ready ? " is-ready" : "");
}

async function checkForUpdates() {
  if (busy) return;
  if (pendingUpdate) return installUpdate();
  $("update").disabled = true;
  updateButton("checking…");
  try {
    const me = await thisBuild();
    const r = await fetch(RELEASES, { headers: { Accept: "application/vnd.github+json" } });
    if (!r.ok) throw new Error(`GitHub answered ${r.status}`);
    const rel = await r.json();
    const asset = (rel.assets || []).find((a) => a.name === me.file);
    if (!asset || !newer(parseVersion(rel.tag_name), me.release)) {
      updateButton("check for updates");
      status(`up to date · ${me.version} is the latest`, "done");
    } else {
      pendingUpdate = { tag: String(rel.tag_name).replace(/^v/, ""), url: asset.browser_download_url };
      updateButton(`update to ${pendingUpdate.tag}`, true);
      status(`${pendingUpdate.tag} is out · update downloads it and opens Creative Cloud’s installer`, "done");
    }
  } catch (e) {
    console.error(e);
    updateButton("check for updates");
    status("couldn’t reach GitHub · check the connection and try again", "error");
  }
  $("update").disabled = false;
}

async function installUpdate() {
  const u = pendingUpdate, uxp = require("uxp");
  $("update").disabled = true;
  updateButton(`downloading ${u.tag}…`);
  let saved = "";
  try {
    const r = await fetch(u.url);
    if (!r.ok) throw new Error(`the download answered ${r.status}`);
    const buf = await r.arrayBuffer();
    const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
    if (buf.byteLength < 100000 || head[0] !== 0x50 || head[1] !== 0x4b) throw new Error("the download isn’t a plugin file");
    const folder = await uxp.storage.localFileSystem.getDataFolder();
    const file = await folder.createFile(`film-dust-${u.tag}.ccx`, { overwrite: true });
    await file.write(buf, { format: uxp.storage.formats.binary });
    saved = file.nativePath;
    const err = await uxp.shell.openPath(saved, "Opens the film dust update in Creative Cloud’s installer.");
    if (err) throw new Error(err);
    pendingUpdate = null;
    updateButton("check for updates");
    status(`the installer is open · approve it, then restart Photoshop`, "done");
  } catch (e) {
    console.error(e);
    updateButton(`update to ${u.tag}`, true);
    status(saved ? `${u.tag} is saved at ${saved} · double-click it to install` : `couldn’t download ${u.tag} · ${(e && e.message) || e}`, "error");
  }
  $("update").disabled = false;
}

$("update").addEventListener("click", checkForUpdates);

// ---------- the run --------------------------------------------------------------------------

function setBusy(on) {
  busy = on;
  $("remove").disabled = on;
  $("remove").textContent = on ? "Removing…" : "Remove dust";
}

async function run() {
  if (busy) return;
  const doc = app.activeDocument;
  if (!doc) return status("open a scan first", "error");
  const why = unsupported(doc);
  if (why) return status(why, "error");
  const t0 = Date.now();
  setBusy(true);
  let result = null, k = 1, subject = null;
  try {
    await core.executeAsModal(async (ctx) => {
      // the engine reports progress often and yields at tick(); the panel redraws a few times a second
      let frac = 0, words = "reading the scan", shown = 0;
      const show = () => {
        status(`${words}…  ${Math.round(frac * 100)}%  ${clock(Date.now() - t0)}`);
        try { ctx.reportProgress({ value: Math.min(0.999, frac), commandName: words }); } catch (e) { /* older hosts */ }
      };
      const tick = async () => {
        if (ctx.isCancelled) throw new Error("cancelled");
        if (Date.now() - shown > 250) { show(); shown = Date.now(); }
        await new Promise((r) => setTimeout(r, 0));   // let Photoshop draw and take a cancel
      };
      show();
      const { W, H } = docSize(doc);
      k = S.workingFactor(W, H);
      const big = k > 1;
      // 0-8 %: reading · then finding (big scans: the healing inside only helps the second look) ·
      // big scans heal at full size in the last stretch
      const src = await readWorking(doc, k, (f) => { frac = 0.08 * f; words = big ? `reading the scan at ${scaleWords(k)}` : "reading the scan"; }, tick);
      words = "finding the subject"; show();
      subject = await readSubject(ctx, doc, src, tick);
      const span = big ? 0.72 : 0.9;
      result = await cleanScan(src.img, src.W, src.H, src.C, Object.assign({ heal: true, subject, scanMp: (W * H) / 1e6 }, SETTINGS), {
        tick, progress: (f, w) => { frac = 0.08 + span * f; words = big ? w.replace(/^healing/, "healing a working copy") : w; },
      });
      if (!result.stats.healed) return;   // nothing found: the document stays exactly as it was
      frac = big ? 0.8 : 0.98; words = big ? "healing at full size" : "making the layer"; show();
      const suspension = await ctx.hostControl.suspendHistory({ documentID: doc.id, name: "Remove dust" });
      try {
        if (big) {
          src.img = null;   // the working copy is no longer needed: free it before full-size tiles
          await healOntoLayer(doc, src, result, { tick, progress: (f) => { frac = 0.8 + 0.18 * f; words = "healing at full size"; } });
        } else {
          await writeLayer(doc, src, result.mask, tick);
        }
        // the doubtful spots get their own hidden layer; a failure there never undoes the healing
        if (result.doubtful && result.doubtful.length) {
          try {
            words = "marking the doubtful spots"; show();
            await writeDoubtLayer(doc, src, result.doubtful, tick);
            // back to "dust removed", so painting its mask works straight away
            await action.batchPlay([{ _obj: "select", _target: [{ _ref: "layer", _id: writeLayer.last.id }], makeVisible: false }], {});
          } catch (e) {
            console.error(e);
            result.doubtLayerFailed = true;
          }
        }
      } catch (e) {
        await ctx.hostControl.resumeHistory(suspension, false);
        throw e;
      }
      await ctx.hostControl.resumeHistory(suspension, true);
    }, { commandName: "Remove dust" });
  } catch (e) {
    setBusy(false);
    if (String(e && e.message).includes("cancelled")) return status("stopped · nothing was changed", "error");
    console.error(e);
    return status(describe(e), "error");
  }
  setBusy(false);
  showEngine();
  recordEngine();
  const s = result.stats;
  $("results").hidden = false;
  $("r-healed").textContent = thousands(s.healed);
  $("r-hairs").textContent = thousands(s.hairs);
  $("r-soft").textContent = thousands(s.soft - (s.discs || 0));
  $("r-discs").textContent = thousands(s.discs || 0);
  $("r-kept").textContent = thousands(s.left_alone);
  $("r-doubt").textContent = thousands(s.doubtful || 0);
  $("r-subject").textContent = subject ? "found" : "none found";
  $("r-scale").textContent = scaleWords(k);
  $("r-time").textContent = clock(Date.now() - t0);
  // fibres taken for having the scan’s own dust colour: crisp ones left as soft, soft lines left as tinted, and soft
  // fibres among the faint discs
  const nf = (s.fibres || 0) + (s.soft_fibres_tinted || 0) + (s.discs_fibre || 0);
  // a small scan (a lab scan, a web copy) gets the careful rules (maps.js res_guard): say so, with its size
  const note = (nf ? ` · ${nf} fibre${nf > 1 ? "s" : ""} of the scan’s own dust colour` : "")
    + (s.small_scan ? ` · small scan (${s.small_scan} MP): only clear dust is taken` : "");
  if (!s.healed) status("no dust found · nothing was changed" + note, "done");
  else if (s.doubtful && !result.doubtLayerFailed) status(`${thousands(s.healed)} marks healed · ${s.doubtful} doubtful ringed on the hidden layer “doubtful spots” · paint black on the “dust removed” mask to bring one back` + note, "done");
  else status(`${thousands(s.healed)} marks healed · new layer “dust removed” · paint black on its mask to bring one back` + note, "done");
}

function describe(e) {
  const m = String((e && e.message) || e);
  if (/memory/i.test(m)) return "not enough memory for a scan this size · close other documents and try again";
  if (/modal/i.test(m)) return "Photoshop is busy with another task · try again when it finishes";
  return `could not finish · ${m}`;
}

try {
  action.addNotificationListener(["select", "open", "close", "make"], () => { if (!busy) idleStatus(); });
} catch (e) { /* older hosts: the idle line updates on the next run */ }
idleStatus();

// for the Node checks in test/
module.exports = { putMaskStrips };

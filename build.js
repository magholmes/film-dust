"use strict";
// Package the plugin: copy the runtime files into dist/film-dust and zip them as a .ccx
// (a zip with manifest.json at its root; double-clicking it installs through Creative Cloud).
//   node build.js
// The zip is written here rather than with Compress-Archive, which stores Windows backslashes
// in entry names; installers on macOS read those as part of the file name.
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

const root = __dirname;
const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"));
const files = ["index.html", "styles.css", "panel.js", "NOTICES.txt", "LICENSE.txt"];
const dirs = { engine: /\.js$/, icons: /\.png$/ };
const ADDON = path.join(root, "native", "build", "filmdust.uxpaddon");
const nativeVersion = manifest.version.replace(/(\d+)$/, (d) => String(Number(d) + 1));

// two packages from the same files:
//   film-dust-<v>.ccx         plain JS engine (installs without admin rights)
//   film-dust-<v>-native.ccx  + the native filters as a hybrid addon (win/x64/filmdust.uxpaddon,
//                             manifest 6 with enableAddon); installing it asks for admin rights
function stage(variant) {
  const out = path.join(root, "dist", variant === "native" ? "film-dust-native" : "film-dust");
  fs.rmSync(out, { recursive: true, force: true });
  fs.mkdirSync(out, { recursive: true });
  const entries = [];
  const add = (rel) => entries.push({ name: rel.split(path.sep).join("/"), file: path.join(out, rel) });
  const man = JSON.parse(JSON.stringify(manifest));
  if (variant === "native") {
    // one patch ahead, so the native package installs over the plain one of the same release
    man.version = nativeVersion;
    man.manifestVersion = 6;
    man.requiredPermissions = Object.assign({}, man.requiredPermissions, { enableAddon: true });
    man.addon = { name: "filmdust.uxpaddon" };
    fs.mkdirSync(path.join(out, "win", "x64"), { recursive: true });
    fs.copyFileSync(ADDON, path.join(out, "win", "x64", "filmdust.uxpaddon"));
    add(path.join("win", "x64", "filmdust.uxpaddon"));
  }
  fs.writeFileSync(path.join(out, "manifest.json"), JSON.stringify(man, null, 2) + "\n");
  add("manifest.json");
  for (const f of files) if (fs.existsSync(path.join(root, f))) { fs.copyFileSync(path.join(root, f), path.join(out, f)); add(f); }
  for (const [d, re] of Object.entries(dirs)) {
    fs.mkdirSync(path.join(out, d), { recursive: true });
    for (const f of fs.readdirSync(path.join(root, d)).sort()) if (re.test(f)) { fs.copyFileSync(path.join(root, d, f), path.join(out, d, f)); add(path.join(d, f)); }
  }
  // the panel shows the manifest's version (panel.js adds "native" when the addon runs)
  const html = path.join(out, "index.html");
  fs.writeFileSync(html, fs.readFileSync(html, "utf8").replace(/(<span class="meta" id="version">)[^<]*(<\/span>)/, `$1${man.version}$2`));
  return entries;
}

// ---- minimal zip writer (deflate, no extras)
const crcTable = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
const crc32 = (buf) => { let c = -1; for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8); return (c ^ -1) >>> 0; };
const now = new Date();
const dosTime = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
const dosDate = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
function zip(entries, ccx) {
const parts = [], central = [];
let offset = 0;
for (const e of entries) {
  const data = fs.readFileSync(e.file);
  const comp = zlib.deflateRawSync(data, { level: 9 });
  const name = Buffer.from(e.name, "utf8");
  const crc = crc32(data);
  const local = Buffer.alloc(30);
  local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0x0800, 6); local.writeUInt16LE(8, 8);
  local.writeUInt16LE(dosTime, 10); local.writeUInt16LE(dosDate, 12); local.writeUInt32LE(crc, 14);
  local.writeUInt32LE(comp.length, 18); local.writeUInt32LE(data.length, 22); local.writeUInt16LE(name.length, 26); local.writeUInt16LE(0, 28);
  parts.push(local, name, comp);
  const cen = Buffer.alloc(46);
  cen.writeUInt32LE(0x02014b50, 0); cen.writeUInt16LE(20, 4); cen.writeUInt16LE(20, 6); cen.writeUInt16LE(0x0800, 8); cen.writeUInt16LE(8, 10);
  cen.writeUInt16LE(dosTime, 12); cen.writeUInt16LE(dosDate, 14); cen.writeUInt32LE(crc, 16); cen.writeUInt32LE(comp.length, 20);
  cen.writeUInt32LE(data.length, 24); cen.writeUInt16LE(name.length, 28); cen.writeUInt32LE(offset, 42);
  central.push(cen, name);
  offset += local.length + name.length + comp.length;
}
const cenSize = central.reduce((a, b) => a + b.length, 0);
const end = Buffer.alloc(22);
end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
end.writeUInt32LE(cenSize, 12); end.writeUInt32LE(offset, 16);
fs.writeFileSync(ccx, Buffer.concat([...parts, ...central, end]));
console.log(`built ${path.relative(root, ccx)} (${(fs.statSync(ccx).size / 1024).toFixed(0)} KB, ${entries.length} files)`);
}

zip(stage("plain"), path.join(root, "dist", `film-dust-${manifest.version}.ccx`));
if (fs.existsSync(ADDON)) zip(stage("native"), path.join(root, "dist", `film-dust-${nativeVersion}-native.ccx`));
else console.log("no native/build/filmdust.uxpaddon: run native/build.sh for the native package");

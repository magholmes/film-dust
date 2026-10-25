"use strict";
// Find and heal: the same two passes as clean_scan.py.
//   pass 1: detect (auto polarity unless told) -> heal
//   pass 2: detect again on the healed image, same polarity; sharp marks must be 15% stronger and
//           anything pass 1 left as skin texture stays (healing dust lowers busy-ness and thins
//           fields, which would otherwise unshield photo marks) -> heal
// img is a Uint16Array (H*W*C, interleaved, 0..65535) and is healed in place. Pixels outside the
// returned mask are never changed.
//
// hooks: progress(fraction, words) · tick() -> Promise (yield to the host; throw to cancel) · log(text)
const DT = require("./detect");
const { detect } = DT;
const { heal } = require("./heal");
const { DEFAULTS } = require("./maps");

// share of the run each stage takes (measured on a 65 MP scan): the plain JS engine, and with the
// native filters, where the filter-heavy stages shrink most
const NAT = require("./native");
const WEIGHTS_JS = { grain: 23, white: 71, black: 33, soft: 40, discs: 4, heal: 3 };
const WEIGHTS_NATIVE = { grain: 5, white: 17, black: 9, soft: 21, discs: 2, heal: 3 };
const WORDS = {
  grain: "measuring the grain", white: "looking for white dust", black: "looking for black dust",
  soft: "looking for out-of-focus dust", discs: "looking for faint soft dust", heal: "healing",
};

function makePlan(polarity, passes, healing) {
  const plan = [];
  for (let n = 1; n <= passes; n++) {
    const pols = n === 1 ? (polarity === "auto" || polarity === "both" ? ["white", "black"] : [polarity === "dark" ? "black" : "white"]) : ["white"];
    plan.push([n, "grain"]);
    for (const p of pols) plan.push([n, p]);
    if (n === 1) plan.push([n, "soft"]);   // the second look is sharp marks only
    if (n === 1 && healing) plan.push([n, "discs"]);
    if (healing) plan.push([n, "heal"]);
    if (!healing) break;
  }
  return plan;
}

// The spots worth a second look: heals the learned decision wasn't sure of (possible photo damage),
// most photo-like first, then out-of-focus fibres and big learned overrules (a quarter at most), then clearly
// visible marks (z 10+) it leaned toward dust but that were left (possible misses). At most 40.
function doubtful(marks, max = 40) {
  const open = (m) => !m.zone;
  const L = marks.filter((m) => m.learned != null && open(m) && (m.verdict === "defect" || m.verdict === "keep-photo"));
  const heals = L.filter((m) => m.verdict === "defect" && m.learned < 0.7).sort((a, b) => a.learned - b.learned);
  // heals with the most at stake when wrong: out-of-focus fibres (rules only: twigs, paths, print slip
  // through) and big marks the learned decision turned into dust; largest first
  const risky = marks.filter((m) => m.verdict === "defect" && open(m) && (String(m.why).startsWith("soft fibre")
    || ((m.area || 0) >= 1000 && String(m.why).includes("decision learned")))).sort((a, b) => (b.area || 0) - (a.area || 0));
  const misses = L.filter((m) => m.verdict === "keep-photo" && m.learned > 0.5 && (m.z || 0) >= 10).sort((a, b) => b.learned - a.learned);
  const nRisky = Math.min(risky.length, Math.floor(max / 4)), nMiss = Math.min(misses.length, Math.floor(max / 4));
  const out = heals.slice(0, max - nRisky - nMiss);
  const out2 = out.concat(risky.slice(0, max - out.length - nMiss));
  return out2.concat(misses.slice(0, max - out2.length));
}

async function cleanScan(img, W, H, C, opts = {}, hooks = {}) {
  const t0 = Date.now();
  const log = hooks.log || (() => {});
  const tick = hooks.tick || (async () => {});
  const progress = hooks.progress || (() => {});
  const healing = opts.heal !== false;
  const passes = healing ? (opts.passes ?? 2) : 1;
  const base = {
    polarity: opts.polarity || "auto", sensitivity: opts.sensitivity || 1.0,
    soft_dust: !!opts.softDust, soft_path: opts.softPath !== false,
    // out-of-focus fibres follow the same switch as out-of-focus dust unless asked for separately
    soft_lines: opts.softLines !== undefined ? !!opts.softLines : opts.softPath !== false,
    // Photoshop's Select Subject at the working size (Uint8Array, 1 = subject), or null
    subject: opts.subject || null,
  };
  // the scan's own size when this is a working copy of a bigger one (engine/scale.js): the small-scan guard goes by it
  if (opts.scanMp) base.res_scan_mp = opts.scanMp;
  // engine settings for experiments (test/run_scan.js --params '{"learned_lo": 0.5}'); the panel passes none
  if (opts.params) Object.assign(base, opts.params);
  const plan = makePlan(base.polarity, passes, healing);
  const WEIGHTS = NAT.available() ? WEIGHTS_NATIVE : WEIGHTS_JS;
  const weight = (st) => WEIGHTS[st[1]] * (st[0] > 1 ? 0.9 : 1);
  const total = plan.reduce((a, st) => a + weight(st), 0);
  let pass = 1;
  const report = (name, f, say) => {
    let done = 0, idx = plan.findIndex((st) => st[0] === pass && st[1] === name);
    if (idx < 0) return;
    for (let i = 0; i < idx; i++) done += weight(plan[i]);
    const frac = (done + weight(plan[idx]) * Math.min(1, Math.max(0, f))) / total;
    const words = say || (passes > 1 && healing ? `${WORDS[name]} · pass ${pass} of ${passes}` : WORDS[name]);
    progress(0.98 * frac, words);
  };
  const dhooks = { log, tick, timing: !!hooks.timing, stage: (name, f) => report(name === "white" || name === "black" ? name : name, f) };

  const r1 = await detect(img, W, H, C, base, dhooks);
  const marks = r1.marks;
  for (const m of marks) m.pass = 1;
  // faint soft discs on smooth ground (engine/discs.js): found and judged now, taken away by subtracting their
  // veil before the patch heals; bright dust only (inverted negatives)
  const P = Object.assign({}, DEFAULTS, base);
  const discs = [], bumps = new Map();
  let rescued = 0, near = 0, fibreDiscs = 0, fibres = 0, tintedTaken = 0;
  const DS = require("./discs");
  if (healing && P.soft_discs && r1.polarities.includes("bright")) {
    report("discs", 0);
    const found = DS.find(img, W, H, C, { outside: r1.outside, marks, core: r1.core, subject: base.subject }, P, true);
    const whys = new Map();
    for (const c of found.cands) {
      const why = DS.rules(c, P);
      if (why) { whys.set(c, why); continue; }
      // the faint-disc network (training/discnet) vetoes a disc it calls picture: on the benchmark this kept 110 of
      // the 113 dust the rules take and left 8 of their 10 photo (cloud wisps, mist)
      c.net = DS.score(img, W, H, C, c);
      if (c.net !== null && c.net < DS.netNeeded(c, P)) continue;
      const b = DS.bumpFit(img, W, H, C, c, r1.core, r1.outside);
      if (!b.ok) continue;
      // small scans: no disc bigger than dust ever is, none on the frame line (detect.js resGuard)
      if (DT.resGuard(P, W, H, c.cx - c.r, c.cy - c.r, c.cx + c.r, c.cy + c.r, Math.PI * c.r * c.r, "disc")) continue;
      const m = DS.mark(c, "defect", "a faint soft disc on smooth ground (out-of-focus dust), its veil taken away");
      discs.push(m); marks.push(m); bumps.set(m, b);
    }
    // a scan with its own dust (5+ discs taken with certainty) gives their colour; a disc turned down only for its
    // shape (a soft fibre, discs run together) and of that colour goes too, its veil taken in its own shape (a sharp
    // core it would leave, a hair in focus, stays the detector's)
    const pop = DS.population(discs.map((m) => m.disc), found.cands, whys, P);
    for (const c of whys.keys()) {
      const viaShape = DS.rescue(c, pop, P), viaFibre = DS.fibreRescue(c, pop, P);
      if (!viaShape && !viaFibre) continue;
      c.net = DS.score(img, W, H, C, c);
      const netOk = c.net === null || c.net >= DS.netNeeded(c, P);
      // a soft fibre of the dust's colour goes past the disc network's veto (disc_rescue_fibre)
      if (!(viaShape && netOk) && !viaFibre) continue;
      const isFibre = !(viaShape && netOk);
      // a small disc a little short of the plainest ground, or one beside an edge (disc_rescue_near)
      const isNear = !isFibre && !DS.rescue(c, pop, Object.assign({}, P, { disc_rescue_near: false }));
      const b = DS.bumpShape(img, W, H, C, c, r1.core, r1.outside);
      // the same dust as a veil already taken (two candidates on discs run together) is not taken twice
      if (!b.ok || DS.overlap(b, [...bumps.values()]) > 0.1) continue;
      if (DT.resGuard(P, W, H, b.box[0], b.box[1], b.box[2], b.box[3], Math.PI * c.r * c.r, "disc")) continue;
      const m = Object.assign(DS.mark(c, "defect", isFibre ? "a soft fibre of the scan's own dust colour, its veil taken away in its own shape"
        : isNear ? "a small soft disc, or one beside an edge, of the scan's own dust colour, its veil taken away in its own shape"
        : "soft dust run together or a soft fibre, of the scan's own dust colour, its veil taken away in its own shape"), { x0: b.box[0], y0: b.box[1], x1: b.box[2], y1: b.box[3] });
      c.rescued = true;
      if (isNear) { m.nearRescue = true; near++; }
      if (isFibre) { m.fibreDisc = true; fibreDiscs++; }
      discs.push(m); marks.push(m); bumps.set(m, b); rescued++;
    }
    // crisp fibres the detector left as photo only for being soft or tinted, of the scan's own dust colour, on smooth
    // ground: healed like a speck through the detector's core and mask (fibre_rescue)
    if (pop && pop.theta && P.fibre_rescue) {
      for (const m of r1.marks) {
        if (!DS.fibreMark(m)) continue;
        // on smooth ground as a disc must be: at its middle and its box's corners (sky between branches, glints on
        // water and foliage are not)
        const smoothMax = P.disc_smooth ?? 1.5;
        if ([[m.cx, m.cy], [m.x0, m.y0], [m.x1, m.y0], [m.x0, m.y1], [m.x1, m.y1]].some(([x, y]) => !(found.noiseRatio(x, y) <= smoothMax))) continue;
        const R = DS.fibreRadius(m);
        const b = DS.bumpShape(img, W, H, C, { cx: m.cx, cy: m.cy, r: R }, r1.core, r1.outside);
        const how = DS.fibreTake(b, pop, P);
        if (!how || DT.resGuard(P, W, H, m.x0, m.y0, m.x1, m.y1, 0, "hair")) continue;
        m.fibre = how;
        // healed like a speck, its halo with it (a sixth of its fine peak, one dilation wider than the detector's)
        m.verdict = "defect"; m.why = "a fibre of the scan's own dust colour, healed like a speck";
        const core = new Uint8Array(b.w * b.h);
        for (let i = 0; i < core.length; i++) if (b.core[i] && !(r1.outside && r1.outside[(b.y0 + Math.floor(i / b.w)) * W + b.x0 + (i % b.w)])) core[i] = 1;
        const d = (base.dilate ?? DEFAULTS.dilate) + 1, grown = require("./filters").dilateMask(core, b.w, b.h, 2 * d + 1, true);
        for (let i = 0; i < core.length; i++) {
          const g = (b.y0 + Math.floor(i / b.w)) * W + b.x0 + (i % b.w);
          if (core[i]) r1.core[g] = 1;
          if (grown[i]) r1.mask[g] = 1;
        }
        fibres++;
      }
    }
    // soft fibres the detector left only for their tint, when the tint is the scan's own dust colour (line_colour):
    // healed like its other soft fibres, through the core and the mask
    if (pop && pop.theta && P.line_colour && r1.tintedLines && r1.tintedLines.length) {
      const dilate = base.dilate ?? DEFAULTS.dilate;
      for (const t of r1.tintedLines) {
        if (t.mark.detail > (P.line_colour_detail ?? 0.2)) continue;
        if (DT.resGuard(P, W, H, t.mark.x0, t.mark.y0, t.mark.x1, t.mark.y1, 0, "line")) continue;
        const v = t.lift.map((l, i) => l / Math.max(t.ground[i], 1e-3)), n = Math.hypot(v[0], v[1], v[2]);
        if (!(n > 0) || v.some((x) => x <= 0)) continue;
        const u = v.map((x) => x / n);
        const a = (Math.acos(Math.min(1, u[0] * pop.dir[0] + u[1] * pop.dir[1] + u[2] * pop.dir[2])) * 180) / Math.PI;
        if (a > (P.line_colour_angle ?? 1) * pop.theta) continue;
        const grown = dilate > 0 ? require("./filters").dilateMask(t.band, t.w, t.h, 2 * dilate + 1, true) : t.band;
        for (let i = 0; i < t.band.length; i++) {
          const g = (t.y0 + Math.floor(i / t.w)) * W + t.x0 + (i % t.w);
          if (r1.outside && r1.outside[g]) continue;
          if (t.band[i]) r1.core[g] = 1;
          if (grown[i]) r1.mask[g] = 1;
        }
        marks.push(Object.assign(t.mark, { verdict: "defect", why: "a soft fibre of the scan's own dust colour", colour: Math.round(a * 10) / 10, polarity: "bright" }));
        tintedTaken++;
      }
    }
    report("discs", 1);
    log(`[discs] ${found.cands.length} faint discs found, ${discs.length} to take away${rescued ? ` (${rescued} of the scan's own dust colour${near ? `, ${near} of them small or beside an edge` : ""}${fibreDiscs ? `, ${fibreDiscs} soft fibres` : ""})` : ""}${pop ? `; dust colour from ${pop.sure} discs, ${pop.theta ? `within ${pop.theta} deg` : "a grey scan: colour off"}` : ""}${fibres ? `; ${fibres} soft fibres of that colour` : ""}${tintedTaken ? `; ${tintedTaken} tinted soft lines of that colour` : ""}`);
  }
  // the discs still called dust lose their veil; where that changed the image is kept apart from the patch heals'
  // mask (healLarge must not patch-heal a disc on a large scan), and joins it in the mask the panel shows
  const discBumps = discs.filter((m) => m.verdict === "defect").map((m) => bumps.get(m));
  const discMask = discBumps.length ? new Uint8Array(W * H) : null;
  if (discBumps.length) require("./discs").subtract(img, W, H, C, discBumps, discMask);
  const mask = Uint8Array.from(r1.mask);
  if (healing) {
    report("heal", 0);
    await heal(img, W, H, C, r1.mask, r1.outside, (f) => report("heal", f), tick);
    const pol = r1.polarities.length === 1 ? r1.polarities[0] : "both";
    for (let n = 2; n <= passes; n++) {
      pass = n;
      // skin texture left in the first pass stays as it is
      const skin = marks.filter((m) => m.verdict === "keep-photo" && (m.why || "").startsWith("on skin")).map((m) => [m.cx, m.cy]);
      // the second look: sharp marks only (soft dust is judged in the first; here it added 0-3
      // marks per scan for a quarter of the run)
      const r2 = await detect(img, W, H, C, Object.assign({}, base, { polarity: pol, second_pass: true, protect: skin, soft_path: false }), dhooks);
      const fresh = r2.marks.filter((m) => m.verdict === "defect");
      log(`pass ${n}: ${fresh.length} more marks`);
      if (!fresh.length) break;
      report("heal", 0);
      await heal(img, W, H, C, r2.mask, r2.outside, (f) => report("heal", f), tick);
      for (const m of fresh) { m.pass = n; marks.push(m); }
      for (let i = 0; i < mask.length; i++) if (r2.mask[i]) mask[i] = 1;
    }
  }
  const healMask = discMask ? Uint8Array.from(mask) : mask;
  if (discMask) for (let i = 0; i < mask.length; i++) if (discMask[i]) mask[i] = 1;
  const byKind = {};
  let healed = 0, left = 0, px = 0;
  for (const m of marks) {
    if (m.verdict === "defect") { healed++; byKind[m.kind] = (byKind[m.kind] || 0) + 1; } else if (m.verdict === "keep-photo") left++;
  }
  for (let i = 0; i < mask.length; i++) px += mask[i];
  const doubt = doubtful(marks);
  const stats = {
    healed, left_alone: left, by_kind: byKind, healed_pixels: px, doubtful: doubt.length,
    hairs: byKind["hair/fibre"] || 0, soft: byKind.soft || 0,
    polarity: r1.polarities, seconds: Math.round((Date.now() - t0) / 100) / 10,
  };
  if (discs.length) stats.discs = discs.filter((m) => m.verdict === "defect" && m.disc && !m.disc.fibre).length;
  if (rescued) stats.discs_shaped = rescued - near - fibreDiscs;
  if (near) stats.discs_near = near;
  if (fibreDiscs) stats.discs_fibre = fibreDiscs;
  if (fibres) stats.fibres = fibres;
  if (tintedTaken) stats.soft_fibres_tinted = tintedTaken;
  // a scan under res_full_mp: the size guard is on and only clear dust is taken (the panel says so)
  const scanMp = Math.max(W * H / 1e6, P.res_scan_mp || 0);
  if (P.res_guard && scanMp < P.res_full_mp) stats.small_scan = Math.round(scanMp * 10) / 10;
  // mask: everything that changed (the layer's mask); healMask: the patch heals only; discBumps: the veils taken
  // away, for healing a large scan at full size (engine/discs.js subtractFull)
  return { mask, healMask, discBumps, marks, doubtful: doubt, outside: r1.outside, polarities: r1.polarities, stats };
}

module.exports = { cleanScan, makePlan, doubtful };

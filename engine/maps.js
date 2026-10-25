"use strict";
// Strength maps and per-mark measurements (port of the first half of detect.py).
// Images are Float32Array, row-major, values 0..1. Masks are Uint8Array (0/1).
const F = require("./filters");
const NAT = require("./native");
const R = require("./regions");

const DEFAULTS = {
  polarity: "auto", seed: 7.0, tiny_area: 6, tiny_seed: 9.0, grow: 3.0, grow_px: 6, link_px: 3, min_area: 3,
  max_area: 60000, field_radius: 150, field_level: 0.5, field_count: 12, stand_area: 4.0, stand_len: 3.0,
  stand_z: 2.2, stand_white: 0.35, stand_min_area: 300, stand_hair_width: 3.5, busy_start: 1.3, busy_gain: 1.2,
  busy_cap: 2.2, busy_field: 1.45, skin_guard: true, skin_count: 5, max_edge_width: 6.5, edge_fine: true, soft_dust: false,
  blob_flat: 0.55, blob_edge: 4.5, blob_contrast: 0.12, fibre_extend: 30, trace_hairs: true, soft_path: true, soft_seed: 9.0,
  soft_sim_max: 40, soft_sim_big: 10, soft_stronger_max: 5, soft_area_max: 1000, soft_fine_max: 3.0, soft_socket: 0.7, soft_sure_photo: true, soft_bar_fine: 1.6, soft_bar_elong: 1.8, soft_bar_z: 20,
  soft_elong_max: 4.5, soft_plateau: 0.5, soft_cos_white: 0.95, soft_row: true, soft_tex_sim: 5, soft_tex_near: 40, partners: true, soft_fill: 0.12, trace_max: 2500,
  trace_min_z: 2.0, trace_turn: 35.0, chain_px: 14, absorb_px: 26, texture_guard: true, line_guard: true,
  chain_gaps: 0.25, chain_per: 0.36, edge_one: 0.9, edge_frac: 0.65, zone_guard: true, zone_margin: 25,
  zone_white: 0.5, zone_crowd_z: 3.0, straight_max: 0.012, straight_len: 150, soft_crisp: 0.72, strong_crisp: 0.40,
  paper_white: 0.75, detail_max: 1.2, tex_min: 0.45, detail_crisp: 0.9, detail_gain: 1.2, broad_min: 0.3,
  // taught by Magnus's answers (2026-10-16): strong specks beside dark edges, strong crisp specks on skin
  // and the strength a lone hair needs
  broad_speck_z: 20.0, hair_seed: 9.0, skin_speck_z: 12.0, skin_speck_crisp: 1.1, skin_speck_sharp: 0.85,
  detail_speck_z: 15.0, detail_speck_crisp: 1.5,
  // the dust network (trained on fake dust laid onto his photos): an opinion both learned decisions read
  dustnet: true,
  // 10-18 (natural highlights first): a heal the learned decision calls photo up to 0.45 is left (was
  // 0.3), a left mark needs 0.75 to be flipped to dust (was 0.7): on his answers, scans held out, photo
  // heals 23 -> 20 and 21 -> 16 (soft) for 9 of 302 dust; on boy it leaves chain links, hair strands,
  // fabric ribbing and leaf highlights it used to heal
  learned: true, learned_lo: 0.45, learned_lo_compact: 0.65, learned_big_area: 400, learned_thin_max: 20, learned_flip_white: 0, learned_flip_white_area: 100, learned_hi: 0.75, learned_reach: true,
  // the second network (patch + 160 px context, fine-tuned on his answers) blended into the sharp decision
  dustnet2: true, learned_net2_weight: 0.5, learned_net2_hairs: true, learned_net2_hair_len: 60,
  // the soft step's learned decision (his soft answers): heal -> photo at P <= lo; left for a well-covered
  // reason -> dust at P >= hi and z >= floor. 10-18: the heal side now uses trees also fed Claude's labels of soft blobs
  // on unseen scans (learned_soft_heal_model.js), whose P runs lower: lo 0.55 -> 0.45 (picked on one half of his answers,
  // tested on the other: the same dust with 4-6 fewer photo heals each time)
  soft_learned_lo: 0.45, soft_learned_hi: 0.7, soft_learned_floor: 10.0, soft_learned_floor_area: 50, soft_learned_floor2: 15.0, soft_learned_big_area: 2500, soft_learned_big_z: 40, soft_learned_big_fine: 1.5, soft_learned_big_near: 10,
  // ... but never a heal -> photo over a sharp mark this near paper white (paper black for dark dust)
  soft_learned_white: 0.9,
  // faint soft discs on smooth ground (discs.js): search z, smooth = finest noise within this x the quietest,
  // and the rules for taking one away (edge across its ring, roundness, z, contrast)
  soft_discs: true, disc_z: 5, disc_smooth: 1.5, disc_edge: 1.0, disc_elong: 1.8, disc_elong_wide: 2.0, disc_z_heal: 6, disc_contrast: 0.05, disc_net_min: -0.5,
  // 1.8.46 (benchmark 110 -> 180 of 249 dust for the same 2 of 103 photo): small discs on the plainest ground when the
  // network is sure; an edge = what the ring holds beyond a gradient (and no step over 1.5 x the lift); z 5 when the
  // network is sure; skin only when saturated like skin (a warm grey sky is not)
  disc_small: true, disc_small_noise: 1.25, disc_small_z: 7, disc_small_net: 1, disc_edge_flat: 0.6, disc_edge_cap: 1.5, disc_z_net: 0, disc_skin_sat: 0.22,
  // faint dust on the subject's clothes (never its skin) when the network is surer (2+): portraits' shirts
  disc_subject_net: 2,
  // 1.8.48: a scan with its own dust (5+ discs taken at z 8+ and network 1+) gives the dust's colour (their median
  // colour direction; the angle a tenth of the scan's picture candidates fall within, at most 5 degrees, off below 2:
  // grey scans); a disc turned down only for its shape (elong up to 2.5, band up to 3) within twice that angle is
  // taken, its veil in its own shape (benchmark 180 -> 198 of 249 dust, the same 2 photo)
  disc_rescue: true, disc_population: 5, disc_sure_z: 8, disc_sure_net: 1, disc_colour: 5, disc_rescue_colour: 2, disc_rescue_elong: 2.5, disc_rescue_wide: 3.0,
  // ... and a crisp fibre the detector left only as soft or tinted, of that colour, on smooth ground and bright all
  // through: healed like a speck (7285's wavy fibre on the shirt; 4 fibres on 30 scans, all dust)
  fibre_rescue: true,
  // small scans (under res_full_mp by the scan's own size, res_scan_mp when the panel finds dust on a copy of a bigger one;
  // lab scans, web copies): a heal bigger than dust ever is at full
  // size (res_cap_area px at res_ref_mp, faint discs res_cap_disc, scaled by the scan's size) or within res_edge_px of the
  // scan's edge is the picture or the frame line. 10-24, his photo (2.6 / 6.3 MP) lost a logo, a cap's highlight, an ear
  // and a jacket's sheen; on lab-style copies of his scans (24-2.6 MP JPEGs) the heals the full-size run never made were
  // a boat's hull, clouds taken as discs, railings, rings and frame lines. Full-size scans: off (unchanged).
  // (hairs res_cap_hair, soft fibres res_cap_line: just above the biggest of each healed at full size on 31 scans,
  // 3,820 and 6,288 px)
  res_guard: true, res_full_mp: 40, res_ref_mp: 65, res_cap_area: 6000, res_cap_disc: 8000, res_cap_hair: 4000, res_cap_line: 8000, res_edge_px: 10,
  // ... twice that for a speck on plain ground (detail and busyness at most these): none of the big picture heals on the
  // lab copies or his photo was on ground this plain; a soft heal on very plain ground (the ring's detail at most
  // res_soft_plain_detail) gets res_soft_plain_mult times: q_04's Y-shaped flake on a plain sky (2.45 x at 4.3 MP; no soft
  // heal on the 40 lab copies went past 1.14 x)
  res_plain_detail: 0.1, res_plain_busy: 1.25, res_plain_mult: 2, res_soft_plain_detail: 0.05, res_soft_plain_mult: 3,
  // 1.8.50: small discs a little short of the plainest ground (noise up to 2, z 6+) and discs with an edge beside them
  // rather than across (what the ring holds beyond a gradient up to 1.5, no step over 3 x the lift), of the dust's
  // colour within 1.5 x the angle (benchmark +3 dust +7 unsure +16 unlabelled, no photo; 30 survey scans: 46 more, every
  // one by eye soft dust; digital photos have no dust colour)
  disc_rescue_near: true, disc_near_noise: 2, disc_near_z: 6, disc_near_edge: 1.5, disc_near_cap: 3, disc_near_colour: 1.5,
  // ... and soft fibres among the candidates (turned down only for their shape) of the dust's colour within the angle,
  // past the disc network's veto (benchmark +7 dust, 1 faint blob Claude called photo, 5 unlabelled fibres)
  disc_rescue_fibre: true, disc_fibre_elong: 4, disc_fibre_wide: 5, disc_fibre_colour: 1,
  // a crowd of same-size soft blobs is the picture unless it is faint specks scattered over plain ground
  soft_scatter_near: 30, soft_scatter_area: 80,
  // soft fibres (pale soft lines alone on plain ground)
  soft_lines: true, line_z: 6.0,
  line_len: 25, line_zmed: 9.5, line_scale: 3.0, line_white: 0.5, line_busy: 4.0, line_busy_joint: 3.0, line_detail_joint: 0.2, line_photo_p: 0.3, line_edge_max: 50, line_two: 0.5,
  line_near_px: 60, line_alone_px: 200, line_alone_max: 2, line_max_len: 500, line_taken_max: 0.9,
  // a soft line is photo when its ridge runs on line_reach_px past an end (twig, stem, strand) or it is
  // ruler-straight (line_straight_len+ px, bowing <= line_straight_max of its length)
  line_reach_px: 40.0, line_cont_z: 3.0, line_cont_max: 120.0, line_straight_len: 60, line_straight_max: 0.03,
  // photo detail beside a soft line (gradient energy at sigma 2.5 x plain grain): more = bark, foliage, fabric
  line_detail_max: 0.6,
  // a nearly straight soft line (bowing under line_bent of its length) on textured ground (detail over
  // line_bent_detail or busy over line_bent_busy) is the picture
  line_bent: 0.065, line_bent_detail: 0.15, line_bent_busy: 2.5,
  // a soft line left only for its tint is healed when the tint is the scan's own dust colour (within
  // line_colour_angle x the angle the discs calibrate) and it lies on plain ground (detail up to line_colour_detail:
  // a bright spot on a cloud's edge shares the colour)
  line_colour: true, line_colour_angle: 1, line_colour_detail: 0.2,
  // a flake on smooth ground: crisp against a perfectly sharp mark of its own shape, compact, neutral,
  // alone, no detail around, not on skin, and the learned decision leaning toward dust
  flake_rel: 0.9, flake_area: 60, flake_width: 6.0, flake_elong: 2.0, flake_white: 0.7, flake_field: 2, flake_z: 1.3,
  flake_detail: 0.2, flake_learned: 0.5,
  // around the subject (Photoshop's Select Subject): extra careful with highlights and texture
  subject_margin: 24, subject_tex: 0.75, subject_hi: 0.8, subject_lo: 0.8,
  // on the subject a soft blob over a sharp mark with this many look-alikes around is a highlight (glasses, jewellery)
  subject_soft_field: 3, subject_soft_field_max: 10,
  dilate: 2, exclude_outside: true, sensitivity: 1.0, second_pass: false, second_gain: 1.15,
};

function params(over = {}) {
  const p = Object.assign({}, DEFAULTS, over);
  p.seed = p.seed / p.sensitivity;
  p.tiny_seed = p.tiny_seed / p.sensitivity;
  p.soft_seed = p.soft_seed / p.sensitivity;
  p.hair_seed = p.hair_seed / p.sensitivity;
  p.broad_speck_z = p.broad_speck_z / p.sensitivity;
  p.skin_speck_z = p.skin_speck_z / p.sensitivity;
  p.detail_speck_z = p.detail_speck_z / p.sensitivity;
  p.line_z = p.line_z / p.sensitivity;
  p.line_zmed = p.line_zmed / p.sensitivity;
  p.min_fine = p.second_pass ? p.second_gain * p.seed : 0.0;
  return p;
}

// ---------- basics ---------------------------------------------------------------------------

// interleaved 16-bit (C = 1, 3 or 4) -> luma 0..1
function lumaFromU16(img, W, H, C) {
  const Y = new Float32Array(W * H);
  if (C === 1) { for (let i = 0; i < Y.length; i++) Y[i] = img[i] / 65535; return Y; }
  for (let i = 0, j = 0; i < Y.length; i++, j += C) Y[i] = (0.2126 * img[j] + 0.7152 * img[j + 1] + 0.0722 * img[j + 2]) / 65535;
  return Y;
}

function median8f(Y, W, H, k) {
  if (NAT.big(W, H) && Y instanceof Float32Array) { const r = NAT.median8f(Y, W, H, k); if (r) return r; }
  const Y8 = new Uint8Array(W * H);
  for (let i = 0; i < Y8.length; i++) { const v = Math.floor(Y[i] * 255 + 0.5); Y8[i] = v < 0 ? 0 : v > 255 ? 255 : v; }
  const m = F.median8(Y8, W, H, k);
  const out = new Float32Array(W * H);
  for (let i = 0; i < out.length; i++) out[i] = m[i] / 255;
  return out;
}

function opening(Y, W, H, k, bright) { return F.ellipseOpenClose(Y, W, H, k, bright); }

// clipped, grain-free areas touching the frame (scanner light / holder beyond the film)
function outsideMask(Y, W, H) {
  if (NAT.big(W, H) && Y instanceof Float32Array) { const r = NAT.outsideMask(Y, W, H); if (r) return r; }
  const mean = F.boxBlur(Y, W, H, 7);
  const sq = new Float32Array(W * H);
  for (let i = 0; i < sq.length; i++) sq[i] = Y[i] * Y[i];
  const msq = F.boxBlur(sq, W, H, 7);
  const flat = new Uint8Array(W * H);
  for (let i = 0; i < flat.length; i++) {
    const sd = Math.sqrt(Math.max(msq[i] - mean[i] * mean[i], 0));
    flat[i] = sd < 0.0015 && (Y[i] > 0.85 || Y[i] < 0.02) ? 1 : 0;
  }
  const { lab, n } = R.label(flat, W, H, 4);
  if (!n) return new Uint8Array(W * H);
  const size = new Int32Array(n + 1);
  for (let i = 0; i < lab.length; i++) if (lab[i]) size[lab[i]]++;
  const keep = new Uint8Array(n + 1);
  const touch = (l) => { if (l && size[l] > 0.002 * H * W) keep[l] = 1; };
  for (let x = 0; x < W; x++) { touch(lab[x]); touch(lab[(H - 1) * W + x]); }
  for (let y = 0; y < H; y++) { touch(lab[y * W]); touch(lab[y * W + W - 1]); }
  const out = new Uint8Array(W * H);
  for (let i = 0; i < out.length; i++) out[i] = keep[lab[i]];
  return F.dilateMask(out, W, H, 9, false);
}

// robust grain sigma as a function of brightness
function noiseByLevel(bg, resid, valid, W, H, bins = 48) {
  if (NAT.big(W, H) && bg instanceof Float32Array && resid instanceof Float32Array && valid instanceof Uint8Array) { const r = NAT.noiseByLevel(bg, resid, valid, W, H, bins); if (r) return r; }
  const buckets = Array.from({ length: bins }, () => []);
  for (let y = 0; y < H; y += 3) {
    for (let x = 0; x < W; x += 3) {
      const i = y * W + x;
      if (!valid[i]) continue;
      let l = Math.floor(bg[i] * bins);
      if (l < 0) l = 0; if (l > bins - 1) l = bins - 1;
      buckets[l].push(resid[i]);
    }
  }
  let sig = new Float64Array(bins).fill(NaN);
  for (let b = 0; b < bins; b++) {
    const rb = buckets[b];
    if (rb.length > 400) {
      const md = R.median(rb);
      const dev = rb.map((v) => Math.abs(v - md));
      sig[b] = 1.4826 * R.median(dev);
    }
  }
  const good = [];
  for (let b = 0; b < bins; b++) if (!Number.isNaN(sig[b])) good.push(b);
  if (!good.length) sig.fill(0.01);
  else {
    const filled = new Float64Array(bins);
    for (let b = 0; b < bins; b++) filled[b] = interp1(b, good, good.map((g) => sig[g]));
    sig = filled;
  }
  // ndi.median_filter(size 3, mode reflect) then floor
  const med3 = new Float64Array(bins);
  for (let b = 0; b < bins; b++) {
    const a = sig[b === 0 ? 0 : b - 1], c = sig[b === bins - 1 ? bins - 1 : b + 1];
    med3[b] = Math.max([a, sig[b], c].sort((u, v) => u - v)[1], 0.002);
  }
  const centers = Array.from({ length: bins }, (_, i) => (i + 0.5) / bins);
  const out = new Float32Array(W * H);
  const xs = centers, ys = Array.from(med3);
  // np.interp on uniform centres: guess the interval, then settle it exactly
  const nb = xs.length, x0 = xs[0], xl = xs[nb - 1];
  for (let i = 0; i < out.length; i++) {
    const x = bg[i];
    if (x <= x0) { out[i] = ys[0]; continue; }
    if (x >= xl) { out[i] = ys[nb - 1]; continue; }
    let lo = Math.floor((x - x0) * bins);
    if (lo < 0) lo = 0; else if (lo > nb - 2) lo = nb - 2;
    while (lo > 0 && xs[lo] > x) lo--;
    while (lo < nb - 2 && xs[lo + 1] <= x) lo++;
    const t = (x - xs[lo]) / (xs[lo + 1] - xs[lo]);
    out[i] = ys[lo] + t * (ys[lo + 1] - ys[lo]);
  }
  return out;
}

// numpy.interp for sorted xs
function interp1(x, xs, ys) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

// per-block robust statistic, then a small median + Gaussian and bilinear upsampling
// per 64-px block statistics: fn(vals, valid, n, nvalid) with scratch arrays (reused)
function blockStat(resid, valid, W, H, block, fn) {
  const hb = Math.max(1, Math.floor(H / block)), wb = Math.max(1, Math.floor(W / block));
  const grid = new Float32Array(hb * wb);
  const n = block * block;
  const vals = new Float64Array(n), vok = new Uint8Array(n);
  for (let by = 0; by < hb; by++) {
    for (let bx = 0; bx < wb; bx++) {
      let k = 0, nvalid = 0;
      for (let y = by * block; y < (by + 1) * block; y++) {
        const off = y * W;
        for (let x = bx * block; x < (bx + 1) * block; x++) {
          vals[k] = resid[off + x];
          const v = valid[off + x] ? 1 : 0;
          vok[k++] = v;
          nvalid += v;
        }
      }
      grid[by * wb + bx] = fn(vals, vok, n, nvalid);
    }
  }
  return { grid, hb, wb };
}

const natBlocks = (resid, valid, W, H) => NAT.big(W, H) && W >= 64 && H >= 64 && resid instanceof Float32Array && valid instanceof Uint8Array;

function noiseLocal(resid, valid, W, H, block = 64) {
  const buf = new Float64Array(block * block);
  const natGrid = natBlocks(resid, valid, W, H) ? NAT.blockStats(resid, valid, W, H, block, 0, true) : null;
  const { grid, hb, wb } = natGrid
    ? { grid: natGrid, hb: Math.max(1, Math.floor(H / block)), wb: Math.max(1, Math.floor(W / block)) }
    : blockStat(resid, valid, W, H, block, (vals, vok, n) => {
    let c = 0;
    for (let i = 0; i < n; i++) if (vok[i]) buf[c++] = vals[i];
    if (!c) return NaN;
    const md = R.medianOf(buf, c);
    let c2 = 0;
    for (let i = 0; i < n; i++) if (vok[i]) buf[c2++] = Math.abs(vals[i] - md);
    return 1.4826 * R.medianOf(buf, c2);
  });
  const fin = Array.from(grid).filter((v) => !Number.isNaN(v));
  const fill = fin.length ? R.median(fin) : 0.01;
  for (let i = 0; i < grid.length; i++) if (Number.isNaN(grid[i])) grid[i] = fill;
  const sm = F.gaussianBlur(median3f(grid, wb, hb), wb, hb, 1.0);
  return F.resizeLinear(sm, wb, hb, W, H);
}

function busyness(resid, valid, W, H, bright, block = 64) {
  const sgn = bright ? 1 : -1;
  const n = block * block;
  const buf = new Float64Array(n);
  const natGrid = natBlocks(resid, valid, W, H) ? NAT.blockStats(resid, valid, W, H, block, 1, bright) : null;
  const { grid, hb, wb } = natGrid
    ? { grid: natGrid, hb: Math.max(1, Math.floor(H / block)), wb: Math.max(1, Math.floor(W / block)) }
    : blockStat(resid, valid, W, H, block, (vals, vok, nn, nvalid) => {
    for (let i = 0; i < n; i++) buf[i] = sgn * vals[i];
    const md = R.medianOf(buf, n);
    for (let i = 0; i < n; i++) buf[i] = sgn * vals[i];
    const p99 = R.selectK(buf, n, Math.floor(0.99 * n));
    for (let i = 0; i < n; i++) buf[i] = Math.abs(sgn * vals[i] - md);
    const mad = 1.4826 * R.medianOf(buf, n);
    let T = (p99 - md) / (2.33 * Math.max(mad, 1e-5));
    if (nvalid / n < 0.5) T = 1.0;
    return T;
  });
  return { T: median3f(grid, wb, hb), hb, wb };
}

// cv2.medianBlur(ksize 3) on a float grid (replicate border)
function median3f(g, w, h) {
  const out = new Float32Array(w * h);
  const v = new Float32Array(9);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let k = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) v[k++] = g[F.clampIdx(y + dy, h) * w + F.clampIdx(x + dx, w)];
    out[y * w + x] = Float32Array.from(v).sort()[4];
  }
  return out;
}

// z-score maps at three scales for one polarity (+ shared pieces cached between polarities)
async function residualMaps(Y, W, H, bright, outside, cache = {}, tick = async () => {}) {
  const sgn = bright ? 1 : -1;
  let tl = Date.now();
  const times = [];
  const lap = (name) => { const now = Date.now(); times.push(name + " " + ((now - tl) / 1000).toFixed(1)); tl = now; };
  const m15 = cache.m15 || (cache.m15 = median8f(Y, W, H, 15));
  await tick(0.05);
  const m41 = cache.m41 || (cache.m41 = median8f(Y, W, H, 41));
  lap("m41");
  await tick(0.15);
  const o15 = opening(Y, W, H, 15, bright);
  lap("o15");
  await tick(0.25);
  const o41 = opening(Y, W, H, 41, bright);
  lap("o41");
  await tick(0.5);
  const half = cache.half || (cache.half = F.downscaleArea2(Y, W, H));
  const mh = cache.mh || (cache.mh = median8f(half.data, half.w, half.h, 61));
  lap("half+m61");
  await tick(0.55);
  const oh = opening(half.data, half.w, half.h, 61, bright);
  lap("o61");
  await tick(0.65);
  const bh = new Float32Array(half.w * half.h);
  for (let i = 0; i < bh.length; i++) bh[i] = bright ? Math.max(mh[i], oh[i]) : Math.min(mh[i], oh[i]);
  const bxl = F.resizeLinear(bh, half.w, half.h, W, H);
  lap("resize");
  const valid = new Uint8Array(W * H);
  for (let i = 0; i < valid.length; i++) valid[i] = outside[i] ? 0 : 1;
  const resid = cache.resid || (() => { const r = new Float32Array(W * H); for (let i = 0; i < r.length; i++) r[i] = Y[i] - m15[i]; return (cache.resid = r); })();
  const sigLevel = cache.sigLevel || (cache.sigLevel = noiseByLevel(m41, resid, valid, W, H));
  const sigLocal = cache.sigLocal || (cache.sigLocal = noiseLocal(resid, valid, W, H));
  const sigma = cache.sigma || (() => { const s = new Float32Array(W * H); for (let i = 0; i < s.length; i++) s[i] = Math.max(sigLevel[i], sigLocal[i]); return (cache.sigma = s); })();
  lap("noise");
  await tick(0.75);
  const busy = busyness(resid, valid, W, H, bright);
  lap("busy");
  await tick(0.8);
  const natZ = NAT.big(W, H) && outside instanceof Uint8Array ? NAT.zMaps(Y, m15, o15, m41, o41, bxl, sigma, outside, bright) : null;
  if (natZ) {
    const [zs, zl, zx] = natZ;
    lap("z");
    return { sigma, sigma_level: sigLevel, busy, z_s: zs, z_l: zl, z_xl: zx, times };
  }
  const zs = new Float32Array(W * H), zl = new Float32Array(W * H), zx = new Float32Array(W * H);
  for (let i = 0; i < zs.length; i++) {
    const b15 = bright ? Math.max(m15[i], o15[i]) : Math.min(m15[i], o15[i]);
    const bb = bright ? Math.max(m41[i], o41[i]) : Math.min(m41[i], o41[i]);
    if (outside[i]) continue;
    zs[i] = (sgn * (Y[i] - b15)) / sigma[i];
    zl[i] = (sgn * (Y[i] - bb)) / sigma[i];
    zx[i] = (sgn * (Y[i] - bxl[i])) / sigma[i];
  }
  lap("z");
  return { sigma, sigma_level: sigLevel, busy, z_s: zs, z_l: zl, z_xl: zx, times };
}

// ---------- per-mark measurements ------------------------------------------------------------

// local window helpers: m is a Uint8Array mask of size mw x mh placed at (y0, x0)
function edgeWidth(Y, G, W, m, mw, mh, y0, x0) {
  const inner = F.erodeMask(m, mw, mh, 3);
  let ninner = 0;
  for (let i = 0; i < inner.length; i++) ninner += inner[i];
  const core = ninner >= 3 ? inner : m;
  const d15 = F.dilateMask(m, mw, mh, 15), d7 = F.dilateMask(m, mw, mh, 7), d5 = F.dilateMask(m, mw, mh, 5), e5 = F.erodeMask(m, mw, mh, 5);
  const coreV = [], ringV = [], bandV = [];
  for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) {
    const i = y * mw + x, gi = (y0 + y) * W + (x0 + x);
    if (core[i]) coreV.push(Y[gi]);
    if (d15[i] && !d7[i]) ringV.push(Y[gi]);
    if (d5[i] && !e5[i]) bandV.push(G[gi]);
  }
  if (!ringV.length || !bandV.length) return 0;
  const C = Math.abs(R.percentile(coreV, 75) - R.median(ringV));
  const E = R.percentile(bandV, 90);
  return E > 1e-6 ? C / E : 99;
}

function kindOf(area, length, width) {
  const w = Math.max(width, 1);
  if (width <= 9 && length >= 12 && (area / (w * w) >= 4.0 || (length >= 25 && area <= 9 * length))) return "hair/fibre";
  if (area >= 60) return "dust";
  return "speck";
}

function skinLike(bg) {
  if (!bg || bg.length < 3) return false;
  const [r, g, b] = bg;
  return r >= 0.18 && r > g && g > b && r - b >= 0.05 && (r - b) / r <= 0.7;
}

// image_box: picture area inside the dark film rebate
function imageBox(Y, W, H, dark = 0.2, frac = 0.65, search = 0.12, step = 4) {
  const w = Math.max(1, Math.floor(W / step)), h = Math.max(1, Math.floor(H / step));
  const small = F.resizeArea(Y, W, H, w, h);
  const sm = F.gaussianBlur(small, w, h, 1.5);
  const d = (x, y) => (sm[y * w + x] < dark ? 1 : 0);
  const inner = (prof, n) => {
    const lim = Math.floor(search * n);
    const idx = [];
    for (let i = 0; i < lim; i++) if (prof[i] >= frac) idx.push(i);
    if (!idx.length) return 0;
    let last = idx[0];
    for (let k = 1; k < idx.length; k++) if (idx[k] - last <= 3) last = idx[k];
    return last + 1;
  };
  const c0 = Math.floor(0.2 * w), c1 = Math.floor(0.8 * w), r0 = Math.floor(0.2 * h), r1 = Math.floor(0.8 * h);
  const rowProf = (rev) => { const out = []; for (let i = 0; i < h; i++) { const y = rev ? h - 1 - i : i; let s = 0; for (let x = c0; x < c1; x++) s += d(x, y); out.push(s / (c1 - c0)); } return out; };
  const colProf = (rev) => { const out = []; for (let i = 0; i < w; i++) { const x = rev ? w - 1 - i : i; let s = 0; for (let y = r0; y < r1; y++) s += d(x, y); out.push(s / (r1 - r0)); } return out; };
  // past the edge print: the innermost solid dark row (0.9+ dark) within a quarter of the frame, with the rows between it
  // and the edge mostly dark (0.5+: the rebate and its print) and the next 3% inward picture (under 0.35 dark). 10-24,
  // q_03: the box stopped at the outermost black strip, so the orange edge codes and "OA" counted as picture and the
  // learned decision healed them. Unchanged on all 192 of his scans.
  const past = (prof, n) => {
    const lim = Math.floor(0.25 * n), side = Math.max(2, Math.ceil(0.03 * n));
    let best = -1, sum = 0;
    for (let i = 0; i < lim; i++) {
      sum += prof[i];
      if (prof[i] < 0.9 || sum / (i + 1) < 0.5) continue;
      let s2 = 0;
      for (let j = i + 1; j <= i + side && j < n; j++) s2 += prof[j];
      if (s2 / side < 0.35) best = i;
    }
    return best + 1;
  };
  const edge = (prof, n) => { const a = inner(prof, n); return [a, Math.max(a, past(prof, n))]; };
  const [t, b, l, r] = [edge(rowProf(false), h), edge(rowProf(true), h), edge(colProf(false), w), edge(colProf(true), w)];
  const box = [t[1] * step, H - b[1] * step, l[1] * step, W - r[1] * step];
  box.outer = [t[0] * step, H - b[0] * step, l[0] * step, W - r[0] * step];   // the box as before (without the print)
  return box;
}

module.exports = { DEFAULTS, params, lumaFromU16, median8f, opening, outsideMask, noiseByLevel, noiseLocal, busyness, residualMaps, edgeWidth, kindOf, skinLike, imageBox, interp1, median3f };

"use strict";
// Faint soft discs on smooth ground: dust far enough from the film's focus to show only as a pale disc a few
// tens of pixels across, lifting the sky by about 1% (the soft stage needs a stronger blob and, at full size,
// the grain hides it). At a quarter of the working size the grain averages out and such a disc is a clean blob:
// a difference of Gaussians at six scales finds it, each scale measured against the local noise of that scale,
// and only where the ground is smooth (no detail finer than the disc around it).
//   find(): the candidates, with what they look like (size, contrast, roundness, tint, edges near)
const F = require("./filters");
const { skinLike } = require("./maps");

const Q = 4;                                        // the search runs at 1/Q of the working size
const SIGMAS = [1.2, 1.9, 3.0, 4.8, 7.7, 12.3];      // Gaussians at 1/Q size; a DoG between neighbours per scale
const RADIUS_PER_SIGMA = 1.6;                        // a disc's radius (1/Q px) is ~1.6 x the sigma that finds it best

// 1/Q-size planes: luminance (Rec. 709, 0..1) and the three channels
function quarter(img, W, H, C) {
  const w = Math.floor(W / Q), h = Math.floor(H / Q), n = w * h, inv = 1 / (Q * Q * 65535);
  const Y = new Float32Array(n), R = new Float32Array(n), G = new Float32Array(n), B = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0;
      for (let dy = 0; dy < Q; dy++) {
        let o = ((y * Q + dy) * W + x * Q) * C;
        for (let dx = 0; dx < Q; dx++, o += C) {
          if (C >= 3) { r += img[o]; g += img[o + 1]; b += img[o + 2]; } else { r += img[o]; g += img[o]; b += img[o]; }
        }
      }
      const i = y * w + x;
      R[i] = r * inv; G[i] = g * inv; B[i] = b * inv;
      Y[i] = 0.2126 * R[i] + 0.7152 * G[i] + 0.0722 * B[i];
    }
  }
  return { w, h, Y, R, G, B };
}

// robust noise of a response map in blocks of size b (median |v| x 1.4826 over every 2nd pixel), each block then
// the median of its 3 x 3 neighbourhood, so a disc never raises the noise it is measured against
function blockNoise(D, w, h, b, inside) {
  const bw = Math.ceil(w / b), bh = Math.ceil(h / b), raw = new Float32Array(bw * bh).fill(NaN);
  const vals = [];
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      vals.length = 0;
      for (let y = by * b; y < Math.min(h, (by + 1) * b); y += 2) {
        for (let x = bx * b; x < Math.min(w, (bx + 1) * b); x += 2) { const i = y * w + x; if (!inside || inside[i]) vals.push(Math.abs(D[i])); }
      }
      if (vals.length >= 16) { vals.sort((p, q) => p - q); raw[by * bw + bx] = vals[vals.length >> 1] * 1.4826; }
    }
  }
  const out = new Float32Array(bw * bh);
  const nb = [];
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      nb.length = 0;
      for (let yy = Math.max(0, by - 1); yy <= Math.min(bh - 1, by + 1); yy++) for (let xx = Math.max(0, bx - 1); xx <= Math.min(bw - 1, bx + 1); xx++) { const v = raw[yy * bw + xx]; if (v === v) nb.push(v); }
      nb.sort((p, q) => p - q);
      out[by * bw + bx] = nb.length ? nb[nb.length >> 1] : Infinity;
    }
  }
  return { b, bw, bh, out };
}
const noiseAt = (N, x, y) => N.out[Math.min(N.bh - 1, Math.floor(y / N.b)) * N.bw + Math.min(N.bw - 1, Math.floor(x / N.b))];

// The candidates. r1: the first detection (outside mask, marks: a disc already judged by the soft stage is left to
// it). p: engine settings (disc_z, disc_smooth, disc_edge_frac). bright: the polarity searched.
// Returns {cands, q} with cands in working-size pixels.
function find(img, W, H, C, r1, p, bright = true) {
  const q = quarter(img, W, H, C);
  const { w, h, Y } = q, n = w * h;
  // inside: not the film border (the outside mask, one quarter-pixel in) and not the outer 2% of the frame
  const inside = new Uint8Array(n).fill(1);
  const mx = Math.round(0.02 * w), my = Math.round(0.02 * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x < mx || y < my || x >= w - mx || y >= h - my) { inside[y * w + x] = 0; continue; }
      if (r1.outside) {
        let o = 0;
        for (let dy = 0; dy < Q && !o; dy++) for (let dx = 0; dx < Q && !o; dx++) o = r1.outside[(y * Q + dy) * W + x * Q + dx];
        if (o) inside[y * w + x] = 0;
      }
    }
  }
  const Gs = SIGMAS.map((s) => F.gaussianBlur(Y, w, h, s));
  const sign = bright ? 1 : -1;
  const D = [], N = [];
  for (let k = 0; k + 1 < Gs.length; k++) {
    const d = new Float32Array(n);
    for (let i = 0; i < n; i++) d[i] = sign * (Gs[k][i] - Gs[k + 1][i]);
    D.push(d);
    N.push(blockNoise(d, w, h, Math.max(24, Math.round(6 * SIGMAS[k])), inside));
  }
  // smooth ground: the finest scale's noise near the image's quietest (the 10th percentile of its blocks)
  const fine = N[0].out.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  const floor = fine.length ? fine[Math.floor(0.1 * fine.length)] : Infinity;
  const smoothMax = (p.disc_smooth ?? 1.5) * floor;
  // best scale per pixel (z against that scale's local noise), then local maxima
  const Z = new Float32Array(n), S = new Uint8Array(n);
  for (let k = 0; k < D.length; k++) {
    const d = D[k], Nk = N[k];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!inside[i]) continue;
        const z = d[i] / noiseAt(Nk, x, y);
        if (z > Z[i]) { Z[i] = z; S[i] = k; }
      }
    }
  }
  const zMin = p.disc_z ?? 5;
  const peaks = [];
  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const i = y * w + x, z = Z[i];
      if (z < zMin || !inside[i]) continue;
      let top = true;
      for (let dy = -2; dy <= 2 && top; dy++) for (let dx = -2; dx <= 2; dx++) if ((dy || dx) && Z[i + dy * w + dx] > z) { top = false; break; }
      if (top) peaks.push({ x, y, z, k: S[i] });
    }
  }
  peaks.sort((a, b) => b.z - a.z);
  // one candidate per disc: a weaker peak inside a kept one's radius is the same disc
  const kept = [];
  for (const pk of peaks) {
    const r = RADIUS_PER_SIGMA * SIGMAS[pk.k];
    if (kept.some((o) => (o.x - pk.x) ** 2 + (o.y - pk.y) ** 2 < (1.2 * Math.max(r, RADIUS_PER_SIGMA * SIGMAS[o.k])) ** 2)) continue;
    kept.push(pk);
  }
  // a disc that is already one of the detector's marks (a speck seen small, a soft blob already judged) is left to it
  const known = r1.marks || [];
  const cands = [];
  for (const pk of kept) {
    const { x, y, k } = pk;
    const rq = RADIUS_PER_SIGMA * SIGMAS[k];            // radius at 1/Q size
    const smooth = noiseAt(N[0], x, y) <= smoothMax;
    // the disc's lift over its surroundings (the Gaussian two scales up stands for the ground), per channel
    const bgK = Math.min(Gs.length - 1, k + 2), i = y * w + x;
    const bgY = Gs[bgK][i], lift = sign * (Gs[Math.max(0, k - 1)][i] - bgY);
    const contrast = lift / Math.max(1e-4, bgY);
    // per channel: the lift of the disc's middle over its ring, and the ring's own colour (the ground)
    const ground = [0, 0, 0];
    const chan = ["R", "G", "B"].map((c, ci) => {
      const P = q[c];
      let s = 0, cnt = 0, sb = 0, cb = 0;
      const R0 = Math.max(1, Math.round(0.7 * rq)), R1 = Math.round(1.8 * rq), R2 = Math.round(2.6 * rq);
      for (let yy = Math.max(0, y - R2); yy <= Math.min(h - 1, y + R2); yy++) {
        for (let xx = Math.max(0, x - R2); xx <= Math.min(w - 1, x + R2); xx++) {
          const dd = (yy - y) ** 2 + (xx - x) ** 2, v = P[yy * w + xx];
          if (dd <= R0 * R0) { s += v; cnt++; } else if (dd >= R1 * R1 && dd <= R2 * R2) { sb += v; cb++; }
        }
      }
      ground[ci] = cb ? sb / cb : 0;
      return cnt && cb ? s / cnt - sb / cb : 0;
    });
    // roundness: second moments of the lift above the ring's level within 1.6 radii
    let m00 = 0, m20 = 0, m02 = 0, m11 = 0;
    const Rm = Math.max(2, Math.round(1.6 * rq)), base = bgY;
    for (let yy = Math.max(0, y - Rm); yy <= Math.min(h - 1, y + Rm); yy++) {
      for (let xx = Math.max(0, x - Rm); xx <= Math.min(w - 1, x + Rm); xx++) {
        const dy = yy - y, dx = xx - x;
        if (dx * dx + dy * dy > Rm * Rm) continue;
        const v = Math.max(0, sign * (Gs[0][yy * w + xx] - base));
        m00 += v; m20 += v * dx * dx; m02 += v * dy * dy; m11 += v * dx * dy;
      }
    }
    let elong = 1;
    if (m00 > 0) {
      const a = m20 / m00, c = m02 / m00, b = m11 / m00, tr = a + c, det = a * c - b * b;
      const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det)), l1 = tr / 2 + disc, l2 = Math.max(1e-9, tr / 2 - disc);
      elong = Math.sqrt(l1 / l2);
    }
    // roundness further out: the second moments of the scale's own positive response within 3 radii. A disc's
    // response ends at its rim (a dark ring follows); a band or line it lies on runs on (d_007's bright band)
    let w00 = 0, w20 = 0, w02 = 0, w11 = 0;
    const Rw = Math.max(3, Math.round(3 * rq)), Dk = D[k];
    for (let yy = Math.max(0, y - Rw); yy <= Math.min(h - 1, y + Rw); yy++) {
      for (let xx = Math.max(0, x - Rw); xx <= Math.min(w - 1, x + Rw); xx++) {
        const dy = yy - y, dx = xx - x;
        if (dx * dx + dy * dy > Rw * Rw) continue;
        const v = Math.max(0, Dk[yy * w + xx]);
        w00 += v; w20 += v * dx * dx; w02 += v * dy * dy; w11 += v * dx * dy;
      }
    }
    let elongWide = 1;
    if (w00 > 0) {
      const a = w20 / w00, c = w02 / w00, b = w11 / w00, tr = a + c, det = a * c - b * b;
      const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det)), l1 = tr / 2 + disc, l2 = Math.max(1e-9, tr / 2 - disc);
      elongWide = Math.sqrt(l1 / l2);
    }
    // edges near: the strongest step of the smooth image across the ring, against the disc's own lift; and what
    // the ring holds besides a plain gradient (sky brightening toward the sun is no edge): the 16 ring samples less
    // their best tilted plane (the first harmonic round the circle), the median of what is left, against the lift
    let edge = 0;
    const Gm = Gs[Math.min(Gs.length - 1, k + 1)], Re = Math.round(2.2 * rq);
    const ring = [];
    for (let t = 0; t < 16; t++) {
      const a = (t / 16) * Math.PI * 2, xa = Math.round(x + Re * Math.cos(a)), ya = Math.round(y + Re * Math.sin(a));
      if (xa >= 0 && ya >= 0 && xa < w && ya < h) ring.push([a, Gm[ya * w + xa]]);
      const xb = Math.round(x - Re * Math.cos(a)), yb = Math.round(y - Re * Math.sin(a));
      if (xa < 0 || ya < 0 || xa >= w || ya >= h || xb < 0 || yb < 0 || xb >= w || yb >= h) continue;
      edge = Math.max(edge, Math.abs(Gm[ya * w + xa] - Gm[yb * w + xb]));
    }
    let edgeFlat = 99;
    if (ring.length >= 12) {
      let m0 = 0, mc = 0, ms = 0;
      for (const [a, v] of ring) { m0 += v; mc += v * Math.cos(a); ms += v * Math.sin(a); }
      m0 /= ring.length; mc *= 2 / ring.length; ms *= 2 / ring.length;
      const res = ring.map(([a, v]) => Math.abs(v - m0 - mc * Math.cos(a) - ms * Math.sin(a))).sort((p, q) => p - q);
      edgeFlat = res[res.length >> 1] / Math.max(1e-6, lift);
    }
    const cx = x * Q + Q / 2, cy = y * Q + Q / 2, r = rq * Q;
    // the detector's own marks there. A small disc that is one of them is that mark seen small: left to it. A
    // bigger one may hold a healed speck (its veil is taken away round the speck, which heals as usual), or lie
    // where the soft stage left a blob only for a crowd or by its learned decision (faint discs scattered over a
    // sky are a crowd): it is judged here. Any other mark the detector left as picture keeps it.
    const near = known.filter((m) => Math.abs(m.cx - cx) < Math.max(r, 8) && Math.abs(m.cy - cy) < Math.max(r, 8));
    const weak = (m) => m.verdict === "keep-photo" && m.kind === "soft" && /one of many same-size blobs|photo, by the decision learned/.test(String(m.why));
    const overrule = p.disc_overrule !== false;
    if (near.length && (!overrule || k < 1 || near.some((m) => m.verdict === "keep-photo" && !weak(m)))) continue;
    // nor one a mark runs through (a hair being healed, a speck at its rim): any of the detector's core within it
    // (for a bigger disc only when its own marks are not what the core holds)
    if (r1.core && (!overrule || k < 1)) {
      let hit = 0;
      const rr = Math.ceil(r);
      for (let yy = Math.max(0, Math.round(cy - rr)); yy <= Math.min(H - 1, Math.round(cy + rr)) && !hit; yy += 2) {
        for (let xx = Math.max(0, Math.round(cx - rr)); xx <= Math.min(W - 1, Math.round(cx + rr)); xx += 2) {
          if ((xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr && r1.core[yy * W + xx]) { hit = 1; break; }
        }
      }
      if (hit) continue;
    }
    const holds = near.filter((m) => m.verdict === "defect").length, crowd = near.some(weak);
    const onSubject = !!(r1.subject && r1.subject[Math.min(H - 1, Math.round(cy)) * W + Math.min(W - 1, Math.round(cx))]);
    cands.push({
      cx, cy, r: Math.round(r * 10) / 10, z: Math.round(pk.z * 10) / 10, scale: k, smooth,
      contrast: Math.round(contrast * 1e4) / 1e4, elong: Math.round(elong * 100) / 100, elongWide: Math.round(elongWide * 100) / 100,
      edge: Math.round((edge / Math.max(1e-6, lift)) * 100) / 100, edgeFlat: Math.round(edgeFlat * 100) / 100, tint: chan.map((v) => Math.round(v * 65535)),
      noise: Math.round(noiseAt(N[0], x, y) / floor * 100) / 100, bg: ground.map((v) => Math.round(v * 1e4) / 1e4), onSubject, holds, crowd,
    });
  }
  // the finest scale's noise at a working-size point against the image's quietest (a candidate's noise): for judging
  // the ground under the detector's own marks (engine/pipeline.js, fibres) as the discs' is judged
  const noiseRatio = (px, py) => noiseAt(N[0], Math.min(w - 1, Math.max(0, Math.floor(px / Q))), Math.min(h - 1, Math.max(0, Math.floor(py / Q)))) / floor;
  return { cands, q: { w, h, floor }, noiseRatio };
}

// ---------- the faint-disc network ----------------------------------------------------------------------------
// The dust network's two-view design (engine/dustnet.js computes it), trained on soft discs painted onto his film
// scans and film-like digital photos against those photos' own soft candidates (training/discnet). Its two views
// of a candidate: the windows 4 and 12 radii across (at least 40 and 120 px) centred on it, edges repeated,
// area-averaged to 40 x 40 like cv2.resize INTER_AREA, each normalised like the dust network's.
let NETD = null;
try { NETD = require("./discnet_weights"); } catch (e) { NETD = null; }
let discNet = null;

function views(img, W, H, C, c) {
  const DN = require("./dustnet"), S = 40;
  return [Math.max(40, Math.round(4 * c.r)), Math.max(120, Math.round(12 * c.r))].map((side) => {
    const x0 = Math.floor(c.cx - side / 2 + 0.5), y0 = Math.floor(c.cy - side / 2 + 0.5);
    const plane = new Float32Array(side * side), vals = [0, 1, 2].map(() => new Float64Array(S * S));
    for (let ch = 0; ch < 3; ch++) {
      const cc = C >= 3 ? ch : 0;
      for (let y = 0; y < side; y++) {
        const sy = Math.min(H - 1, Math.max(0, y0 + y));
        for (let x = 0; x < side; x++) plane[y * side + x] = Math.fround(img[(sy * W + Math.min(W - 1, Math.max(0, x0 + x))) * C + cc] / 65535);
      }
      const small = side === S ? plane : F.resizeArea(plane, side, side, S, S);
      for (let k = 0; k < S * S; k++) vals[ch][k] = small[k];
    }
    return DN.normalise(vals, true);
  });
}

// the network's opinion of a candidate (a logit, > 0 = soft dust; 3 decimals), or null without a trained network
function score(img, W, H, C, c) {
  if (!NETD) return null;
  if (!discNet) discNet = require("./dustnet").make(NETD, -1);
  return Math.round(discNet.forward(views(img, W, H, C, c)) * 1000) / 1000;
}

// The rules a candidate must pass to be soft dust (before any learned model): on smooth ground, clear of
// the noise, faint, round, and with no edge of the picture across its ring. Returns "" or the reason it is left.
function rules(c, p = {}) {
  // spots the size of the smallest scale (catchlights, flowers, highlights on metal) are the soft stage's to judge,
  // unless small discs are on: then only on the plainest ground and well clear of the grain (and the network must
  // be sure: netNeeded)
  if (c.scale < 1 && !p.disc_small) return "smaller than a faint disc (the soft stage judges those)";
  if (c.scale < 1 && (c.noise > (p.disc_small_noise ?? 1.25) || c.z < (p.disc_small_z ?? 7))) return "a small disc not on the plainest ground or not clear of the grain";
  if (!c.smooth) return "not on smooth ground";
  // the subject Select Subject found: its skin always stays (sheen, highlights); its clothes only when set
  // (disc_subject_net: the network must be surer there, netNeeded), so faint dust on a shirt can go
  if (c.onSubject && (!p.disc_subject_net || skinLike(c.bg))) return "on the subject Select Subject found (sheen and highlights belong there)";
  // skin off the subject: skin-coloured ground, and (when set) saturated like skin: a warm grey sky is not skin
  if (skinLike(c.bg) && (!p.disc_skin_sat || (c.bg[0] - c.bg[2]) / c.bg[0] >= p.disc_skin_sat)) return "on skin (sheen and highlights belong there)";
  // an edge: (when set) what the ring holds beyond a plain gradient, else the strongest step across it
  if (p.disc_edge_flat ? (c.edgeFlat > p.disc_edge_flat || c.edge > (p.disc_edge_cap ?? 3)) : c.edge > (p.disc_edge ?? 1.0)) return "an edge of the picture runs past it";
  if (c.elong > (p.disc_elong ?? 1.8)) return "not round";
  if (c.elongWide > (p.disc_elong_wide ?? 2.0)) return "part of a line or band";
  if (c.z < (p.disc_z_net ? 5 : (p.disc_z_heal ?? 6))) return "too faint to tell from the grain";
  if (c.contrast > (p.disc_contrast ?? 0.05)) return "too bright for soft dust (a light, the sun, a bright cloud)";
  return "";
}

// how sure the faint-disc network must be (its logit) for a disc the rules let through: a small disc, or one
// only a little above the grain, needs the network to be sure; any other only that it does not call it picture
function netNeeded(c, p = {}) {
  let need = p.disc_net_min ?? 0;
  if (c.scale < 1) need = Math.max(need, p.disc_small_net ?? 1);
  if (c.z < (p.disc_z_heal ?? 6)) need = Math.max(need, p.disc_z_net ?? 1);
  if (c.onSubject) need = Math.max(need, p.disc_subject_net ?? 2);
  return need;
}

// ---------- healing by subtraction ----------------------------------------------------------------------------
// Soft dust is a faint veil over the picture, not a hole in it: the grain under it is intact. So instead of
// copying a patch over it, the veil itself is measured and taken away. The ground under the disc is fitted from a
// ring around it (1.5 to 2.3 radii; a quadratic per channel, robust to grain and specks, the detector's marks
// left out); what stands above that ground, smoothed so only the veil and not the grain is taken, faded out
// between 1.3 and 2 radii, is the bump. Subtracting it leaves the original grain exactly where it was.

// a 3 x 3 linear system (a plane's least squares), or null
function solve3(A, b) {
  const M = A.map((r, i) => r.concat([b[i]]));
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 3; r++) { if (r === c) continue; const f = M[r][c] / M[c][c]; for (let k = c; k <= 3; k++) M[r][k] -= f * M[c][k]; }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}

// least squares with weights for the six terms of a quadratic in (u, v)
function solve6(A, b) {
  const n = 6, M = A.map((row, i) => row.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

// The bump of one disc d = {cx, cy, r} at working size. core: the detector's marks (left out of the fit and the
// veil), outside: the film border. Returns {x0, y0, w, h, C, data (Float32 per pixel per channel, 16-bit units),
// peak, ring, ok, why}: ok only when the ground fits (no edge in the ring) and the veil clearly stands above it.
function bump(img, W, H, C, d, core, outside) {
  const R = d.r, half = Math.ceil(2.4 * R) + 2, cx = d.cx, cy = d.cy;
  const x0 = Math.max(0, Math.round(cx) - half), y0 = Math.max(0, Math.round(cy) - half);
  const x1 = Math.min(W - 1, Math.round(cx) + half), y1 = Math.min(H - 1, Math.round(cy) + half);
  const w = x1 - x0 + 1, h = y1 - y0 + 1, n = w * h;
  const dist = new Float32Array(n), skip = new Uint8Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, g = (y0 + y) * W + x0 + x;
      dist[i] = Math.hypot(x0 + x - cx, y0 + y - cy) / R;
      if ((core && core[g]) || (outside && outside[g])) skip[i] = 1;
    }
  }
  const nc = Math.min(C, 3), data = new Float32Array(n * C);
  const out = { x0, y0, w, h, C, data, peak: 0, ring: 0, ok: false, why: "" };
  const lift = [];   // the smoothed lift per channel
  for (let c = 0; c < nc; c++) {
    const val = new Float32Array(n);
    for (let i = 0; i < n; i++) val[i] = img[((y0 + Math.floor(i / w)) * W + x0 + (i % w)) * C + c];
    // the ground: a quadratic over the ring, three passes reweighting by the residual (Tukey, 4 x the robust spread)
    const ring = [];
    for (let i = 0; i < n; i += 1) if (!skip[i] && dist[i] >= 1.5 && dist[i] <= 2.3) ring.push(i);
    if (ring.length < 60) { out.why = "too little ground around it"; return out; }
    let coef = null, wts = new Float64Array(ring.length).fill(1);
    for (let it = 0; it < 3; it++) {
      const A = [0, 1, 2, 3, 4, 5].map(() => [0, 0, 0, 0, 0, 0]), b = [0, 0, 0, 0, 0, 0];
      for (let j = 0; j < ring.length; j++) {
        const i = ring[j], u = (x0 + (i % w) - cx) / R, v = (y0 + Math.floor(i / w) - cy) / R, f = [1, u, v, u * u, u * v, v * v], wj = wts[j];
        if (!wj) continue;
        for (let a = 0; a < 6; a++) { b[a] += wj * f[a] * val[i]; for (let q = 0; q < 6; q++) A[a][q] += wj * f[a] * f[q]; }
      }
      coef = solve6(A, b);
      if (!coef) { out.why = "the ground does not fit"; return out; }
      const res = ring.map((i) => { const u = (x0 + (i % w) - cx) / R, v = (y0 + Math.floor(i / w) - cy) / R; return val[i] - (coef[0] + coef[1] * u + coef[2] * v + coef[3] * u * u + coef[4] * u * v + coef[5] * v * v); });
      const sorted = res.map(Math.abs).sort((p, q) => p - q), s = Math.max(1, sorted[sorted.length >> 1] * 1.4826);
      for (let j = 0; j < ring.length; j++) { const t = res[j] / (4 * s); wts[j] = Math.abs(t) < 1 ? (1 - t * t) ** 2 : 0; }
    }
    const resid = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      if (skip[i]) continue;
      const u = (x0 + (i % w) - cx) / R, v = (y0 + Math.floor(i / w) - cy) / R;
      resid[i] = val[i] - (coef[0] + coef[1] * u + coef[2] * v + coef[3] * u * u + coef[4] * u * v + coef[5] * v * v);
    }
    lift.push(F.gaussianBlur(resid, w, h, Math.max(1.5, R / 5)));
  }
  // luminance of the lift: its peak near the centre against its wobble on the ring
  const Lw = nc === 3 ? [0.2126, 0.7152, 0.0722] : [1];
  const L = new Float32Array(n);
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++) L[i] += Lw[c] * lift[c][i];
  let peak = -Infinity, ss = 0, cnt = 0;
  for (let i = 0; i < n; i++) {
    if (dist[i] <= 0.7 && L[i] > peak) peak = L[i];
    if (dist[i] >= 1.5 && dist[i] <= 2.3 && !skip[i]) { ss += L[i] * L[i]; cnt++; }
  }
  out.peak = Math.round(peak);
  out.ring = Math.round(Math.sqrt(ss / Math.max(1, cnt)));
  if (!(peak > 3 * Math.max(out.ring, 4))) { out.why = "no veil stands clearly above the ground"; return out; }
  for (let i = 0; i < n; i++) {
    const t = dist[i] <= 1.3 ? 1 : dist[i] >= 2.0 ? 0 : 0.5 + 0.5 * Math.cos(((dist[i] - 1.3) / 0.7) * Math.PI);
    if (!t) continue;
    for (let c = 0; c < C; c++) data[i * C + c] = t * lift[Math.min(c, nc - 1)][i];
  }
  out.ok = true;
  return out;
}

// The bump at the disc's own size: a disc bigger than the scale that found it puts the ground ring on its own flank
// and most of its veil into the "ground". So the window grows (1.5 x, then 2.25 x the radius found) while the veil
// it measures keeps growing by a sixth or more and still fits; an edge in the bigger ring stops it.
function bumpFit(img, W, H, C, d, core, outside) {
  let best = bump(img, W, H, C, d, core, outside);
  if (!best.ok) return best;
  for (const f of [1.5, 2.25]) {
    const b = bump(img, W, H, C, Object.assign({}, d, { r: d.r * f }), core, outside);
    if (!b.ok || b.peak < 1.15 * best.peak) break;
    best = b;
  }
  return best;
}

// Take each bump away from the image (in place); mark where it changed anything (> 1/8 of an 8-bit level).
function subtract(img, W, H, C, bumps, mask) {
  for (const b of bumps) {
    for (let y = 0; y < b.h; y++) {
      for (let x = 0; x < b.w; x++) {
        const i = y * b.w + x, g = (b.y0 + y) * W + b.x0 + x;
        let big = false;
        for (let c = 0; c < C; c++) {
          const v = b.data[i * C + c];
          if (Math.abs(v) >= 32) big = true;
          const o = g * C + c, nv = Math.round(img[o] - v);
          img[o] = nv < 0 ? 0 : nv > 65535 ? 65535 : nv;
        }
        if (big && mask) mask[g] = 1;
      }
    }
  }
}

// The same bumps taken away from a full-size tile of a large scan (the working copy is k times smaller): each
// full-size pixel takes the bump at its place on the working grid, bilinearly (the veil is smooth, so nothing of
// the grain is resampled). tile: chunky Uint16, (x1 - x0) x (y1 - y0) x C, full-size coordinates.
function subtractFull(tile, x0, y0, x1, y1, C, k, bumps) {
  const tw = x1 - x0;
  for (const b of bumps) {
    const fx0 = Math.max(x0, Math.floor(b.x0 * k)), fx1 = Math.min(x1, Math.ceil((b.x0 + b.w) * k));
    const fy0 = Math.max(y0, Math.floor(b.y0 * k)), fy1 = Math.min(y1, Math.ceil((b.y0 + b.h) * k));
    if (fx1 <= fx0 || fy1 <= fy0) continue;
    const at = (i, j, c) => (i < 0 || j < 0 || i >= b.w || j >= b.h ? 0 : b.data[(j * b.w + i) * C + c]);
    for (let Y = fy0; Y < fy1; Y++) {
      const v = (Y + 0.5) / k - 0.5 - b.y0, j0 = Math.floor(v), fj = v - j0;
      for (let X = fx0; X < fx1; X++) {
        const u = (X + 0.5) / k - 0.5 - b.x0, i0 = Math.floor(u), fi = u - i0;
        for (let c = 0; c < C; c++) {
          const val = (1 - fi) * (1 - fj) * at(i0, j0, c) + fi * (1 - fj) * at(i0 + 1, j0, c) + (1 - fi) * fj * at(i0, j0 + 1, c) + fi * fj * at(i0 + 1, j0 + 1, c);
          if (!val) continue;
          const o = ((Y - y0) * tw + (X - x0)) * C + c, nv = Math.round(tile[o] - val);
          tile[o] = nv < 0 ? 0 : nv > 65535 ? 65535 : nv;
        }
      }
    }
  }
}

// A disc that missed the rules only narrowly (an edge a little near, a little oval, a little bright), on smooth
// ground, big enough, off skin and the subject: worth Claude's look when there is one, never taken away without it.
function nearMiss(c, p = {}) {
  if ((c.scale < 1 && !p.disc_small) || !c.smooth || (c.onSubject && (!p.disc_subject_net || skinLike(c.bg)))) return false;
  if (skinLike(c.bg) && (!p.disc_skin_sat || (c.bg[0] - c.bg[2]) / c.bg[0] >= p.disc_skin_sat)) return false;
  const edgeOk = p.disc_edge_flat ? c.edgeFlat <= p.disc_edge_flat * 1.5 && c.edge <= (p.disc_edge_cap ?? 3) * 1.3 : c.edge <= (p.disc_edge ?? 1.0) * 1.6;
  return edgeOk && c.elong <= (p.disc_elong ?? 1.8) * 1.35 && c.elongWide <= (p.disc_elong_wide ?? 2.0) * 1.25 && c.contrast <= (p.disc_contrast ?? 0.05) * 1.5;
}

// ---------- the scan's own dust --------------------------------------------------------------------------------
// A speck's shadow dims the negative's three colours by one factor, so after inversion every real disc in a scan
// lifts the channels in the same proportion (the colour consistency Zhou & Lin, CVPR 2007, use for sensor dust);
// the picture's soft blobs rarely share it (on the benchmark: within 5 degrees of the scan's own, 124 of 244 dust,
// 6 of 66 photo). The fingerprint is the median colour direction of the discs taken with the most certainty (z 8+,
// network 1+, at least 5 of them: a scan with its own dust). Its angle is calibrated on the scan's picture candidates
// (not on smooth ground, or an edge across): the angle a tenth of them fall within, at most 5 degrees; below 2
// degrees the colour tells nothing (a black-and-white scan: everything is grey) and the test is off.
function colourDir(c) {
  if (!c.tint || c.tint.length < 3 || !c.bg) return null;
  const v = c.tint.map((t, i) => t / 65535 / Math.max(c.bg[i], 1e-3)), n = Math.hypot(v[0], v[1], v[2]);
  return n > 0 ? v.map((x) => x / n) : null;
}
const angle = (a, b) => (Math.acos(Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])) * 180) / Math.PI;

// taken: the candidates taken away (with their network score); cands: all of them; whys: the rules' reason per
// candidate. Returns {dir, theta, sure} or null (no dust population, or a grey scan: theta 0)
function population(taken, cands, whys, p = {}) {
  const sure = taken.filter((c) => c.z >= (p.disc_sure_z ?? 8) && c.net !== null && c.net !== undefined && c.net >= (p.disc_sure_net ?? 1)).map(colourDir).filter(Boolean);
  if (sure.length < (p.disc_population ?? 5)) return null;
  const m = [0, 1, 2].map((i) => sure.map((u) => u[i]).sort((a, b) => a - b)[sure.length >> 1]), n = Math.hypot(m[0], m[1], m[2]);
  if (!(n > 0)) return null;
  const dir = m.map((x) => x / n);
  const pic = cands.filter((c) => !c.smooth || /edge of the picture/.test(whys.get(c) || "")).map(colourDir).filter(Boolean).map((u) => angle(u, dir)).sort((a, b) => a - b);
  const theta = pic.length >= 20 ? Math.min(p.disc_colour ?? 5, pic[Math.floor(0.1 * pic.length)]) : 0;
  return { dir, theta: theta >= 2 ? Math.round(theta * 10) / 10 : 0, sure: sure.length };
}

// A disc turned down only for its shape (a soft fibre, two discs run together: not round, or running on like a band)
// in a scan with its own dust, of the dust's colour (within twice the calibrated angle). On the benchmark: 180 -> 198
// of 249 dust, the same 2 photo; the 27 digital photos have no dust population and are untouched.
// With disc_rescue_near, one turned down only for being small and a little short of the plainest ground (noise up to
// disc_near_noise, z disc_near_z+) or for an edge beside it rather than across it (a cloud's rim, a neighbouring disc:
// what the ring holds beyond a gradient up to disc_near_edge, no step over disc_near_cap) goes too when of the dust's
// colour within disc_near_colour x the angle.
function rescue(c, pop, p = {}) {
  if (!pop || !pop.theta || p.disc_rescue === false) return false;
  const u = colourDir(c);
  if (!u) return false;
  const a = angle(u, pop.dir);
  const shape = Object.assign({}, p, { disc_elong: p.disc_rescue_elong ?? 2.5, disc_elong_wide: p.disc_rescue_wide ?? 3.0 });
  if (!rules(c, shape)) return a <= (p.disc_rescue_colour ?? 2) * pop.theta;
  if (!p.disc_rescue_near) return false;
  const near = Object.assign(shape, { disc_small_noise: p.disc_near_noise ?? 1.5, disc_small_z: p.disc_near_z ?? 6, disc_edge_flat: p.disc_near_edge ?? 1.0, disc_edge_cap: p.disc_near_cap ?? 2.5 });
  return !rules(c, near) && a <= (p.disc_near_colour ?? 1) * pop.theta;
}

// A soft fibre among the candidates: turned down only for its shape (not round, or a band, up to disc_fibre_elong /
// disc_fibre_wide) and of the dust's colour within disc_fibre_colour x the angle. Taken by its shaped veil even when
// the disc network calls it picture: that network was taught round discs and calls fibres picture (on the benchmark
// +7 dust and 1 faint soft blob Claude called picture).
function fibreRescue(c, pop, p = {}) {
  if (!pop || !pop.theta || !p.disc_rescue_fibre) return false;
  if (rules(c, Object.assign({}, p, { disc_elong: p.disc_fibre_elong ?? 4, disc_elong_wide: p.disc_fibre_wide ?? 5 }))) return false;
  const u = colourDir(c);
  return !!u && angle(u, pop.dir) <= (p.disc_fibre_colour ?? 1) * pop.theta;
}

// A fibre the detector left as photo only for being soft ("not crisp enough") or tinted, in a scan with its own dust
// (pipeline: on smooth ground at its middle and corners): taken when it is crisp (the veil of its own shape would
// leave a sharp core), bright all through (no dark line inside: a scratch's halo), of the dust's colour (within twice
// the angle) and not on skin; healed like a speck. Returns "patch" or "". On 30 scans it takes 4 fibres, all dust,
// 7285's wavy fibre on the shirt among them; taking the soft ones by their veil left dark traces beside bright blobs
// (smoke340), without the smooth-ground test it took sky between branches, foliage and glints on water, and without
// the dark test a scratch's bright halo (path295).
function fibreTake(b, pop, p = {}) {
  if (!pop || !pop.theta || !p.fibre_rescue || !b || !b.tint) return "";
  if (b.ok || !/sharp core/.test(b.why) || !(b.dark <= 0.02)) return "";
  if (skinLike(b.bg) && (!p.disc_skin_sat || (b.bg[0] - b.bg[2]) / b.bg[0] >= p.disc_skin_sat)) return "";
  const u = colourDir(b);
  if (!u || angle(u, pop.dir) > (p.disc_rescue_colour ?? 2) * pop.theta) return "";
  return "patch";
}
// the detector's marks a fibre check is for: bright, left only as soft or tinted, at least 3 times as long as wide
// and at most 600 px long (longer ones are the film's edge or the picture's lines, and slow to measure)
const fibreMark = (m) => !m.disc && m.verdict === "keep-photo" && m.polarity !== "dark" && (m.kind === "dust" || m.kind === "hair/fibre")
  && /not crisp enough|soft, tinted mark/.test(String(m.why)) && m.length >= 3 * m.width && m.length <= 600;
// the radius a fibre's shaped veil is measured at (its window 3 radii, the ground 1-2 radii out)
const fibreRadius = (m) => Math.max(10, 2 * m.width, m.length / 6);

// distance (px) from a region on a grid: two chamfer passes (3-4), divided by 3
function chamfer(reg, w, h) {
  const d = new Float32Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = reg[i] ? 0 : 1e9;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let v = d[i];
      if (x > 0) v = Math.min(v, d[i - 1] + 3);
      if (y > 0) { v = Math.min(v, d[i - w] + 3); if (x > 0) v = Math.min(v, d[i - w - 1] + 4); if (x < w - 1) v = Math.min(v, d[i - w + 1] + 4); }
      d[i] = v;
    }
  }
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      let v = d[i];
      if (x < w - 1) v = Math.min(v, d[i + 1] + 3);
      if (y < h - 1) { v = Math.min(v, d[i + w] + 3); if (x < w - 1) v = Math.min(v, d[i + w + 1] + 4); if (x > 0) v = Math.min(v, d[i + w - 1] + 4); }
      d[i] = v;
    }
  }
  for (let i = 0; i < d.length; i++) d[i] /= 3;
  return d;
}

// The veil of a disc that is not round (a soft fibre, discs run together): as bump(), but the veil's region is the
// lift's own shape instead of a circle. The lift: luminance less a broad blur (3 radii), smoothed at a fifth of the
// radius; the region: what stays above a third of its peak near the centre, connected to it. The veil is taken in full
// to 0.4 radii round the region, faded out by 0.9; the ground is a band 1 to 2 radii out, fitted per channel as for a
// disc. A region that runs out of a window 13 radii across (a line of the picture), or one bigger than 30 radii
// squared, is left.
function bumpShape(img, W, H, C, d, core, outside) {
  const R = d.r, cx = d.cx, cy = d.cy, sig = Math.max(1.5, R / 5), nc = Math.min(C, 3);
  const Lw = nc === 3 ? [0.2126, 0.7152, 0.0722] : [1];
  const fail = (why) => ({ ok: false, why, peak: 0, ring: 0 });
  let half = Math.ceil(3 * R) + 2;
  for (let tries = 0; tries < 3; tries++, half = Math.ceil(half * 1.5)) {
    const x0 = Math.max(0, Math.round(cx) - half), y0 = Math.max(0, Math.round(cy) - half);
    const x1 = Math.min(W - 1, Math.round(cx) + half), y1 = Math.min(H - 1, Math.round(cy) + half);
    const w = x1 - x0 + 1, h = y1 - y0 + 1, n = w * h;
    const Y = new Float32Array(n), skip = new Uint8Array(n);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x, g = (y0 + y) * W + x0 + x;
        let v = 0;
        for (let c = 0; c < nc; c++) v += Lw[c] * img[g * C + c];
        Y[i] = v;
        if ((core && core[g]) || (outside && outside[g])) skip[i] = 1;
      }
    }
    const broad = F.gaussianBlur(Y, w, h, 3 * R);
    for (let i = 0; i < n; i++) Y[i] -= broad[i];
    const L = F.gaussianBlur(Y, w, h, sig);
    // the peak near the centre, then the region connected to it above a third of the peak
    const ccx = Math.round(cx) - x0, ccy = Math.round(cy) - y0, r7 = Math.max(2, Math.round(0.7 * R));
    let peak = 0, pi = -1;
    for (let y = Math.max(0, ccy - r7); y <= Math.min(h - 1, ccy + r7); y++) {
      for (let x = Math.max(0, ccx - r7); x <= Math.min(w - 1, ccx + r7); x++) {
        if ((x - ccx) ** 2 + (y - ccy) ** 2 <= r7 * r7 && L[y * w + x] > peak) { peak = L[y * w + x]; pi = y * w + x; }
      }
    }
    if (pi < 0) return fail("no lift at its centre");
    const reg = new Uint8Array(n), stack = [pi];
    reg[pi] = 1;
    let area = 0, touches = false;
    while (stack.length) {
      const i = stack.pop(), x = i % w, y = (i - x) / w;
      area++;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) touches = true;
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1];
      for (const j of nb) if (j >= 0 && !reg[j] && L[j] >= peak / 3) { reg[j] = 1; stack.push(j); }
    }
    if (area > 30 * R * R) return fail("too big a region for dust");
    if (touches) { if (tries < 2) continue; return fail("it runs on beyond 13 radii (a line of the picture)"); }
    const dist = chamfer(reg, w, h);
    // how wide it is (a fibre is narrower than the disc's radius): the region's half-width hw (its deepest point from
    // its edge) is ~1.48 x the spread of a Gaussian profile across it, widened by the smoothing at sig; so its own
    // spread s = sqrt((hw / 1.48)^2 - sig^2). The veil is smoothed at 0.4 s, within 1.5 .. sig: a thin fibre is not
    // taken as a broader, fainter veil (a dark fringe and a bright core left), a broad one keeps its grain
    const outsideReg = new Uint8Array(n);
    for (let i = 0; i < n; i++) outsideReg[i] = reg[i] ? 0 : 1;
    const inner = chamfer(outsideReg, w, h);
    let hw = 0;
    for (let i = 0; i < n; i++) if (reg[i] && inner[i] > hw) hw = inner[i];
    const sv = Math.min(sig, Math.max(1.5, 0.4 * Math.sqrt(Math.max(0, (hw / 1.48) ** 2 - sig * sig))));
    const ring = [];
    for (let i = 0; i < n; i++) if (!skip[i] && dist[i] >= R && dist[i] <= 2 * R) ring.push(i);
    if (ring.length < 60) return fail("too little ground around it");
    const lift = [];
    for (let c = 0; c < nc; c++) {
      const val = new Float32Array(n);
      for (let i = 0; i < n; i++) val[i] = img[((y0 + Math.floor(i / w)) * W + x0 + (i % w)) * C + c];
      let coef = null;
      const wts = new Float64Array(ring.length).fill(1);
      const fq = (i) => { const u = (x0 + (i % w) - cx) / half, v = (y0 + Math.floor(i / w) - cy) / half; return [1, u, v, u * u, u * v, v * v]; };
      for (let it = 0; it < 3; it++) {
        const A = [0, 1, 2, 3, 4, 5].map(() => [0, 0, 0, 0, 0, 0]), b = [0, 0, 0, 0, 0, 0];
        for (let j = 0; j < ring.length; j++) {
          const wj = wts[j];
          if (!wj) continue;
          const f = fq(ring[j]), v = val[ring[j]];
          for (let a = 0; a < 6; a++) { b[a] += wj * f[a] * v; for (let q = 0; q < 6; q++) A[a][q] += wj * f[a] * f[q]; }
        }
        coef = solve6(A, b);
        if (!coef) return fail("the ground does not fit");
        const res = ring.map((i) => { const f = fq(i); return val[i] - f.reduce((s, t, k) => s + t * coef[k], 0); });
        const sorted = res.map(Math.abs).sort((p, q) => p - q), s = Math.max(1, sorted[sorted.length >> 1] * 1.4826);
        for (let j = 0; j < ring.length; j++) { const t = res[j] / (4 * s); wts[j] = Math.abs(t) < 1 ? (1 - t * t) ** 2 : 0; }
      }
      const resid = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        if (skip[i] || dist[i] > 2 * R) continue;
        const f = fq(i);
        resid[i] = val[i] - f.reduce((s, t, k) => s + t * coef[k], 0);
      }
      // a plane over the same ground with the same weights: where the quadratic bends away from it under the region,
      // it is extrapolating a glow or a gradient's curve, not the ground (checked below)
      const A3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b3 = [0, 0, 0];
      for (let j = 0; j < ring.length; j++) {
        const wj = wts[j];
        if (!wj) continue;
        const f = fq(ring[j]), v = val[ring[j]], g = [1, f[1], f[2]];
        for (let a = 0; a < 3; a++) { b3[a] += wj * g[a] * v; for (let q = 0; q < 3; q++) A3[a][q] += wj * g[a] * g[q]; }
      }
      const plane = solve3(A3, b3) || [coef[0], coef[1], coef[2]];
      lift.push({ resid, veil: F.gaussianBlur(resid, w, h, sv), ground: coef[0], coef, plane, fq });
    }
    const taper = new Float32Array(n);
    for (let i = 0; i < n; i++) taper[i] = skip[i] ? 0 : dist[i] <= 0.4 * R ? 1 : dist[i] >= 0.9 * R ? 0 : 0.5 + 0.5 * Math.cos(((dist[i] - 0.4 * R) / (0.5 * R)) * Math.PI);
    const Ll = new Float32Array(n), El = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < nc; c++) { Ll[i] += Lw[c] * lift[c].veil[i]; El[i] += Lw[c] * (lift[c].resid[i] - taper[i] * lift[c].veil[i]); }
    }
    let top = -Infinity, ss = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
      if (reg[i] && Ll[i] > top) top = Ll[i];
      if (!skip[i] && dist[i] >= R && dist[i] <= 2 * R) { ss += Ll[i] * Ll[i]; cnt++; }
    }
    let bx0 = w, by0 = h, bx1 = 0, by1 = 0;
    for (let i = 0; i < n; i++) if (reg[i]) { const x = i % w, y = (i - x) / w; bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x); by0 = Math.min(by0, y); by1 = Math.max(by1, y); }
    const out = {
      x0, y0, w, h, C, data: new Float32Array(n * C), peak: Math.round(top), ring: Math.round(Math.sqrt(ss / Math.max(1, cnt))), ok: false, why: "", shaped: true, area,
      sv: Math.round(sv * 10) / 10, box: [x0 + bx0, y0 + by0, x0 + bx1, y0 + by1],
    };
    // its colour (the mean lift over the region per channel against that channel's ground at the centre, as a
    // candidate's tint and bg) and its elongation (second moments of the region)
    const tint = [0, 0, 0], bg = [0, 0, 0];
    let m00 = 0, mx = 0, my = 0;
    for (let i = 0; i < n; i++) if (reg[i]) { const x = i % w, y = (i - x) / w; m00++; mx += x; my += y; for (let c = 0; c < 3; c++) tint[c] += lift[Math.min(c, nc - 1)].veil[i]; }
    mx /= m00; my /= m00;
    let a = 0, b = 0, cc = 0;
    for (let i = 0; i < n; i++) if (reg[i]) { const x = i % w - mx, y = (i - (i % w)) / w - my; a += x * x; b += x * y; cc += y * y; }
    const tr = (a + cc) / m00, det = (a * cc - b * b) / (m00 * m00), dd = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    out.elong = Math.round(Math.sqrt((tr / 2 + dd) / Math.max(1e-9, tr / 2 - dd)) * 100) / 100;
    for (let c = 0; c < 3; c++) { tint[c] = Math.round(tint[c] / m00); bg[c] = Math.round((lift[Math.min(c, nc - 1)].ground / 65535) * 1e4) / 1e4; }
    out.tint = tint; out.bg = bg;
    // the fibre with its halo (the lift smoothed only 1.5 px above a sixth of its peak, inside the region), for a
    // patch heal when the veil would leave a sharp core
    const Lf = F.gaussianBlur(Y, w, h, 1.5);
    let pf = 0;
    for (let i = 0; i < n; i++) if (reg[i] && Lf[i] > pf) pf = Lf[i];
    out.core = new Uint8Array(n);
    for (let i = 0; i < n; i++) if (reg[i] && Lf[i] >= pf / 6) out.core[i] = 1;
    // dark detail in or beside it (a scratch's dark line by its bright halo): pixels within 0.4 radii of the region
    // whose fine lift falls below 4 x the band's spread under the ground
    const bandF = [];
    for (let i = 0; i < n; i++) if (!skip[i] && dist[i] >= R && dist[i] <= 2 * R) bandF.push(Math.abs(Lf[i]));
    bandF.sort((p, q) => p - q);
    const sF = Math.max(1, (bandF[bandF.length >> 1] || 0) * 1.4826);
    let darkN = 0, regN = 0;
    for (let i = 0; i < n; i++) if (!skip[i] && dist[i] <= 0.4 * R) { regN++; if (Lf[i] < -4 * sF) darkN++; }
    out.dark = Math.round((darkN / Math.max(1, regN)) * 1000) / 1000;
    if (!(top > 3 * Math.max(out.ring, 4))) { out.why = "no veil stands clearly above the ground"; return out; }
    // the ground's bend under the region: the quadratic less the plane, mean over the region (luminance); more than a
    // third of the veil's peak either way and the ground is not what the ring says (a faint big disc on a glow came
    // out as a dark blob)
    let bend = 0, bn = 0;
    for (let i = 0; i < n; i++) {
      if (!reg[i]) continue;
      let d = 0;
      for (let c = 0; c < nc; c++) { const L = lift[c], f = L.fq(i); d += Lw[c] * (f.reduce((s2, t, k) => s2 + t * L.coef[k], 0) - (L.plane[0] + L.plane[1] * f[1] + L.plane[2] * f[2])); }
      bend += d; bn++;
    }
    out.bend = Math.round((bend / Math.max(1, bn) / Math.max(1, top)) * 100) / 100;
    if (Math.abs(out.bend) > 0.33) { out.why = "the ground bends under it (a glow or a gradient's curve)"; return out; }
    // the veil
    for (let i = 0; i < n; i++) {
      if (!taper[i]) continue;
      for (let c = 0; c < C; c++) out.data[i * C + c] = taper[i] * lift[Math.min(c, nc - 1)].veil[i];
    }
    // what the veil leaves must be grain: after it, fine detail (1.5 px) inside the region beyond the band's own
    // 99th percentile on more than 2% of it is a sharp core (a hair in focus: the detector's to heal)
    const Eb = F.gaussianBlur(El, w, h, 1.5), band = [];
    for (let i = 0; i < n; i++) if (!skip[i] && dist[i] >= R && dist[i] <= 2 * R) band.push(Math.abs(Eb[i]));
    band.sort((p, q) => p - q);
    const p99 = band[Math.floor(0.99 * (band.length - 1))];
    let over = 0, cnt2 = 0;
    for (let i = 0; i < n; i++) if (!skip[i] && dist[i] <= 0.4 * R) { cnt2++; if (Math.abs(Eb[i]) > p99) over++; }
    out.left = Math.round((over / Math.max(1, cnt2)) * 1000) / 1000;
    if (out.left > 0.02) { out.why = "a sharp core would stay (a hair in focus: the detector's)"; out.data.fill(0); return out; }
    out.ok = true;
    return out;
  }
  return fail("it runs on beyond 13 radii (a line of the picture)");
}

// How much of veil b (where it takes away 32+ of 65535 in its first channel) another taken veil already covers there:
// discs run together are found as two candidates whose shaped veils are one shape, and taking both took it twice
function overlap(b, others) {
  let own = 0, both = 0;
  for (let y = 0; y < b.h; y++) {
    for (let x = 0; x < b.w; x++) {
      if (Math.abs(b.data[(y * b.w + x) * b.C]) < 32) continue;
      own++;
      const gx = b.x0 + x, gy = b.y0 + y;
      for (const e of others) {
        const ex = gx - e.x0, ey = gy - e.y0;
        if (ex < 0 || ey < 0 || ex >= e.w || ey >= e.h) continue;
        if (Math.abs(e.data[(ey * e.w + ex) * e.C]) >= 32) { both++; break; }
      }
    }
  }
  return own ? both / own : 0;
}

// a mark for a disc (working-size box from its radius)
function mark(c, verdict, why) {
  const r = Math.round(c.r), cx = Math.round(c.cx), cy = Math.round(c.cy);
  return {
    id: -3, kind: "soft", cx, cy, x0: cx - r, x1: cx + r, y0: cy - r, y1: cy + r, area: Math.round(Math.PI * c.r * c.r), z: c.z,
    learned: null, zone: false, polarity: "bright", pass: 1, verdict, why, disc: c,
  };
}

module.exports = {
  find, rules, netNeeded, nearMiss, mark, views, score, bump, bumpFit, bumpShape, subtract, subtractFull, quarter, colourDir, population, rescue,
  fibreRescue, fibreTake, fibreMark, fibreRadius, overlap,
  SIGMAS, RADIUS_PER_SIGMA, Q, hasNet: () => !!NETD,
};

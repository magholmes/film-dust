// The functions JS calls, written once for both hosts. The including file defines:
#include <cstring>
#include <vector>
//   ENV, VAL, CBINFO       the host's handle types
//   ARGS(env, info, n)     fills `VAL argv[n]` and `size_t argc`
//   AB(env, v, off, T, n)  -> T* into an ArrayBuffer at byte offset `off`, checked for n items (or null)
//   I32(env, v), F64(env, v), BOOL(env, v)
//   NEW_AB(env, bytes, &ptr) -> VAL (a new ArrayBuffer; ptr receives its memory)
//   NUM(env, x) -> VAL, THROW(env, msg)
// Images travel as (ArrayBuffer, byteOffset) pairs: UXP's addon API has no typed-array accessor.

#define FD_FN(name) static VAL name(ENV env, CBINFO info)

#define FD_IMG_IN(idx, ptr, W, H)                                              \
    const float* ptr = AB(env, argv[idx], (size_t)F64(env, argv[idx + 1]), float, (size_t)(W) * (H)); \
    if (!ptr) { THROW(env, "film dust native: image buffer too small"); return nullptr; }

FD_FN(fd_median8f) {
    ARGS(env, info, 5);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]), k = I32(env, argv[4]);
    FD_IMG_IN(0, src, w, h);
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)w * h * 4, &out);
    fd::median8f(src, w, h, k, (float*)out);
    return res;
}

FD_FN(fd_ellipse) {   // (buf, off, w, h, k, hwBuf, hwOff, mode) mode: 0 erode, 1 dilate, 2 open, 3 close
    ARGS(env, info, 8);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]), k = I32(env, argv[4]), mode = I32(env, argv[7]);
    FD_IMG_IN(0, src, w, h);
    const int32_t* hw = AB(env, argv[5], (size_t)F64(env, argv[6]), int32_t, (size_t)k);
    if (!hw) { THROW(env, "film dust native: bad ellipse widths"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)w * h * 4, &out);
    if (mode < 2) fd::ellipse_morph(src, w, h, k, hw, mode == 1, (float*)out);
    else fd::ellipse_open_close(src, w, h, k, hw, mode == 2, (float*)out);
    return res;
}

FD_FN(fd_sep_conv) {   // (buf, off, w, h, kerBuf, kerOff, klen)
    ARGS(env, info, 7);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]), klen = I32(env, argv[6]);
    FD_IMG_IN(0, src, w, h);
    const float* ker = AB(env, argv[4], (size_t)F64(env, argv[5]), float, (size_t)klen);
    if (!ker) { THROW(env, "film dust native: bad kernel"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)w * h * 4, &out);
    fd::sep_conv(src, w, h, ker, klen, (float*)out);
    return res;
}

FD_FN(fd_box3) {   // (buf, off, w, h, radiiBuf, radiiOff, n)
    ARGS(env, info, 7);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]), n = I32(env, argv[6]);
    FD_IMG_IN(0, src, w, h);
    const int32_t* radii = AB(env, argv[4], (size_t)F64(env, argv[5]), int32_t, (size_t)n);
    if (!radii) { THROW(env, "film dust native: bad radii"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)w * h * 4, &out);
    fd::box3(src, w, h, radii, n, (float*)out);
    return res;
}

FD_FN(fd_box_blur) {   // (buf, off, w, h, k)
    ARGS(env, info, 5);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]), k = I32(env, argv[4]);
    FD_IMG_IN(0, src, w, h);
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)w * h * 4, &out);
    fd::box_blur(src, w, h, k, (float*)out);
    return res;
}

FD_FN(fd_sobel) {   // (buf, off, w, h, dx, dy)
    ARGS(env, info, 6);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]), dx = I32(env, argv[4]), dy = I32(env, argv[5]);
    FD_IMG_IN(0, src, w, h);
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)w * h * 4, &out);
    fd::sobel(src, w, h, dx, dy, (float*)out);
    return res;
}

FD_FN(fd_net_load) {   // (buf, off, nDoubles, towers, hidden, slot) -> 1 when the weights fit
    ARGS(env, info, 6);
    const size_t n = (size_t)F64(env, argv[2]);
    const double* w = AB(env, argv[0], (size_t)F64(env, argv[1]), double, n);
    if (!w) { THROW(env, "film dust native: bad weights"); return nullptr; }
    return NUM(env, fd::net_load(I32(env, argv[5]), w, n, I32(env, argv[3]), I32(env, argv[4])) ? 1.0 : 0.0);
}

FD_FN(fd_net_forward) {   // (buf, off, N, slot) -> ArrayBuffer of N float64 logits
    ARGS(env, info, 4);
    const int N = I32(env, argv[2]), slot = I32(env, argv[3]);
    const int T = fd::net_towers(slot);
    if (T <= 0) { THROW(env, "film dust native: no network loaded"); return nullptr; }
    const double* in = AB(env, argv[0], (size_t)F64(env, argv[1]), double, (size_t)N * T * 4800);
    if (!in) { THROW(env, "film dust native: input buffer too small"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)N * 8, &out);
    fd::net_forward(slot, in, N, (double*)out);
    return res;
}

FD_FN(fd_net_forward_at) {   // (img u16, off, W, H, C, centres i32, off, N, bright, slot) -> N float64 logits
    ARGS(env, info, 10);
    const int W = I32(env, argv[2]), H = I32(env, argv[3]), C = I32(env, argv[4]), N = I32(env, argv[7]), slot = I32(env, argv[9]);
    if (fd::net_towers(slot) <= 0) { THROW(env, "film dust native: no network loaded"); return nullptr; }
    const uint16_t* img = AB(env, argv[0], (size_t)F64(env, argv[1]), uint16_t, (size_t)W * H * C);
    const int32_t* cen = AB(env, argv[5], (size_t)F64(env, argv[6]), int32_t, (size_t)2 * N);
    if (!img || !cen || N <= 0) { THROW(env, "film dust native: bad centres"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)N * 8, &out);
    fd::net_forward_at(slot, img, W, H, C, cen, N, I32(env, argv[8]) != 0, (double*)out);
    return res;
}

FD_FN(fd_hypot_into) {   // (gy f32, off, gx f32, off, n): gy = hypot(gx, gy) in place
    ARGS(env, info, 5);
    const size_t n = (size_t)F64(env, argv[4]);
    float* gy = AB(env, argv[0], (size_t)F64(env, argv[1]), float, n);
    const float* gx = AB(env, argv[2], (size_t)F64(env, argv[3]), float, n);
    if (!gy || !gx) { THROW(env, "film dust native: bad gradients"); return nullptr; }
    fd::hypot_into(gy, gx, (int64_t)n);
    return NUM(env, 1.0);
}

FD_FN(fd_blob_tiles) {   // (Y, off, sl, off, W, H, T, sq, sgn, ker f32, off, klen, k, mg, tiles i32, off, n)
    // -> one buffer: n counts, then per tile its ys, xs, vals (count each) and its inner response
    ARGS(env, info, 17);
    const int W = I32(env, argv[4]), H = I32(env, argv[5]), T = I32(env, argv[6]), klen = I32(env, argv[11]);
    const int k = I32(env, argv[12]), mg = I32(env, argv[13]), n = I32(env, argv[16]);
    FD_IMG_IN(0, Y, W, H);
    FD_IMG_IN(2, sl, W, H);
    const float* ker = AB(env, argv[9], (size_t)F64(env, argv[10]), float, (size_t)klen);
    const int32_t* tl = AB(env, argv[14], (size_t)F64(env, argv[15]), int32_t, (size_t)2 * n);
    if (!ker || !tl || n <= 0 || T <= 0) { THROW(env, "film dust native: bad blob tiles"); return nullptr; }
    std::vector<fd::BlobTile> tiles;
    fd::blob_tiles(Y, sl, W, H, T, F64(env, argv[7]), I32(env, argv[8]), ker, klen, k, mg, tl, n, tiles);
    size_t words = (size_t)n;
    for (const auto& t : tiles) words += 3 * t.ys.size() + t.inner.size();
    void* out = nullptr;
    VAL res = NEW_AB(env, words * 4, &out);
    int32_t* o = (int32_t*)out;
    for (int i = 0; i < n; i++) o[i] = (int32_t)tiles[i].ys.size();
    size_t p = (size_t)n;
    for (const auto& t : tiles) {
        const size_t c = t.ys.size();
        if (c) {
            std::memcpy(o + p, t.ys.data(), c * 4); p += c;
            std::memcpy(o + p, t.xs.data(), c * 4); p += c;
            std::memcpy(o + p, t.vals.data(), c * 4); p += c;
        }
        std::memcpy(o + p, t.inner.data(), t.inner.size() * 4); p += t.inner.size();
    }
    return res;
}

FD_FN(fd_skeletonize) {   // (mask u8, off, w, h) -> u8 skeleton
    ARGS(env, info, 4);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]);
    const uint8_t* m = AB(env, argv[0], (size_t)F64(env, argv[1]), uint8_t, (size_t)w * h);
    if (!m || w <= 0 || h <= 0) { THROW(env, "film dust native: bad mask"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)w * h, &out);
    fd::skeletonize(m, w, h, (uint8_t*)out);
    return res;
}

FD_FN(fd_block_stats) {   // (resid, off, valid, voff, W, H, block, mode, bright) -> f32 grid hb x wb
    ARGS(env, info, 9);
    const int W = I32(env, argv[4]), H = I32(env, argv[5]), block = I32(env, argv[6]), mode = I32(env, argv[7]);
    const bool bright = I32(env, argv[8]) != 0;
    if (block <= 0 || W < block || H < block) { THROW(env, "film dust native: image smaller than a block"); return nullptr; }
    FD_IMG_IN(0, resid, W, H);
    const uint8_t* valid = AB(env, argv[2], (size_t)F64(env, argv[3]), uint8_t, (size_t)W * H);
    if (!valid) { THROW(env, "film dust native: bad mask"); return nullptr; }
    const int hb = W > 0 && H / block > 0 ? H / block : 1, wb = W / block > 0 ? W / block : 1;
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)hb * wb * 4, &out);
    fd::block_stats(resid, valid, W, H, block, mode, bright, (float*)out);
    return res;
}

FD_FN(fd_noise_by_level) {   // (bg, off, resid, roff, valid, voff, W, H, bins) -> f32 W x H
    ARGS(env, info, 9);
    const int W = I32(env, argv[6]), H = I32(env, argv[7]), bins = I32(env, argv[8]);
    FD_IMG_IN(0, bg, W, H);
    FD_IMG_IN(2, resid, W, H);
    const uint8_t* valid = AB(env, argv[4], (size_t)F64(env, argv[5]), uint8_t, (size_t)W * H);
    if (!valid) { THROW(env, "film dust native: bad mask"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)W * H * 4, &out);
    fd::noise_by_level(bg, resid, valid, W, H, bins, (float*)out);
    return res;
}

FD_FN(fd_z_maps) {   // (Y, m15, o15, m41, o41, bxl, sigma: 7 x (buf, off); outside, ooff; N; bright) -> f32 3N
    ARGS(env, info, 18);
    const size_t N = (size_t)F64(env, argv[16]);
    const bool bright = I32(env, argv[17]) != 0;
    const float* in[7];
    for (int i = 0; i < 7; i++) {
        in[i] = AB(env, argv[2 * i], (size_t)F64(env, argv[2 * i + 1]), float, N);
        if (!in[i]) { THROW(env, "film dust native: map buffer too small"); return nullptr; }
    }
    const uint8_t* outside = AB(env, argv[14], (size_t)F64(env, argv[15]), uint8_t, N);
    if (!outside) { THROW(env, "film dust native: bad mask"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, N * 3 * 4, &out);
    float* o = (float*)out;
    fd::z_maps(in[0], in[1], in[2], in[3], in[4], in[5], in[6], outside, (int64_t)N, bright, o, o + N, o + 2 * N);
    return res;
}

FD_FN(fd_np_gradient) {   // (buf, off, w, h) -> f32 2N (gy, then gx)
    ARGS(env, info, 4);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]);
    if (w < 2 || h < 2) { THROW(env, "film dust native: gradient needs 2 x 2"); return nullptr; }
    FD_IMG_IN(0, src, w, h);
    const size_t N = (size_t)w * h;
    void* out = nullptr;
    VAL res = NEW_AB(env, N * 2 * 4, &out);
    fd::np_gradient(src, w, h, (float*)out, (float*)out + N);
    return res;
}

FD_FN(fd_resize_linear) {   // (buf, off, sw, sh, dw, dh) -> f32 dw x dh
    ARGS(env, info, 6);
    const int sw = I32(env, argv[2]), sh = I32(env, argv[3]), dw = I32(env, argv[4]), dh = I32(env, argv[5]);
    FD_IMG_IN(0, src, sw, sh);
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)dw * dh * 4, &out);
    fd::resize_linear(src, sw, sh, dw, dh, (float*)out);
    return res;
}

FD_FN(fd_label) {   // (buf, off, w, h, conn) -> Int32 w*h labels + 1 (the count, last)
    ARGS(env, info, 5);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]), conn = I32(env, argv[4]);
    const uint8_t* m = AB(env, argv[0], (size_t)F64(env, argv[1]), uint8_t, (size_t)w * h);
    if (!m) { THROW(env, "film dust native: bad mask"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, ((size_t)w * h + 1) * 4, &out);
    int32_t* lab = (int32_t*)out;
    lab[(size_t)w * h] = fd::label(m, w, h, conn, lab);
    return res;
}

FD_FN(fd_bin_morph) {   // (buf, off, w, h, k, hwBuf, hwOff, mode) mode 0 rect erode, 1 rect dilate, 2 ellipse erode, 3 ellipse dilate
    ARGS(env, info, 8);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]), k = I32(env, argv[4]), mode = I32(env, argv[7]);
    const uint8_t* m = AB(env, argv[0], (size_t)F64(env, argv[1]), uint8_t, (size_t)w * h);
    if (!m) { THROW(env, "film dust native: bad mask"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)w * h, &out);
    if (mode < 2) {
        fd::bin_rect(m, w, h, k, mode == 1, (uint8_t*)out);
    } else {
        const int32_t* hw = AB(env, argv[5], (size_t)F64(env, argv[6]), int32_t, (size_t)k);
        if (!hw) { THROW(env, "film dust native: bad ellipse widths"); return nullptr; }
        fd::ellipse_morph_u8(m, w, h, k, hw, mode == 3, (uint8_t*)out);
    }
    return res;
}

FD_FN(fd_outside_mask) {   // (Y, off, W, H) -> Uint8 W*H
    ARGS(env, info, 4);
    const int W = I32(env, argv[2]), H = I32(env, argv[3]);
    FD_IMG_IN(0, Y, W, H);
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)W * H, &out);
    fd::outside_mask(Y, W, H, (uint8_t*)out);
    return res;
}

FD_FN(fd_ridge_maps) {   // (Y, off, outside, ooff, sig, soff, W, H, bright, scales f64, sOff, kers f32, kOff, klens i32, lOff, ns) -> f32 2N (best, bestS)
    ARGS(env, info, 16);
    const int W = I32(env, argv[6]), H = I32(env, argv[7]), ns = I32(env, argv[15]);
    const bool bright = I32(env, argv[8]) != 0;
    FD_IMG_IN(0, Y, W, H);
    const uint8_t* outside = AB(env, argv[2], (size_t)F64(env, argv[3]), uint8_t, (size_t)W * H);
    FD_IMG_IN(4, sig, W, H);
    const double* scales = AB(env, argv[9], (size_t)F64(env, argv[10]), double, (size_t)ns);
    const int32_t* klens = AB(env, argv[13], (size_t)F64(env, argv[14]), int32_t, (size_t)ns);
    if (!outside || !scales || !klens) { THROW(env, "film dust native: bad ridge arguments"); return nullptr; }
    size_t total = 0;
    for (int i = 0; i < ns; i++) total += (size_t)klens[i];
    const float* kers = AB(env, argv[11], (size_t)F64(env, argv[12]), float, total);
    if (!kers) { THROW(env, "film dust native: bad kernels"); return nullptr; }
    const size_t N = (size_t)W * H;
    void* out = nullptr;
    VAL res = NEW_AB(env, N * 2 * 4, &out);
    fd::ridge_maps(Y, W, H, outside, bright, sig, ns, scales, kers, klens, (float*)out, (float*)out + N);
    return res;
}

FD_FN(fd_rect_morph) {   // (buf, off, w, h, k, isMax) -> f32
    ARGS(env, info, 6);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]), k = I32(env, argv[4]);
    FD_IMG_IN(0, src, w, h);
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)w * h * 4, &out);
    fd::rect_morph(src, w, h, k, I32(env, argv[5]) != 0, (float*)out);
    return res;
}

FD_FN(fd_average_strip) {   // (src, off, eight, srcW, rows, C, k, dst, doff, Wd, yd0) -> rows written (dst filled in place)
    ARGS(env, info, 11);
    const bool eight = I32(env, argv[2]) != 0;
    const int srcW = I32(env, argv[3]), rows = I32(env, argv[4]), C = I32(env, argv[5]), k = I32(env, argv[6]);
    const int Wd = I32(env, argv[9]), yd0 = I32(env, argv[10]);
    const size_t n = (size_t)srcW * rows * C;
    const void* src = eight ? (const void*)AB(env, argv[0], (size_t)F64(env, argv[1]), uint8_t, n)
                            : (const void*)AB(env, argv[0], (size_t)F64(env, argv[1]), uint16_t, n);
    uint16_t* dst = AB(env, argv[7], (size_t)F64(env, argv[8]), uint16_t, (size_t)(yd0 + rows / k) * Wd * C);
    if (!src || !dst) { THROW(env, "film dust native: bad strip"); return nullptr; }
    return NUM(env, (double)fd::average_strip(src, eight, srcW, rows, C, k, dst, Wd, yd0));
}

FD_FN(fd_fill_holes) {   // (buf, off, w, h) -> Uint8 w*h
    ARGS(env, info, 4);
    const int w = I32(env, argv[2]), h = I32(env, argv[3]);
    const uint8_t* m = AB(env, argv[0], (size_t)F64(env, argv[1]), uint8_t, (size_t)w * h);
    if (!m) { THROW(env, "film dust native: bad mask"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)w * h, &out);
    fd::fill_holes(m, w, h, (uint8_t*)out);
    return res;
}

// ellipse half-width tables for sizes 0..127: hwPack (int32, all tables back to back) and hwStart[k] (-1 = none)
struct FdHws { const int32_t* t[128] = {}; };
static bool fd_hws(ENV env, VAL pack, VAL poff, VAL start, VAL soff, FdHws& out) {
    const int32_t* st = AB(env, start, (size_t)F64(env, soff), int32_t, 128);
    if (!st) return false;
    size_t need = 0;
    for (int k = 0; k < 128; k++) if (st[k] >= 0 && (size_t)st[k] + (size_t)k > need) need = (size_t)st[k] + (size_t)k;
    const int32_t* pk = need ? AB(env, pack, (size_t)F64(env, poff), int32_t, need) : nullptr;
    if (need && !pk) return false;
    for (int k = 0; k < 128; k++) out.t[k] = st[k] >= 0 ? pk + st[k] : nullptr;
    return true;
}

// (Y, off, sig, off, glab, off, cand, off, W, H, n, boxes, off, ids, off, bright, k1, off, k1n, hwPack, off, hwStart, off,
//  dk, offs f64, off) -> one ArrayBuffer: solid bytes (offs[n]) | objd bytes | pad to 8 | how i32 n | area i32 n | width f64 n
FD_FN(fd_solid_batch) {
    ARGS(env, info, 26);
    const int W = I32(env, argv[8]), H = I32(env, argv[9]), n = I32(env, argv[10]);
    FD_IMG_IN(0, Y, W, H);
    FD_IMG_IN(2, sig, W, H);
    const int32_t* glab = AB(env, argv[4], (size_t)F64(env, argv[5]), int32_t, (size_t)W * H);
    const uint8_t* cand = AB(env, argv[6], (size_t)F64(env, argv[7]), uint8_t, (size_t)W * H);
    const int32_t* boxes = AB(env, argv[11], (size_t)F64(env, argv[12]), int32_t, (size_t)4 * n);
    const int32_t* ids = AB(env, argv[13], (size_t)F64(env, argv[14]), int32_t, (size_t)n);
    const int k1n = I32(env, argv[18]);
    const float* k1 = AB(env, argv[16], (size_t)F64(env, argv[17]), float, (size_t)k1n);
    const double* offd = AB(env, argv[24], (size_t)F64(env, argv[25]), double, (size_t)n + 1);
    FdHws hw;
    if (!glab || !cand || !boxes || !ids || !k1 || !offd || !fd_hws(env, argv[19], argv[20], argv[21], argv[22], hw)) { THROW(env, "film dust native: bad solid batch"); return nullptr; }
    std::vector<int64_t> offs((size_t)n + 1);
    for (int i = 0; i <= n; i++) offs[i] = (int64_t)offd[i];
    const size_t tot = (size_t)offs[n], pad = (8 - (2 * tot) % 8) % 8;
    void* out = nullptr;
    VAL res = NEW_AB(env, 2 * tot + pad + (size_t)n * 16, &out);
    uint8_t* o = (uint8_t*)out;
    fd::solid_batch(Y, sig, glab, cand, W, H, n, boxes, ids, I32(env, argv[15]) != 0, k1, k1n, hw.t, I32(env, argv[23]), offs.data(),
                    o, o + tot, (int32_t*)(o + 2 * tot + pad), (int32_t*)(o + 2 * tot + pad + (size_t)n * 4), (double*)(o + 2 * tot + pad + (size_t)n * 8));
    return res;
}

// (Y, off, G3, off, E25, off, E1, off, hasE1, sig, off, cand, off, cgi, off, cidx, off, nc, W, H, n, boxes, off, offs f64, off,
//  objd, off, bright, gradCal, k1, off, k1n, k3, off, k3n, hwPack, off, hwStart, off) -> f64 n x 10
FD_FN(fd_cues_batch) {
    ARGS(env, info, 40);
    const int nc = I32(env, argv[17]);
    const int W = I32(env, argv[18]), H = I32(env, argv[19]), n = I32(env, argv[20]);
    FD_IMG_IN(0, Y, W, H);
    FD_IMG_IN(2, G3, W, H);
    FD_IMG_IN(4, E25, W, H);
    const bool hasE1 = I32(env, argv[8]) != 0;
    const float* E1 = hasE1 ? AB(env, argv[6], (size_t)F64(env, argv[7]), float, (size_t)W * H) : nullptr;
    FD_IMG_IN(9, sig, W, H);
    const uint8_t* cand = AB(env, argv[11], (size_t)F64(env, argv[12]), uint8_t, (size_t)W * H);
    const int32_t* cgi = AB(env, argv[13], (size_t)F64(env, argv[14]), int32_t, (size_t)std::max(1, nc));
    const int32_t* cidx = AB(env, argv[15], (size_t)F64(env, argv[16]), int32_t, (size_t)std::max(1, nc));
    const int32_t* boxes = AB(env, argv[21], (size_t)F64(env, argv[22]), int32_t, (size_t)4 * n);
    const double* offd = AB(env, argv[23], (size_t)F64(env, argv[24]), double, (size_t)n + 1);
    const int k1n = I32(env, argv[31]), k3n = I32(env, argv[34]);
    const float* k1 = AB(env, argv[29], (size_t)F64(env, argv[30]), float, (size_t)k1n);
    const float* k3 = AB(env, argv[32], (size_t)F64(env, argv[33]), float, (size_t)k3n);
    FdHws hw;
    if ((hasE1 && !E1) || !cand || !cgi || !cidx || !boxes || !offd || !k1 || !k3 || !fd_hws(env, argv[35], argv[36], argv[37], argv[38], hw)) { THROW(env, "film dust native: bad cues batch"); return nullptr; }
    std::vector<int64_t> offs((size_t)n + 1);
    for (int i = 0; i <= n; i++) offs[i] = (int64_t)offd[i];
    const uint8_t* objd = AB(env, argv[25], (size_t)F64(env, argv[26]), uint8_t, (size_t)offs[n]);
    if (!objd) { THROW(env, "film dust native: bad cues masks"); return nullptr; }
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)n * 10 * 8, &out);
    fd::cues_batch(Y, G3, E25, E1, sig, cand, cgi, cidx, nc, W, H, n, boxes, offs.data(), objd, I32(env, argv[27]) != 0, F64(env, argv[28]),
                   k1, k1n, k3, k3n, hw.t, (double*)out);
    return res;
}

// (Y, off, sig, off, img u16, off, C, Ci, W, H, bright, n, seeds i32, off, k15, off, k15n, k08, off, k08n, k25, off, k25n,
//  softFill, socketOn) -> fixed f64 n x 24 | offsets i32 3n + 1 (padded to 8) | lists i32
FD_FN(fd_soft_batch) {
    ARGS(env, info, 27);
    const int C = I32(env, argv[6]), Ci = I32(env, argv[7]), W = I32(env, argv[8]), H = I32(env, argv[9]), n = I32(env, argv[11]);
    FD_IMG_IN(0, Y, W, H);
    FD_IMG_IN(2, sig, W, H);
    const uint16_t* img = AB(env, argv[4], (size_t)F64(env, argv[5]), uint16_t, (size_t)W * H * C);
    const int32_t* seeds = AB(env, argv[12], (size_t)F64(env, argv[13]), int32_t, (size_t)7 * n);
    const int k15n = I32(env, argv[16]), k08n = I32(env, argv[19]), k25n = I32(env, argv[22]);
    const float* k15 = AB(env, argv[14], (size_t)F64(env, argv[15]), float, (size_t)k15n);
    const float* k08 = AB(env, argv[17], (size_t)F64(env, argv[18]), float, (size_t)k08n);
    const float* k25 = AB(env, argv[20], (size_t)F64(env, argv[21]), float, (size_t)k25n);
    if (!img || !seeds || !k15 || !k08 || !k25 || n <= 0) { THROW(env, "film dust native: bad soft batch"); return nullptr; }
    std::vector<double> fixed((size_t)n * 24);
    std::vector<std::vector<int64_t>> lists;
    fd::soft_batch(Y, sig, img, C, Ci, W, H, I32(env, argv[10]) != 0, n, seeds, k15, k15n, k08, k08n, k25, k25n,
                   F64(env, argv[23]), I32(env, argv[24]) != 0, fixed.data(), lists);
    size_t total = 0;
    for (const auto& l : lists) total += l.size();
    const size_t no = (size_t)3 * n + 1, offBytes = ((no * 4 + 7) / 8) * 8;
    void* out = nullptr;
    VAL res = NEW_AB(env, (size_t)n * 24 * 8 + offBytes + total * 4, &out);
    uint8_t* o = (uint8_t*)out;
    std::memcpy(o, fixed.data(), (size_t)n * 24 * 8);
    int32_t* offs = (int32_t*)(o + (size_t)n * 24 * 8);
    int32_t* data = (int32_t*)(o + (size_t)n * 24 * 8 + offBytes);
    size_t q = 0;
    for (size_t i = 0; i < lists.size(); i++) {
        offs[i] = (int32_t)q;
        for (int64_t g : lists[i]) data[q++] = (int32_t)g;
    }
    offs[lists.size()] = (int32_t)q;
    return res;
}

FD_FN(fd_threads) {   // ([n]) -> threads in use (sets them when n > 0)
    ARGS(env, info, 1);
    if (argc >= 1) {
        const int n = I32(env, argv[0]);
        if (n > 0) fd::set_threads(n);
    }
    return NUM(env, (double)fd::threads());
}

FD_FN(fd_version) {
    (void)info;
    return NUM(env, 1.0);
}

struct FdEntry { const char* name; VAL (*fn)(ENV, CBINFO); };
static const FdEntry FD_FUNCTIONS[] = {
    {"median8f", fd_median8f}, {"ellipse", fd_ellipse}, {"sepConv", fd_sep_conv}, {"box3", fd_box3},
    {"boxBlur", fd_box_blur}, {"sobel", fd_sobel}, {"threads", fd_threads}, {"version", fd_version},
    {"netLoad", fd_net_load}, {"netForward", fd_net_forward}, {"netForwardAt", fd_net_forward_at}, {"hypotInto", fd_hypot_into}, {"blobTiles", fd_blob_tiles}, {"skeletonize", fd_skeletonize}, {"blockStats", fd_block_stats},
    {"noiseByLevel", fd_noise_by_level}, {"zMaps", fd_z_maps}, {"npGradient", fd_np_gradient},
    {"resizeLinear", fd_resize_linear}, {"label", fd_label}, {"binMorph", fd_bin_morph},
    {"outsideMask", fd_outside_mask}, {"ridgeMaps", fd_ridge_maps},
    {"rectMorph", fd_rect_morph}, {"averageStrip", fd_average_strip}, {"fillHoles", fd_fill_holes},
    {"solidBatch", fd_solid_batch}, {"cuesBatch", fd_cues_batch}, {"softBatch", fd_soft_batch},
};

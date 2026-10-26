// The functions JS calls, written once for both hosts. The including file defines namespace host: its value
// types (Env, Val, Info) and a handful of calls (get_args, arraybuffer, get_i32, get_f64, get_named, get_element,
// new_arraybuffer, number, throw_error). Images travel as (ArrayBuffer, byteOffset) pairs: UXP's addon API has no
// typed-array accessor. Functions with a few arguments take them in order; the ones with many take one object of
// named fields (Opts).
// Every function runs inside guarded(): a bad argument or memory running out throws a C++ exception, which comes
// back to JS as an error (engine/native.js then switches native off and the JS filters carry on). Nothing
// thrown here can reach the host, so a failure never takes Photoshop down.
#include <algorithm>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using host::Env;
using host::Info;
using host::Val;

struct ArgError : std::runtime_error {
    explicit ArgError(const std::string& what) : std::runtime_error("film dust native: " + what) {}
};

// the call's arguments, read with checks (a failed check throws ArgError)
class Args {
public:
    Args(Env env, Info info, size_t n) : env_(env) {
        if (n > kMax) throw ArgError("too many arguments");
        argc_ = n;
        if (!host::get_args(env, info, &argc_, argv_)) throw ArgError("bad call");
    }
    size_t count() const { return argc_; }
    Val arg(size_t i) const { return argv_[i]; }
    int i32(size_t i) const {
        int32_t r = 0;
        if (!host::get_i32(env_, argv_[i], &r)) throw ArgError("argument " + std::to_string(i) + " is not a number");
        return r;
    }
    double f64(size_t i) const {
        double r = 0;
        if (!host::get_f64(env_, argv_[i], &r)) throw ArgError("argument " + std::to_string(i) + " is not a number");
        return r;
    }
    bool flag(size_t i) const { return i32(i) != 0; }
    // the (ArrayBuffer, byteOffset) pair at arguments i, i + 1, holding at least n items of T
    template <typename T>
    T* buf(size_t i, size_t n, const char* what) const {
        void* data = nullptr;
        size_t len = 0;
        if (!host::arraybuffer(env_, argv_[i], &data, &len) || (!data && n)) throw ArgError(std::string("bad ") + what);
        const double off = f64(i + 1);
        if (!(off >= 0) || off + (double)n * sizeof(T) > (double)len) throw ArgError(std::string(what) + " buffer too small");
        return reinterpret_cast<T*>(static_cast<char*>(data) + (size_t)off);
    }

private:
    static const size_t kMax = 40;
    Env env_;
    Val argv_[kMax] = {};
    size_t argc_ = 0;
};

// named arguments: one JS object of numbers and, for arrays, [ArrayBuffer, byteOffset] pairs (a failed check throws)
class Opts {
public:
    Opts(Env env, Info info) : env_(env) {
        size_t argc = 1;
        if (!host::get_args(env, info, &argc, &obj_) || argc < 1) throw ArgError("bad call");
    }
    Opts(Env env, Val obj) : env_(env), obj_(obj) {}
    double f64(const char* name) const {
        double r = 0;
        if (!host::get_f64(env_, field(name), &r)) throw ArgError(std::string(name) + " is not a number");
        return r;
    }
    int i32(const char* name) const {
        int32_t r = 0;
        if (!host::get_i32(env_, field(name), &r)) throw ArgError(std::string(name) + " is not a number");
        return r;
    }
    bool flag(const char* name) const { return i32(name) != 0; }
    // the field's [ArrayBuffer, byteOffset] pair, holding at least n items of T
    template <typename T>
    T* buf(const char* name, size_t n) const {
        T* p = opt_buf<T>(name, n);
        if (!p && n) throw ArgError(std::string("bad ") + name);
        return p;
    }
    // the same, or nullptr when the field holds no such pair (null)
    template <typename T>
    T* opt_buf(const char* name, size_t n) const {
        Val pair = field(name), ab = nullptr, off = nullptr;
        void* data = nullptr;
        size_t len = 0;
        double o = 0;
        if (!host::get_element(env_, pair, 0, &ab) || !host::get_element(env_, pair, 1, &off) ||
            !host::arraybuffer(env_, ab, &data, &len) || !host::get_f64(env_, off, &o))
            return nullptr;
        if ((!data && n) || !(o >= 0) || o + (double)n * sizeof(T) > (double)len) throw ArgError(std::string(name) + " buffer too small");
        return reinterpret_cast<T*>(static_cast<char*>(data) + (size_t)o);
    }

private:
    Val field(const char* name) const {
        Val v = nullptr;
        if (!host::get_named(env_, obj_, name, &v)) throw ArgError(std::string("missing ") + name);
        return v;
    }
    Env env_;
    Val obj_ = nullptr;
};

// w x h as a pixel count (both at least 1)
size_t area(int w, int h) {
    if (w <= 0 || h <= 0) throw ArgError("empty image");
    return (size_t)w * (size_t)h;
}

// a new ArrayBuffer for n items of T; *data points into it
template <typename T>
Val new_buffer(Env env, size_t n, T** data) {
    void* p = nullptr;
    Val v = host::new_arraybuffer(env, n * sizeof(T), &p);
    if (!v || (n && !p)) throw std::bad_alloc();
    *data = static_cast<T*>(p);
    return v;
}

template <typename F>
Val guarded(Env env, F&& body) {
    try {
        return body();
    } catch (const std::bad_alloc&) {
        host::throw_error(env, "film dust native: out of memory");
    } catch (const std::exception& e) {
        host::throw_error(env, e.what());
    } catch (...) {
        host::throw_error(env, "film dust native: unexpected error");
    }
    return nullptr;
}

#define FD_FN(name) Val name(Env env, Info info)

FD_FN(fd_median8f) {   // (buf, off, w, h, k) -> f32 w x h
    return guarded(env, [&] {
        Args a(env, info, 5);
        const int w = a.i32(2), h = a.i32(3), k = a.i32(4);
        const float* src = a.buf<float>(0, area(w, h), "image");
        float* out;
        Val res = new_buffer(env, area(w, h), &out);
        fd::median8f(src, w, h, k, out);
        return res;
    });
}

FD_FN(fd_ellipse) {   // (buf, off, w, h, k, hwBuf, hwOff, mode) mode: 0 erode, 1 dilate, 2 open, 3 close
    return guarded(env, [&] {
        Args a(env, info, 8);
        const int w = a.i32(2), h = a.i32(3), k = a.i32(4), mode = a.i32(7);
        const float* src = a.buf<float>(0, area(w, h), "image");
        const int32_t* hw = a.buf<int32_t>(5, (size_t)k, "ellipse widths");
        float* out;
        Val res = new_buffer(env, area(w, h), &out);
        if (mode < 2) fd::ellipse_morph(src, w, h, k, hw, mode == 1, out);
        else fd::ellipse_open_close(src, w, h, k, hw, mode == 2, out);
        return res;
    });
}

FD_FN(fd_sep_conv) {   // (buf, off, w, h, kerBuf, kerOff, klen)
    return guarded(env, [&] {
        Args a(env, info, 7);
        const int w = a.i32(2), h = a.i32(3), klen = a.i32(6);
        const float* src = a.buf<float>(0, area(w, h), "image");
        const float* ker = a.buf<float>(4, (size_t)klen, "kernel");
        float* out;
        Val res = new_buffer(env, area(w, h), &out);
        fd::sep_conv(src, w, h, ker, klen, out);
        return res;
    });
}

FD_FN(fd_box3) {   // (buf, off, w, h, radiiBuf, radiiOff, n)
    return guarded(env, [&] {
        Args a(env, info, 7);
        const int w = a.i32(2), h = a.i32(3), n = a.i32(6);
        const float* src = a.buf<float>(0, area(w, h), "image");
        const int32_t* radii = a.buf<int32_t>(4, (size_t)n, "radii");
        float* out;
        Val res = new_buffer(env, area(w, h), &out);
        fd::box3(src, w, h, radii, n, out);
        return res;
    });
}

FD_FN(fd_box_blur) {   // (buf, off, w, h, k)
    return guarded(env, [&] {
        Args a(env, info, 5);
        const int w = a.i32(2), h = a.i32(3), k = a.i32(4);
        const float* src = a.buf<float>(0, area(w, h), "image");
        float* out;
        Val res = new_buffer(env, area(w, h), &out);
        fd::box_blur(src, w, h, k, out);
        return res;
    });
}

FD_FN(fd_sobel) {   // (buf, off, w, h, dx, dy)
    return guarded(env, [&] {
        Args a(env, info, 6);
        const int w = a.i32(2), h = a.i32(3), dx = a.i32(4), dy = a.i32(5);
        const float* src = a.buf<float>(0, area(w, h), "image");
        float* out;
        Val res = new_buffer(env, area(w, h), &out);
        fd::sobel(src, w, h, dx, dy, out);
        return res;
    });
}

FD_FN(fd_net_load) {   // (buf, off, nDoubles, towers, hidden, slot) -> 1 when the weights fit
    return guarded(env, [&] {
        Args a(env, info, 6);
        const size_t n = (size_t)a.f64(2);
        const double* w = a.buf<double>(0, n, "weights");
        return host::number(env, fd::net_load(a.i32(5), w, n, a.i32(3), a.i32(4)) ? 1.0 : 0.0);
    });
}

FD_FN(fd_net_forward) {   // (buf, off, N, slot) -> N float64 logits
    return guarded(env, [&] {
        Args a(env, info, 4);
        const int N = a.i32(2), slot = a.i32(3);
        const int T = fd::net_towers(slot);
        if (T <= 0) throw ArgError("no network loaded");
        if (N < 0) throw ArgError("bad count");
        const double* in = a.buf<double>(0, (size_t)N * T * 4800, "input");
        double* out;
        Val res = new_buffer(env, (size_t)N, &out);
        fd::net_forward(slot, in, N, out);
        return res;
    });
}

FD_FN(fd_net_forward_at) {   // (img u16, off, W, H, C, centres i32, off, N, bright, slot) -> N float64 logits
    return guarded(env, [&] {
        Args a(env, info, 10);
        const int W = a.i32(2), H = a.i32(3), C = a.i32(4), N = a.i32(7), slot = a.i32(9);
        if (fd::net_towers(slot) <= 0) throw ArgError("no network loaded");
        if (N <= 0 || C <= 0) throw ArgError("bad centres");
        const uint16_t* img = a.buf<uint16_t>(0, area(W, H) * C, "image");
        const int32_t* cen = a.buf<int32_t>(5, (size_t)2 * N, "centres");
        double* out;
        Val res = new_buffer(env, (size_t)N, &out);
        fd::net_forward_at(slot, img, W, H, C, cen, N, a.flag(8), out);
        return res;
    });
}

FD_FN(fd_hypot_into) {   // (gy f32, off, gx f32, off, n): gy = hypot(gx, gy) in place
    return guarded(env, [&] {
        Args a(env, info, 5);
        const size_t n = (size_t)a.f64(4);
        float* gy = a.buf<float>(0, n, "gradient");
        const float* gx = a.buf<float>(2, n, "gradient");
        fd::hypot_into(gy, gx, (int64_t)n);
        return host::number(env, 1.0);
    });
}

FD_FN(fd_blob_tiles) {   // {Y, sl, W, H, T, sq, sgn, ker, klen, k, mg, tiles (ty, tx pairs), n}
    // -> one buffer: n counts, then per tile its ys, xs, vals (count each) and its inner response
    return guarded(env, [&] {
        const Opts o(env, info);
        const int W = o.i32("W"), H = o.i32("H"), T = o.i32("T"), klen = o.i32("klen"), k = o.i32("k"), mg = o.i32("mg"), n = o.i32("n");
        if (n <= 0 || T <= 0) throw ArgError("bad blob tiles");
        const float* Y = o.buf<float>("Y", area(W, H));
        const float* sl = o.buf<float>("sl", area(W, H));
        const float* ker = o.buf<float>("ker", (size_t)klen);
        const int32_t* tl = o.buf<int32_t>("tiles", (size_t)2 * n);
        std::vector<fd::BlobTile> tiles;
        fd::blob_tiles(Y, sl, W, H, T, o.f64("sq"), o.i32("sgn"), ker, klen, k, mg, tl, n, tiles);
        size_t words = (size_t)n;
        for (const auto& t : tiles) words += 3 * t.ys.size() + t.inner.size();
        int32_t* out;
        Val res = new_buffer(env, words, &out);
        for (int i = 0; i < n; i++) out[i] = (int32_t)tiles[i].ys.size();
        size_t p = (size_t)n;
        for (const auto& t : tiles) {
            const size_t c = t.ys.size();
            if (c) {
                std::memcpy(out + p, t.ys.data(), c * 4); p += c;
                std::memcpy(out + p, t.xs.data(), c * 4); p += c;
                std::memcpy(out + p, t.vals.data(), c * 4); p += c;
            }
            if (!t.inner.empty()) std::memcpy(out + p, t.inner.data(), t.inner.size() * 4);
            p += t.inner.size();
        }
        return res;
    });
}

FD_FN(fd_skeletonize) {   // (mask u8, off, w, h) -> u8 skeleton
    return guarded(env, [&] {
        Args a(env, info, 4);
        const int w = a.i32(2), h = a.i32(3);
        const uint8_t* m = a.buf<uint8_t>(0, area(w, h), "mask");
        uint8_t* out;
        Val res = new_buffer(env, area(w, h), &out);
        fd::skeletonize(m, w, h, out);
        return res;
    });
}

FD_FN(fd_block_stats) {   // (resid, off, valid, voff, W, H, block, mode, bright) -> f32 grid hb x wb
    return guarded(env, [&] {
        Args a(env, info, 9);
        const int W = a.i32(4), H = a.i32(5), block = a.i32(6), mode = a.i32(7);
        if (block <= 0 || W < block || H < block) throw ArgError("image smaller than a block");
        const float* resid = a.buf<float>(0, area(W, H), "image");
        const uint8_t* valid = a.buf<uint8_t>(2, area(W, H), "mask");
        const int hb = H / block, wb = W / block;
        float* out;
        Val res = new_buffer(env, (size_t)hb * wb, &out);
        fd::block_stats(resid, valid, W, H, block, mode, a.flag(8), out);
        return res;
    });
}

FD_FN(fd_noise_by_level) {   // (bg, off, resid, roff, valid, voff, W, H, bins) -> f32 W x H
    return guarded(env, [&] {
        Args a(env, info, 9);
        const int W = a.i32(6), H = a.i32(7), bins = a.i32(8);
        if (bins < 1) throw ArgError("bad bins");
        const float* bg = a.buf<float>(0, area(W, H), "image");
        const float* resid = a.buf<float>(2, area(W, H), "image");
        const uint8_t* valid = a.buf<uint8_t>(4, area(W, H), "mask");
        float* out;
        Val res = new_buffer(env, area(W, H), &out);
        fd::noise_by_level(bg, resid, valid, W, H, bins, out);
        return res;
    });
}

FD_FN(fd_z_maps) {   // (Y, m15, o15, m41, o41, bxl, sigma: 7 x (buf, off); outside, ooff; N; bright) -> f32 3N
    return guarded(env, [&] {
        Args a(env, info, 18);
        const size_t N = (size_t)a.f64(16);
        const float* in[7];
        for (int i = 0; i < 7; i++) in[i] = a.buf<float>(2 * i, N, "map");
        const uint8_t* outside = a.buf<uint8_t>(14, N, "mask");
        float* o;
        Val res = new_buffer(env, N * 3, &o);
        fd::z_maps(in[0], in[1], in[2], in[3], in[4], in[5], in[6], outside, (int64_t)N, a.flag(17), o, o + N, o + 2 * N);
        return res;
    });
}

// the same z loop written over o15, o41 and bxl (z_s, z_l, z_xl in that order): each pixel reads its inputs before
// its results are stored, so no new maps are made (the caller no longer needs those three)
FD_FN(fd_z_maps_into) {   // (Y, m15, o15, m41, o41, bxl, sigma: 7 x (buf, off); outside, ooff; N; bright) -> 1
    return guarded(env, [&] {
        Args a(env, info, 18);
        const size_t N = (size_t)a.f64(16);
        float* in[7];
        for (int i = 0; i < 7; i++) in[i] = a.buf<float>(2 * i, N, "map");
        const uint8_t* outside = a.buf<uint8_t>(14, N, "mask");
        fd::z_maps(in[0], in[1], in[2], in[3], in[4], in[5], in[6], outside, (int64_t)N, a.flag(17), in[2], in[4], in[5]);
        return host::number(env, 1.0);
    });
}

FD_FN(fd_np_gradient) {   // (buf, off, w, h) -> f32 2N (gy, then gx)
    return guarded(env, [&] {
        Args a(env, info, 4);
        const int w = a.i32(2), h = a.i32(3);
        if (w < 2 || h < 2) throw ArgError("gradient needs 2 x 2");
        const float* src = a.buf<float>(0, area(w, h), "image");
        const size_t N = area(w, h);
        float* out;
        Val res = new_buffer(env, N * 2, &out);
        fd::np_gradient(src, w, h, out, out + N);
        return res;
    });
}

FD_FN(fd_resize_linear) {   // (buf, off, sw, sh, dw, dh) -> f32 dw x dh
    return guarded(env, [&] {
        Args a(env, info, 6);
        const int sw = a.i32(2), sh = a.i32(3), dw = a.i32(4), dh = a.i32(5);
        const float* src = a.buf<float>(0, area(sw, sh), "image");
        float* out;
        Val res = new_buffer(env, area(dw, dh), &out);
        fd::resize_linear(src, sw, sh, dw, dh, out);
        return res;
    });
}

FD_FN(fd_label) {   // (buf, off, w, h, conn) -> Int32 w*h labels + 1 (the count, last)
    return guarded(env, [&] {
        Args a(env, info, 5);
        const int w = a.i32(2), h = a.i32(3), conn = a.i32(4);
        const uint8_t* m = a.buf<uint8_t>(0, area(w, h), "mask");
        int32_t* lab;
        Val res = new_buffer(env, area(w, h) + 1, &lab);
        lab[area(w, h)] = fd::label(m, w, h, conn, lab);
        return res;
    });
}

FD_FN(fd_bin_morph) {   // (buf, off, w, h, k, hwBuf, hwOff, mode) mode 0 rect erode, 1 rect dilate, 2 ellipse erode, 3 ellipse dilate
    return guarded(env, [&] {
        Args a(env, info, 8);
        const int w = a.i32(2), h = a.i32(3), k = a.i32(4), mode = a.i32(7);
        const uint8_t* m = a.buf<uint8_t>(0, area(w, h), "mask");
        const int32_t* hw = mode < 2 ? nullptr : a.buf<int32_t>(5, (size_t)k, "ellipse widths");
        uint8_t* out;
        Val res = new_buffer(env, area(w, h), &out);
        if (mode < 2) fd::bin_rect(m, w, h, k, mode == 1, out);
        else fd::ellipse_morph_u8(m, w, h, k, hw, mode == 3, out);
        return res;
    });
}

FD_FN(fd_outside_mask) {   // (Y, off, W, H) -> Uint8 W*H
    return guarded(env, [&] {
        Args a(env, info, 4);
        const int W = a.i32(2), H = a.i32(3);
        const float* Y = a.buf<float>(0, area(W, H), "image");
        uint8_t* out;
        Val res = new_buffer(env, area(W, H), &out);
        fd::outside_mask(Y, W, H, out);
        return res;
    });
}

FD_FN(fd_ridge_maps) {   // {Y, outside, sig, W, H, bright, scales f64, kers f32 (back to back), klens i32, ns} -> f32 2N (best, bestS)
    return guarded(env, [&] {
        const Opts o(env, info);
        const int W = o.i32("W"), H = o.i32("H"), ns = o.i32("ns");
        if (ns < 0) throw ArgError("bad ridge scales");
        const size_t N = area(W, H);
        const float* Y = o.buf<float>("Y", N);
        const uint8_t* outside = o.buf<uint8_t>("outside", N);
        const float* sig = o.buf<float>("sig", N);
        const double* scales = o.buf<double>("scales", (size_t)ns);
        const int32_t* klens = o.buf<int32_t>("klens", (size_t)ns);
        size_t total = 0;
        for (int i = 0; i < ns; i++) {
            if (klens[i] <= 0) throw ArgError("bad kernel length");
            total += (size_t)klens[i];
        }
        const float* kers = o.buf<float>("kers", total);
        float* out;
        Val res = new_buffer(env, N * 2, &out);
        fd::ridge_maps(Y, W, H, outside, o.flag("bright"), sig, ns, scales, kers, klens, out, out + N);
        return res;
    });
}

FD_FN(fd_rect_morph) {   // (buf, off, w, h, k, isMax) -> f32
    return guarded(env, [&] {
        Args a(env, info, 6);
        const int w = a.i32(2), h = a.i32(3), k = a.i32(4);
        const float* src = a.buf<float>(0, area(w, h), "image");
        float* out;
        Val res = new_buffer(env, area(w, h), &out);
        fd::rect_morph(src, w, h, k, a.flag(5), out);
        return res;
    });
}

FD_FN(fd_average_strip) {   // (src, off, eight, srcW, rows, C, k, dst, doff, Wd, yd0) -> rows written (dst filled in place)
    return guarded(env, [&] {
        Args a(env, info, 11);
        const bool eight = a.flag(2);
        const int srcW = a.i32(3), rows = a.i32(4), C = a.i32(5), k = a.i32(6), Wd = a.i32(9), yd0 = a.i32(10);
        if (k <= 0 || C <= 0 || yd0 < 0 || srcW < 0 || rows < 0 || Wd < 0) throw ArgError("bad strip");
        const size_t n = (size_t)srcW * rows * C;
        const void* src = eight ? (const void*)a.buf<uint8_t>(0, n, "strip") : (const void*)a.buf<uint16_t>(0, n, "strip");
        uint16_t* dst = a.buf<uint16_t>(7, (size_t)(yd0 + rows / k) * Wd * C, "strip");
        return host::number(env, (double)fd::average_strip(src, eight, srcW, rows, C, k, dst, Wd, yd0));
    });
}

FD_FN(fd_fill_holes) {   // (buf, off, w, h) -> Uint8 w*h
    return guarded(env, [&] {
        Args a(env, info, 4);
        const int w = a.i32(2), h = a.i32(3);
        const uint8_t* m = a.buf<uint8_t>(0, area(w, h), "mask");
        uint8_t* out;
        Val res = new_buffer(env, area(w, h), &out);
        fd::fill_holes(m, w, h, out);
        return res;
    });
}

// ellipse half-width tables for sizes 0..127: hwPack (int32, all tables back to back) and hwStart[k] (-1 = none)
struct EllipseTables {
    const int32_t* t[128] = {};
};
EllipseTables ellipse_tables(const Opts& o) {
    const int32_t* st = o.buf<int32_t>("hwStart", 128);
    size_t need = 0;
    for (int k = 0; k < 128; k++)
        if (st[k] >= 0 && (size_t)st[k] + (size_t)k > need) need = (size_t)st[k] + (size_t)k;
    EllipseTables out;
    const int32_t* pk = need ? o.buf<int32_t>("hwPack", need) : nullptr;
    for (int k = 0; k < 128; k++) out.t[k] = st[k] >= 0 ? pk + st[k] : nullptr;
    return out;
}

// per-object offsets into the masks buffers: n + 1 doubles (from JS) as int64
std::vector<int64_t> offsets(const Opts& o, int n) {
    const double* offd = o.buf<double>("offs", (size_t)n + 1);
    std::vector<int64_t> offs((size_t)n + 1);
    for (int k = 0; k <= n; k++) {
        if (!(offd[k] >= 0) || (k && offd[k] < offd[k - 1])) throw ArgError("bad offsets");
        offs[k] = (int64_t)offd[k];
    }
    return offs;
}

// {Y, sig, glab, cand, W, H, n, boxes, ids, bright, k1, k1n, hwPack, hwStart, dk, offs}
// -> one ArrayBuffer: solid bytes (offs[n]) | objd bytes | pad to 8 | how i32 n | area i32 n | width f64 n
FD_FN(fd_solid_batch) {
    return guarded(env, [&] {
        const Opts o(env, info);
        const int W = o.i32("W"), H = o.i32("H"), n = o.i32("n"), k1n = o.i32("k1n");
        if (n < 0) throw ArgError("bad solid batch");
        const size_t N = area(W, H);
        const float* Y = o.buf<float>("Y", N);
        const float* sig = o.buf<float>("sig", N);
        const int32_t* glab = o.buf<int32_t>("glab", N);
        const uint8_t* cand = o.buf<uint8_t>("cand", N);
        const int32_t* boxes = o.buf<int32_t>("boxes", (size_t)4 * n);
        const int32_t* ids = o.buf<int32_t>("ids", (size_t)n);
        const float* k1 = o.buf<float>("k1", (size_t)k1n);
        const EllipseTables hw = ellipse_tables(o);
        const std::vector<int64_t> offs = offsets(o, n);
        const size_t tot = (size_t)offs[n], pad = (8 - (2 * tot) % 8) % 8;
        uint8_t* out;
        Val res = new_buffer(env, 2 * tot + pad + (size_t)n * 16, &out);
        fd::solid_batch(Y, sig, glab, cand, W, H, n, boxes, ids, o.flag("bright"), k1, k1n, hw.t, o.i32("dk"), offs.data(),
                        out, out + tot, (int32_t*)(out + 2 * tot + pad), (int32_t*)(out + 2 * tot + pad + (size_t)n * 4),
                        (double*)(out + 2 * tot + pad + (size_t)n * 8));
        return res;
    });
}

// {Y, G3, E25, E1 (or null), sig, cand, cgi, cidx, nc, W, H, n, boxes, offs, objd, bright, gradCal, k1, k1n, k3, k3n,
//  hwPack, hwStart} -> f64 n x 10
FD_FN(fd_cues_batch) {
    return guarded(env, [&] {
        const Opts o(env, info);
        const int nc = o.i32("nc"), W = o.i32("W"), H = o.i32("H"), n = o.i32("n"), k1n = o.i32("k1n"), k3n = o.i32("k3n");
        if (n < 0 || nc < 0) throw ArgError("bad cues batch");
        const size_t N = area(W, H);
        const float* Y = o.buf<float>("Y", N);
        const float* G3 = o.buf<float>("G3", N);
        const float* E25 = o.buf<float>("E25", N);
        const float* E1 = o.opt_buf<float>("E1", N);
        const float* sig = o.buf<float>("sig", N);
        const uint8_t* cand = o.buf<uint8_t>("cand", N);
        const int32_t* cgi = o.buf<int32_t>("cgi", (size_t)nc);
        const int32_t* cidx = o.buf<int32_t>("cidx", (size_t)nc);
        const int32_t* boxes = o.buf<int32_t>("boxes", (size_t)4 * n);
        const std::vector<int64_t> offs = offsets(o, n);
        const uint8_t* objd = o.buf<uint8_t>("objd", (size_t)offs[n]);
        const float* k1 = o.buf<float>("k1", (size_t)k1n);
        const float* k3 = o.buf<float>("k3", (size_t)k3n);
        const EllipseTables hw = ellipse_tables(o);
        double* out;
        Val res = new_buffer(env, (size_t)n * 10, &out);
        fd::cues_batch(Y, G3, E25, E1, sig, cand, cgi, cidx, nc, W, H, n, boxes, offs.data(), objd, o.flag("bright"), o.f64("gradCal"),
                       k1, k1n, k3, k3n, hw.t, out);
        return res;
    });
}

// {Y, sig, img u16, C, Ci, W, H, bright, n, seeds i32, k15, k15n, k08, k08n, k25, k25n, softFill, socketOn}
// -> fixed f64 n x 24 | offsets i32 3n + 1 (padded to 8) | lists i32
FD_FN(fd_soft_batch) {
    return guarded(env, [&] {
        const Opts o(env, info);
        const int C = o.i32("C"), Ci = o.i32("Ci"), W = o.i32("W"), H = o.i32("H"), n = o.i32("n");
        if (n <= 0 || C <= 0) throw ArgError("empty soft batch");
        const size_t N = area(W, H);
        const float* Y = o.buf<float>("Y", N);
        const float* sig = o.buf<float>("sig", N);
        const uint16_t* img = o.buf<uint16_t>("img", N * C);
        const int32_t* seeds = o.buf<int32_t>("seeds", (size_t)7 * n);
        const int k15n = o.i32("k15n"), k08n = o.i32("k08n"), k25n = o.i32("k25n");
        const float* k15 = o.buf<float>("k15", (size_t)k15n);
        const float* k08 = o.buf<float>("k08", (size_t)k08n);
        const float* k25 = o.buf<float>("k25", (size_t)k25n);
        std::vector<double> fixed((size_t)n * 24);
        std::vector<std::vector<int64_t>> lists;
        fd::soft_batch(Y, sig, img, C, Ci, W, H, o.flag("bright"), n, seeds, k15, k15n, k08, k08n, k25, k25n, o.f64("softFill"),
                       o.flag("socketOn"), fixed.data(), lists);
        size_t total = 0;
        for (const auto& l : lists) total += l.size();
        const size_t no = (size_t)3 * n + 1, offBytes = ((no * 4 + 7) / 8) * 8;
        uint8_t* out;
        Val res = new_buffer(env, (size_t)n * 24 * 8 + offBytes + total * 4, &out);
        std::memcpy(out, fixed.data(), (size_t)n * 24 * 8);
        int32_t* offs = (int32_t*)(out + (size_t)n * 24 * 8);
        int32_t* data = (int32_t*)(out + (size_t)n * 24 * 8 + offBytes);
        size_t q = 0;
        for (size_t i = 0; i < lists.size(); i++) {
            offs[i] = (int32_t)q;
            for (int64_t g : lists[i]) data[q++] = (int32_t)g;
        }
        offs[lists.size()] = (int32_t)q;
        return res;
    });
}

FD_FN(fd_threads) {   // ([n]) -> threads in use (sets them when n > 0)
    return guarded(env, [&] {
        Args a(env, info, 1);
        if (a.count() >= 1) {
            const int n = a.i32(0);
            if (n > 0) fd::set_threads(n);
        }
        return host::number(env, (double)fd::threads());
    });
}

FD_FN(fd_version) {
    (void)info;
    return host::number(env, 1.0);
}

// (kind) 1 runs out of memory on the calling thread, 2 inside a worker thread, 3 throws a plain exception: each
// must come back to JS as an error (test/native_errors.js). (4, {n, v: [buffer, offset]}) -> n + v[0]: the host
// reads named fields, which engine/native.js checks once at load before using the functions that take them.
FD_FN(fd_self_test) {
    return guarded(env, [&] {
        Args a(env, info, 2);
        const int kind = a.i32(0);
        if (kind == 1) throw std::bad_alloc();
        if (kind == 2) fd::run_parallel(64, 4, [](int, int) { throw std::bad_alloc(); });
        if (kind == 3) throw std::runtime_error("film dust native: self test");
        if (kind == 4) {
            const Opts o(env, a.arg(1));
            return host::number(env, o.i32("n") + o.buf<double>("v", 1)[0]);
        }
        return host::number(env, 0.0);
    });
}

#undef FD_FN

struct FdEntry {
    const char* name;
    Val (*fn)(Env, Info);
};
const FdEntry FD_FUNCTIONS[] = {
    {"median8f", fd_median8f}, {"ellipse", fd_ellipse}, {"sepConv", fd_sep_conv}, {"box3", fd_box3},
    {"boxBlur", fd_box_blur}, {"sobel", fd_sobel}, {"threads", fd_threads}, {"version", fd_version},
    {"netLoad", fd_net_load}, {"netForward", fd_net_forward}, {"netForwardAt", fd_net_forward_at},
    {"hypotInto", fd_hypot_into}, {"blobTiles", fd_blob_tiles}, {"skeletonize", fd_skeletonize},
    {"blockStats", fd_block_stats}, {"noiseByLevel", fd_noise_by_level}, {"zMaps", fd_z_maps}, {"zMapsInto", fd_z_maps_into},
    {"npGradient", fd_np_gradient}, {"resizeLinear", fd_resize_linear}, {"label", fd_label},
    {"binMorph", fd_bin_morph}, {"outsideMask", fd_outside_mask}, {"ridgeMaps", fd_ridge_maps},
    {"rectMorph", fd_rect_morph}, {"averageStrip", fd_average_strip}, {"fillHoles", fd_fill_holes},
    {"solidBatch", fd_solid_batch}, {"cuesBatch", fd_cues_batch}, {"softBatch", fd_soft_batch},
    {"selfTest", fd_self_test},
};

}  // namespace

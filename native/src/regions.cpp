// Native ports of engine/regions.js label and engine/filters.js binary morphology, and of maps.js
// outsideMask: exact operations (component numbering, counts, min/max), so any correct algorithm gives
// the JS results; these follow the JS ones closely anyway.
#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

#include "filters.h"

namespace fd {

static void par(int n, const std::function<void(int, int)>& fn, int min_chunk = 32) {
    int t = std::min(threads(), std::max(1, n / std::max(1, min_chunk)));
    if (t <= 1) {
        fn(0, n);
        return;
    }
    run_parallel(n, t, fn);
}

// scipy.ndimage.label (regions.js label): conn 4 or 8, components numbered by first appearance in
// raster order. Returns the number of components.
int label(const uint8_t* mask, int w, int h, int conn, int32_t* lab) {
    std::vector<int32_t> parent(1, 0);
    parent.reserve(1 << 16);
    auto find = [&](int32_t a) {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    };
    auto join = [&](int32_t l, int32_t b) {
        const int32_t rb = find(b);
        if (rb == l) return l;
        if (rb < l) {
            parent[l] = rb;
            return rb;
        }
        parent[rb] = l;
        return l;
    };
    const bool eight = conn == 8;
    for (int y = 0; y < h; y++) {
        const int64_t off = (int64_t)y * w;
        for (int x = 0; x < w; x++) {
            const int64_t i = off + x;
            if (!mask[i]) {
                lab[i] = 0;
                continue;
            }
            const int32_t a = x > 0 ? lab[i - 1] : 0;
            const int32_t b = y > 0 ? lab[i - w] : 0;
            const int32_t c = eight && y > 0 && x > 0 ? lab[i - w - 1] : 0;
            const int32_t d = eight && y > 0 && x < w - 1 ? lab[i - w + 1] : 0;
            if (!a && !b && !c && !d) {
                const int32_t nl = (int32_t)parent.size();
                parent.push_back(nl);
                lab[i] = nl;
                continue;
            }
            int32_t l = find(a ? a : b ? b : c ? c : d);
            if (a) l = join(l, a);
            if (b) l = join(l, b);
            if (c) l = join(l, c);
            if (d) l = join(l, d);
            lab[i] = l;
        }
    }
    std::vector<int32_t> remap(parent.size(), 0);
    int32_t n = 0;
    const int64_t N = (int64_t)w * h;
    for (int64_t i = 0; i < N; i++) {
        const int32_t l = lab[i];
        if (!l) continue;
        const int32_t r = find(l);
        if (!remap[r]) remap[r] = ++n;
        lab[i] = remap[r];
    }
    return n;
}

// filters.js binRect: binary square erode / dilate with running counts (outside never counts)
void bin_rect(const uint8_t* mask, int w, int h, int k, bool dilate, uint8_t* out) {
    const int r = k >> 1;
    std::vector<uint8_t> hz((size_t)w * h);
    par(h, [&](int ya, int yb) {
        std::vector<int32_t> Pc(w + 1, 0);
        for (int y = ya; y < yb; y++) {
            const int64_t o = (int64_t)y * w;
            for (int x = 0; x < w; x++) Pc[x + 1] = Pc[x] + (dilate ? (mask[o + x] ? 1 : 0) : (mask[o + x] ? 0 : 1));
            for (int x = 0; x < w; x++) {
                const int c = Pc[x + r + 1 > w ? w : x + r + 1] - Pc[x - r < 0 ? 0 : x - r];
                hz[o + x] = dilate ? (c > 0 ? 1 : 0) : (c == 0 ? 1 : 0);
            }
        }
    });
    par(w, [&](int xa, int xb) {
        std::vector<int32_t> cnt(xb - xa, 0);
        const int top = std::min(r, h - 1);
        for (int yy = 0; yy <= top; yy++) {
            const int64_t o = (int64_t)yy * w;
            for (int x = xa; x < xb; x++) cnt[x - xa] += dilate ? hz[o + x] : 1 - hz[o + x];
        }
        for (int y = 0; y < h; y++) {
            const int64_t o = (int64_t)y * w;
            for (int x = xa; x < xb; x++) out[o + x] = dilate ? (cnt[x - xa] > 0 ? 1 : 0) : (cnt[x - xa] == 0 ? 1 : 0);
            const int add = y + r + 1, sub = y - r;
            if (add < h) {
                const int64_t a = (int64_t)add * w;
                for (int x = xa; x < xb; x++) cnt[x - xa] += dilate ? hz[a + x] : 1 - hz[a + x];
            }
            if (sub >= 0) {
                const int64_t b = (int64_t)sub * w;
                for (int x = xa; x < xb; x++) cnt[x - xa] -= dilate ? hz[b + x] : 1 - hz[b + x];
            }
        }
    }, 64);
}

// filters.js ellipseMorph on a 0/1 mask: max / min over the ellipse, outside never counts
void ellipse_morph_u8(const uint8_t* src, int w, int h, int k, const int32_t* hw, bool is_max, uint8_t* dst) {
    // the float version is exact for 0/1 data; run it in strips to keep memory small
    const int r = k >> 1;
    const int strip = 512;
    par((h + strip - 1) / strip, [&](int sa, int sb) {
        for (int s = sa; s < sb; s++) {
            const int y0 = s * strip, y1 = std::min(h, y0 + strip);
            const int ya = std::max(0, y0 - r), yb = std::min(h, y1 + r);
            const int sh = yb - ya;
            std::vector<float> a((size_t)sh * w), o((size_t)sh * w);
            for (int64_t i = 0; i < (int64_t)sh * w; i++) a[i] = src[(int64_t)ya * w + i];
            set_threads_local(1);
            ellipse_morph(a.data(), w, sh, k, hw, is_max, o.data());
            set_threads_local(0);
            for (int y = y0; y < y1; y++) {
                const float* row = &o[(size_t)(y - ya) * w];
                uint8_t* d = dst + (int64_t)y * w;
                for (int x = 0; x < w; x++) d[x] = (uint8_t)row[x];
            }
        }
    }, 1);
}

// maps.js outsideMask: clipped, grain-free areas touching the frame (scanner light beyond the film)
void outside_mask(const float* Y, int W, int H, uint8_t* out) {
    const int64_t N = (int64_t)W * H;
    std::vector<float> mean(N), sq(N), msq(N);
    box_blur(Y, W, H, 7, mean.data());
    par(H, [&](int ya, int yb) {
        for (int64_t i = (int64_t)ya * W; i < (int64_t)yb * W; i++) sq[i] = (float)((double)Y[i] * (double)Y[i]);
    });
    box_blur(sq.data(), W, H, 7, msq.data());
    std::vector<uint8_t> flat(N);
    par(H, [&](int ya, int yb) {
        for (int64_t i = (int64_t)ya * W; i < (int64_t)yb * W; i++) {
            const double sd = std::sqrt(std::max((double)msq[i] - (double)mean[i] * (double)mean[i], 0.0));
            flat[i] = sd < 0.0015 && ((double)Y[i] > 0.85 || (double)Y[i] < 0.02) ? 1 : 0;
        }
    });
    std::vector<int32_t> lab(N);
    const int n = label(flat.data(), W, H, 4, lab.data());
    std::vector<uint8_t> core(N, 0);
    if (n) {
        std::vector<int64_t> size(n + 1, 0);
        for (int64_t i = 0; i < N; i++)
            if (lab[i]) size[lab[i]]++;
        std::vector<uint8_t> keep(n + 1, 0);
        auto touch = [&](int32_t l) {
            if (l && (double)size[l] > 0.002 * H * W) keep[l] = 1;
        };
        for (int x = 0; x < W; x++) {
            touch(lab[x]);
            touch(lab[(int64_t)(H - 1) * W + x]);
        }
        for (int y = 0; y < H; y++) {
            touch(lab[(int64_t)y * W]);
            touch(lab[(int64_t)y * W + W - 1]);
        }
        for (int64_t i = 0; i < N; i++) core[i] = keep[lab[i]];
    }
    bin_rect(core.data(), W, H, 9, true, out);
}

}  // namespace fd

namespace fd {

// filters.js rectMorph: square max / min (float), outside never counts; rows then columns
void rect_morph(const float* src, int w, int h, int k, bool is_max, float* dst) {
    const int r = k >> 1;
    const float pad = is_max ? -INFINITY : INFINITY;
    std::vector<float> tmp((size_t)w * h);
    auto run_line = [&](const float* in, int n, int stride, float* out, int ostride, std::vector<float>& P,
                        std::vector<float>& f, std::vector<float>& g) {
        const int kk = 2 * r + 1, len = n + 2 * r;
        for (int i = 0; i < r; i++) P[i] = pad;
        for (int i = 0; i < n; i++) P[r + i] = in[(int64_t)i * stride];
        for (int i = 0; i < r; i++) P[r + n + i] = pad;
        for (int b0 = 0; b0 < len; b0 += kk) {
            const int b1 = b0 + kk < len ? b0 + kk : len;
            float m = pad;
            for (int i = b0; i < b1; i++) {
                const float v = P[i];
                if (is_max ? v > m : v < m) m = v;
                f[i] = m;
            }
            m = pad;
            for (int i = b1 - 1; i >= b0; i--) {
                const float v = P[i];
                if (is_max ? v > m : v < m) m = v;
                g[i] = m;
            }
        }
        for (int i = 0; i < n; i++) {
            const float a = g[i], b = f[i + kk - 1];
            out[(int64_t)i * ostride] = is_max ? (a > b ? a : b) : (a < b ? a : b);
        }
    };
    par(h, [&](int ya, int yb) {
        std::vector<float> P(w + 2 * r), f(w + 4 * r + 1), g(w + 4 * r + 1);
        for (int y = ya; y < yb; y++) run_line(src + (int64_t)y * w, w, 1, tmp.data() + (int64_t)y * w, 1, P, f, g);
    });
    par(w, [&](int xa, int xb) {
        std::vector<float> P(h + 2 * r), f(h + 4 * r + 1), g(h + 4 * r + 1);
        for (int x = xa; x < xb; x++) run_line(tmp.data() + x, h, w, dst + x, w, P, f, g);
    }, 16);
}

// scale.js averageStrip: k x k block means of a strip into the working image (whole blocks only),
// rounded as the JS does: floor((sum * scale + half) / kk). Returns the rows written.
int average_strip(const void* src, bool eight, int srcW, int rows, int C, int k, uint16_t* dst, int Wd, int yd0) {
    const int kk = k * k, half = kk >> 1;
    const double scale = eight ? 257 : 1;
    const int rowsOut = rows / k;
    par(rowsOut, [&](int ra, int rb) {
        std::vector<double> acc((size_t)Wd * C);
        for (int r = ra; r < rb; r++) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int dy = 0; dy < k; dy++) {
                int64_t s = (int64_t)(r * k + dy) * srcW * C;
                for (int xd = 0; xd < Wd; xd++) {
                    const int64_t o = (int64_t)xd * C;
                    for (int dx = 0; dx < k; dx++) {
                        for (int c = 0; c < C; c++)
                            acc[o + c] += eight ? (double)((const uint8_t*)src)[s + c] : (double)((const uint16_t*)src)[s + c];
                        s += C;
                    }
                }
            }
            uint16_t* d = dst + (int64_t)(yd0 + r) * Wd * C;
            for (int64_t i = 0; i < (int64_t)Wd * C; i++) d[i] = (uint16_t)std::floor((acc[i] * scale + half) / kk);
        }
    }, 4);
    return rowsOut;
}

}  // namespace fd

namespace fd {

// regions.js fillHoles (scipy binary_fill_holes): background not reachable from the border (4-connected) is filled
void fill_holes(const uint8_t* mask, int w, int h, uint8_t* out) {
    const int64_t N = (int64_t)w * h;
    std::vector<uint8_t> bg(N, 0);
    std::vector<int64_t> stack;
    stack.reserve(1 << 16);
    auto push = [&](int64_t i) {
        if (!mask[i] && !bg[i]) {
            bg[i] = 1;
            stack.push_back(i);
        }
    };
    for (int x = 0; x < w; x++) {
        push(x);
        push((int64_t)(h - 1) * w + x);
    }
    for (int y = 0; y < h; y++) {
        push((int64_t)y * w);
        push((int64_t)y * w + w - 1);
    }
    while (!stack.empty()) {
        const int64_t i = stack.back();
        stack.pop_back();
        const int x = (int)(i % w);
        const int y = (int)(i / w);
        if (x > 0) push(i - 1);
        if (x < w - 1) push(i + 1);
        if (y > 0) push(i - w);
        if (y < h - 1) push(i + w);
    }
    for (int64_t i = 0; i < N; i++) out[i] = mask[i] || !bg[i] ? 1 : 0;
}

}  // namespace fd

// See filters.h. Each function is a line-by-line port of its JS original (engine/filters.js,
// engine/maps.js), split over threads by rows (or column strips where the JS runs down columns).
// Built with -ffp-contract=off and without fast-math so every double/float operation rounds
// exactly as V8's does.
#include "filters.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

namespace fd {

static int g_threads = 0;
static thread_local int tl_threads = 0;   // > 0: this thread runs nested work with that many threads

void set_threads_local(int n) { tl_threads = n; }

int threads() {
    if (tl_threads > 0) return tl_threads;
    if (g_threads <= 0) {
        unsigned n = std::thread::hardware_concurrency();
        g_threads = n ? (int)std::min(n, 32u) : 4;
    }
    return g_threads;
}

void set_threads(int n) { g_threads = n; }

// split [0, n) into one contiguous chunk per thread (at least min_chunk items each)
static void parallel_for(int n, const std::function<void(int, int)>& fn, int min_chunk = 32) {
    int t = std::min(threads(), std::max(1, n / std::max(1, min_chunk)));
    if (t <= 1) {
        fn(0, n);
        return;
    }
    run_parallel(n, t, fn);
}

static inline int reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

static inline int clamp_idx(int i, int n) { return i < 0 ? 0 : i >= n ? n - 1 : i; }

// ---------- median ---------------------------------------------------------------------------

void median8(const uint8_t* src, int w, int h, int k, uint8_t* dst) {
    const int r = k >> 1;
    const int half = (k * k) >> 1;
    parallel_for(h, [&](int ya, int yb) {
        int32_t hist[257];
        std::vector<int64_t> rows(k);
        for (int y = ya; y < yb; y++) {
            for (int i = 0; i < k; i++) rows[i] = (int64_t)clamp_idx(y + i - r, h) * w;
            std::fill(hist, hist + 257, 0);
            for (int i = 0; i < k; i++) {
                const int64_t ro = rows[i];
                for (int dx = -r; dx <= r; dx++) hist[src[ro + clamp_idx(dx, w)]]++;
            }
            int med = 0, below = 0;
            while (below + hist[med] <= half) {
                below += hist[med];
                med++;
            }
            dst[(int64_t)y * w] = (uint8_t)med;
            for (int x = 1; x < w; x++) {
                const int xo = clamp_idx(x - r - 1, w), xi = clamp_idx(x + r, w);
                if (xo != xi) {
                    for (int i = 0; i < k; i++) {
                        const int64_t ro = rows[i];
                        const int vo = src[ro + xo], vi = src[ro + xi];
                        if (vo == vi) continue;
                        hist[vo]--;
                        if (vo < med) below--;
                        hist[vi]++;
                        if (vi < med) below++;
                    }
                }
                while (below > half) {
                    med--;
                    below -= hist[med];
                }
                while (below + hist[med] <= half) {
                    below += hist[med];
                    med++;
                }
                dst[(int64_t)y * w + x] = (uint8_t)med;
            }
        }
    }, 8);
}

void median8f(const float* Y, int w, int h, int k, float* out) {
    const int64_t N = (int64_t)w * h;
    std::vector<uint8_t> y8(N), m(N);
    parallel_for(h, [&](int ya, int yb) {
        for (int64_t i = (int64_t)ya * w; i < (int64_t)yb * w; i++) {
            const double v = std::floor((double)Y[i] * 255.0 + 0.5);
            y8[i] = std::isnan(v) ? 0 : v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
        }
    });
    median8(y8.data(), w, h, k, m.data());
    parallel_for(h, [&](int ya, int yb) {
        for (int64_t i = (int64_t)ya * w; i < (int64_t)yb * w; i++) out[i] = (float)((double)m[i] / 255.0);
    });
}

// ---------- ellipse morphology -----------------------------------------------------------------

// van Herk / Gil-Werman running min or max over windows of 2r+1 on a padded line P
static void vh_line(const float* P, int n, int r, bool is_max, float* out, float* f, float* g) {
    const int k = 2 * r + 1, len = n + 2 * r;
    if (is_max) {
        for (int b0 = 0; b0 < len; b0 += k) {
            const int b1 = b0 + k < len ? b0 + k : len;
            float m = -std::numeric_limits<float>::infinity();
            for (int i = b0; i < b1; i++) {
                const float v = P[i];
                if (v > m) m = v;
                f[i] = m;
            }
            m = -std::numeric_limits<float>::infinity();
            for (int i = b1 - 1; i >= b0; i--) {
                const float v = P[i];
                if (v > m) m = v;
                g[i] = m;
            }
        }
        for (int i = 0; i < n; i++) {
            const float a = g[i], b = f[i + k - 1];
            out[i] = a > b ? a : b;
        }
    } else {
        for (int b0 = 0; b0 < len; b0 += k) {
            const int b1 = b0 + k < len ? b0 + k : len;
            float m = std::numeric_limits<float>::infinity();
            for (int i = b0; i < b1; i++) {
                const float v = P[i];
                if (v < m) m = v;
                f[i] = m;
            }
            m = std::numeric_limits<float>::infinity();
            for (int i = b1 - 1; i >= b0; i--) {
                const float v = P[i];
                if (v < m) m = v;
                g[i] = m;
            }
        }
        for (int i = 0; i < n; i++) {
            const float a = g[i], b = f[i + k - 1];
            out[i] = a < b ? a : b;
        }
    }
}

void ellipse_morph(const float* src, int w, int h, int k, const int32_t* hw, bool is_max, float* dst) {
    const int r = k >> 1;
    std::vector<int> widths;
    for (int i = 0; i < k; i++)
        if (hw[i] >= 0) widths.push_back(hw[i]);
    std::sort(widths.begin(), widths.end());
    widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
    const int m = (int)widths.size();
    std::vector<int> widx(k, -1);
    for (int i = 0; i < k; i++)
        if (hw[i] >= 0) widx[i] = (int)(std::find(widths.begin(), widths.end(), hw[i]) - widths.begin());
    const int max_r = widths[m - 1];
    const float pad = is_max ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    parallel_for(h, [&](int ya, int yb) {
        std::vector<float> ring((size_t)k * m * w);
        std::vector<int> ring_row(k, -1);
        std::vector<float> P(w + 2 * max_r), f(w + 4 * max_r + 1), g(w + 4 * max_r + 1), out(w), acc(w);
        auto fill = [&](int yy) {
            const int slot = yy % k;
            const size_t base = (size_t)slot * m * w;
            const float* srow = src + (int64_t)yy * w;
            for (int j = 0; j < m; j++) {
                const int wj = widths[j];
                float* o = &ring[base + (size_t)j * w];
                if (wj == 0) {
                    std::copy(srow, srow + w, o);
                    continue;
                }
                const int prev = j > 0 ? widths[j - 1] : -1, d = wj - prev;
                if (j > 0 && d <= prev) {
                    const float* po = &ring[base + (size_t)(j - 1) * w];
                    const int lo = std::min(d, w), hi = std::max(lo, w - d);
                    if (is_max) {
                        for (int x = 0; x < lo; x++) {
                            const float a = po[0], b = po[x + d >= w ? w - 1 : x + d];
                            o[x] = a > b ? a : b;
                        }
                        for (int x = lo; x < hi; x++) {
                            const float a = po[x - d], b = po[x + d];
                            o[x] = a > b ? a : b;
                        }
                        for (int x = hi; x < w; x++) {
                            const float a = po[x - d < 0 ? 0 : x - d], b = po[w - 1];
                            o[x] = a > b ? a : b;
                        }
                    } else {
                        for (int x = 0; x < lo; x++) {
                            const float a = po[0], b = po[x + d >= w ? w - 1 : x + d];
                            o[x] = a < b ? a : b;
                        }
                        for (int x = lo; x < hi; x++) {
                            const float a = po[x - d], b = po[x + d];
                            o[x] = a < b ? a : b;
                        }
                        for (int x = hi; x < w; x++) {
                            const float a = po[x - d < 0 ? 0 : x - d], b = po[w - 1];
                            o[x] = a < b ? a : b;
                        }
                    }
                } else {
                    std::fill(P.begin(), P.begin() + wj, pad);
                    std::copy(srow, srow + w, P.begin() + wj);
                    std::fill(P.begin() + wj + w, P.begin() + w + 2 * wj, pad);
                    vh_line(P.data(), w, wj, is_max, out.data(), f.data(), g.data());
                    std::copy(out.begin(), out.end(), o);
                }
            }
            ring_row[slot] = yy;
        };
        for (int y = ya; y < yb; y++) {
            std::fill(acc.begin(), acc.end(), pad);
            for (int i = 0; i < k; i++) {
                const int j = widx[i];
                if (j < 0) continue;
                const int yy = y + i - r;
                if (yy < 0 || yy >= h) continue;
                const int slot = yy % k;
                if (ring_row[slot] != yy) fill(yy);
                const float* o = &ring[(size_t)slot * m * w + (size_t)j * w];
                if (is_max) {
                    for (int x = 0; x < w; x++) {
                        const float v = o[x];
                        if (v > acc[x]) acc[x] = v;
                    }
                } else {
                    for (int x = 0; x < w; x++) {
                        const float v = o[x];
                        if (v < acc[x]) acc[x] = v;
                    }
                }
            }
            std::copy(acc.begin(), acc.end(), dst + (int64_t)y * w);
        }
    }, std::max(64, 4 * k));
}

void ellipse_open_close(const float* src, int w, int h, int k, const int32_t* hw, bool open, float* dst) {
    std::vector<float> a((size_t)w * h);
    ellipse_morph(src, w, h, k, hw, !open, a.data());
    ellipse_morph(a.data(), w, h, k, hw, open, dst);
}

void hypot_into(float* gy, const float* gx, int64_t n) {
    const int64_t rows = (n + 4095) / 4096;
    parallel_for((int)rows, [&](int a, int b) {
        const int64_t e = std::min(n, (int64_t)b * 4096);
        for (int64_t i = (int64_t)a * 4096; i < e; i++) gy[i] = (float)v8_hypot(gx[i], gy[i]);
    });
}

// ---------- linear filters ----------------------------------------------------------------------

void sep_conv(const float* src, int w, int h, const float* ker, int klen, float* dst) {
    const int r = klen >> 1;
    const double kc = ker[r];
    std::vector<float> tmp((size_t)w * h);
    parallel_for(h, [&](int ya, int yb) {
        std::vector<float> row(w + 2 * r);
        for (int y = ya; y < yb; y++) {
            const int64_t off = (int64_t)y * w;
            for (int x = -r; x < w + r; x++) row[x + r] = src[off + reflect101(x, w)];
            for (int x = 0; x < w; x++) {
                const int c = x + r;
                double acc = (double)row[c] * kc;
                for (int j = 1; j <= r; j++) acc += (double)ker[r - j] * ((double)row[c - j] + (double)row[c + j]);
                tmp[off + x] = (float)acc;
            }
        }
    });
    parallel_for(h, [&](int ya, int yb) {
        std::vector<double> acc(w);
        for (int y = ya; y < yb; y++) {
            const int64_t oc = (int64_t)y * w;
            for (int x = 0; x < w; x++) acc[x] = (double)tmp[oc + x] * kc;
            for (int j = 1; j <= r; j++) {
                const int64_t ra = (int64_t)reflect101(y - j, h) * w, rb = (int64_t)reflect101(y + j, h) * w;
                const double kv = ker[r - j];
                for (int x = 0; x < w; x++) acc[x] += kv * ((double)tmp[ra + x] + (double)tmp[rb + x]);
            }
            for (int x = 0; x < w; x++) dst[oc + x] = (float)acc[x];
        }
    });
}

static void box_h(const float* src, float* dst, int w, int h, int r) {
    const double inv = 1.0 / (2 * r + 1);
    parallel_for(h, [&](int ya, int yb) {
        for (int y = ya; y < yb; y++) {
            const int64_t off = (int64_t)y * w;
            double acc = 0;
            for (int x = -r; x <= r; x++) acc += src[off + reflect101(x, w)];
            for (int x = 0; x < w; x++) {
                dst[off + x] = (float)(acc * inv);
                acc += (double)src[off + reflect101(x + r + 1, w)] - (double)src[off + reflect101(x - r, w)];
            }
        }
    });
}

static void box_v(const float* src, float* dst, int w, int h, int r) {
    const double inv = 1.0 / (2 * r + 1);
    parallel_for(w, [&](int xa, int xb) {
        std::vector<double> acc(xb - xa, 0.0);
        for (int y = -r; y <= r; y++) {
            const int64_t o = (int64_t)reflect101(y, h) * w;
            for (int x = xa; x < xb; x++) acc[x - xa] += src[o + x];
        }
        for (int y = 0; y < h; y++) {
            const int64_t d = (int64_t)y * w, a = (int64_t)reflect101(y + r + 1, h) * w, b = (int64_t)reflect101(y - r, h) * w;
            for (int x = xa; x < xb; x++) {
                dst[d + x] = (float)(acc[x - xa] * inv);
                acc[x - xa] += (double)src[a + x] - (double)src[b + x];
            }
        }
    }, 64);
}

void box3(const float* src, int w, int h, const int32_t* radii, int npass, float* dst) {
    std::vector<float> a(src, src + (size_t)w * h), b((size_t)w * h);
    for (int i = 0; i < npass; i++) {
        box_h(a.data(), b.data(), w, h, radii[i]);
        box_v(b.data(), a.data(), w, h, radii[i]);
    }
    std::copy(a.begin(), a.end(), dst);
}

void box_blur(const float* src, int w, int h, int k, float* dst) {
    std::vector<float> tmp((size_t)w * h);
    box_h(src, tmp.data(), w, h, k >> 1);
    box_v(tmp.data(), dst, w, h, k >> 1);
}

void sobel(const float* src, int w, int h, int dx, int dy, float* dst) {
    static const double kd1[3] = {-1, 0, 1}, kd2[3] = {1, -2, 1}, ks[3] = {1, 2, 1};
    const double* kx = dx == 1 ? kd1 : dx == 2 ? kd2 : ks;
    const double* ky = dy == 1 ? kd1 : dy == 2 ? kd2 : ks;
    std::vector<float> tmp((size_t)w * h);
    parallel_for(h, [&](int ya, int yb) {
        for (int y = ya; y < yb; y++) {
            const int64_t off = (int64_t)y * w;
            for (int x = 0; x < w; x++) {
                double t = kx[0] * (double)src[off + reflect101(x - 1, w)] + kx[1] * (double)src[off + x];
                t = t + kx[2] * (double)src[off + reflect101(x + 1, w)];
                tmp[off + x] = (float)t;
            }
        }
    });
    parallel_for(h, [&](int ya, int yb) {
        for (int y = ya; y < yb; y++) {
            const int64_t up = (int64_t)reflect101(y - 1, h) * w, mid = (int64_t)y * w, dn = (int64_t)reflect101(y + 1, h) * w;
            for (int x = 0; x < w; x++) {
                double t = ky[0] * (double)tmp[up + x] + ky[1] * (double)tmp[mid + x];
                t = t + ky[2] * (double)tmp[dn + x];
                dst[mid + x] = (float)t;
            }
        }
    });
}

}  // namespace fd

// See filters.h. Linear filters (Gaussian, box, Sobel) do the JS's arithmetic in the JS's order for every pixel,
// built with -ffp-contract=off and without fast-math so each operation rounds as V8's does; whole rows a step, so
// the loops vectorise. Min / max filters involve no rounding, so they use faster algorithms with the same answer
// (cache-sized tiles). Big images go in bands of rows, so no second whole image is held where a band will do.
#include "cpu.h"
#include "filters.h"

#include <algorithm>
#include <atomic>
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
    parallel_rows(h, (int64_t)w * k, [&](int ya, int yb) {
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
    parallel_rows(h, w, [&](int ya, int yb) {
        for (int64_t i = (int64_t)ya * w; i < (int64_t)yb * w; i++) {
            const double v = std::floor((double)Y[i] * 255.0 + 0.5);
            y8[i] = std::isnan(v) ? 0 : v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
        }
    });
    median8(y8.data(), w, h, k, m.data());
    parallel_rows(h, w, [&](int ya, int yb) {
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

namespace {

// the ellipse's distinct half-widths (ascending) and, per row offset i, the index of its half-width (-1: none)
struct Shape {
    std::vector<int> widths, widx;
    Shape(int k, const int32_t* hw) : widx(k, -1) {
        for (int i = 0; i < k; i++)
            if (hw[i] >= 0) widths.push_back(hw[i]);
        std::sort(widths.begin(), widths.end());
        widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
        for (int i = 0; i < k; i++)
            if (hw[i] >= 0) widx[i] = (int)(std::lower_bound(widths.begin(), widths.end(), hw[i]) - widths.begin());
    }
};

// no NaN and no -0: then a max / min over a window has one answer whatever the order it is taken in
bool plain_floats(const float* a, int64_t n) {
    std::atomic<bool> plain{true};
    const int64_t chunk = 1 << 16;
    parallel_rows((int)((n + chunk - 1) / chunk), chunk, [&](int ca, int cb) {
        for (int64_t i = (int64_t)ca * chunk; i < std::min(n, (int64_t)cb * chunk); i++)
            if (std::isnan(a[i]) || (a[i] == 0 && std::signbit(a[i]))) { plain = false; return; }
    }, 1);
    return plain;
}

// The ellipse in tiles of rows x 128 columns, each with its own ring of k rows x widths x 128 floats, small enough
// to stay in the processor's cache (a ring the image's full width does not). Each row segment's running max at
// every half-width is a van Herk pass over the segment and its margins, so tiles need nothing from each other.
// Rows [ya, yb) of the result for an image h rows tall: src holds rows from src_row0 on (all the rows those
// output rows reach, [ya - k/2, yb + k/2) within the image), row y goes to dst + (y - dst_row0) w.
void ellipse_tiles(const float* src, int src_row0, int w, int h, int k, const Shape& sh, bool is_max, float* dst, int dst_row0,
                   int ya0, int yb0) {
    const int r = k >> 1, m = (int)sh.widths.size(), max_r = sh.widths[m - 1];
    const float pad = is_max ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    const int TW = 128, band = std::max(256, 8 * k);
    const int ntx = (w + TW - 1) / TW, nby = (yb0 - ya0 + band - 1) / band;
    if (nby <= 0) return;
    const int ntasks = ntx * nby;
    run_parallel(ntasks, std::min(threads(), ntasks), [&](int ta, int tb) {
        std::vector<float> ring((size_t)k * m * TW), P(TW + 2 * max_r), f(TW + 2 * max_r + 1), g(TW + 2 * max_r + 1), acc(TW);
        std::vector<int> ring_row(k);
        for (int task = ta; task < tb; task++) {
            const int x0 = (task % ntx) * TW, x1 = std::min(w, x0 + TW), tw = x1 - x0;
            const int ya = ya0 + (task / ntx) * band, yb = std::min(yb0, ya + band);
            std::fill(ring_row.begin(), ring_row.end(), -1);
            auto fill = [&](int yy) {
                const int slot = yy % k;
                const float* srow = src + (int64_t)(yy - src_row0) * w;
                for (int j = 0; j < m; j++) {
                    const int wj = sh.widths[j];
                    float* o = &ring[((size_t)slot * m + j) * TW];
                    if (wj == 0) {
                        std::copy(srow + x0, srow + x1, o);
                        continue;
                    }
                    const int len = tw + 2 * wj;
                    for (int i = 0; i < len; i++) {
                        const int x = x0 - wj + i;
                        P[i] = x >= 0 && x < w ? srow[x] : pad;
                    }
                    vh_line(P.data(), tw, wj, is_max, o, f.data(), g.data());
                }
                ring_row[slot] = yy;
            };
            for (int y = ya; y < yb; y++) {
                std::fill(acc.begin(), acc.begin() + tw, pad);
                for (int i = 0; i < k; i++) {
                    const int j = sh.widx[i], yy = y + i - r;
                    if (j < 0 || yy < 0 || yy >= h) continue;
                    const int slot = yy % k;
                    if (ring_row[slot] != yy) fill(yy);
                    const float* o = &ring[((size_t)slot * m + j) * TW];
                    if (is_max) {
                        for (int x = 0; x < tw; x++)
                            if (o[x] > acc[x]) acc[x] = o[x];
                    } else {
                        for (int x = 0; x < tw; x++)
                            if (o[x] < acc[x]) acc[x] = o[x];
                    }
                }
                std::copy(acc.begin(), acc.begin() + tw, dst + (int64_t)(y - dst_row0) * w + x0);
            }
        }
    }, ntasks);
}

}  // namespace

// filters.js ellipseMorph. Plain values (the engine's images) go through ellipse_tiles; an image holding NaN or
// -0 keeps the JS's own order of comparisons (ellipse_rows), so it too matches bit for bit.
void ellipse_morph(const float* src, int w, int h, int k, const int32_t* hw, bool is_max, float* dst) {
    const Shape sh(k, hw);
    if (sh.widths.empty()) return;
    if (plain_floats(src, (int64_t)w * h)) ellipse_tiles(src, 0, w, h, k, sh, is_max, dst, 0, 0, h);
    else ellipse_rows(src, w, h, k, hw, is_max, dst);
}

// the JS's algorithm, row by row: each source row filtered once per distinct half-width (each grown from the
// previous one where it can be) and kept in a ring of k rows the image's full width
void ellipse_rows(const float* src, int w, int h, int k, const int32_t* hw, bool is_max, float* dst) {
    const int r = k >> 1;
    const Shape sh(k, hw);
    const std::vector<int>& widths = sh.widths;
    const std::vector<int>& widx = sh.widx;
    const int m = (int)widths.size();
    const int max_r = widths[m - 1];
    const float pad = is_max ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    parallel_rows(h, (int64_t)w * (k + 2 * m), [&](int ya, int yb) {
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

// filters.js ellipseOpenClose: open = erode then dilate (bright), close = dilate then erode (dark). For plain
// values in bands of output rows: each band's first pass covers the k/2 rows either side its second pass reads,
// so the intermediate image is never held whole.
void ellipse_open_close(const float* src, int w, int h, int k, const int32_t* hw, bool open, float* dst) {
    const Shape sh(k, hw);
    if (sh.widths.empty()) return;
    if (!plain_floats(src, (int64_t)w * h)) {
        std::vector<float> a((size_t)w * h);
        ellipse_morph(src, w, h, k, hw, !open, a.data());
        ellipse_morph(a.data(), w, h, k, hw, open, dst);
        return;
    }
    const int r = k >> 1, band = std::max(512, 8 * k);
    std::vector<float> mid((size_t)w * std::min(h, band + 2 * r));
    for (int y0 = 0; y0 < h; y0 += band) {
        const int y1 = std::min(h, y0 + band), ra = std::max(0, y0 - r), rb = std::min(h, y1 + r);
        ellipse_tiles(src, 0, w, h, k, sh, !open, mid.data(), ra, ra, rb);
        ellipse_tiles(mid.data(), ra, w, h, k, sh, open, dst, 0, y0, y1);
    }
}

void hypot_into(float* gy, const float* gx, int64_t n) {
    const int64_t rows = (n + 4095) / 4096;
    parallel_rows((int)rows, 4096 * 8, [&](int a, int b) {
        const int64_t e = std::min(n, (int64_t)b * 4096);
        for (int64_t i = (int64_t)a * 4096; i < e; i++) gy[i] = (float)v8_hypot(gx[i], gy[i]);
    });
}

// ---------- linear filters ----------------------------------------------------------------------

namespace {

// Gaussian rows: out[x] = row[x + r] kc + the sum over j = 1 .. r of ker[r - j] (row[x + r - j] + row[x + r + j]),
// each pixel's terms added in the JS order, a whole row a term so the loop vectorises (row: r mirrored pixels each
// side)
FD_INLINE void conv_row(const float* row, int w, int r, const float* ker, double* acc, float* out) {
    const double kc = ker[r];
    for (int x = 0; x < w; x++) acc[x] = (double)row[x + r] * kc;
    for (int j = 1; j <= r; j++) {
        const double kj = ker[r - j];
        const float *a = row + r - j, *b = row + r + j;
        for (int x = 0; x < w; x++) acc[x] += kj * ((double)a[x] + (double)b[x]);
    }
    for (int x = 0; x < w; x++) out[x] = (float)acc[x];
}

FD_INLINE void conv_rows(const float* src, int w, int ya, int yb, int r, const float* ker, float* row, double* acc, float* dst, int dst_row0) {
    for (int y = ya; y < yb; y++) {
        const float* s = src + (int64_t)y * w;
        for (int x = -r; x < w + r; x++) row[x + r] = s[reflect101(x, w)];
        conv_row(row, w, r, ker, acc, dst + (int64_t)(y - dst_row0) * w);
    }
}

// Gaussian columns for rows [ya, yb): rows y - j and y + j of the row pass (mirrored at the image's edges), which
// tmp holds from row tmp_row0 on
FD_INLINE void conv_cols(const float* tmp, int tmp_row0, int w, int h, int ya, int yb, int r, const float* ker, double* acc, float* dst, int dst_row0) {
    const double kc = ker[r];
    for (int y = ya; y < yb; y++) {
        const float* c = tmp + (int64_t)(y - tmp_row0) * w;
        for (int x = 0; x < w; x++) acc[x] = (double)c[x] * kc;
        for (int j = 1; j <= r; j++) {
            const float *a = tmp + (int64_t)(reflect101(y - j, h) - tmp_row0) * w, *b = tmp + (int64_t)(reflect101(y + j, h) - tmp_row0) * w;
            const double kv = ker[r - j];
            for (int x = 0; x < w; x++) acc[x] += kv * ((double)a[x] + (double)b[x]);
        }
        float* o = dst + (int64_t)(y - dst_row0) * w;
        for (int x = 0; x < w; x++) o[x] = (float)acc[x];
    }
}

void conv_rows_plain(const float* src, int w, int ya, int yb, int r, const float* ker, float* row, double* acc, float* dst, int dst_row0) {
    conv_rows(src, w, ya, yb, r, ker, row, acc, dst, dst_row0);
}
FD_AVX2 void conv_rows_avx2(const float* src, int w, int ya, int yb, int r, const float* ker, float* row, double* acc, float* dst, int dst_row0) {
    conv_rows(src, w, ya, yb, r, ker, row, acc, dst, dst_row0);
}
void conv_cols_plain(const float* tmp, int tmp_row0, int w, int h, int ya, int yb, int r, const float* ker, double* acc, float* dst, int dst_row0) {
    conv_cols(tmp, tmp_row0, w, h, ya, yb, r, ker, acc, dst, dst_row0);
}
FD_AVX2 void conv_cols_avx2(const float* tmp, int tmp_row0, int w, int h, int ya, int yb, int r, const float* ker, double* acc, float* dst, int dst_row0) {
    conv_cols(tmp, tmp_row0, w, h, ya, yb, r, ker, acc, dst, dst_row0);
}

}  // namespace

// The Gaussian in bands of output rows: each band's row pass covers the band and the r rows either side its
// column pass reads (the rows it needs at the image's edges, mirrored, fall inside that range too), so only one
// band's worth of the row pass is ever held instead of a second whole image.
void sep_conv(const float* src, int w, int h, const float* ker, int klen, float* dst) {
    const int r = klen >> 1;
    const bool avx2 = use_avx2();
    const int band = std::max(256, 16 * r);
    std::vector<float> tmp((size_t)w * std::min(h, band + 2 * r));
    for (int y0 = 0; y0 < h; y0 += band) {
        const int y1 = std::min(h, y0 + band), ra = std::max(0, y0 - r), rb = std::min(h, y1 + r);
        parallel_rows(rb - ra, (int64_t)w * klen, [&](int a, int b) {
            std::vector<float> row(w + 2 * r);
            std::vector<double> acc(w);
            if (avx2) conv_rows_avx2(src, w, ra + a, ra + b, r, ker, row.data(), acc.data(), tmp.data(), ra);
            else conv_rows_plain(src, w, ra + a, ra + b, r, ker, row.data(), acc.data(), tmp.data(), ra);
        });
        parallel_rows(y1 - y0, (int64_t)w * klen, [&](int a, int b) {
            std::vector<double> acc(w);
            if (avx2) conv_cols_avx2(tmp.data(), ra, w, h, y0 + a, y0 + b, r, ker, acc.data(), dst, 0);
            else conv_cols_plain(tmp.data(), ra, w, h, y0 + a, y0 + b, r, ker, acc.data(), dst, 0);
        });
    }
}

static void box_h(const float* src, float* dst, int w, int h, int r) {
    const double inv = 1.0 / (2 * r + 1);
    parallel_rows(h, (int64_t)w * 2, [&](int ya, int yb) {
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
    parallel_rows(w, (int64_t)h * 2, [&](int xa, int xb) {
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

namespace {

// Sobel along x for rows [ya, yb): k0 left + k1 centre, then + k2 right (the JS's order), edges mirrored (P: w + 2);
// row y goes to tmp + (y - tmp_row0) w
FD_INLINE void sobel_x(const float* src, int w, int ya, int yb, const double* kx, float* P, float* tmp, int tmp_row0) {
    for (int y = ya; y < yb; y++) {
        const float* s = src + (int64_t)y * w;
        P[0] = s[reflect101(-1, w)];
        std::copy(s, s + w, P + 1);
        P[w + 1] = s[reflect101(w, w)];
        float* o = tmp + (int64_t)(y - tmp_row0) * w;
        for (int x = 0; x < w; x++) {
            double t = kx[0] * (double)P[x] + kx[1] * (double)P[x + 1];
            t = t + kx[2] * (double)P[x + 2];
            o[x] = (float)t;
        }
    }
}

// Sobel along y for rows [ya, yb), rows mirrored at the edges (tmp holds the x pass from row tmp_row0 on)
FD_INLINE void sobel_y(const float* tmp, int tmp_row0, int w, int h, int ya, int yb, const double* ky, float* dst, int dst_row0) {
    for (int y = ya; y < yb; y++) {
        const float *up = tmp + (int64_t)(reflect101(y - 1, h) - tmp_row0) * w, *mid = tmp + (int64_t)(y - tmp_row0) * w,
                    *dn = tmp + (int64_t)(reflect101(y + 1, h) - tmp_row0) * w;
        float* o = dst + (int64_t)(y - dst_row0) * w;
        for (int x = 0; x < w; x++) {
            double t = ky[0] * (double)up[x] + ky[1] * (double)mid[x];
            t = t + ky[2] * (double)dn[x];
            o[x] = (float)t;
        }
    }
}

void sobel_x_plain(const float* src, int w, int ya, int yb, const double* kx, float* P, float* tmp, int r0) { sobel_x(src, w, ya, yb, kx, P, tmp, r0); }
FD_AVX2 void sobel_x_avx2(const float* src, int w, int ya, int yb, const double* kx, float* P, float* tmp, int r0) { sobel_x(src, w, ya, yb, kx, P, tmp, r0); }
void sobel_y_plain(const float* tmp, int r0, int w, int h, int ya, int yb, const double* ky, float* dst, int d0) { sobel_y(tmp, r0, w, h, ya, yb, ky, dst, d0); }
FD_AVX2 void sobel_y_avx2(const float* tmp, int r0, int w, int h, int ya, int yb, const double* ky, float* dst, int d0) { sobel_y(tmp, r0, w, h, ya, yb, ky, dst, d0); }

const double SOBEL_K[3][3] = {{1, 2, 1}, {-1, 0, 1}, {1, -2, 1}};   // smoothing, first and second derivative

}  // namespace

// rows [y0, y1) of the Sobel filter, written from out (row y at out + (y - y0) w); tmp: y1 - y0 + 2 rows of scratch
void sobel_rows(const float* src, int w, int h, int dx, int dy, int y0, int y1, float* out, float* tmp) {
    const double *kx = SOBEL_K[dx], *ky = SOBEL_K[dy];
    const bool avx2 = use_avx2();
    const int ra = std::max(0, y0 - 1), rb = std::min(h, y1 + 1);
    parallel_rows(rb - ra, (int64_t)w * 3, [&](int a, int b) {
        std::vector<float> P(w + 2);
        if (avx2) sobel_x_avx2(src, w, ra + a, ra + b, kx, P.data(), tmp, ra);
        else sobel_x_plain(src, w, ra + a, ra + b, kx, P.data(), tmp, ra);
    });
    parallel_rows(y1 - y0, (int64_t)w * 3, [&](int a, int b) {
        if (avx2) sobel_y_avx2(tmp, ra, w, h, y0 + a, y0 + b, ky, out, y0);
        else sobel_y_plain(tmp, ra, w, h, y0 + a, y0 + b, ky, out, y0);
    });
}

// filters.js sobel (ksize 3, REFLECT_101), in bands of rows so its x pass never needs a second whole image
void sobel(const float* src, int w, int h, int dx, int dy, float* dst) {
    const int band = 512;
    std::vector<float> tmp((size_t)w * (std::min(h, band) + 2));
    for (int y0 = 0; y0 < h; y0 += band) {
        const int y1 = std::min(h, y0 + band);
        sobel_rows(src, w, h, dx, dy, y0, y1, dst + (int64_t)y0 * w, tmp.data());
    }
}

}  // namespace fd

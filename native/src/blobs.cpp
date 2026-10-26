// detect.js BlobField.tile for several tiles at once: the scale-normalised DoH response on the tile and its margin,
// its local maxima (response >= 1 and equal to its k x k dilation) inside the tile, and the tile's own response for
// respAt. The JS path's filters (filters.cpp sep_conv, sobel x 3, regions.cpp rect_morph) with their sums written
// out as there, fused, on every thread, in buffers kept between calls (a fresh buffer per filter cost more than the
// filter): bit for bit the JS. Calls come one at a time from the JS thread.
#include <algorithm>
#include <cmath>
#include <vector>

#include "filters.h"

namespace fd {

namespace {

std::vector<float> g_a, g_b, g_resp;   // one tile's working images

inline int refl(int i, int n) {   // filters.cpp reflect101
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

const double KS[3] = {1, 2, 1}, KD1[3] = {-1, 0, 1}, KD2[3] = {1, -2, 1};

// sobel's first pass on one row: [k0 k1 k2] along x, edges mirrored, rounded to float
inline void xpass(const float* s, int w, const double* k, float* o) {
    for (int x = 0; x < w; x++) {
        double t = k[0] * (double)s[refl(x - 1, w)] + k[1] * (double)s[x];
        t = t + k[2] * (double)s[refl(x + 1, w)];
        o[x] = (float)t;
    }
}

// sobel's second pass at one pixel from the three first-pass rows
inline float ypass(const double* k, float u, float m, float d) {
    double t = k[0] * (double)u + k[1] * (double)m;
    t = t + k[2] * (double)d;
    return (float)t;
}

// regions.cpp rect_morph's van Herk line (max): n values at stride, padded by r each side
void vh_max(const float* in, int n, int stride, int r, float* out, int ostride, std::vector<float>& P, std::vector<float>& f,
            std::vector<float>& g) {
    const float pad = -INFINITY;
    const int kk = 2 * r + 1, len = n + 2 * r;
    for (int i = 0; i < r; i++) P[i] = pad;
    for (int i = 0; i < n; i++) P[r + i] = in[(int64_t)i * stride];
    for (int i = 0; i < r; i++) P[r + n + i] = pad;
    for (int b0 = 0; b0 < len; b0 += kk) {
        const int b1 = b0 + kk < len ? b0 + kk : len;
        float m = pad;
        for (int i = b0; i < b1; i++) {
            const float v = P[i];
            if (v > m) m = v;
            f[i] = m;
        }
        m = pad;
        for (int i = b1 - 1; i >= b0; i--) {
            const float v = P[i];
            if (v > m) m = v;
            g[i] = m;
        }
    }
    for (int i = 0; i < n; i++) {
        const float a = g[i], b = f[i + kk - 1];
        out[(int64_t)i * ostride] = a > b ? a : b;
    }
}

}  // namespace

void blob_tiles(const float* Y, const float* sl, int W, int H, int T, double sq, int sgn, const float* ker, int klen,
                int k, int mg, const int32_t* tiles, int n, std::vector<BlobTile>& out) {
    out.assign(n, BlobTile());
    const double s2 = sq * sq, kc = ker[klen >> 1];
    const int r = klen >> 1, rm = k >> 1;
    for (int i = 0; i < n; i++) {
        const int ty = tiles[2 * i], tx = tiles[2 * i + 1];
        const int y0 = ty * T, x0 = tx * T, y1 = std::min(H, y0 + T), x1 = std::min(W, x0 + T);
        const int ya = std::max(0, y0 - mg), yb = std::min(H, y1 + mg), xa = std::max(0, x0 - mg), xb = std::min(W, x1 + mg);
        const int w = xb - xa, h = yb - ya;
        const size_t np = (size_t)w * h;
        if (g_a.size() < np) {
            g_a.resize(np);
            g_b.resize(np);
            g_resp.resize(np);
        }
        float* A = g_a.data();
        float* B = g_b.data();
        float* resp = g_resp.data();
        // the Gaussian (sep_conv on the window's crop, edges mirrored at the window): rows into A, columns into B
        parallel_rows(h, (int64_t)w * klen, [&](int a, int b) {
            std::vector<float> row(w + 2 * r);
            for (int y = a; y < b; y++) {
                const float* s = Y + (int64_t)(ya + y) * W + xa;
                for (int x = -r; x < w + r; x++) row[x + r] = s[refl(x, w)];
                float* o = A + (size_t)y * w;
                for (int x = 0; x < w; x++) {
                    const int c = x + r;
                    double acc = (double)row[c] * kc;
                    for (int j = 1; j <= r; j++) acc += (double)ker[r - j] * ((double)row[c - j] + (double)row[c + j]);
                    o[x] = (float)acc;
                }
            }
        });
        parallel_rows(h, (int64_t)w * klen, [&](int a, int b) {
            std::vector<double> acc(w);
            for (int y = a; y < b; y++) {
                const size_t oc = (size_t)y * w;
                for (int x = 0; x < w; x++) acc[x] = (double)A[oc + x] * kc;
                for (int j = 1; j <= r; j++) {
                    const size_t ra = (size_t)refl(y - j, h) * w, rb = (size_t)refl(y + j, h) * w;
                    const double kv = ker[r - j];
                    for (int x = 0; x < w; x++) acc[x] += kv * ((double)A[ra + x] + (double)A[rb + x]);
                }
                for (int x = 0; x < w; x++) B[oc + x] = (float)acc[x];
            }
        });
        // Lxx (d2 along x, smoothed along y), Lyy (smoothed along x, d2 along y), Lxy (d1, d1) and the response
        parallel_rows(h, (int64_t)w * 30, [&](int a, int b) {
            std::vector<float> bufs((size_t)9 * w);
            float *d2u = &bufs[0], *d2m = d2u + w, *d2d = d2m + w, *su = d2d + w, *sm = su + w, *sd = sm + w, *d1u = sd + w,
                  *d1m = d1u + w, *d1d = d1m + w;
            for (int y = a; y < b; y++) {
                const float *gu = B + (size_t)refl(y - 1, h) * w, *gm = B + (size_t)y * w, *gd = B + (size_t)refl(y + 1, h) * w;
                xpass(gu, w, KD2, d2u); xpass(gm, w, KD2, d2m); xpass(gd, w, KD2, d2d);
                xpass(gu, w, KS, su); xpass(gm, w, KS, sm); xpass(gd, w, KS, sd);
                xpass(gu, w, KD1, d1u); xpass(gm, w, KD1, d1m); xpass(gd, w, KD1, d1d);
                const float* sv = sl + (int64_t)(ya + y) * W + xa;
                float* o = resp + (size_t)y * w;
                for (int x = 0; x < w; x++) {
                    const float lxx = ypass(KS, d2u[x], d2m[x], d2d[x]);
                    const float lyy = ypass(KD2, su[x], sm[x], sd[x]);
                    const float lxy = ypass(KD1, d1u[x], d1m[x], d1d[x]);
                    const double p = sgn * (double)lxx, q = sgn * (double)lyy;
                    float v = 0.0f;
                    if (p < 0 && q < 0) {
                        const double c = (double)lxy;
                        const double det = p * q - c * c;
                        if (det > 0) v = (float)((std::sqrt(det) * s2) / std::max((double)sv[x], 1e-4));
                    }
                    o[x] = v;
                }
            }
        });
        // its k x k dilation (rect_morph: rows into A, columns into B)
        parallel_rows(h, (int64_t)w * 3, [&](int a, int b) {
            std::vector<float> P(w + 2 * rm), f(w + 4 * rm + 1), g(w + 4 * rm + 1);
            for (int y = a; y < b; y++) vh_max(resp + (size_t)y * w, w, 1, rm, A + (size_t)y * w, 1, P, f, g);
        });
        parallel_rows(w, (int64_t)h * 3, [&](int a, int b) {
            std::vector<float> P(h + 2 * rm), f(h + 4 * rm + 1), g(h + 4 * rm + 1);
            for (int x = a; x < b; x++) vh_max(A + x, h, w, rm, B + x, w, P, f, g);
        }, 16);
        BlobTile& t = out[i];
        for (int y = y0; y < y1; y++) {
            const size_t o = (size_t)(y - ya) * w - xa;
            for (int x = x0; x < x1; x++) {
                const float v = resp[o + x];
                if (v >= 1.0f && v == B[o + x]) {
                    t.ys.push_back(y);
                    t.xs.push_back(x);
                    t.vals.push_back(v);
                }
            }
        }
        const int iw = x1 - x0;
        t.inner.resize((size_t)(y1 - y0) * iw);
        for (int y = y0; y < y1; y++) {
            const size_t o = (size_t)(y - ya) * w + (x0 - xa);
            std::copy(resp + o, resp + o + iw, t.inner.begin() + (size_t)(y - y0) * iw);
        }
    }
}

}  // namespace fd

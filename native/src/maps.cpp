// Native ports of engine/maps.js's per-block statistics and full-image loops, and of
// filters.js npGradient / resizeLinear: the same arithmetic in the same order as the JS.
// Order statistics (medians, k-th values) are unique values, so std::nth_element gives exactly
// what the JS selection gives.
#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

#include "filters.h"
#include "stats.h"

namespace fd {

// maps.js blockStat + noiseLocal's / busyness's per-block statistic.
// mode 0: noiseLocal (1.4826 x MAD over valid pixels, NaN when none); mode 1: busyness T
void block_stats(const float* resid, const uint8_t* valid, int W, int H, int block, int mode, bool bright, float* grid) {
    const int hb = std::max(1, H / block), wb = std::max(1, W / block);
    const int n = block * block;
    const double sgn = bright ? 1 : -1;
    parallel_rows(hb * wb, (int64_t)n * 8, [&](int a, int b) {
        std::vector<double> vals(n), buf(n);
        std::vector<uint8_t> vok(n);
        for (int bi = a; bi < b; bi++) {
            const int by = bi / wb, bx = bi % wb;
            int k = 0, nvalid = 0;
            for (int y = by * block; y < (by + 1) * block; y++) {
                const int64_t off = (int64_t)y * W;
                for (int x = bx * block; x < (bx + 1) * block; x++) {
                    vals[k] = resid[off + x];
                    const int v = valid[off + x] ? 1 : 0;
                    vok[k++] = (uint8_t)v;
                    nvalid += v;
                }
            }
            double r;
            if (mode == 0) {
                int c = 0;
                for (int i = 0; i < n; i++)
                    if (vok[i]) buf[c++] = vals[i];
                if (!c) {
                    r = NAN;
                } else {
                    const double md = median_of(buf.data(), c);
                    int c2 = 0;
                    for (int i = 0; i < n; i++)
                        if (vok[i]) buf[c2++] = std::fabs(vals[i] - md);
                    r = 1.4826 * median_of(buf.data(), c2);
                }
            } else {
                for (int i = 0; i < n; i++) buf[i] = sgn * vals[i];
                const double md = median_of(buf.data(), n);
                for (int i = 0; i < n; i++) buf[i] = sgn * vals[i];
                const double p99 = kth(buf.data(), n, (int)std::floor(0.99 * n));
                for (int i = 0; i < n; i++) buf[i] = std::fabs(sgn * vals[i] - md);
                const double mad = 1.4826 * median_of(buf.data(), n);
                double T = (p99 - md) / (2.33 * std::max(mad, 1e-5));
                if ((double)nvalid / n < 0.5) T = 1.0;
                r = T;
            }
            grid[bi] = (float)r;
        }
    }, 4);
}

// maps.js noiseByLevel: robust grain sigma per brightness level of bg, interpolated per pixel
void noise_by_level(const float* bg, const float* resid, const uint8_t* valid, int W, int H, int bins, float* out) {
    std::vector<std::vector<double>> buckets(bins);
    for (int y = 0; y < H; y += 3)
        for (int x = 0; x < W; x += 3) {
            const int64_t i = (int64_t)y * W + x;
            if (!valid[i]) continue;
            int l = (int)std::floor((double)bg[i] * bins);
            if (l < 0) l = 0;
            if (l > bins - 1) l = bins - 1;
            buckets[l].push_back(resid[i]);
        }
    std::vector<double> sig(bins, NAN);
    parallel_rows(bins, 1 << 16, [&](int a, int b) {
        for (int k = a; k < b; k++) {
            auto& rb = buckets[k];
            if ((int)rb.size() > 400) {
                std::vector<double> c(rb);
                const double md = percentile(c.data(), (int)c.size(), 50);
                for (size_t i = 0; i < rb.size(); i++) c[i] = std::fabs(rb[i] - md);
                sig[k] = 1.4826 * percentile(c.data(), (int)c.size(), 50);
            }
        }
    }, 1);
    std::vector<int> good;
    for (int b = 0; b < bins; b++)
        if (!std::isnan(sig[b])) good.push_back(b);
    std::vector<double> filled(bins);
    if (good.empty()) {
        std::fill(filled.begin(), filled.end(), 0.01);
    } else {
        // interp1(b, good, sig[good]) (maps.js)
        const int n = (int)good.size();
        for (int b = 0; b < bins; b++) {
            const double x = b;
            if (x <= good[0]) { filled[b] = sig[good[0]]; continue; }
            if (x >= good[n - 1]) { filled[b] = sig[good[n - 1]]; continue; }
            int lo = 0, hi = n - 1;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (good[mid] <= x) lo = mid; else hi = mid;
            }
            const double t = (x - good[lo]) / (double)(good[hi] - good[lo]);
            filled[b] = sig[good[lo]] + t * (sig[good[hi]] - sig[good[lo]]);
        }
    }
    std::vector<double> med3(bins);
    for (int b = 0; b < bins; b++) {
        double v[3] = {filled[b == 0 ? 0 : b - 1], filled[b], filled[b == bins - 1 ? bins - 1 : b + 1]};
        std::sort(v, v + 3);
        med3[b] = std::max(v[1], 0.002);
    }
    std::vector<double> xs(bins);
    for (int i = 0; i < bins; i++) xs[i] = (i + 0.5) / bins;
    const double x0 = xs[0], xl = xs[bins - 1];
    parallel_rows(H, W, [&](int ya, int yb) {
        for (int64_t i = (int64_t)ya * W; i < (int64_t)yb * W; i++) {
            const double x = bg[i];
            if (x <= x0) { out[i] = (float)med3[0]; continue; }
            if (x >= xl) { out[i] = (float)med3[bins - 1]; continue; }
            int lo = (int)std::floor((x - x0) * bins);
            if (lo < 0) lo = 0; else if (lo > bins - 2) lo = bins - 2;
            while (lo > 0 && xs[lo] > x) lo--;
            while (lo < bins - 2 && xs[lo + 1] <= x) lo++;
            const double t = (x - xs[lo]) / (xs[lo + 1] - xs[lo]);
            out[i] = (float)(med3[lo] + t * (med3[lo + 1] - med3[lo]));
        }
    });
}

// maps.js residualMaps' last loop: strength in grain units at three scales
void z_maps(const float* Y, const float* m15, const float* o15, const float* m41, const float* o41, const float* bxl,
            const float* sigma, const uint8_t* outside, int64_t N, bool bright, float* zs, float* zl, float* zx) {
    const double sgn = bright ? 1 : -1;
    const int64_t chunk = 1 << 16;
    const int nchunks = (int)((N + chunk - 1) / chunk);
    parallel_rows(nchunks, chunk, [&](int a, int b) {
        for (int64_t i = (int64_t)a * chunk; i < std::min(N, (int64_t)b * chunk); i++) {
            const float b15 = bright ? std::max(m15[i], o15[i]) : std::min(m15[i], o15[i]);
            const float bb = bright ? std::max(m41[i], o41[i]) : std::min(m41[i], o41[i]);
            if (outside[i]) {   // the JS leaves these at their initial 0
                zs[i] = zl[i] = zx[i] = 0.0f;
                continue;
            }
            zs[i] = (float)((sgn * ((double)Y[i] - (double)b15)) / (double)sigma[i]);
            zl[i] = (float)((sgn * ((double)Y[i] - (double)bb)) / (double)sigma[i]);
            zx[i] = (float)((sgn * ((double)Y[i] - (double)bxl[i])) / (double)sigma[i]);
        }
    }, 1);
}

// filters.js npGradient: [gy, gx], central differences inside, one-sided at the edges
void np_gradient(const float* src, int w, int h, float* gy, float* gx) {
    parallel_rows(h, w, [&](int ya, int yb) {
        for (int y = ya; y < yb; y++) {
            const int64_t off = (int64_t)y * w;
            for (int x = 0; x < w; x++) {
                if (x == 0) gx[off] = (float)((double)src[off + 1] - (double)src[off]);
                else if (x == w - 1) gx[off + x] = (float)((double)src[off + x] - (double)src[off + x - 1]);
                else gx[off + x] = (float)(0.5 * ((double)src[off + x + 1] - (double)src[off + x - 1]));
                if (y == 0) gy[off + x] = (float)((double)src[off + w + x] - (double)src[off + x]);
                else if (y == h - 1) gy[off + x] = (float)((double)src[off + x] - (double)src[off - w + x]);
                else gy[off + x] = (float)(0.5 * ((double)src[off + w + x] - (double)src[off - w + x]));
            }
        }
    });
}

// filters.js resizeLinear (cv2.resize INTER_LINEAR)
void resize_linear(const float* src, int sw, int sh, int dw, int dh, float* dst) {
    const double fx = (double)sw / dw, fy = (double)sh / dh;
    parallel_rows(dh, (int64_t)dw * 2, [&](int ya, int yb) {
        for (int y = ya; y < yb; y++) {
            double sy = (y + 0.5) * fy - 0.5;
            if (sy < 0) sy = 0;
            int y0 = (int)std::floor(sy);
            if (y0 >= sh - 1) y0 = sh - 1;
            const int y1 = std::min(y0 + 1, sh - 1);
            const double ay = sy - y0;
            for (int x = 0; x < dw; x++) {
                double sx = (x + 0.5) * fx - 0.5;
                if (sx < 0) sx = 0;
                int x0 = (int)std::floor(sx);
                if (x0 >= sw - 1) x0 = sw - 1;
                const int x1 = std::min(x0 + 1, sw - 1);
                const double ax = sx - x0;
                const double a = (double)src[(int64_t)y0 * sw + x0] * (1 - ax) + (double)src[(int64_t)y0 * sw + x1] * ax;
                const double b = (double)src[(int64_t)y1 * sw + x0] * (1 - ax) + (double)src[(int64_t)y1 * sw + x1] * ax;
                dst[(int64_t)y * dw + x] = (float)(a * (1 - ay) + b * ay);
            }
        }
    });
}

}  // namespace fd

namespace fd {

// detect.js ridgeMaps (soft fibres): bright-line strength at each scale against the local grain, the
// strongest per pixel and the scale it came from; float32 arithmetic exactly as the JS writes it.
// kers: the Gaussian kernels from filters.js gaussKernel, one after another (klens long).
// Memory: the blurred image and its second derivative along x are kept whole (the scale's grain constant k is a
// median over the whole image's x derivative); the other two derivatives are made a band of rows at a time.
void ridge_maps(const float* Y, int W, int H, const uint8_t* outside, bool bright, const float* sig, int ns,
                const double* scales, const float* kers, const int32_t* klens, float* best, float* bestS) {
    const int64_t N = (int64_t)W * H;
    std::fill(best, best + N, 0.0f);
    std::fill(bestS, bestS + N, 0.0f);
    const float tiny = (float)1e-5, c1 = (float)1.4826;
    const int band = 256;
    std::vector<float> g(N), xx(N), yy((size_t)W * std::min(H, band)), xy((size_t)W * std::min(H, band)),
        tmp((size_t)W * (std::min(H, band) + 2));
    size_t koff = 0;
    for (int si = 0; si < ns; si++) {
        const double s = scales[si];
        sep_conv(Y, W, H, kers + koff, klens[si], g.data());
        koff += klens[si];
        sobel(g.data(), W, H, 2, 0, xx.data());
        if (!bright)
            for (int64_t i = 0; i < N; i++) xx[i] = -xx[i];
        const float s2 = (float)(s * s);
        std::vector<float> samp;
        for (int y = 0; y < H; y += 5)
            for (int x = 0; x < W; x += 5) {
                const int64_t i = (int64_t)y * W + x;
                if (outside[i]) continue;
                const float a = (float)((double)c1 * std::fabs((double)xx[i]));
                const float b = (float)((double)a * (double)s2);
                samp.push_back((float)((double)b / (double)std::max(sig[i], tiny)));
            }
        float k = NAN;
        const size_t n = samp.size();
        if (n) {
            const size_t h = (n - 1) / 2;
            std::nth_element(samp.begin(), samp.begin() + h, samp.end());
            if (n % 2) {
                k = samp[h];
            } else {
                float u = INFINITY;
                for (size_t i = h + 1; i < n; i++)
                    if (samp[i] < u) u = samp[i];
                const float sum = (float)((double)samp[h] + (double)u);
                k = (float)((double)sum / 2);
            }
        }
        const float sf = (float)s;
        for (int y0 = 0; y0 < H; y0 += band) {
            const int y1 = std::min(H, y0 + band);
            sobel_rows(g.data(), W, H, 0, 2, y0, y1, yy.data(), tmp.data());
            sobel_rows(g.data(), W, H, 1, 1, y0, y1, xy.data(), tmp.data());
            parallel_rows(y1 - y0, (int64_t)W * 8, [&](int ra, int rb) {
                for (int64_t q = (int64_t)ra * W; q < (int64_t)rb * W; q++) {
                    const int64_t i = (int64_t)y0 * W + q;
                    const float yv = bright ? yy[q] : -yy[q], xv = bright ? xy[q] : -xy[q];
                    const float tr = (float)((double)xx[i] + (double)yv);
                    const float hh = (float)((double)tr / 2);
                    const float det = (float)((double)(float)((double)xx[i] * (double)yv) - (double)(float)((double)xv * (double)xv));
                    const float hh2 = (float)((double)hh * (double)hh);
                    const float dd = (float)((double)hh2 - (double)det);
                    const float disc = (float)std::sqrt(std::max((double)dd, 0.0));
                    const float a = (float)((double)hh - (double)disc);
                    const float b = (float)((double)hh + (double)disc);
                    const float e = (float)(-(double)a - std::fabs((double)b));
                    const float r = (float)(std::max((double)e, 0.0) * (double)s2);
                    const float den = (float)((double)k * (double)std::max(sig[i], tiny));
                    const float z = (float)((double)r / (double)den);
                    if (z > best[i]) {
                        best[i] = z;
                        bestS[i] = sf;
                    }
                }
            });
        }
    }
    for (int64_t i = 0; i < N; i++)
        if (outside[i]) best[i] = 0.0f;
}

}  // namespace fd

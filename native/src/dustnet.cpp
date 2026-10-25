// The dust network's forward pass (engine/dustnet.js), batched over marks and split over threads.
// The same loops in the same order as the JS, in double, so the logits are bit-identical.
// Weights arrive once (net_load) as one flat array: per tower (1 = v1, 2 = v2: local view, then
// context view) c1w c1b ... c5w c5b with channels 3-8-8-16-16-16, then d1w d1b d2w d2b.
#include <algorithm>
#include <functional>
#include <thread>
#include <algorithm>
#include <cmath>
#include <vector>

#include "filters.h"

namespace fd {

namespace {
const int CH[6] = {3, 8, 8, 16, 16, 16};
struct Conv { std::vector<double> w, b; int ci, co; };
struct Net { int towers = 0, hidden = 0; std::vector<Conv> conv; std::vector<double> d1w, d1b, d2w, d2b; };
Net g_nets[4];   // slots: 0 the shipped v1 network, 1 the v2 network (patch + context)
thread_local const Net* tl_net = nullptr;

void conv(const std::vector<double>& x, int ci, int n, const Conv& c, std::vector<double>& out) {
    const int nn = n * n;
    out.assign((size_t)c.co * nn, 0.0);
    for (int o = 0; o < c.co; o++) {
        double* acc = &out[(size_t)o * nn];
        for (int k = 0; k < nn; k++) acc[k] = c.b[o];
        for (int i = 0; i < ci; i++) {
            const double* xi = &x[(size_t)i * nn];
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    const double wv = c.w[(((size_t)o * ci + i) * 3 + ky) * 3 + kx];
                    for (int y = 0; y < n; y++) {
                        const int sy = y + ky - 1;
                        if (sy < 0 || sy >= n) {
                            for (int xx = 0; xx < n; xx++) acc[y * n + xx] += wv * 0.0;
                            continue;
                        }
                        for (int xx = 0; xx < n; xx++) {
                            const int sx = xx + kx - 1;
                            acc[y * n + xx] += wv * (sx < 0 || sx >= n ? 0.0 : xi[sy * n + sx]);
                        }
                    }
                }
        }
        for (int k = 0; k < nn; k++)
            if (!(acc[k] > 0)) acc[k] = 0.0;
    }
}

void pool(const std::vector<double>& x, int c, int n, std::vector<double>& out) {
    const int m = n >> 1;
    out.assign((size_t)c * m * m, 0.0);
    for (int ch = 0; ch < c; ch++) {
        const double* a = &x[(size_t)ch * n * n];
        double* p = &out[(size_t)ch * m * m];
        for (int y = 0; y < m; y++)
            for (int xx = 0; xx < m; xx++) {
                const int i = 2 * y * n + 2 * xx;
                p[y * m + xx] = std::max(std::max(a[i], a[i + 1]), std::max(a[i + n], a[i + n + 1]));
            }
    }
}

// one tower on a (3, 40, 40) input -> 48 features (mean, max, centre of the last 16 channels)
void tower(const double* in, int t, double* feats) {
    std::vector<double> a(in, in + 3 * 1600), b;
    const Conv* c = &tl_net->conv[(size_t)t * 5];
    conv(a, 3, 40, c[0], b);
    conv(b, 8, 40, c[1], a);
    pool(a, 8, 40, b);
    conv(b, 8, 20, c[2], a);
    pool(a, 16, 20, b);
    conv(b, 16, 10, c[3], a);
    pool(a, 16, 10, b);
    conv(b, 16, 5, c[4], a);
    for (int ch = 0; ch < 16; ch++) {
        const double* v = &a[(size_t)ch * 25];
        double s = 0.0;
        for (int k = 0; k < 25; k++) s += v[k];
        feats[ch] = s / 25;
        double m = -1.0 / 0.0;
        for (int k = 0; k < 25; k++)
            if (v[k] > m) m = v[k];
        feats[16 + ch] = m;
        feats[32 + ch] = v[2 * 5 + 2];
    }
}
}  // namespace

bool net_load(int slot, const double* flat, size_t n, int towers, int hidden) {
    if (slot < 0 || slot >= 4) return false;
    Net net;
    net.towers = towers;
    net.hidden = hidden;
    size_t p = 0;
    auto take = [&](size_t k, std::vector<double>& v) {
        if (p + k > n) return false;
        v.assign(flat + p, flat + p + k);
        p += k;
        return true;
    };
    for (int t = 0; t < towers; t++)
        for (int l = 0; l < 5; l++) {
            Conv c;
            c.ci = CH[l];
            c.co = CH[l + 1];
            if (!take((size_t)c.co * c.ci * 9, c.w) || !take((size_t)c.co, c.b)) return false;
            net.conv.push_back(std::move(c));
        }
    const size_t nf = (size_t)48 * towers;
    if (!take((size_t)hidden * nf, net.d1w) || !take((size_t)hidden, net.d1b) || !take((size_t)hidden, net.d2w) ||
        !take(1, net.d2b) || p != n)
        return false;
    g_nets[slot] = std::move(net);
    return true;
}

int net_towers(int slot) { return slot >= 0 && slot < 4 ? g_nets[slot].towers : 0; }

namespace {
// regions.js medianOf: the middle order statistic, or the mean of the two (a scratch copy is reordered)
double median_of(std::vector<double>& a) {
    const int n = (int)a.size(), h = (n - 1) >> 1;
    std::nth_element(a.begin(), a.begin() + h, a.end());
    const double v = a[h];
    if (n % 2) return v;
    double u = 1.0 / 0.0;
    for (int i = h + 1; i < n; i++) if (a[i] < u) u = a[i];
    return (v + u) / 2;
}

// dustnet.js normalise: each channel less its median, over the patch's grain (1.0483 x the median |difference| of
// horizontal neighbours in its luminance), sign-flipped for dark dust, clipped to +-12, x 0.25
void normalise(const double* vals, bool bright, double* x) {
    const double sgn = bright ? 1.0 : -1.0;
    std::vector<double> d;
    d.reserve(40 * 39);
    for (int r = 0; r < 40; r++) for (int q = 0; q < 39; q++) {
        const int k = r * 40 + q;
        const double a = 0.2126 * vals[k] + 0.7152 * vals[1600 + k] + 0.0722 * vals[3200 + k];
        const double b = 0.2126 * vals[k + 1] + 0.7152 * vals[1600 + k + 1] + 0.0722 * vals[3200 + k + 1];
        d.push_back(std::fabs(b - a));
    }
    const double s = std::max(median_of(d) * 1.0483, 1e-4);
    std::vector<double> c(1600);
    for (int ch = 0; ch < 3; ch++) {
        c.assign(vals + ch * 1600, vals + ch * 1600 + 1600);
        const double m = median_of(c);
        for (int k = 0; k < 1600; k++) x[ch * 1600 + k] = std::min(12.0, std::max(-12.0, sgn * (vals[ch * 1600 + k] - m) / s)) * 0.25;
    }
}

inline double px(const uint16_t* img, int C, int64_t o, int c) { return (double)(float)((double)img[o + (C == 1 ? 0 : c)] / 65535); }

// dustnet.js patchInput: the 40 x 40 patch round (cy, cx), edges repeated
void patch_input(const uint16_t* img, int W, int H, int C, int cy, int cx, bool bright, double* x) {
    std::vector<double> vals(4800);
    for (int r = 0; r < 40; r++) {
        const int y = std::min(H - 1, std::max(0, cy - 20 + r));
        for (int q = 0; q < 40; q++) {
            const int xx = std::min(W - 1, std::max(0, cx - 20 + q));
            const int64_t o = ((int64_t)y * W + xx) * C;
            for (int c = 0; c < 3; c++) vals[c * 1600 + r * 40 + q] = px(img, C, o, c);
        }
    }
    normalise(vals.data(), bright, x);
}

// dustnet.js contextInput: the 160 x 160 neighbourhood, each 4 x 4 block summed in row order, / 16
void context_input(const uint16_t* img, int W, int H, int C, int cy, int cx, bool bright, double* x) {
    int xs[160];
    for (int q = 0; q < 160; q++) xs[q] = std::min(W - 1, std::max(0, cx - 80 + q));
    std::vector<double> acc(4800, 0.0);
    for (int by = 0; by < 40; by++) {
        for (int dy = 0; dy < 4; dy++) {
            const int y = std::min(H - 1, std::max(0, cy - 80 + 4 * by + dy));
            const int64_t ro = (int64_t)y * W;
            for (int bx = 0; bx < 40; bx++) {
                const int k = by * 40 + bx;
                for (int dx = 0; dx < 4; dx++) {
                    const int64_t o = (ro + xs[4 * bx + dx]) * C;
                    for (int c = 0; c < 3; c++) acc[c * 1600 + k] += px(img, C, o, c);
                }
            }
        }
    }
    for (int k = 0; k < 4800; k++) acc[k] = acc[k] / 16.0;
    normalise(acc.data(), bright, x);
}
}  // namespace

// the logits of marks at centres (cy, cx pairs), their inputs made here as dustnet.js makes them
void net_forward_at(int slot, const uint16_t* img, int W, int H, int C, const int32_t* centres, int N, bool bright, double* out) {
    const Net& g_net = g_nets[slot];
    const int T = g_net.towers, Hd = g_net.hidden;
    auto run = [&](int a, int b) {
        tl_net = &g_net;
        std::vector<double> in((size_t)T * 4800), feats((size_t)48 * T), h(Hd);
        for (int s = a; s < b; s++) {
            const int cy = centres[2 * s], cx = centres[2 * s + 1];
            patch_input(img, W, H, C, cy, cx, bright, in.data());
            if (T >= 2) context_input(img, W, H, C, cy, cx, bright, in.data() + 4800);
            for (int t = 0; t < T; t++) tower(in.data() + (size_t)t * 4800, t, &feats[(size_t)t * 48]);
            for (int o = 0; o < Hd; o++) {
                double v = g_net.d1b[o];
                for (size_t i = 0; i < feats.size(); i++) v += g_net.d1w[(size_t)o * feats.size() + i] * feats[i];
                h[o] = v > 0 ? v : 0.0;
            }
            double v = g_net.d2b[0];
            for (int i = 0; i < Hd; i++) v += g_net.d2w[i] * h[i];
            out[s] = v;
        }
    };
    run_parallel(N, std::min(threads(), std::max(1, N / 4)), run);
}

void net_forward(int slot, const double* inputs, int N, double* out) {
    const Net& g_net = g_nets[slot];
    const int T = g_net.towers, Hd = g_net.hidden;
    const size_t per = (size_t)T * 3 * 1600;
    auto run = [&](int a, int b) {
        tl_net = &g_net;
        std::vector<double> feats((size_t)48 * T), h(Hd);
        for (int s = a; s < b; s++) {
            for (int t = 0; t < T; t++) tower(inputs + (size_t)s * per + (size_t)t * 4800, t, &feats[(size_t)t * 48]);
            for (int o = 0; o < Hd; o++) {
                double v = g_net.d1b[o];
                for (size_t i = 0; i < feats.size(); i++) v += g_net.d1w[(size_t)o * feats.size() + i] * feats[i];
                h[o] = v > 0 ? v : 0.0;
            }
            double v = g_net.d2b[0];
            for (int i = 0; i < Hd; i++) v += g_net.d2w[i] * h[i];
            out[s] = v;
        }
    };
    run_parallel(N, std::min(threads(), std::max(1, N / 4)), run);
}

}  // namespace fd

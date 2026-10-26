// The dust network's forward pass (engine/dustnet.js), batched over marks and split over threads.
// The same sums in the same order as the JS, in double, so the logits are bit-identical.
// Weights arrive once (net_load) as one flat array: per tower (1 = v1, 2 = v2: local view, then
// context view) c1w c1b ... c5w c5b with channels 3-8-8-16-16-16, then d1w d1b d2w d2b.
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "cpu.h"
#include "filters.h"
#include "stats.h"

namespace fd {

namespace {
const int CH[6] = {3, 8, 8, 16, 16, 16};
struct Conv { std::vector<double> w, b; int ci, co; };
struct Net { int towers = 0, hidden = 0; std::vector<Conv> conv; std::vector<double> d1w, d1b, d2w, d2b; };
Net g_nets[4];   // slots: 0 the shipped v1 network, 1 the v2 network (patch + context)

// one thread's working memory for a run of marks
struct Scratch {
    std::vector<double> in, pad, a, b, feats, h, d, c;
    Scratch(int towers, int hidden)
        : in((size_t)towers * 4800), pad((size_t)8 * 42 * 42), a(8 * 1600), b(8 * 1600), feats((size_t)48 * towers),
          h(hidden), d(40 * 39), c(1600) {}
};

// 3 x 3 convolution with a zero border, then ReLU: out (co, n, n) from x (ci, n, n). Each output pixel adds its
// terms in the JS order (input channel, kernel row, kernel column), the border's included as w x 0, so the result
// is the JS's bit for bit; the innermost loop runs along a row, so it vectorises.
FD_INLINE void conv(const double* x, int ci, int n, const Conv& c, double* out, double* pad) {
    const int np = n + 2, nn = n * n;
    std::fill(pad, pad + (size_t)ci * np * np, 0.0);
    for (int i = 0; i < ci; i++)
        for (int y = 0; y < n; y++)
            std::copy(x + ((size_t)i * n + y) * n, x + ((size_t)i * n + y + 1) * n, pad + ((size_t)i * np + y + 1) * np + 1);
    for (int o = 0; o < c.co; o++) {
        double* acc = out + (size_t)o * nn;
        std::fill(acc, acc + nn, c.b[o]);
        for (int i = 0; i < ci; i++) {
            const double* xp = pad + (size_t)i * np * np;
            const double* wk = &c.w[((size_t)o * ci + i) * 9];
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    const double wv = wk[ky * 3 + kx];
                    for (int y = 0; y < n; y++) {
                        const double* s = xp + (size_t)(y + ky) * np + kx;
                        double* r = acc + (size_t)y * n;
                        for (int q = 0; q < n; q++) r[q] += wv * s[q];
                    }
                }
        }
        for (int k = 0; k < nn; k++)
            if (!(acc[k] > 0)) acc[k] = 0.0;
    }
}

// 2 x 2 max pooling: (c, n, n) -> (c, n/2, n/2)
FD_INLINE void pool(const double* x, int c, int n, double* out) {
    const int m = n >> 1;
    for (int ch = 0; ch < c; ch++) {
        const double* a = x + (size_t)ch * n * n;
        double* p = out + (size_t)ch * m * m;
        for (int y = 0; y < m; y++)
            for (int q = 0; q < m; q++) {
                const int i = 2 * y * n + 2 * q;
                p[y * m + q] = std::max(std::max(a[i], a[i + 1]), std::max(a[i + n], a[i + n + 1]));
            }
    }
}

// one tower on a (3, 40, 40) input -> 48 features (mean, max, centre of the last 16 channels)
FD_INLINE void tower(const Net& net, const double* in, int t, double* feats, Scratch& s) {
    const Conv* c = &net.conv[(size_t)t * 5];
    conv(in, 3, 40, c[0], s.a.data(), s.pad.data());
    conv(s.a.data(), 8, 40, c[1], s.b.data(), s.pad.data());
    pool(s.b.data(), 8, 40, s.a.data());
    conv(s.a.data(), 8, 20, c[2], s.b.data(), s.pad.data());
    pool(s.b.data(), 16, 20, s.a.data());
    conv(s.a.data(), 16, 10, c[3], s.b.data(), s.pad.data());
    pool(s.b.data(), 16, 10, s.a.data());
    conv(s.a.data(), 16, 5, c[4], s.b.data(), s.pad.data());
    for (int ch = 0; ch < 16; ch++) {
        const double* v = &s.b[(size_t)ch * 25];
        double sum = 0.0;
        for (int k = 0; k < 25; k++) sum += v[k];
        feats[ch] = sum / 25;
        double m = -1.0 / 0.0;
        for (int k = 0; k < 25; k++)
            if (v[k] > m) m = v[k];
        feats[16 + ch] = m;
        feats[32 + ch] = v[2 * 5 + 2];
    }
}

// the towers' features through the two dense layers -> the logit
FD_INLINE double head(const Net& net, Scratch& s) {
    const size_t nf = s.feats.size();
    for (int o = 0; o < net.hidden; o++) {
        double v = net.d1b[o];
        for (size_t i = 0; i < nf; i++) v += net.d1w[(size_t)o * nf + i] * s.feats[i];
        s.h[o] = v > 0 ? v : 0.0;
    }
    double v = net.d2b[0];
    for (int i = 0; i < net.hidden; i++) v += net.d2w[i] * s.h[i];
    return v;
}

// dustnet.js normalise: each channel less its median, over the patch's grain (1.0483 x the median |difference| of
// horizontal neighbours in its luminance), sign-flipped for dark dust, clipped to +-12, x 0.25
void normalise(const double* vals, bool bright, double* x, Scratch& s) {
    const double sgn = bright ? 1.0 : -1.0;
    double* d = s.d.data();
    int nd = 0;
    for (int r = 0; r < 40; r++)
        for (int q = 0; q < 39; q++) {
            const int k = r * 40 + q;
            const double a = 0.2126 * vals[k] + 0.7152 * vals[1600 + k] + 0.0722 * vals[3200 + k];
            const double b = 0.2126 * vals[k + 1] + 0.7152 * vals[1600 + k + 1] + 0.0722 * vals[3200 + k + 1];
            d[nd++] = std::fabs(b - a);
        }
    const double g = std::max(median_of(d, nd) * 1.0483, 1e-4);
    double* c = s.c.data();
    for (int ch = 0; ch < 3; ch++) {
        std::copy(vals + ch * 1600, vals + ch * 1600 + 1600, c);
        const double m = median_of(c, 1600);
        for (int k = 0; k < 1600; k++) x[ch * 1600 + k] = std::min(12.0, std::max(-12.0, sgn * (vals[ch * 1600 + k] - m) / g)) * 0.25;
    }
}

inline double px(const uint16_t* img, int C, int64_t o, int c) { return (double)(float)((double)img[o + (C == 1 ? 0 : c)] / 65535); }

// dustnet.js patchInput: the 40 x 40 patch round (cy, cx), edges repeated
void patch_input(const uint16_t* img, int W, int H, int C, int cy, int cx, bool bright, double* x, Scratch& s) {
    double* vals = s.a.data();   // free until the towers run
    for (int r = 0; r < 40; r++) {
        const int y = std::min(H - 1, std::max(0, cy - 20 + r));
        for (int q = 0; q < 40; q++) {
            const int xx = std::min(W - 1, std::max(0, cx - 20 + q));
            const int64_t o = ((int64_t)y * W + xx) * C;
            for (int c = 0; c < 3; c++) vals[c * 1600 + r * 40 + q] = px(img, C, o, c);
        }
    }
    normalise(vals, bright, x, s);
}

// dustnet.js contextInput: the 160 x 160 neighbourhood, each 4 x 4 block summed in row order, / 16
void context_input(const uint16_t* img, int W, int H, int C, int cy, int cx, bool bright, double* x, Scratch& s) {
    int xs[160];
    for (int q = 0; q < 160; q++) xs[q] = std::min(W - 1, std::max(0, cx - 80 + q));
    double* acc = s.a.data();
    std::fill(acc, acc + 4800, 0.0);
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
    normalise(acc, bright, x, s);
}

// the logits of marks [a, b): inputs made from the image at their centres (img) or taken as given (inputs)
FD_INLINE void logits(const Net& net, const uint16_t* img, int W, int H, int C, const int32_t* centres, bool bright,
                      const double* inputs, int a, int b, double* out) {
    Scratch s(net.towers, net.hidden);
    const size_t per = (size_t)net.towers * 4800;
    for (int m = a; m < b; m++) {
        const double* in = inputs ? inputs + (size_t)m * per : s.in.data();
        if (!inputs) {
            const int cy = centres[2 * m], cx = centres[2 * m + 1];
            patch_input(img, W, H, C, cy, cx, bright, s.in.data(), s);
            if (net.towers >= 2) context_input(img, W, H, C, cy, cx, bright, s.in.data() + 4800, s);
        }
        for (int t = 0; t < net.towers; t++) tower(net, in + (size_t)t * 4800, t, &s.feats[(size_t)t * 48], s);
        out[m] = head(net, s);
    }
}

void logits_plain(const Net& net, const uint16_t* img, int W, int H, int C, const int32_t* centres, bool bright,
                  const double* inputs, int a, int b, double* out) {
    logits(net, img, W, H, C, centres, bright, inputs, a, b, out);
}

FD_AVX2 void logits_avx2(const Net& net, const uint16_t* img, int W, int H, int C, const int32_t* centres, bool bright,
                         const double* inputs, int a, int b, double* out) {
    logits(net, img, W, H, C, centres, bright, inputs, a, b, out);
}

// marks in chunks of 8 (the scratch memory is made once a chunk), taken by whichever thread is free
void run_marks(const Net& net, const uint16_t* img, int W, int H, int C, const int32_t* centres, bool bright,
               const double* inputs, int N, double* out) {
    const bool avx2 = use_avx2();
    run_parallel(N, std::min(threads(), std::max(1, N / 4)), [&](int a, int b) {
        if (avx2) logits_avx2(net, img, W, H, C, centres, bright, inputs, a, b, out);
        else logits_plain(net, img, W, H, C, centres, bright, inputs, a, b, out);
    }, (N + 7) / 8);
}
}  // namespace

bool net_load(int slot, const double* flat, size_t n, int towers, int hidden) {
    if (slot < 0 || slot >= 4 || towers < 1 || towers > 2 || hidden < 1 || hidden > 4096) return false;
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

// the logits of marks at centres (cy, cx pairs), their inputs made here as dustnet.js makes them
void net_forward_at(int slot, const uint16_t* img, int W, int H, int C, const int32_t* centres, int N, bool bright, double* out) {
    run_marks(g_nets[slot], img, W, H, C, centres, bright, nullptr, N, out);
}

// the logits of marks whose inputs (towers x 3 x 40 x 40 each) JS made
void net_forward(int slot, const double* inputs, int N, double* out) {
    run_marks(g_nets[slot], nullptr, 0, 0, 0, nullptr, true, inputs, N, out);
}

}  // namespace fd

// detect.js find()'s per-object stage 3 for every object at once, split over the worker threads: the solid fill
// (marks.js solidify / growFlake) and the cues (marks.js markCues). Each object is computed exactly as the JS does
// it alone (the same operations in the same order and precision); the JS side applies the fills' claims in object
// order between the two calls and rounds the cues as before, so the results are bit for bit the same.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "filters.h"
#include "stats.h"

namespace fd {

namespace {

int count(const std::vector<uint8_t>& m) { int n = 0; for (uint8_t v : m) n += v ? 1 : 0; return n; }

std::vector<double> vals(const std::vector<float>& img, const std::vector<uint8_t>& m) {
    std::vector<double> out;
    for (size_t i = 0; i < m.size(); i++) if (m[i]) out.push_back(img[i]);
    return out;
}

std::vector<float> crop(const float* src, int W, int y0, int y1, int x0, int x1) {
    const int w = x1 - x0, h = y1 - y0;
    std::vector<float> out((size_t)w * h);
    for (int y = 0; y < h; y++) std::memcpy(&out[(size_t)y * w], src + (size_t)(y0 + y) * W + x0, sizeof(float) * w);
    return out;
}

std::vector<uint8_t> and_not(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    std::vector<uint8_t> o(a.size());
    for (size_t i = 0; i < a.size(); i++) o[i] = a[i] && !b[i] ? 1 : 0;
    return o;
}

// filters.js dilateMask / erodeMask: square (binRect) or ellipse (half-widths from the JS side)
struct Morph { const int32_t* hw[128] = {}; };
std::vector<uint8_t> morph(const std::vector<uint8_t>& m, int w, int h, int k, bool dilate, const Morph& el, bool ellipse) {
    std::vector<uint8_t> out((size_t)w * h);
    if (ellipse) ellipse_morph_u8(m.data(), w, h, k, el.hw[k], dilate, out.data());
    else bin_rect(m.data(), w, h, k, dilate, out.data());
    return out;
}

std::vector<uint8_t> fill(const std::vector<uint8_t>& m, int w, int h) {
    std::vector<uint8_t> out((size_t)w * h);
    fill_holes(m.data(), w, h, out.data());
    return out;
}

std::vector<float> blur(const std::vector<float>& src, int w, int h, const float* ker, int klen) {
    std::vector<float> out((size_t)w * h);
    sep_conv(src.data(), w, h, ker, klen, out.data());
    return out;
}

// regions.js distanceTransform (chamfer 0.955 / 1.3693, float32 storage, double arithmetic): detect.js maxDist
double max_dist(const std::vector<uint8_t>& m, int w, int h) {
    const int pw = w + 2, ph = h + 2;
    std::vector<float> d((size_t)pw * ph, 0.0f);
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) if (m[(size_t)y * w + x]) d[(size_t)(y + 1) * pw + x + 1] = 1e9f;
    const double a = 0.955, b = 1.3693;
    auto get = [&](int x, int y) -> double { return x < 0 || y < 0 || x >= pw || y >= ph ? 0.0 : (double)d[(size_t)y * pw + x]; };
    for (int y = 0; y < ph; y++) for (int x = 0; x < pw; x++) {
        float& v = d[(size_t)y * pw + x];
        if (!v) continue;
        double t = v;
        t = std::min({t, get(x - 1, y) + a, get(x, y - 1) + a, get(x - 1, y - 1) + b, get(x + 1, y - 1) + b});
        v = (float)t;
    }
    for (int y = ph - 1; y >= 0; y--) for (int x = pw - 1; x >= 0; x--) {
        float& v = d[(size_t)y * pw + x];
        if (!v) continue;
        double t = v;
        t = std::min({t, get(x + 1, y) + a, get(x, y + 1) + a, get(x + 1, y + 1) + b, get(x - 1, y + 1) + b});
        v = (float)t;
    }
    double best = 0;
    for (float v : d) if (v > best) best = v;
    return best;
}

// marks.js growFlake: the raw mark grown to the flat region around it, or empty
std::vector<uint8_t> grow_flake(const std::vector<float>& Yw, const std::vector<uint8_t>& raw, int w, int h, float sgn,
                                const std::vector<float>& sig, const Morph& el, const float* k1, int k1n) {
    if (count(raw) < 20) return {};
    std::vector<float> yy(Yw.size());
    for (size_t i = 0; i < Yw.size(); i++) yy[i] = Yw[i] * sgn;
    const auto d7 = morph(raw, w, h, 7, true, el, false), d25 = morph(raw, w, h, 25, true, el, false), d15 = morph(raw, w, h, 15, true, el, false);
    const auto near_ = and_not(d7, raw), far_ = and_not(d25, d15);
    if (count(near_) < 8 || count(far_) < 8) return {};
    auto vr = vals(yy, raw), vf = vals(yy, far_);
    const double rim = percentile(vr, 75), bgl = percentile(vf, 30);
    if (rim - bgl <= 0) return {};
    std::vector<uint8_t> reg((size_t)w * h);
    const double thr = bgl + 0.6 * (rim - bgl);
    for (size_t i = 0; i < reg.size(); i++) reg[i] = (double)yy[i] >= thr || raw[i] ? 1 : 0;
    std::vector<int32_t> lab((size_t)w * h);
    const int nl = label(reg.data(), w, h, 8, lab.data());
    std::vector<uint8_t> keep(nl + 1, 0);
    for (size_t i = 0; i < raw.size(); i++) if (raw[i] && lab[i]) keep[lab[i]] = 1;
    std::vector<uint8_t> region((size_t)w * h);
    for (size_t i = 0; i < region.size(); i++) region[i] = keep[lab[i]] && lab[i] ? 1 : 0;
    const int extra = count(and_not(region, raw)), rawN = count(raw), regN = count(region);
    if (extra < 0.25 * rawN) return {};
    for (int x = 0; x < w; x++) if (region[x] || region[(size_t)(h - 1) * w + x]) return {};
    for (int y = 0; y < h; y++) if (region[(size_t)y * w] || region[(size_t)y * w + w - 1]) return {};
    if (regN > 4 * rawN + 1500) return {};
    const auto inner = morph(region, w, h, 7, false, el, true);
    if (count(inner) < 15) return {};
    const auto g1 = blur(Yw, w, h, k1, k1n);
    std::vector<double> hp, sg;
    for (size_t i = 0; i < inner.size(); i++) if (inner[i]) { hp.push_back((double)Yw[i] - (double)g1[i]); sg.push_back(sig[i]); }
    const double flat = stdev(hp) / std::max(median(sg), 1e-5);
    if (!(flat <= 0.6)) return {};
    return fill(region, w, h);
}

}  // namespace

// Phase 1, per object: raw = its own candidate pixels in its window (glab == id and cand), then marks.js solidify.
// boxes: n x (y0, y1, x0, x1); out: how (0 kept, 1 grown, 2 filled), the solid mask and its dilation (objd: ellipse
// k = dk, or the solid itself when dk == 0) at offs[i] in the masks buffers, its area and 2 x maxDist.
void solid_batch(const float* Y, const float* sigLevel, const int32_t* glab, const uint8_t* cand, int W, int H, int n,
                 const int32_t* boxes, const int32_t* ids, bool bright, const float* k1, int k1n, const int32_t* const* hws,
                 int dk, const int64_t* offs, uint8_t* solid, uint8_t* objd, int32_t* how, int32_t* area, double* width) {
    Morph el;
    for (int k = 0; k < 128; k++) el.hw[k] = hws[k];
    const float sgn = bright ? 1.0f : -1.0f;
    run_parallel(n, std::min(threads(), std::max(1, n / 8)), [&](int a, int b) {
        for (int i = a; i < b; i++) {
            const int y0 = boxes[4 * i], y1 = boxes[4 * i + 1], x0 = boxes[4 * i + 2], x1 = boxes[4 * i + 3];
            const int w = x1 - x0, h = y1 - y0, g = ids[i];
            std::vector<uint8_t> raw((size_t)w * h);
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
                const int64_t gi = (int64_t)(y0 + y) * W + x0 + x;
                raw[(size_t)y * w + x] = glab[gi] == g && cand[gi] ? 1 : 0;
            }
            const auto Yw = crop(Y, W, y0, y1, x0, x1), sig = crop(sigLevel, W, y0, y1, x0, x1);
            std::vector<uint8_t> res;
            int hw_ = 0;
            auto grown = grow_flake(Yw, raw, w, h, sgn, sig, el, k1, k1n);
            if (!grown.empty()) { res = std::move(grown); hw_ = 1; }
            else {
                const auto closed = morph(morph(raw, w, h, 5, true, el, true), w, h, 5, false, el, true);
                const auto filled = fill(closed, w, h);
                const auto hole = and_not(filled, raw);
                res = raw;
                if (count(hole) >= 6) {
                    const auto outside = and_not(morph(filled, w, h, 9, true, el, false), morph(filled, w, h, 3, true, el, false));
                    if (count(outside) >= 8) {
                        std::vector<float> yy(Yw.size());
                        for (size_t k = 0; k < Yw.size(); k++) yy[k] = Yw[k] * sgn;
                        auto vr = vals(yy, raw), vo = vals(yy, outside), vh = vals(yy, hole);
                        const double rim = percentile(vr, 75), out = median(vo), hv = median(vh);
                        if (rim - out > 0 && hv >= out + 0.5 * (rim - out)) { res = filled; hw_ = 2; }
                    }
                }
            }
            how[i] = hw_;
            std::memcpy(solid + offs[i], res.data(), res.size());
            const auto d = dk > 0 ? morph(res, w, h, dk, true, el, true) : res;
            std::memcpy(objd + offs[i], d.data(), d.size());
            area[i] = count(res);
            width[i] = hw_ ? 2 * max_dist(res, w, h) : NAN;
        }
    }, n);   // one object a chunk
}

// Phase 3, per object: marks.js markCues on its window, with cand as the JS loop saw it at that object: cand after
// every fill less the fills claimed by objects after it (the claims sorted by pixel cgi, the claiming object's index
// cidx, nc of them). out: n x 10 doubles (NaN = not set): crisp, crisp_rel, core_y, bg_y, ring_range, flat, broad,
// detail, tex, ok
void cues_batch(const float* Y, const float* G3, const float* E25, const float* E1, const float* sigLevel, const uint8_t* cand,
                const int32_t* cgi, const int32_t* cidx, int nc, int W, int H, int n, const int32_t* boxes, const int64_t* offs, const uint8_t* objd,
                bool bright, double gradCal, const float* k1, int k1n, const float* k3, int k3n, const int32_t* const* hws, double* out) {
    Morph el;
    for (int k = 0; k < 128; k++) el.hw[k] = hws[k];
    const float sgn = bright ? 1.0f : -1.0f;
    run_parallel(n, std::min(threads(), std::max(1, n / 8)), [&](int a, int b) {
        for (int i = a; i < b; i++) {
            double* o = out + (size_t)10 * i;
            for (int k = 0; k < 10; k++) o[k] = NAN;
            const int y0 = boxes[4 * i], y1 = boxes[4 * i + 1], x0 = boxes[4 * i + 2], x1 = boxes[4 * i + 3];
            const int ww = x1 - x0, wh = y1 - y0;
            const size_t nn = (size_t)ww * wh;
            std::vector<uint8_t> obj(objd + offs[i], objd + offs[i] + nn);
            if (count(obj) < 4) continue;
            std::vector<uint8_t> allc(nn);
            for (int y = 0; y < wh; y++) for (int x = 0; x < ww; x++) allc[(size_t)y * ww + x] = cand[(int64_t)(y0 + y) * W + x0 + x] ? 1 : 0;
            for (int y = 0; y < wh && nc; y++) {
                const int32_t a = (y0 + y) * W + x0, b = (y0 + y) * W + x1;
                for (const int32_t* q = std::lower_bound(cgi, cgi + nc, a); q < cgi + nc && *q < b; q++)
                    if (cidx[q - cgi] > i) allc[(size_t)y * ww + (*q - a)] = 0;
            }
            const auto Yw = crop(Y, W, y0, y1, x0, x1);
            std::vector<float> yy(nn);
            for (size_t k = 0; k < nn; k++) yy[k] = Yw[k] * sgn;
            auto excl = morph(allc, ww, wh, 7, true, el, false);
            for (size_t k = 0; k < nn; k++) excl[k] = excl[k] || obj[k] ? 1 : 0;
            const auto ring = and_not(morph(obj, ww, wh, 21, true, el, false), excl);
            if (count(ring) < 10) continue;
            o[9] = 1;
            auto vo = vals(yy, obj), vr = vals(yy, ring);
            const double core = percentile(vo, 85), bgm = median(vr);
            const double con = std::max(core - bgm, 1e-4);
            const auto g3 = crop(G3, W, y0, y1, x0, x1);
            std::vector<double> ad;
            for (size_t k = 0; k < nn; k++) if (obj[k]) ad.push_back(std::fabs((double)Yw[k] - (double)g3[k]));
            const double crisp = percentile(ad, 95) / con;
            o[0] = crisp;
            std::vector<float> of(nn);
            for (size_t k = 0; k < nn; k++) of[k] = obj[k] ? 1.0f : 0.0f;
            const auto gb = blur(of, ww, wh, k3, k3n);
            std::vector<double> ad2;
            for (size_t k = 0; k < nn; k++) if (obj[k]) ad2.push_back((double)(float)std::fabs((double)of[k] - (double)gb[k]));
            o[1] = crisp / std::max(percentile(ad2, 95), 1e-3);
            auto vy = vals(Yw, obj);
            o[2] = percentile(vy, bright ? 85 : 15);
            auto vyr = vals(Yw, ring);
            o[3] = median(vyr);
            auto rv = vals(yy, ring), rv2 = rv;
            o[4] = (percentile(rv, 90) - percentile(rv2, 10)) / con;
            const auto sl = crop(sigLevel, W, y0, y1, x0, x1);
            const auto inner = morph(obj, ww, wh, 7, false, el, true);
            if (count(inner) >= 15) {
                const auto g1 = blur(Yw, ww, wh, k1, k1n);
                std::vector<double> hp, sg;
                for (size_t k = 0; k < nn; k++) if (inner[k]) { hp.push_back((double)Yw[k] - (double)g1[k]); sg.push_back(sl[k]); }
                o[5] = stdev(hp) / std::max(median(sg), 1e-5);
            }
            const auto big = and_not(morph(obj, ww, wh, 81, true, el, false), morph(allc, ww, wh, 9, true, el, false));
            if (count(big) >= 50) {
                auto vb = vals(yy, big);
                o[6] = (core - median(vb)) / con;
                auto vs = vals(sl, big);
                const double ms = median(vs), s2 = gradCal * (ms * ms);
                const auto e25w = crop(E25, W, y0, y1, x0, x1);
                auto ve = vals(e25w, big);
                const double e25 = median(ve) / s2;
                o[7] = e25;
                if (E1) {
                    const auto e1w = crop(E1, W, y0, y1, x0, x1);
                    auto v1 = vals(e1w, big);
                    const double e1 = median(v1) / s2;
                    o[8] = e1 - 1 - e25;
                }
            }
        }
    }, n);   // one object a chunk
}

}  // namespace fd

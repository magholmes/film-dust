// detect.js judgeSoft's measurements (steps 1-4: its own scale, the whole object, what is measured around it, the
// fill it would get) for every soft seed at once, split over the worker threads. Each seed is measured exactly as the
// JS measures it alone, window by window (the same windows, blurs, floods in the same order); what depends on the
// judging loop's state (the crowd of look-alikes, the decision) stays in JS.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "filters.h"

namespace fd {

namespace {

// V8's Math.hypot for two numbers (normalised by the larger, Kahan-summed squares, sqrt, scaled back)

// regions.js percentile / percentileOf (selection; numpy's linear interpolation)
double percentile_of(double* a, int n, double q) {
    if (!n) return NAN;
    const double pos = (q / 100) * (n - 1);
    const int lo = (int)std::floor(pos), hi = (int)std::ceil(pos);
    std::nth_element(a, a + lo, a + n);
    const double vlo = a[lo];
    if (hi == lo) return vlo;
    double vhi = INFINITY;
    for (int i = lo + 1; i < n; i++) if (a[i] < vhi) vhi = a[i];
    return vlo + (vhi - vlo) * (pos - lo);
}
double pct(std::vector<double> a, double q) { return percentile_of(a.data(), (int)a.size(), q); }

// the annulus offsets (dy, dx pairs) of rIn <= hypot < ro, rows then columns
const std::vector<int16_t>& annulus(int rIn, int ro) {
    static const int R[6] = {15, 25, 40, 60, 85, 115};
    static std::vector<int16_t> cache[6];
    static bool made = false;
    if (!made) {
        for (int i = 0; i < 6; i++) {
            const int a = R[i], b = R[i] + 30;
            for (int dy = -b; dy <= b; dy++) for (int dx = -b; dx <= b; dx++) {
                const double d = v8_hypot(dy, dx);
                if (d >= a && d < b) { cache[i].push_back((int16_t)dy); cache[i].push_back((int16_t)dx); }
            }
        }
        made = true;
    }
    for (int i = 0; i < 6; i++) if (R[i] == rIn && R[i] + 30 == ro) return cache[i];
    return cache[0];
}

struct Box { int y0, y1, x0, x1; };

struct Judge {
    // inputs
    const float* Y; const float* sigLevel; const uint16_t* img; int C, Ci, W, H; bool bright;
    const float *k15, *k08, *k25; int k15n, k08n, k25n;
    double soft_fill; bool socket_on;
    // window state
    Box W0{}, win{};
    bool have = false;
    int ww = 0, hh = 0;
    std::vector<float> Yg;

    bool setWin(int y0, int y1, int x0, int x1) {
        y0 = std::max(W0.y0, y0); y1 = std::min(W0.y1, y1); x0 = std::max(W0.x0, x0); x1 = std::min(W0.x1, x1);
        if (have && y0 >= win.y0 && y1 <= win.y1 && x0 >= win.x0 && x1 <= win.x1) return false;
        if (have) { y0 = std::min(y0, win.y0); y1 = std::max(y1, win.y1); x0 = std::min(x0, win.x0); x1 = std::max(x1, win.x1); }
        win = {y0, y1, x0, x1};
        have = true;
        ww = x1 - x0; hh = y1 - y0;
        const float sgn = bright ? 1.0f : -1.0f;
        std::vector<float> Yw((size_t)ww * hh);
        for (int y = 0; y < hh; y++) { const int64_t o = (int64_t)(y0 + y) * W + x0; for (int x = 0; x < ww; x++) Yw[(size_t)y * ww + x] = sgn * Y[o + x]; }
        Yg.assign((size_t)ww * hh, 0.0f);
        sep_conv(Yw.data(), ww, hh, k15, k15n, Yg.data());
        return true;
    }
    bool safe(int y, int x, int m) const {
        return (win.y0 == W0.y0 || y >= m) && (win.y1 == W0.y1 || y < hh - m) && (win.x0 == W0.x0 || x >= m) && (win.x1 == W0.x1 || x < ww - m);
    }
    bool atW0(int y, int x) const {
        return (win.y0 == W0.y0 && y == 0) || (win.y1 == W0.y1 && y == hh - 1) || (win.x0 == W0.x0 && x == 0) || (win.x1 == W0.x1 && x == ww - 1);
    }
    int loc(int64_t g) const { const int yy = (int)(g / W); return (yy - win.y0) * ww + (int)(g - (int64_t)yy * W - win.x0); }

    // detect.js flood: 4-connected, depth first (a stack: left, right, up, down pushed in that order), over global
    // pixels; stop: 0 none, 1 cap, 2 edge, 3 far
    template <class Ok, class Visit>
    int flood(const std::vector<int64_t>& seeds, Ok ok, Visit visit, bool guard, std::vector<int64_t>& px) {
        std::vector<uint8_t> stamp((size_t)ww * hh, 0);
        std::vector<int64_t> stack;
        px.clear();
        for (int64_t g : seeds) { const int k = loc(g); if (stamp[k]) continue; stamp[k] = 1; stack.push_back(g); }
        int stop = 0;
        while (!stack.empty() && !stop) {
            const int64_t g = stack.back(); stack.pop_back();
            const int gyy = (int)(g / W), gxx = (int)(g - (int64_t)gyy * W);
            const int ly = gyy - win.y0, lx = gxx - win.x0, k = ly * ww + lx;
            if (guard && !safe(ly, lx, 7)) { stop = 2; break; }
            if (!ok(k)) continue;
            px.push_back(g);
            stop = visit(ly, lx, k, (int)px.size());
            if (stop) break;
            if (lx > 0 && !stamp[k - 1]) { stamp[k - 1] = 1; stack.push_back(g - 1); }
            if (lx < ww - 1 && !stamp[k + 1]) { stamp[k + 1] = 1; stack.push_back(g + 1); }
            if (ly > 0 && !stamp[k - ww]) { stamp[k - ww] = 1; stack.push_back(g - W); }
            if (ly < hh - 1 && !stamp[k + ww]) { stamp[k + ww] = 1; stack.push_back(g + W); }
        }
        return stop;
    }
    bool growTo(const std::vector<int64_t>& pxs, int pad) {
        int y0 = 1 << 30, y1 = -(1 << 30), x0 = 1 << 30, x1 = -(1 << 30);
        for (int64_t g : pxs) { const int yy = (int)(g / W), xx = (int)(g - (int64_t)yy * W); y0 = std::min(y0, yy); y1 = std::max(y1, yy); x0 = std::min(x0, xx); x1 = std::max(x1, xx); }
        return setWin(y0 - pad, y1 + pad + 1, x0 - pad, x1 + pad + 1);
    }
    double rgb(int y, int x, int c) const { return C == 1 ? (double)img[(int64_t)y * W + x] / 65535 : (double)img[((int64_t)y * W + x) * C + c] / 65535; }
};

std::vector<uint8_t> dil(const std::vector<uint8_t>& m, int w, int h, int k) {
    std::vector<uint8_t> o((size_t)w * h);
    bin_rect(m.data(), w, h, k, true, o.data());
    return o;
}

}  // namespace

// For each seed i: gy, gx, peak and its 320-px window W0 (y0, y1, x0, x1). out (fixed, 24 doubles per seed):
//  0 status (0: no own scale, part of a bigger bright area; 1 measured), 1 area, 2 elong, 3 plateau, 4 fine, 5 cosw,
//  6 broad, 7 socket raw (NaN: not measured), 8 r2ok, 9-12 step-3 window (y0, x0, w, h), 13 top, 14 bg, 15 con,
//  16-19 fill box (x0, x1, y0, y1, ends exclusive), 20 length, 21 width, 22 nRg, 23 nObj
// lists: per seed its own region Rg, the whole object objG (when r2ok) and its fill, as global indices
void soft_batch(const float* Y, const float* sigLevel, const uint16_t* img, int C, int Ci, int W, int H, bool bright,
                int n, const int32_t* seeds, const float* k15, int k15n, const float* k08, int k08n, const float* k25, int k25n,
                double soft_fill, bool socket_on, double* fixed, std::vector<std::vector<int64_t>>& lists) {
    lists.assign((size_t)n * 3, {});
    annulus(15, 45);   // made once, before the threads
    run_parallel(n, std::min(threads(), std::max(1, n / 4)), [&](int a, int b) {
        for (int i = a; i < b; i++) {
            double* o = fixed + (size_t)24 * i;
            for (int k = 0; k < 24; k++) o[k] = NAN;
            o[0] = 0;
            const int gy = seeds[7 * i], gx = seeds[7 * i + 1];
            Judge J{Y, sigLevel, img, C, Ci, W, H, bright, k15, k08, k25, k15n, k08n, k25n, soft_fill, socket_on};
            J.W0 = {seeds[7 * i + 3], seeds[7 * i + 4], seeds[7 * i + 5], seeds[7 * i + 6]};
            // 1. its own scale
            J.setWin(gy - 62, gy + 63, gx - 62, gx + 63);
            int qy = gy - J.win.y0, qx = gx - J.win.x0;
            double top = -INFINITY;
            for (int y = std::max(0, qy - 2); y < std::min(J.hh, qy + 3); y++) for (int x = std::max(0, qx - 2); x < std::min(J.ww, qx + 3); x++) top = std::max(top, (double)J.Yg[(size_t)y * J.ww + x]);
            std::vector<int64_t> Rg, px;
            bool haveR = false;
            double bg = 0, con = 0;
            static const int RIN[6] = {15, 25, 40, 60, 85, 115};
            std::vector<double> buf;
            for (int ri = 0; ri < 6; ri++) {
                const int rIn = RIN[ri], ro = rIn + 30;
                if (ro + 7 > 62 && J.setWin(J.W0.y0, J.W0.y1, J.W0.x0, J.W0.x1)) { qy = gy - J.win.y0; qx = gx - J.win.x0; }
                const auto& offs = annulus(rIn, ro);
                buf.clear();
                for (size_t t = 0; t < offs.size(); t += 2) {
                    const int y = qy + offs[t], x = qx + offs[t + 1];
                    if (y >= 0 && y < J.hh && x >= 0 && x < J.ww) buf.push_back(J.Yg[(size_t)y * J.ww + x]);
                }
                if (buf.size() < 50) break;
                bg = percentile_of(buf.data(), (int)buf.size(), 50); con = top - bg;
                if (con <= 0) break;
                const double thr = bg + 0.5 * con;
                if (!((double)J.Yg[(size_t)qy * J.ww + qx] >= thr)) break;
                const int lim = rIn - 2;
                const std::vector<int64_t> s0 = {(int64_t)gy * W + gx};
                const int st = J.flood(s0, [&](int k) { return (double)J.Yg[k] >= thr; },
                    [&](int ly, int lx, int, int) { return v8_hypot(ly - qy, lx - qx) >= lim ? 3 : 0; }, false, px);
                if (!st) { Rg = px; haveR = true; break; }
            }
            if (!haveR) continue;
            o[0] = 1;
            const int area = (int)Rg.size();
            o[1] = area;
            // 2. the whole object
            const int cap2 = 3 * area + 200;
            std::vector<int64_t> Rs(Rg);   // membership by global index through a sorted copy
            std::sort(Rs.begin(), Rs.end());
            auto isR = [&](int64_t g) { return std::binary_search(Rs.begin(), Rs.end(), g); };
            J.growTo(Rg, 48);
            int st2;
            std::vector<int64_t> f2;
            for (;;) {
                st2 = J.flood(Rg, [&](int k) { const int64_t g = (int64_t)J.win.y0 * W + J.win.x0 + (int64_t)(k / J.ww) * W + (k % J.ww); return isR(g) || std::fabs((double)J.Yg[k] - bg) >= 0.3 * con; },
                    [&](int ly, int lx, int, int cnt) { if (J.atW0(ly, lx)) return 1; return cnt > cap2 ? 1 : 0; }, true, f2);
                if (st2 != 2) break;
                const int before = J.ww * J.hh;
                J.setWin(J.win.y0 - 64, J.win.y1 + 64, J.win.x0 - 64, J.win.x1 + 64);
                if (J.ww * J.hh == before) break;
            }
            const bool r2ok = st2 == 0 && (int)f2.size() <= cap2;
            o[8] = r2ok ? 1 : 0;
            const std::vector<int64_t>& objG = r2ok ? f2 : Rg;
            // 3. the window for everything measured around it
            J.growTo(objG, 48);
            qy = gy - J.win.y0; qx = gx - J.win.x0;
            const int ww = J.ww, hh = J.hh, nn = ww * hh;
            o[9] = J.win.y0; o[10] = J.win.x0; o[11] = ww; o[12] = hh; o[13] = top; o[14] = bg; o[15] = con;
            std::vector<uint8_t> obj((size_t)nn, 0);
            for (int64_t g : objG) obj[J.loc(g)] = 1;
            {
                // regions.js shapeStats(ys, xs).elong over Rg (n > 3, else 1)
                double elong = 1.0;
                const int m = (int)Rg.size();
                if (m > 3) {
                    double my = 0, mx = 0;
                    for (int64_t g : Rg) { my += (double)(g / W); mx += (double)(g % W); }
                    my /= m; mx /= m;
                    double sxx = 0, syy = 0, sxy = 0;
                    for (int64_t g : Rg) { const double dx = (double)(g % W) - mx, dy = (double)(g / W) - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
                    sxx /= m - 1; syy /= m - 1; sxy /= m - 1;
                    const double tr = sxx + syy, det = sxx * syy - sxy * sxy;
                    const double disc = std::sqrt(std::max(0.0, (tr * tr) / 4 - det));
                    const double l1 = tr / 2 + disc, l2 = std::max(tr / 2 - disc, 1e-3);
                    elong = std::sqrt(l1 / l2);
                }
                o[2] = elong;
            }
            int plat = 0;
            for (int64_t g : Rg) if ((double)J.Yg[J.loc(g)] >= bg + 0.85 * con) plat++;
            o[3] = (double)plat / area;
            const auto d41 = dil(obj, ww, hh, 41), d9 = dil(obj, ww, hh, 9);
            std::vector<uint8_t> ring((size_t)nn);
            int nRing = 0;
            for (int k = 0; k < nn; k++) { ring[k] = d41[k] && !d9[k] ? 1 : 0; nRing += ring[k]; }
            double fine = 99.0;
            if (nRing > 30) {
                std::vector<float> Yraw((size_t)nn), g08((size_t)nn), g25((size_t)nn);
                for (int y = 0; y < hh; y++) std::memcpy(&Yraw[(size_t)y * ww], Y + (int64_t)(J.win.y0 + y) * W + J.win.x0, sizeof(float) * ww);
                sep_conv(Yraw.data(), ww, hh, k08, k08n, g08.data());
                sep_conv(Yraw.data(), ww, hh, k25, k25n, g25.data());
                std::vector<double> d2, sl;
                for (int k = 0; k < nn; k++) if (ring[k]) {
                    const double d = (double)g08[k] - (double)g25[k];
                    d2.push_back(d * d);
                    sl.push_back(sigLevel[(int64_t)(J.win.y0 + k / ww) * W + J.win.x0 + (k % ww)]);
                }
                const double ms = pct(sl, 50);
                fine = pct(d2, 90) / std::max(ms * ms, 1e-10);
            }
            o[4] = fine;
            bool useCore = false;
            for (int64_t g : Rg) if ((double)J.Yg[J.loc(g)] >= bg + 0.7 * con) { useCore = true; break; }
            double corec[3] = {0, 0, 0};
            int ncc = 0;
            for (int64_t g : Rg) {
                if (useCore && !((double)J.Yg[J.loc(g)] >= bg + 0.7 * con)) continue;
                const int yy = (int)(g / W), xx = (int)(g - (int64_t)yy * W);
                for (int c = 0; c < Ci; c++) corec[c] += J.rgb(yy, xx, c);
                ncc++;
            }
            for (int c = 0; c < Ci; c++) corec[c] /= ncc;
            double bgc[3] = {corec[0], corec[1], corec[2]};
            if (nRing > 30) {
                for (int c = 0; c < Ci; c++) {
                    std::vector<double> v;
                    for (int k = 0; k < nn; k++) if (ring[k]) v.push_back(J.rgb(J.win.y0 + k / ww, J.win.x0 + (k % ww), c));
                    bgc[c] = pct(v, 50);
                }
            }
            double dot = 0, nl = 0, nt = 0;
            for (int c = 0; c < Ci; c++) {
                const double lift = corec[c] - bgc[c], tow = bright ? 1.0 - bgc[c] : -bgc[c];
                dot += lift * tow; nl += lift * lift; nt += tow * tow;
            }
            o[5] = dot / (std::sqrt(nl) * std::sqrt(nt) + 1e-9);
            const auto d81 = dil(obj, ww, hh, 81), d21 = dil(obj, ww, hh, 21);
            std::vector<double> vb;
            for (int k = 0; k < nn; k++) if (d81[k] && !d21[k]) vb.push_back(J.Yg[k]);
            o[6] = vb.size() > 30 ? (top - pct(vb, 50)) / con : 0.0;
            if (bright && socket_on) {
                const double Rb = std::max(2.0, std::sqrt(area / 3.141592653589793));
                const int M = (int)std::ceil(Rb + 62);
                std::vector<double> inner, nearv, far;
                for (int y = std::max(0, gy - M); y <= std::min(H - 1, gy + M); y++) {
                    for (int x = std::max(0, gx - M); x <= std::min(W - 1, gx + M); x++) {
                        const double d = v8_hypot(y - gy, x - gx), v = Y[(int64_t)y * W + x];
                        if (d <= std::max(1.5, 0.7 * Rb)) inner.push_back(v);
                        else if (d >= Rb + 2 && d <= Rb + 8) nearv.push_back(v);
                        else if (d >= Rb + 20 && d <= Rb + 60) far.push_back(v);
                    }
                }
                if (!inner.empty() && !nearv.empty() && !far.empty()) {
                    std::sort(inner.begin(), inner.end());
                    const double pk = inner[(size_t)std::floor(0.9 * (inner.size() - 1))];
                    const double nm = pct(nearv, 50), fm = pct(far, 50);
                    o[7] = (fm - nm) / std::max(pk - nm, 1e-6);
                }
            }
            // 4. the fill it would get, if healed
            const int cap3 = 4 * std::max(area, 30);
            const double thr3 = bg + soft_fill * con;
            int st3;
            bool touch3 = false;
            std::vector<int64_t> f3;
            for (;;) {
                touch3 = false;
                st3 = J.flood(Rg, [&](int k) { return (double)J.Yg[k] >= thr3; },
                    [&](int ly, int lx, int, int cnt) { if (J.atW0(ly, lx)) { touch3 = true; return 1; } return cnt > cap3 ? 1 : 0; }, true, f3);
                if (st3 != 2) break;
                const int before = J.ww * J.hh;
                J.setWin(J.win.y0 - 64, J.win.y1 + 64, J.win.x0 - 64, J.win.x1 + 64);
                if (J.ww * J.hh == before) break;
            }
            const int w2 = J.ww, h2 = J.hh, n2 = w2 * h2;
            std::vector<uint8_t> fillM((size_t)n2, 0);
            if (st3 || (int)f3.size() > cap3 || touch3) {
                std::vector<uint8_t> Rm2((size_t)n2, 0);
                for (int64_t g : Rg) Rm2[J.loc(g)] = 1;
                fillM = dil(Rm2, w2, h2, 7);
            } else {
                for (int64_t g : f3) fillM[J.loc(g)] = 1;
            }
            if (r2ok) {
                std::vector<uint8_t> R2b((size_t)n2, 0);
                for (int64_t g : objG) R2b[J.loc(g)] = 1;
                const auto d5 = dil(R2b, w2, h2, 5);
                for (int k = 0; k < n2; k++) fillM[k] = fillM[k] | d5[k];
            }
            std::vector<uint8_t> filled((size_t)n2);
            fill_holes(fillM.data(), w2, h2, filled.data());
            int fy0 = 1 << 30, fy1 = -1, fx0 = 1 << 30, fx1 = -1;
            std::vector<int64_t> fl;
            for (int k = 0; k < n2; k++) if (filled[k]) {
                const int yy = k / w2, xx = k % w2;
                fy0 = std::min(fy0, yy); fy1 = std::max(fy1, yy); fx0 = std::min(fx0, xx); fx1 = std::max(fx1, xx);
                fl.push_back((int64_t)(J.win.y0 + yy) * W + J.win.x0 + xx);
            }
            o[16] = fx0 + J.win.x0; o[17] = fx1 + J.win.x0 + 1; o[18] = fy0 + J.win.y0; o[19] = fy1 + J.win.y0 + 1;
            o[20] = std::max(fy1 - fy0, fx1 - fx0) + 1; o[21] = std::min(fy1 - fy0, fx1 - fx0) + 1;
            o[22] = (double)Rg.size(); o[23] = (double)objG.size();
            lists[(size_t)3 * i] = std::move(Rg);
            if (r2ok) lists[(size_t)3 * i + 1] = f2;
            lists[(size_t)3 * i + 2] = std::move(fl);
        }
    });
}

}  // namespace fd

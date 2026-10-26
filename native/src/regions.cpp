// Native versions of engine/regions.js label and fillHoles, engine/filters.js binary morphology and rectMorph, and
// maps.js outsideMask. All exact operations (component numbering, counts, min / max of 0s and 1s), so any correct
// algorithm gives the JS's results: masks go as bits, 64 pixels a word, and connected areas as runs of pixels.
#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

#include "filters.h"

namespace fd {

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

namespace {

// A 0/1 mask row as bits: 64 pixels a word, pixel x at bit x + off. The `off` spare bits on each side let a
// dilation reach past the row's ends and come back, as it must for its steps to add up; nothing outside the
// image is ever set to begin with.

// y |= y shifted by a bits towards higher x and towards lower x (n words; t: scratch of n words)
void spread(uint64_t* y, uint64_t* t, int n, int a) {
    std::copy(y, y + n, t);
    const int q = a >> 6, s = a & 63;
    for (int i = 0; i < n; i++) {
        uint64_t v = 0;
        if (i - q >= 0) v |= t[i - q] << s;
        if (s && i - q - 1 >= 0) v |= t[i - q - 1] >> (64 - s);
        if (i + q < n) v |= t[i + q] >> s;
        if (s && i + q + 1 < n) v |= t[i + q + 1] << (64 - s);
        y[i] |= v;
    }
}

// y dilated by the run [-r, r]: each step reaches a further a <= 2c + 1 pixels (c: the reach so far), so the
// covered offsets stay one unbroken run and r is reached in about log2(r) steps
void dilate_bits(uint64_t* y, uint64_t* t, int n, int r) {
    int c = 0;
    while (c < r) {
        const int a = std::min(2 * c + 1, r - c);
        spread(y, t, n, a);
        c += a;
    }
}

// Binary erode / dilate on bits over a shape given by its half-widths per row offset (hw[i] for offset i - k/2,
// -1 = none): an ellipse, or a square when every row has the same half-width. Dilation: a pixel is set when any
// image pixel under the shape is; erosion is the complement's dilation, complemented; the outside never counts.
// binary: any non-zero value counts as 1 (binRect); otherwise a mask with other values than 0 and 1 returns false
// (the float ellipse path keeps them), since max / min of 0s and 1s is all this computes. Exact either way.
bool morph_bits(const uint8_t* src, int w, int h, int k, const int32_t* hw, bool dilate, bool binary, uint8_t* dst) {
    const int64_t N = (int64_t)w * h;
    if (!binary)
        for (int64_t i = 0; i < N; i++)
            if (src[i] > 1) return false;
    const int r = k >> 1;
    std::vector<int> widths;
    for (int i = 0; i < k; i++)
        if (hw[i] >= 0) widths.push_back(hw[i]);
    if (widths.empty()) return false;
    std::sort(widths.begin(), widths.end());
    widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
    const int m = (int)widths.size();
    std::vector<int> widx(k, -1);
    for (int i = 0; i < k; i++)
        if (hw[i] >= 0) widx[i] = (int)(std::lower_bound(widths.begin(), widths.end(), hw[i]) - widths.begin());
    const int off = 64 * ((widths[m - 1] + 63) / 64);
    const int n = (2 * off + w + 63) / 64;
    parallel_rows(h, (int64_t)w + (int64_t)k * n * 8, [&](int ya, int yb) {
        std::vector<uint64_t> ring((size_t)k * m * n), acc(n), tmp(n);
        std::vector<int> ring_row(k, -1);
        auto fill = [&](int yy) {
            const int slot = yy % k;
            uint64_t* base = &ring[(size_t)slot * m * n];
            std::fill(base, base + n, 0);
            const uint8_t* s = src + (int64_t)yy * w;
            for (int x = 0; x < w; x++)
                if ((s[x] != 0) == dilate) base[(x + off) >> 6] |= 1ull << ((x + off) & 63);
            dilate_bits(base, tmp.data(), n, widths[0]);
            for (int j = 1; j < m; j++) {
                uint64_t* cur = base + (size_t)j * n;
                std::copy(cur - n, cur, cur);
                dilate_bits(cur, tmp.data(), n, widths[j] - widths[j - 1]);
            }
            ring_row[slot] = yy;
        };
        for (int y = ya; y < yb; y++) {
            std::fill(acc.begin(), acc.end(), 0);
            for (int i = 0; i < k; i++) {
                const int j = widx[i], yy = y + i - r;
                if (j < 0 || yy < 0 || yy >= h) continue;
                const int slot = yy % k;
                if (ring_row[slot] != yy) fill(yy);
                const uint64_t* o = &ring[((size_t)slot * m + j) * n];
                for (int q = 0; q < n; q++) acc[q] |= o[q];
            }
            uint8_t* d = dst + (int64_t)y * w;
            const uint8_t on = dilate ? 1 : 0;
            for (int x = 0; x < w; x++) d[x] = ((acc[(x + off) >> 6] >> ((x + off) & 63)) & 1) ? on : (uint8_t)(1 - on);
        }
    }, std::max(16, 2 * k));
    return true;
}

}  // namespace

// filters.js binRect: binary square erode / dilate, any non-zero value counting as set, outside never counts.
// Its window is 2 (k / 2) + 1 pixels each way, which morph_bits does as 2 (k / 2) + 1 rows of that half-width.
void bin_rect(const uint8_t* mask, int w, int h, int k, bool dilate, uint8_t* out) {
    const int r = k >> 1;
    std::vector<int32_t> hw(2 * r + 1, r);
    morph_bits(mask, w, h, 2 * r + 1, hw.data(), dilate, true, out);
}

// filters.js ellipseMorph on a 0/1 mask: max / min over the ellipse, outside never counts
void ellipse_morph_u8(const uint8_t* src, int w, int h, int k, const int32_t* hw, bool is_max, uint8_t* dst) {
    if (morph_bits(src, w, h, k, hw, is_max, false, dst)) return;
    // other values than 0 and 1: the float version (exact), in strips to keep memory small
    const int r = k >> 1;
    const int strip = 512;
    parallel_rows((h + strip - 1) / strip, (int64_t)strip * w * k, [&](int sa, int sb) {
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
    // flat: grain-free (the 7 x 7 standard deviation under 0.0015) and clipped. The mean square and the mean are
    // made one after the other, so only one blur's working memory exists at a time.
    std::vector<uint8_t> flat(N);
    {
        std::vector<float> msq(N);
        {
            std::vector<float> sq(N);
            parallel_rows(H, W, [&](int ya, int yb) {
                for (int64_t i = (int64_t)ya * W; i < (int64_t)yb * W; i++) sq[i] = (float)((double)Y[i] * (double)Y[i]);
            });
            box_blur(sq.data(), W, H, 7, msq.data());
        }
        std::vector<float> mean(N);
        box_blur(Y, W, H, 7, mean.data());
        parallel_rows(H, W, [&](int ya, int yb) {
            for (int64_t i = (int64_t)ya * W; i < (int64_t)yb * W; i++) {
                const double sd = std::sqrt(std::max((double)msq[i] - (double)mean[i] * (double)mean[i], 0.0));
                flat[i] = sd < 0.0015 && ((double)Y[i] > 0.85 || (double)Y[i] < 0.02) ? 1 : 0;
            }
        });
    }
    // the flat areas (4-connected) as runs joined to the runs they overlap in the row above: each area's size and
    // whether it touches the frame, without a label per pixel. Kept: areas touching the frame, over 0.2% of it.
    struct Run { int y, x0, x1; };
    std::vector<Run> runs;
    std::vector<int32_t> parent;
    std::vector<int64_t> size;
    std::vector<uint8_t> edge;
    auto find = [&](int32_t a) {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    };
    auto join = [&](int32_t a, int32_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (b < a) std::swap(a, b);
        parent[b] = a;
        size[a] += size[b];
        edge[a] |= edge[b];
    };
    int prev_start = 0, prev_end = 0;
    for (int y = 0; y < H; y++) {
        const int start = (int)runs.size();
        const uint8_t* f = flat.data() + (int64_t)y * W;
        for (int x = 0; x < W;) {
            if (!f[x]) { x++; continue; }
            const int x0 = x;
            while (x < W && f[x]) x++;
            const int32_t id = (int32_t)runs.size();
            runs.push_back({y, x0, x});
            parent.push_back(id);
            size.push_back(x - x0);
            edge.push_back(y == 0 || y == H - 1 || x0 == 0 || x == W ? 1 : 0);
        }
        int p = prev_start;
        for (int c = start; c < (int)runs.size(); c++) {
            while (p < prev_end && runs[p].x1 <= runs[c].x0) p++;
            for (int q = p; q < prev_end && runs[q].x0 < runs[c].x1; q++) join(c, q);
        }
        prev_start = start;
        prev_end = (int)runs.size();
    }
    std::vector<uint8_t> core(N, 0);
    for (int32_t c = 0; c < (int32_t)runs.size(); c++) {
        const int32_t root = find(c);
        if (edge[root] && (double)size[root] > 0.002 * H * W)
            std::fill(core.begin() + (int64_t)runs[c].y * W + runs[c].x0, core.begin() + (int64_t)runs[c].y * W + runs[c].x1, (uint8_t)1);
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
    parallel_rows(h, (int64_t)w * 3, [&](int ya, int yb) {
        std::vector<float> P(w + 2 * r), f(w + 4 * r + 1), g(w + 4 * r + 1);
        for (int y = ya; y < yb; y++) run_line(src + (int64_t)y * w, w, 1, tmp.data() + (int64_t)y * w, 1, P, f, g);
    });
    parallel_rows(w, (int64_t)h * 3, [&](int xa, int xb) {
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
    parallel_rows(rowsOut, (int64_t)Wd * C * k * k, [&](int ra, int rb) {
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

// regions.js fillHoles (scipy binary_fill_holes): background not reachable from the border (4-connected) is filled.
// Works on runs of background: a run joins the runs it overlaps in the row above, and a joined set touching the
// border is reachable; every run of any other set is a hole.
void fill_holes(const uint8_t* mask, int w, int h, uint8_t* out) {
    struct Run { int x0, x1; };
    std::vector<Run> runs;
    std::vector<int32_t> parent;
    std::vector<uint8_t> edge;   // per root: the set touches the border
    std::vector<int> row_start((size_t)h + 1, 0);
    auto find = [&](int32_t a) {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    };
    auto join = [&](int32_t a, int32_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (b < a) std::swap(a, b);
        parent[b] = a;
        edge[a] |= edge[b];
    };
    for (int y = 0; y < h; y++) {
        row_start[y] = (int)runs.size();
        const uint8_t* m = mask + (int64_t)y * w;
        for (int x = 0; x < w;) {
            if (m[x]) { x++; continue; }
            const int x0 = x;
            while (x < w && !m[x]) x++;
            const int32_t id = (int32_t)runs.size();
            runs.push_back({x0, x});
            parent.push_back(id);
            edge.push_back(y == 0 || y == h - 1 || x0 == 0 || x == w ? 1 : 0);
        }
        if (y == 0) continue;
        // runs overlapping in x with the row above are 4-connected
        int p = row_start[y - 1];
        const int pe = row_start[y];
        for (int c = row_start[y]; c < (int)runs.size(); c++) {
            while (p < pe && runs[p].x1 <= runs[c].x0) p++;
            for (int q = p; q < pe && runs[q].x0 < runs[c].x1; q++) join(c, q);
        }
    }
    row_start[h] = (int)runs.size();
    for (int y = 0; y < h; y++) {
        const uint8_t* m = mask + (int64_t)y * w;
        uint8_t* o = out + (int64_t)y * w;
        for (int x = 0; x < w; x++) o[x] = m[x] ? 1 : 0;
        for (int c = row_start[y]; c < row_start[y + 1]; c++)
            if (!edge[find(c)]) std::fill(o + runs[c].x0, o + runs[c].x1, (uint8_t)1);
    }
}

}  // namespace fd

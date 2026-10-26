// Native, multi-threaded ports of engine/filters.js and maps.js hot spots. Every function gives
// bit-identical results to its JS original: the same operations in the same order and precision
// (float32 storage, double sums), no fused multiply-adds, kernels computed by the JS side.
#pragma once
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace fd {

int threads();
void set_threads(int n);
void set_threads_local(int n);   // per calling thread (0 = the global setting)
// fn(a, b) over [0, n) on t threads of the persistent workers (pool.cpp), in `chunks` pieces (at least t) taken
// by whichever thread is free; shutdown_pool joins them
void run_parallel(int n, int t, const std::function<void(int, int)>& fn, int chunks = 0);
// fn over [0, n) items (rows, columns, blocks) costing about `cost` each (pixels touched, times the kernel size
// where there is one), on as many threads as the work fills: at least min_items items and 64k of cost a thread.
// A small job (a mark's window) runs on the calling thread instead of waking the whole pool for microseconds.
void parallel_rows(int n, int64_t cost, const std::function<void(int, int)>& fn, int min_items = 32);
void shutdown_pool();

// cv2.medianBlur on 8-bit (REPLICATE border), Huang's sliding histogram as filters.js median8
void median8(const uint8_t* src, int w, int h, int k, uint8_t* dst);
// maps.js median8f: float luma -> 8-bit -> median -> / 255
void median8f(const float* Y, int w, int h, int k, float* out);
// filters.js ellipseMorph (hw: the ellipse's half-width per row offset, -1 = unused)
void ellipse_morph(const float* src, int w, int h, int k, const int32_t* hw, bool is_max, float* dst);
// the same, always in the JS's own order of comparisons (tests compare the two)
void ellipse_rows(const float* src, int w, int h, int k, const int32_t* hw, bool is_max, float* dst);
// filters.js ellipseOpenClose: open = erode then dilate (bright), close = dilate then erode (dark)
void ellipse_open_close(const float* src, int w, int h, int k, const int32_t* hw, bool open, float* dst);
// filters.js convolveRows + convolveCols with a symmetric float32 kernel (Gaussian FIR path)
void sep_conv(const float* src, int w, int h, const float* ker, int klen, float* dst);
// filters.js gaussianBlur's 3-pass box approximation (radii per pass)
void box3(const float* src, int w, int h, const int32_t* radii, int npass, float* dst);
// filters.js boxBlur (cv2.blur, REFLECT_101)
void box_blur(const float* src, int w, int h, int k, float* dst);
// filters.js sobel (ksize 3, REFLECT_101)
void sobel(const float* src, int w, int h, int dx, int dy, float* dst);
// rows [y0, y1) of it into out (row y at out + (y - y0) w); tmp: (y1 - y0 + 2) x w floats of scratch
void sobel_rows(const float* src, int w, int h, int dx, int dy, int y0, int y1, float* out, float* tmp);
// V8's Math.hypot for two numbers (normalised by the larger, Kahan-summed squares), bit for bit
inline double v8_hypot(double a, double b) {
    a = std::fabs(a); b = std::fabs(b);
    const double mx = a > b ? a : b;
    if (mx == 0) return 0;
    double sum = 0, comp = 0;
    const double v[2] = {a, b};
    for (int i = 0; i < 2; i++) {
        const double n = v[i] / mx;
        const double summand = n * n - comp;
        const double prelim = sum + summand;
        comp = (prelim - sum) - summand;
        sum = prelim;
    }
    return std::sqrt(sum) * mx;
}
// detect.js BlobField tiles at scale sq (tiles: ty, tx pairs): each tile's maxima and its own response
struct BlobTile {
    std::vector<int32_t> ys, xs;
    std::vector<float> vals, inner;
};
void blob_tiles(const float* Y, const float* sl, int W, int H, int T, double sq, int sgn, const float* ker, int klen,
                int k, int mg, const int32_t* tiles, int n, std::vector<BlobTile>& out);
// regions.js skeletonize (0/1 mask in, 0/1 skeleton out)
void skeletonize(const uint8_t* mask, int w, int h, uint8_t* out);
// gy[i] = hypot(gx[i], gy[i]) in place
void hypot_into(float* gy, const float* gx, int64_t n);

// maps.js per-block statistic: mode 0 noiseLocal's MAD (NaN without valid pixels), 1 busyness T
void block_stats(const float* resid, const uint8_t* valid, int W, int H, int block, int mode, bool bright, float* grid);
// maps.js noiseByLevel (bins brightness levels), interpolated for every pixel
void noise_by_level(const float* bg, const float* resid, const uint8_t* valid, int W, int H, int bins, float* out);
// maps.js residualMaps' z loop (zs, zl, zx; 0 outside)
void z_maps(const float* Y, const float* m15, const float* o15, const float* m41, const float* o41, const float* bxl,
            const float* sigma, const uint8_t* outside, int64_t N, bool bright, float* zs, float* zl, float* zx);
// filters.js npGradient and resizeLinear
void np_gradient(const float* src, int w, int h, float* gy, float* gx);
void resize_linear(const float* src, int sw, int sh, int dw, int dh, float* dst);

// regions.js label (conn 4 or 8; numbered by first appearance), returns the count
int label(const uint8_t* mask, int w, int h, int conn, int32_t* lab);
// filters.js binRect (square), ellipseMorph on 0/1 masks; maps.js outsideMask
void bin_rect(const uint8_t* mask, int w, int h, int k, bool dilate, uint8_t* out);
void ellipse_morph_u8(const uint8_t* src, int w, int h, int k, const int32_t* hw, bool is_max, uint8_t* dst);
void outside_mask(const float* Y, int W, int H, uint8_t* out);
void fill_holes(const uint8_t* mask, int w, int h, uint8_t* out);

// detect.js ridgeMaps (kernels from gaussKernel, one per scale, concatenated)
void ridge_maps(const float* Y, int W, int H, const uint8_t* outside, bool bright, const float* sig, int ns,
                const double* scales, const float* kers, const int32_t* klens, float* best, float* bestS);

// filters.js rectMorph (float, square) and scale.js averageStrip
void rect_morph(const float* src, int w, int h, int k, bool is_max, float* dst);
int average_strip(const void* src, bool eight, int srcW, int rows, int C, int k, uint16_t* dst, int Wd, int yd0);

// the dust network (engine/dustnet.js): weights once, then logits for a batch of inputs
// (each input: towers x 3 x 40 x 40 doubles)
bool net_load(int slot, const double* flat, size_t n, int towers, int hidden);
int net_towers(int slot);
void net_forward(int slot, const double* inputs, int N, double* out);
// the same for marks at centres (cy, cx pairs), the inputs made natively from the 16-bit image as dustnet.js makes them
void net_forward_at(int slot, const uint16_t* img, int W, int H, int C, const int32_t* centres, int N, bool bright, double* out);

// detect.js find() stage 3 for all objects (cues.cpp): marks.js solidify per object, then markCues per object
// (hws[k]: ellipse half-widths for size k, or null)
void solid_batch(const float* Y, const float* sigLevel, const int32_t* glab, const uint8_t* cand, int W, int H, int n,
                 const int32_t* boxes, const int32_t* ids, bool bright, const float* k1, int k1n, const int32_t* const* hws,
                 int dk, const int64_t* offs, uint8_t* solid, uint8_t* objd, int32_t* how, int32_t* area, double* width);
void cues_batch(const float* Y, const float* G3, const float* E25, const float* E1, const float* sigLevel, const uint8_t* cand,
                const int32_t* cgi, const int32_t* cidx, int nc, int W, int H, int n, const int32_t* boxes, const int64_t* offs, const uint8_t* objd,
                bool bright, double gradCal, const float* k1, int k1n, const float* k3, int k3n, const int32_t* const* hws, double* out);

// detect.js judgeSoft's measurements for every seed (soft.cpp): seeds n x 7 (gy, gx, 0, W0 y0, y1, x0, x1); fixed
// n x 24 doubles; lists 3 per seed (own region, whole object when r2ok, fill) as global indices
void soft_batch(const float* Y, const float* sigLevel, const uint16_t* img, int C, int Ci, int W, int H, bool bright,
                int n, const int32_t* seeds, const float* k15, int k15n, const float* k08, int k08n, const float* k25, int k25n,
                double soft_fill, bool socket_on, double* fixed, std::vector<std::vector<int64_t>>& lists);

}  // namespace fd

// regions.js skeletonize: the same thinning (two sub-iterations a round, skimage's table), bit for bit. Each
// sub-iteration decides on the image as it was and then removes, as the JS does with its copy.
#include <cstdint>
#include <vector>

#include "filters.h"

namespace fd {

namespace {
const uint8_t THIN_LUT[256] = {
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0,
};
}  // namespace

void skeletonize(const uint8_t* mask, int w, int h, uint8_t* out) {
    const int W2 = w + 2, H2 = h + 2;
    std::vector<uint8_t> s((size_t)W2 * H2, 0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) s[(size_t)(y + 1) * W2 + x + 1] = mask[(size_t)y * w + x] ? 1 : 0;
    std::vector<size_t> del;
    bool removed = true;
    while (removed) {
        removed = false;
        for (int pass = 0; pass < 2; pass++) {
            del.clear();
            for (int y = 1; y < H2 - 1; y++) {
                for (int x = 1; x < W2 - 1; x++) {
                    const size_t i = (size_t)y * W2 + x;
                    if (!s[i]) continue;
                    const int v = THIN_LUT[s[i - W2 - 1] + 2 * s[i - W2] + 4 * s[i - W2 + 1] + 8 * s[i + 1] + 16 * s[i + W2 + 1] +
                                           32 * s[i + W2] + 64 * s[i + W2 - 1] + 128 * s[i - 1]];
                    if (v == 3 || (v == 1 && pass == 0) || (v == 2 && pass == 1)) del.push_back(i);
                }
            }
            if (!del.empty()) removed = true;
            for (const size_t i : del) s[i] = 0;
        }
    }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) out[(size_t)y * w + x] = s[(size_t)(y + 1) * W2 + x + 1];
}

}  // namespace fd

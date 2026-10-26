// The order statistics and moments the JS engine uses (engine/regions.js), each in one place. They work on a
// scratch array, which they reorder. Medians and percentiles are values of the array, unique whatever the order
// they are found in, so std::nth_element gives exactly what the JS's own selection gives.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace fd {

// the k-th smallest of a[0, n) (regions.js selectK)
inline double kth(double* a, int n, int k) {
    std::nth_element(a, a + k, a + n);
    return a[k];
}

// regions.js medianOf: the middle value, or the mean of the middle two; NaN when there are none
inline double median_of(double* a, int n) {
    if (!n) return NAN;
    const int h = (n - 1) >> 1;
    const double v = kth(a, n, h);
    if (n % 2) return v;
    double u = INFINITY;
    for (int i = h + 1; i < n; i++)
        if (a[i] < u) u = a[i];
    return (v + u) / 2;
}

// regions.js percentile: numpy's linear interpolation between the order statistics at floor and ceil of
// q% x (n - 1); NaN when there are none
inline double percentile(double* a, int n, double q) {
    if (!n) return NAN;
    const double pos = (q / 100) * (n - 1);
    const int lo = (int)std::floor(pos), hi = (int)std::ceil(pos);
    const double vlo = kth(a, n, lo);
    if (hi == lo) return vlo;
    double vhi = INFINITY;
    for (int i = lo + 1; i < n; i++)
        if (a[i] < vhi) vhi = a[i];
    return vlo + (vhi - vlo) * (pos - lo);
}
inline double percentile(std::vector<double>& a, double q) { return percentile(a.data(), (int)a.size(), q); }

// regions.js median: percentile 50 (which can differ from medianOf in the last bit on an even count)
inline double median(std::vector<double>& a) { return percentile(a, 50); }

inline double mean(const std::vector<double>& a) {
    double s = 0;
    for (double v : a) s += v;
    return a.empty() ? NAN : s / a.size();
}

// the population standard deviation (numpy's std)
inline double stdev(const std::vector<double>& a) {
    const double m = mean(a);
    double s = 0;
    for (double v : a) {
        const double d = v - m;
        s += d * d;
    }
    return a.empty() ? NAN : std::sqrt(s / a.size());
}

}  // namespace fd

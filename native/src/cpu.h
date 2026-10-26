// Which instructions the hot kernels may use. The addon is built for every 64-bit PC (-march=x86-64); a kernel
// that gains from AVX2 is compiled a second time with FD_AVX2 and runs only when the processor has AVX2. Both
// copies do the same operations in the same order (no fused multiply-adds), so their results are identical.
#pragma once
#include <cstdlib>

#if defined(__GNUC__) && defined(__x86_64__)
#define FD_AVX2 __attribute__((target("avx2")))
#define FD_X86 1
#else
#define FD_AVX2
#define FD_X86 0
#endif
// a body written once and inlined into both copies
#define FD_INLINE inline __attribute__((always_inline))

namespace fd {

// AVX2 present (and not switched off with FILMDUST_NO_AVX2=1, which tests use to run the plain copies)
inline bool use_avx2() {
#if FD_X86
    static const bool on = [] {
        const char* off = std::getenv("FILMDUST_NO_AVX2");
        if (off && *off && *off != '0') return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") != 0;
    }();
    return on;
#else
    return false;
#endif
}

}  // namespace fd

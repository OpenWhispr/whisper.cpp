// OpenWhispr (OpenWhispr/openwhispr#2356): stop at startup on a processor below
// the x86 level ggml's CPU code was built for.
//
// A static build has no run-time CPU dispatch, and ggml faults only when it
// first runs an instruction the processor lacks. For F16C that is the first
// transcription, after the server is already serving. OpenWhispr falls back to
// the build for the next older level when whisper-server dies at startup with
// an illegal instruction, so raise that fault here, before main(). Enabled by
// OPENWHISPR_CPU_LEVEL_CHECK, for the builds for older processors only.

#if defined(__x86_64__) || defined(_M_X64)

#include "ggml-cpu.h"

#include <cstdio>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace {

struct cpuid_result {
    unsigned eax, ebx, ecx, edx;
};

cpuid_result cpuid(unsigned leaf, unsigned subleaf) {
#if defined(_MSC_VER)
    int r[4];
    __cpuidex(r, (int) leaf, (int) subleaf);
    return { (unsigned) r[0], (unsigned) r[1], (unsigned) r[2], (unsigned) r[3] };
#else
    cpuid_result r;
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
    return r;
#endif
}

// AVX, F16C and FMA also need the OS to save the YMM registers (XCR0 bits 1-2)
bool os_saves_ymm(const cpuid_result & leaf1) {
    if (!(leaf1.ecx & (1u << 27))) { // OSXSAVE
        return false;
    }
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
    return (xcr0 & 0x6) == 0x6;
}

bool check_cpu_level() {
    const cpuid_result leaf1 = cpuid(1, 0);
    const cpuid_result leaf7 = cpuid(0, 0).eax >= 7 ? cpuid(7, 0) : cpuid_result{};
    const bool ymm = (leaf1.ecx & (1u << 28)) && os_saves_ymm(leaf1);

    const struct {
        const char * name;
        bool built;
        bool present;
    } features[] = {
        { "AVX",  ggml_cpu_has_avx()  != 0, ymm },
        { "F16C", ggml_cpu_has_f16c() != 0, ymm && (leaf1.ecx & (1u << 29)) },
        { "FMA",  ggml_cpu_has_fma()  != 0, ymm && (leaf1.ecx & (1u << 12)) },
        { "AVX2", ggml_cpu_has_avx2() != 0, ymm && (leaf7.ebx & (1u << 5)) },
        { "BMI2", ggml_cpu_has_bmi2() != 0, (leaf7.ebx & (1u << 8)) != 0 },
    };

    bool supported = true;
    for (const auto & feature : features) {
        if (feature.built && !feature.present) {
            fprintf(stderr, "whisper-server: built for %s, which this processor does not support\n", feature.name);
            supported = false;
        }
    }
    if (!supported) {
        fflush(stderr);
#if defined(_MSC_VER)
        __ud2();
#else
        __builtin_trap();
#endif
    }
    return supported;
}

[[maybe_unused]] const bool cpu_level_supported = check_cpu_level();

} // namespace

#endif

/**
 * @file ids_pi.cpp
 * @brief Lightweight TCN Inference Engine for Raspberry Pi Zero W (CIC-IDS2017)
 * @details Performs native C++ inference using exported model weights without heavy ML frameworks.
 * 
 * Usage:
 *   - Build: g++ -std=c++11 -O2 -Wall -o ids_pi ids_pi.cpp
 *   - Check: ./ids_pi --check (Validates parity with PyTorch output within 1e-4 tolerance)
 *   - Run:   ./ids_pi < flows_test.csv > predictions.csv
 */

#include "model_weights.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <sys/resource.h>

namespace {

/**
 * @brief Performs causal convolution across feature channels and time steps.
 */
void causalConv(const float* x, int cin, int T, const float* w, const float* b,
                int cout, int K, int D, float* out) {
    for (int o = 0; o < cout; ++o)
        for (int t = 0; t < T; ++t) {
            float acc = b[o];
            for (int c = 0; c < cin; ++c)
                for (int i = 0; i < K; ++i) {
                    const int idx = t - (K - 1 - i) * D;
                    if (idx >= 0) acc += w[(o * cin + c) * K + i] * x[c * T + idx];
                }
            out[o * T + t] = acc;
        }
}

/**
 * @brief 1x1 pointwise convolution applied on the residual path.
 */
void pointwise(const float* x, int cin, int T, const float* w, const float* b,
               int cout, float* out) {
    for (int o = 0; o < cout; ++o)
        for (int t = 0; t < T; ++t) {
            float acc = b[o];
            for (int c = 0; c < cin; ++c) acc += w[o * cin + c] * x[c * T + t];
            out[o * T + t] = acc;
        }
}

/**
 * @brief In-place ReLU activation function.
 */
void reluInPlace(float* v, int n) {
    for (int i = 0; i < n; ++i)
        if (v[i] < 0.0f) v[i] = 0.0f;
}

/**
 * @brief Executes the full TCN forward pass.
 * @param x Input tensor in channel-major format [N_FEAT][WIN_LEN].
 * @param logits Output classification logits.
 */
void forward(const float* x, float* logits) {
    const int T = WIN_LEN;
    float a1[C1 * WIN_LEN], a2[C1 * WIN_LEN], r1[C1 * WIN_LEN], h1[C1 * WIN_LEN];
    float b1[C2 * WIN_LEN], b2[C2 * WIN_LEN], r2[C2 * WIN_LEN], h2[C2 * WIN_LEN];

    // Block 1: 8 -> 16 channels, dilation 1
    causalConv(x, N_FEAT, T, B1_C1_W, B1_C1_B, C1, KSIZE, DIL1, a1);
    reluInPlace(a1, C1 * T);
    causalConv(a1, C1, T, B1_C2_W, B1_C2_B, C1, KSIZE, DIL1, a2);
    pointwise(x, N_FEAT, T, B1_DS_W, B1_DS_B, C1, r1);
    for (int i = 0; i < C1 * T; ++i) h1[i] = a2[i] + r1[i];
    reluInPlace(h1, C1 * T);

    // Block 2: 16 -> 32 channels, dilation 2
    causalConv(h1, C1, T, B2_C1_W, B2_C1_B, C2, KSIZE, DIL2, b1);
    reluInPlace(b1, C2 * T);
    causalConv(b1, C2, T, B2_C2_W, B2_C2_B, C2, KSIZE, DIL2, b2);
    pointwise(h1, C1, T, B2_DS_W, B2_DS_B, C2, r2);
    for (int i = 0; i < C2 * T; ++i) h2[i] = b2[i] + r2[i];
    reluInPlace(h2, C2 * T);

    // Classification Head: Maps last time step to linear logits
    float last[C2];
    for (int c = 0; c < C2; ++c) last[c] = h2[c * T + (T - 1)];
    for (int k = 0; k < N_CLASS; ++k) {
        float acc = FC_B[k];
        for (int c = 0; c < C2; ++c) acc += FC_W[k * C2 + c] * last[c];
        logits[k] = acc;
    }
}

/**
 * @brief Applies log-signed transformation and training-fitted standardization.
 */
float scaleFeature(double raw, int k) {
    double v = std::isfinite(raw) ? raw : static_cast<double>(SCALER_MEDIAN[k]);
    double z = (v < 0.0) ? -std::log1p(-v) : std::log1p(v);
    return static_cast<float>((z - SCALER_MEAN[k]) / SCALER_STD[k]);
}

std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    for (char ch : line) {
        if (ch == ',') { out.push_back(cell); cell.clear(); }
        else if (ch != '\r') cell += ch;
    }
    out.push_back(cell);
    return out;
}

double parseField(const std::string& s) {
    if (s.empty()) return NAN;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    return (end == s.c_str()) ? NAN : v;
}

long peakRssKb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;  // Peak resident set size in kilobytes
}

int selfCheck() {
    float logits[N_CLASS];
    forward(CHECK_INPUT, logits);
    double maxDiff = 0.0;
    for (int k = 0; k < N_CLASS; ++k) {
        const double d = std::fabs(static_cast<double>(logits[k]) - CHECK_LOGITS[k]);
        if (d > maxDiff) maxDiff = d;
    }
    std::printf("max |logit diff| vs PyTorch = %.3g\n", maxDiff);
    return maxDiff < 1e-4 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--check") == 0) return selfCheck();

    std::string line;
    if (!std::getline(std::cin, line)) {
        std::fprintf(stderr, "Error: Empty input stream.\n");
        return 1;
    }
    std::vector<std::string> header = splitCsv(line);
    for (auto& h : header) h = trim(h);

    int col[N_FEAT];
    for (int k = 0; k < N_FEAT; ++k) {
        col[k] = -1;
        for (size_t j = 0; j < header.size(); ++j)
            if (header[j] == FEATURE_NAMES[k]) { col[k] = static_cast<int>(j); break; }
        if (col[k] < 0) {
            std::fprintf(stderr, "Error: Missing required column: %s\n", FEATURE_NAMES[k]);
            return 1;
        }
    }

    float win[N_FEAT * WIN_LEN];          // Channel-major sliding window buffer
    std::memset(win, 0, sizeof(win));
    float logits[N_CLASS];
    long rowIdx = 0, windows = 0;
    double totalUs = 0.0;

    while (std::getline(std::cin, line)) {
        if (trim(line).empty()) continue;
        ++rowIdx;
        const std::vector<std::string> f = splitCsv(line);

        for (int k = 0; k < N_FEAT; ++k) {
            float* ch = win + k * WIN_LEN;
            std::memmove(ch, ch + 1, (WIN_LEN - 1) * sizeof(float));
            double raw = NAN;
            if (col[k] < static_cast<int>(f.size())) raw = parseField(trim(f[col[k]]));
            ch[WIN_LEN - 1] = scaleFeature(raw, k);
        }
        if (rowIdx < WIN_LEN) continue;   // Skip until window is fully populated

        const auto t0 = std::chrono::steady_clock::now();
        forward(win, logits);
        const auto t1 = std::chrono::steady_clock::now();

        int best = 0;
        for (int k = 1; k < N_CLASS; ++k)
            if (logits[k] > logits[best]) best = k;

        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        totalUs += us;
        ++windows;
        std::printf("%ld,%s,%.1f\n", rowIdx, CLASS_NAMES[best], us);
    }

    std::fprintf(stderr, "Execution summary: rows=%ld windows=%ld mean_latency_us=%.1f peak_rss_kb=%ld\n",
                 rowIdx, windows, windows ? totalUs / windows : 0.0, peakRssKb());
    return 0;
}

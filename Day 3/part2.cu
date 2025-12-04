#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t e, const char* msg=nullptr) {
    if (e != cudaSuccess) {
        if (msg) std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        else std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void perBankTopKKernel(const unsigned char *digits,
                                     const size_t *offsets,
                                     const size_t *lengths,
                                     unsigned long long *out,
                                     size_t N,
                                     int K) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    size_t off = offsets[idx];
    size_t len = lengths[idx];
    if ((int)len < K) { out[idx] = 0ULL; return; }

    size_t start = off;
    unsigned long long value = 0ULL;
    for (int t = 0; t < K; ++t) {
        size_t maxJ = off + len - (K - t);
        unsigned char bestDigit = 0;
        size_t bestPos = start;
        for (size_t j = start; j <= maxJ; ++j) {
            unsigned char d = digits[j];
            if (d > bestDigit) {
                bestDigit = d;
                bestPos = j;
                if (bestDigit == 9) break;
            }
        }
        value = value * 10ULL + (unsigned long long)bestDigit;
        start = bestPos + 1;
    }
    out[idx] = value;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./part2 input.txt\n";
        return 1;
    }

    std::ifstream fin(argv[1]);
    if (!fin) { std::cerr << "Cannot open " << argv[1] << "\n"; return 1; }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    size_t N = lines.size();
    if (N == 0) { std::cout << 0 << "\n"; return 0; }

    std::vector<size_t> offsets(N);
    std::vector<size_t> lengths(N);
    size_t totalDigits = 0;
    for (size_t i = 0; i < N; ++i) {
        offsets[i] = totalDigits;
        lengths[i] = lines[i].size();
        totalDigits += lengths[i];
    }
    std::vector<unsigned char> hostDigits;
    hostDigits.reserve(totalDigits);
    for (size_t i = 0; i < N; ++i) {
        for (char c : lines[i]) {
            unsigned char digitValue = (unsigned char)(c - '0');
            hostDigits.push_back(digitValue);
        }
    }

    unsigned char *deviceDigits = nullptr;
    size_t *deviceOffsets = nullptr;
    size_t *deviceLengths = nullptr;
    unsigned long long *deviceOut = nullptr;

    checkCuda(cudaMalloc(&deviceDigits, totalDigits * sizeof(unsigned char)), "cudaMalloc digits");
    checkCuda(cudaMalloc(&deviceOffsets, N * sizeof(size_t)), "cudaMalloc offsets");
    checkCuda(cudaMalloc(&deviceLengths, N * sizeof(size_t)), "cudaMalloc lengths");
    checkCuda(cudaMalloc(&deviceOut, N * sizeof(unsigned long long)), "cudaMalloc out");

    checkCuda(cudaMemcpy(deviceDigits, hostDigits.data(), totalDigits * sizeof(unsigned char), cudaMemcpyHostToDevice), "cudaMemcpy digits");
    checkCuda(cudaMemcpy(deviceOffsets, offsets.data(), N * sizeof(size_t), cudaMemcpyHostToDevice), "cudaMemcpy offsets");
    checkCuda(cudaMemcpy(deviceLengths, lengths.data(), N * sizeof(size_t), cudaMemcpyHostToDevice), "cudaMemcpy lengths");

    const int block = 256;
    int grid = (int)((N + block - 1) / block);
    const int K = 12;
    perBankTopKKernel<<<grid, block>>>(deviceDigits, deviceOffsets, deviceLengths, deviceOut, N, K);
    checkCuda(cudaDeviceSynchronize(), "kernel sync");
    checkCuda(cudaGetLastError(), "kernel launch");

    std::vector<unsigned long long> hostOut(N);
    checkCuda(cudaMemcpy(hostOut.data(), deviceOut, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "Memcpy out");

    cudaFree(deviceDigits);
    cudaFree(deviceOffsets);
    cudaFree(deviceLengths);
    cudaFree(deviceOut);

    __int128 grandTotal = 0;
    for (size_t i = 0; i < N; ++i) grandTotal += ( __int128 ) hostOut[i];

    if (grandTotal == 0) { std::cout << "0\n"; return 0; }
    bool neg = false;
    if (grandTotal < 0) { neg = true; grandTotal = -grandTotal; }
    std::string output;
    while (grandTotal > 0) {
        int digit = (int)(grandTotal % 10);
        output.push_back(char('0' + digit));
        grandTotal /= 10;
    }
    if (neg) output.push_back('-');
    std::reverse(output.begin(), output.end());
    std::cout << output << "\n";
    return 0;
}

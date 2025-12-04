#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t e, const char* msg=nullptr) {
    if (e != cudaSuccess) {
        if (msg) std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        else std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void perBankMaxKernel(const unsigned char *digits, const size_t *offsets, const size_t *lengths, unsigned int *out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    size_t off = offsets[idx];
    size_t len = lengths[idx];
    if (len < 2) { out[idx] = 0u; return; }

    int bestRight = -1;
    unsigned int bestVal = 0u;
    for (size_t jj = 0; jj < len; ++jj) {
        size_t j = off + len - 1 - jj;
        int d = (int)digits[j];
        if (bestRight >= 0) {
            unsigned int candidate = (unsigned int)(d * 10 + bestRight);
            if (candidate > bestVal) bestVal = candidate;
        }
        if (d > bestRight) bestRight = d;
    }
    out[idx] = bestVal;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./part1 input.txt\n";
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
    unsigned int *deviceOut = nullptr;

    checkCuda(cudaMalloc(&deviceDigits, totalDigits * sizeof(unsigned char)), "cudaMalloc digits");
    checkCuda(cudaMalloc(&deviceOffsets, N * sizeof(size_t)), "cudaMalloc offsets");
    checkCuda(cudaMalloc(&deviceLengths, N * sizeof(size_t)), "cudaMalloc lengths");
    checkCuda(cudaMalloc(&deviceOut, N * sizeof(unsigned int)), "cudaMalloc out");

    checkCuda(cudaMemcpy(deviceDigits, hostDigits.data(), totalDigits * sizeof(unsigned char), cudaMemcpyHostToDevice), "cudaMemcpy digits");
    checkCuda(cudaMemcpy(deviceOffsets, offsets.data(), N * sizeof(size_t), cudaMemcpyHostToDevice), "cudaMemcpy offsets");
    checkCuda(cudaMemcpy(deviceLengths, lengths.data(), N * sizeof(size_t), cudaMemcpyHostToDevice), "cudaMemcpy lengths");

    int block = 256;
    int grid = (int)((N + block - 1) / block);
    perBankMaxKernel<<<grid, block>>>(deviceDigits, deviceOffsets, deviceLengths, deviceOut, N);
    checkCuda(cudaDeviceSynchronize(), "kernel sync");
    checkCuda(cudaGetLastError(), "kernel launch");

    std::vector<unsigned int> hostOut(N);
    checkCuda(cudaMemcpy(hostOut.data(), deviceOut, N * sizeof(unsigned int), cudaMemcpyDeviceToHost), "Memcpy out");

    cudaFree(deviceDigits);
    cudaFree(deviceOffsets);
    cudaFree(deviceLengths);
    cudaFree(deviceOut);

    unsigned long long total = 0ull;
    for (size_t i = 0; i < N; ++i) total += (unsigned long long)hostOut[i];

    std::cout << total << "\n";
    return 0;
}

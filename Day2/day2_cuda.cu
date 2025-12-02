#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <sstream>
#include <algorithm>

struct Task {
    uint64_t a;
    uint64_t b;
    int k;
    Task() {}
    Task(uint64_t A, uint64_t B, int K): a(A), b(B), k(K) {}
};

__global__ static void computeBoundsKernel(const uint64_t *deviceA, const uint64_t *deviceB, const int *deviceK,
                                             uint64_t *deviceOutLow, uint64_t *deviceOutHigh, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    uint64_t a = deviceA[idx];
    uint64_t b = deviceB[idx];
    int k = deviceK[idx];

    if (a > b) {
        uint64_t temp = a; a = b; b = temp;
    }

    uint64_t sMin = 1;
    for (int i = 1; i < k; ++i) sMin *= 10ULL;
    uint64_t sMax = sMin * 10ULL - 1ULL;

    uint64_t pow10 = 1;
    for (int i = 0; i < k; ++i) pow10 *= 10ULL;
    uint64_t M = pow10 + 1ULL;

    uint64_t low = (a + M - 1ULL) / M;
    uint64_t high = b / M;

    if (low < sMin) low = sMin;
    if (high > sMax) high = sMax;
    if (low > high) {
        deviceOutLow[idx] = 0ULL;
        deviceOutHigh[idx] = 0ULL;
    } else {
        deviceOutLow[idx] = low;
        deviceOutHigh[idx] = high;
    }
}

static std::vector<std::pair<uint64_t,uint64_t>> parseRangesLine(const std::string &line) {
    std::vector<std::pair<uint64_t,uint64_t>> result;
    std::string s = line;
    std::string temp;
    temp.reserve(s.size());
    for (char c: s) if (c != ' ' && c != '\t' && c != '\r' && c != '\n') temp.push_back(c);
    s.swap(temp);
    std::stringstream ss(s);
    while (ss.good()) {
        std::string part;
        if (!std::getline(ss, part, ',')) break;
        if (part.empty()) continue;
        size_t dash = part.find('-');
        if (dash == std::string::npos) continue;
        std::string aStr = part.substr(0, dash);
        std::string bStr = part.substr(dash+1);
        uint64_t a = std::stoull(aStr);
        uint64_t b = std::stoull(bStr);
        result.emplace_back(a,b);
    }
    return result;
}

static std::string int128ToString(__int128 v) {
    if (v == 0) return "0";
    bool neg = false;
    if (v < 0) { neg = true; v = -v; }
    __int128 ten = 10;
    std::string s;
    while (v > 0) {
        int digit = (int)(v % ten);
        s.push_back(char('0' + digit));
        v /= ten;
    }
    if (neg) s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
}

int main(int argc, char** argv) {
    std::string inputLine;
    if (argc >= 2) {
        std::ifstream fin(argv[1]);
        if (!fin) {
            std::cerr << "Cannot open " << argv[1] << "\n";
            return 1;
        }
        std::getline(fin, inputLine);
    } else {
        if (!std::getline(std::cin, inputLine)) {
            std::cerr << "Provide the single-line input as an argument or via stdin.\n";
            return 1;
        }
    }

    auto ranges = parseRangesLine(inputLine);
    if (ranges.empty()) {
        std::cout << "0\n";
        return 0;
    }

    std::vector<Task> tasks;
    tasks.reserve(ranges.size() * 11);
    for (auto &pr : ranges) {
        uint64_t a = pr.first;
        uint64_t b = pr.second;
        uint64_t maxVal = (b > a) ? b : a;
        int maxDigits = 1;
        {
            uint64_t t = maxVal;
            while (t >= 10ULL) { t /= 10ULL; ++maxDigits; }
        }
        int maxK = maxDigits / 2;
        if (maxK < 1) continue;
        for (int k = 1; k <= maxK; ++k) {
            tasks.emplace_back(a, b, k);
        }
    }

    size_t N = tasks.size();
    if (N == 0) {
        std::cout << "0\n";
        return 0;
    }

    thrust::host_vector<uint64_t> hostA(N), hostB(N);
    thrust::host_vector<int> hostK(N);
    for (size_t i = 0; i < N; ++i) {
        hostA[i] = tasks[i].a;
        hostB[i] = tasks[i].b;
        hostK[i] = tasks[i].k;
    }

    thrust::device_vector<uint64_t> deviceA = hostA;
    thrust::device_vector<uint64_t> deviceB = hostB;
    thrust::device_vector<int> deviceK = hostK;
    thrust::device_vector<uint64_t> deviceOutLow(N);
    thrust::device_vector<uint64_t> deviceOutHigh(N);

    const int block = 256;
    int grid = (int)((N + block - 1) / block);
    computeBoundsKernel<<<grid, block>>>(thrust::raw_pointer_cast(deviceA.data()),
                                           thrust::raw_pointer_cast(deviceB.data()),
                                           thrust::raw_pointer_cast(deviceK.data()),
                                           thrust::raw_pointer_cast(deviceOutLow.data()),
                                           thrust::raw_pointer_cast(deviceOutHigh.data()),
                                           N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    thrust::host_vector<uint64_t> hostOutLow = deviceOutLow;
    thrust::host_vector<uint64_t> hostOutHigh = deviceOutHigh;

    __int128 grandTotal = 0;

    for (size_t i = 0; i < N; ++i) {
        uint64_t low = hostOutLow[i];
        uint64_t high = hostOutHigh[i];
        if (low == 0 && high == 0) continue;
        int k = hostK[i];
        unsigned long long pow10 = 1ULL;
        for (int j = 0; j < k; ++j) pow10 *= 10ULL;
        unsigned long long M = pow10 + 1ULL;

        __int128 count = ( __int128 )(high - low + 1ULL);
        __int128 sumS = (__int128)(low + high) * count / 2;
        __int128 add = (__int128)M * sumS;
        grandTotal += add;
    }

    std::string output = int128ToString(grandTotal);
    std::cout << output << "\n";
    return 0;
}

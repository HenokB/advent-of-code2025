#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

using i64 = long long;

struct PrefixToPosition {
    const i64 start;
    PrefixToPosition(i64 s): start(s) {}
    __device__ i64 operator()(const i64 &prefix) const {
        i64 position = (start + prefix) % 100;
        if (position < 0) position += 100;
        return position;
    }
};

struct CountHits {
    __device__ i64 operator()(const thrust::tuple<const i64&, const i64&>& t) const {
        i64 displacement = thrust::get<0>(t);
        i64 startPosition = thrust::get<1>(t);
        i64 distance = displacement >= 0 ? displacement : -displacement;
        if (distance <= 0) return (i64)0;

        i64 remainder;
        if (displacement > 0) {
            remainder = ((100 - (startPosition % 100)) % 100);
        } else {
            remainder = startPosition % 100;
        }
        i64 firstK = (remainder == 0) ? 100 : remainder;
        if (firstK > distance) return (i64)0;
        i64 remaining = distance - firstK;
        return (i64)(1 + (remaining / 100));
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./main input.txt\n";
        return 1;
    }

    std::ifstream fin(argv[1]);
    if (!fin) {
        std::cerr << "Cannot open file " << argv[1] << "\n";
        return 1;
    }

    std::vector<i64> values;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        char direction = line[0];
        size_t pos = 1;
        while (pos < line.size() && isspace((unsigned char)line[pos])) ++pos;
        i64 distance = std::stoll(line.substr(pos));
        i64 signedDistance = (direction == 'R') ? distance : -distance;
        values.push_back(signedDistance);
    }

    if (values.empty()) {
        std::cout << 0 << "\n";
        return 0;
    }

    thrust::device_vector<i64> deviceInput = values;
    thrust::device_vector<i64> devicePrefix(deviceInput.size());

    thrust::exclusive_scan(deviceInput.begin(), deviceInput.end(), devicePrefix.begin(), (i64)0);

    const i64 start = 50;
    thrust::device_vector<i64> devicePosition(devicePrefix.size());
    thrust::transform(devicePrefix.begin(), devicePrefix.end(), devicePosition.begin(), PrefixToPosition(start));

    thrust::device_vector<i64> deviceHits(deviceInput.size());
    auto zipFirst = thrust::make_zip_iterator(thrust::make_tuple(deviceInput.begin(), devicePosition.begin()));
    auto zipLast = thrust::make_zip_iterator(thrust::make_tuple(deviceInput.end(), devicePosition.end()));
    thrust::transform(zipFirst, zipLast, deviceHits.begin(), CountHits());

    i64 total = thrust::reduce(deviceHits.begin(), deviceHits.end(), (i64)0, thrust::plus<i64>());

    std::cout << total << "\n";
    return 0;
}


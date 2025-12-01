#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/count.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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

    std::vector<int> values;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        char direction = line[0];
        size_t pos = 1;
        while (pos < line.size() && isspace((unsigned char)line[pos])) ++pos;
        int distance = std::stoi(line.substr(pos));
        int signedDistance = (direction == 'R' ? distance : -distance);
        values.push_back(signedDistance);
    }

    if (values.empty()) {
        std::cout << 0 << "\n";
        return 0;
    }

    thrust::device_vector<int> deviceInput = values;
    thrust::device_vector<int> devicePrefix(deviceInput.size());

    thrust::inclusive_scan(deviceInput.begin(), deviceInput.end(), devicePrefix.begin());

    const int start = 50;
    const int target = (100 - (start % 100)) % 100;

    thrust::transform(devicePrefix.begin(), devicePrefix.end(), devicePrefix.begin(),
        [] __device__ (int x) {
            int mod = x % 100;
            if (mod < 0) mod += 100;
            return mod;
        });

    long long count = thrust::count(devicePrefix.begin(), devicePrefix.end(), target);
    std::cout << count << "\n";
    return 0;
}


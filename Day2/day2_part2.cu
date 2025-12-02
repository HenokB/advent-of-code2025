#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <cstdint>

using u64 = unsigned long long;
using i128 = __int128_t;

static std::string int128ToString(i128 v) {
    if (v == 0) return "0";
    bool neg = false;
    if (v < 0) { neg = true; v = -v; }
    std::string s;
    while (v > 0) {
        int d = int(v % 10);
        s.push_back(char('0' + d));
        v /= 10;
    }
    if (neg) s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
}

static std::vector<std::pair<u64,u64>> parseRangesLine(const std::string &line) {
    std::vector<std::pair<u64,u64>> result;
    std::string temp;
    temp.reserve(line.size());
    for (char c : line) if (!isspace((unsigned char)c)) temp.push_back(c);
    std::stringstream ss(temp);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        size_t dash = part.find('-');
        if (dash == std::string::npos) continue;
        std::string aStr = part.substr(0, dash);
        std::string bStr = part.substr(dash + 1);
        u64 a = std::stoull(aStr);
        u64 b = std::stoull(bStr);
        result.emplace_back(a,b);
    }
    return result;
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string line;
    if (argc >= 2) {
        std::ifstream fin(argv[1]);
        if (!fin) { std::cerr << "Cannot open " << argv[1] << "\n"; return 1; }
        std::getline(fin, line);
    } else {
        if (!std::getline(std::cin, line)) { std::cerr << "Provide the single-line input via file or stdin\n"; return 1; }
    }

    auto ranges = parseRangesLine(line);
    if (ranges.empty()) { std::cout << "0\n"; return 0; }

    std::unordered_set<std::string> seen;
    seen.reserve(1<<16);

    for (auto &pr : ranges) {
        u64 a0 = pr.first;
        u64 b0 = pr.second;
        u64 a = std::min(a0,b0), b = std::max(a0,b0);

        int maxDigits = 1;
        {
            u64 t = b;
            while (t >= 10ULL) { t /= 10ULL; ++maxDigits; }
        }

        for (int k = 1; k <= maxDigits; ++k) {
            for (int r = 2; k * r <= maxDigits; ++r) {
                i128 pow10K = 1;
                for (int i = 0; i < k; ++i) pow10K *= 10;
                i128 pow10KR = 1;
                for (int i = 0; i < k * r; ++i) pow10KR *= 10;

                i128 denom = pow10K - 1;
                if (denom == 0) continue;
                i128 numer = pow10KR - 1;
                i128 M = numer / denom;

                if (M <= 0) continue;

                i128 a128 = (i128)a;
                i128 b128 = (i128)b;
                i128 low = (a128 + M - 1) / M;
                i128 high = b128 / M;

                i128 sMin = 1;
                for (int i = 1; i < k; ++i) sMin *= 10;
                i128 sMax = sMin * 10 - 1;

                if (low < sMin) low = sMin;
                if (high > sMax) high = sMax;
                if (low > high) continue;

                for (i128 s = low; s <= high; ++s) {
                    i128 value = s * M;
                    std::string key = int128ToString(value);
                    seen.insert(key);
                }
            }
        }
    }

    i128 total = 0;
    for (auto &k : seen) {
        i128 v = 0;
        for (char c : k) { v = v * 10 + (c - '0'); }
        total += v;
    }

    std::cout << int128ToString(total) << "\n";
    return 0;
}

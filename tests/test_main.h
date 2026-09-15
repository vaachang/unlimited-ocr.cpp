#pragma once

// Tiny self-contained test harness (avoids a gtest dependency).

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {

struct Case {
    std::string name;
    std::function<void()> fn;
};

std::vector<Case>& cases();
int run_all();

[[noreturn]] inline void fail(const std::string& msg, const char* file, int line) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + msg);
}

}  // namespace test

#define UOCR_TEST(name)                                              \
    static void name();                                             \
    static const bool name##_registered = []() {                    \
        ::test::cases().push_back({#name, &name});                  \
        return true;                                                \
    }();                                                            \
    static void name()

#define CHECK(cond)                                                              \
    do {                                                                        \
        if (!(cond)) ::test::fail("CHECK failed: " #cond, __FILE__, __LINE__);   \
    } while (0)

#define CHECK_EQ(a, b)                                                                    \
    do {                                                                                  \
        const auto _a = (a);                                                              \
        const auto _b = (b);                                                              \
        if (!(_a == _b))                                                                  \
            ::test::fail("CHECK_EQ failed: " #a " == " #b, __FILE__, __LINE__);           \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                             \
    do {                                                                                  \
        const double _a = (a);                                                            \
        const double _b = (b);                                                            \
        if (std::fabs(_a - _b) > (tol))                                                   \
            ::test::fail("CHECK_NEAR failed: " #a " ~= " #b, __FILE__, __LINE__);         \
    } while (0)

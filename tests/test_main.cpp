#include "test_main.h"

#include <exception>

namespace test {

std::vector<Case>& cases() {
    static std::vector<Case> c;
    return c;
}

int run_all() {
    int failed = 0;
    int passed = 0;
    for (const auto& c : cases()) {
        try {
            c.fn();
            std::printf("[  OK  ] %s\n", c.name.c_str());
            ++passed;
        } catch (const std::exception& e) {
            std::printf("[ FAIL ] %s\n         %s\n", c.name.c_str(), e.what());
            ++failed;
        } catch (...) {
            std::printf("[ FAIL ] %s (unknown exception)\n", c.name.c_str());
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed, %zu total\n", passed, failed, cases().size());
    return failed == 0 ? 0 : 1;
}

}  // namespace test

int main() { return test::run_all(); }

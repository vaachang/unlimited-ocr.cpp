#include "uocr/log.h"

#include <cstdarg>
#include <ctime>

namespace uocr {
namespace log {

namespace {
Level g_level = Level::Info;
}

void set_level(Level lvl) { g_level = lvl; }
Level level() { return g_level; }
bool enabled(Level lvl) { return static_cast<int>(lvl) >= static_cast<int>(g_level); }

void write(Level lvl, const char* file, int line, const char* fmt, ...) {
    static const char* names[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "OFF"};

    std::time_t t = std::time(nullptr);
    std::tm tmv{};
    localtime_r(&t, &tmv);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%H:%M:%S", &tmv);

    std::fprintf(stderr, "[%s][%s] ", ts, names[static_cast<int>(lvl)]);

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fprintf(stderr, "  (%s:%d)\n", file, line);
    std::fflush(stderr);
}

}  // namespace log
}  // namespace uocr

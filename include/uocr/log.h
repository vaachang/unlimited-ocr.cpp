#pragma once

// Minimal header-only-ish logging facility.
//
// prj.md suggests spdlog, but to keep the build self-contained we ship a tiny
// logger instead.  It supports the subset the engine needs: level filtering,
// timestamps, file/line anchors and printf-style formatting.

#include <cstdio>
#include <string>

namespace uocr {
namespace log {

enum class Level { Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4, Off = 5 };

void set_level(Level lvl);
Level level();
bool enabled(Level lvl);

// Not thread safe by itself; the engine only logs from the host thread.
void write(Level lvl, const char* file, int line, const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 4, 5)))
#endif
    ;

}  // namespace log
}  // namespace uocr

#define UOCR_LOG(level, ...)                                              \
    do {                                                                  \
        if (::uocr::log::enabled(level))                                  \
            ::uocr::log::write((level), __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)

#define UOCR_TRACE(...) UOCR_LOG(::uocr::log::Level::Trace, __VA_ARGS__)
#define UOCR_DEBUG(...) UOCR_LOG(::uocr::log::Level::Debug, __VA_ARGS__)
#define UOCR_INFO(...) UOCR_LOG(::uocr::log::Level::Info, __VA_ARGS__)
#define UOCR_WARN(...) UOCR_LOG(::uocr::log::Level::Warn, __VA_ARGS__)
#define UOCR_ERROR(...) UOCR_LOG(::uocr::log::Level::Error, __VA_ARGS__)

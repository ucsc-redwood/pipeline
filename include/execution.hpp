#pragma once

// Check if <execution> header is available
#ifdef __has_include
#if __has_include(<execution>)
#include <execution>
#define HAS_EXECUTION_HEADER 1
#endif
#endif

#ifdef HAS_EXECUTION_HEADER
#define EXE_PAR std::execution::par
#else
#define EXE_PAR
#endif

#ifndef LIBGP_VERSION_H
#define LIBGP_VERSION_H

#define LIBGP_VERSION_MAJOR 0
#define LIBGP_VERSION_MINOR 1
#define LIBGP_VERSION_PATCH 4

#if defined _WIN32 || defined __CYGWIN__
  #ifdef libgp_EXPORTS
    #define LIBGP_EXPORT __declspec(dllexport)
  #else
    #define LIBGP_EXPORT __declspec(dllimport)
  #endif
#else
  #define LIBGP_EXPORT __attribute__((visibility("default")))
#endif

#endif // LIBGP_VERSION_H
#ifndef PERSIST_H_
#define PERSIST_H_

#include <stdint.h>
#include <cstdio>
#include <cstdlib>


//add by cc
#define CLWB 1

#define CAS(_p, _u, _v)                                             \
  (__atomic_compare_exchange_n(_p, _u, _v, false, __ATOMIC_ACQUIRE, \
                               __ATOMIC_ACQUIRE))
#define kCacheLineSize (64)

static inline void CPUPause(void) { __asm__ volatile("pause" ::: "memory"); }
static inline void clflush(void *p) {
#ifdef CLFLUSH
  asm volatile("clflush %0" : "+m"(p));
#elif CLFLUSH_OPT
  asm volatile(".byte 0x66; clflush %0" : "+m"(p));
#elif CLWB
  asm volatile(".byte 0x66; xsaveopt %0" : "+m"(p));
#endif
}
static inline void BARRIER(void *p) { clflush(p); }

static inline void mfence(void) { asm volatile("mfence" ::: "memory"); }

static inline void clflush(char *data, size_t len, bool fence = true) {
  volatile char *ptr = (char *)((unsigned long)data & (~(kCacheLineSize - 1)));
  if (fence) mfence();
  for (; ptr < data + len; ptr += kCacheLineSize) {
#ifdef CLFLUSH
    asm volatile("clflush %0" : "+m"(*(volatile char *)ptr));
#elif CLFLUSH_OPT
    asm volatile(".byte 0x66; clflush %0" : "+m"(*(volatile char *)ptr));
#elif CLWB
    asm volatile(".byte 0x66; xsaveopt %0" : "+m"(*(volatile char *)(ptr)));
#endif
  }
  if (fence) mfence();
}

#endif
#ifndef PERSIST_H_
#define PERSIST_H_

#include <libpmem.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

//add by cc
#define CLWB 1

#define K_CACHE_LINE_SIZE (64)

static inline void clflush(void *p) {
#ifdef CLFLUSH
  asm volatile("clflush %0" : "+m"(p));
#elif CLFLUSH_OPT
  asm volatile(".byte 0x66; clflush %0" : "+m"(p));
#elif CLWB
  asm volatile(".byte 0x66; xsaveopt %0" : "+m"(p));
#endif
}

static inline void mfence(void) { asm volatile("mfence" ::: "memory"); }

static inline void clflush(char *data, size_t len, bool fence = true) {
  volatile char *ptr = (char *)((unsigned long)data & (~(K_CACHE_LINE_SIZE - 1)));
  if (fence) mfence();
  for (; ptr < data + len; ptr += K_CACHE_LINE_SIZE) {
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
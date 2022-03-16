#pragma once

#include <assert.h>
#include <stdio.h>

#define NN_MAX_NDIM 4

#define EXPAND(...) __VA_ARGS__

#define NAME_SEQUENCE_17(_name)                                                \
  _name, _name, _name, _name, _name, _name, _name, _name, _name, _name, _name, \
      _name, _name, _name, _name, _name, _name

#define GET_MACRO_18(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, \
                     _14, _15, _16, _17, _18, NAME, ...)                     \
  NAME

#define nn_assert_without_msg(_expr) assert(_expr)

inline void do_nothing() {}

/*
 * The arguments after _expr are expected to be the arguments for printf.
 * The variables used in the format string should be no more than 16.
 */
#define nn_assert_with_msg(_expr, ...) \
  do {                                 \
    if (!(_expr)) {                    \
      printf(__VA_ARGS__);             \
      abort();                         \
    }                                  \
  } while (0)

#define nn_assert(...)                                                        \
  EXPAND(                                                                     \
      GET_MACRO_18 EXPAND((__VA_ARGS__, NAME_SEQUENCE_17(nn_assert_with_msg), \
                           nn_assert_without_msg)))                           \
  (__VA_ARGS__)

#define nn_throw(...) throw __VA_ARGS__

#define nn_throw_if(_expr, ...) \
  do {                          \
    if (!(_expr)) {             \
      throw __VA_ARGS__;        \
    }                           \
  } while (0)

#define DEF_OP_IMPL_CTOR(_name)   \
 private:                         \
  _name() {}                      \
  _name(const _name&);            \
  _name& operator=(const _name&); \
                                  \
 public:                          \
  static _name* get_instance() {  \
    static _name instance;        \
    return &instance;             \
  }

#pragma once

#include "paddle/extension.h"


#define CHECK_CPU(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")
#define CHECK_INPUT(x) PD_CHECK(x, "Input mismatch")


template <typename T>
class MPTypeTrait {
 public:
  using Type = T;
};

template <>
class MPTypeTrait<phi::dtype::float16> {
 public:
  using Type = float;
};

template <>
class MPTypeTrait<phi::dtype::bfloat16> {
 public:
  using Type = float;
};

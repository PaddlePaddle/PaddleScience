#pragma once

#include "paddle/extension.h"


#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")
#define CHECK_INPUT(x) PD_CHECK(x, "Input mismatch")

///////// Basic Marco ///////////

#define PD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...) \
  case enum_type: {                                                       \
    using HINT = type;                                                    \
    __VA_ARGS__();                                                        \
    break;                                                                \
  }

#define PD_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...) \
  PD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, data_t, __VA_ARGS__)

///////// Floating and Integral Dispatch Marco ///////////

#define PD_DISPATCH_FLOATING_AND_INTEGRAL_AND_2_TYPES(                        \
    SPECIFIED_TYPE1, SPECIFIED_TYPE2, TYPE, NAME, ...)                        \
  PD_VISIT_FLOATING_AND_INTEGRAL_AND_2_TYPES(                                 \
    SPECIFIED_TYPE1, SPECIFIED_TYPE2, TYPE, NAME, __VA_ARGS__)

///////// Floating and Integral Dispatch Marco ///////////

#define PD_VISIT_FLOATING_AND_INTEGRAL_AND_2_TYPES(                           \
    SPECIFIED_TYPE1, SPECIFIED_TYPE2, TYPE, NAME, ...)                        \
  [&] {                                                                       \
    const auto& __dtype__ = TYPE;                                             \
    switch (__dtype__) {                                                      \
      PD_PRIVATE_CASE_TYPE(NAME,                                              \
                           SPECIFIED_TYPE1,                                   \
                           ::phi::DataTypeToCppType<SPECIFIED_TYPE1>::type,   \
                           __VA_ARGS__)                                       \
      PD_PRIVATE_CASE_TYPE(NAME,                                              \
                           SPECIFIED_TYPE2,                                   \
                           ::phi::DataTypeToCppType<SPECIFIED_TYPE2>::type,   \
                           __VA_ARGS__)                                       \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::FLOAT64, double, __VA_ARGS__)             \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::UINT8, uint8_t, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT8, int8_t, __VA_ARGS__)                \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT16, int16_t, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT32, int, __VA_ARGS__)                  \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT64, int64_t, __VA_ARGS__)              \
      default:                                                                \
        PD_THROW("function " #NAME " is not implemented for data type `",     \
                 __dtype__,                                                   \
                 "`");                                                        \
    }                                                                         \
  }()

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

#ifdef USE_ROCM
#define SHFL_UP_SYNC(mask, var, delta) __shfl_up(var, delta)
#define SHFL_DOWN_SYNC(mask, var, delta) __shfl_down(var, delta)
#else
#define SHFL_UP_SYNC __shfl_up_sync
#define SHFL_DOWN_SYNC __shfl_down_sync
#endif

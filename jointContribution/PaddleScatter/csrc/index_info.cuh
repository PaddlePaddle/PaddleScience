#pragma once

#include "paddle/extension.h"

#define MAX_TENSORINFO_DIMS 7

template <typename T> struct TensorInfo {
  TensorInfo(const T *p, int dim, int sz[MAX_TENSORINFO_DIMS],
             int st[MAX_TENSORINFO_DIMS]) {
    data = p;
    dims = dim;
    PD_CHECK(dims < MAX_TENSORINFO_DIMS, "Input dims should be smaller than 7.");

    for (int i = 0; i < dim; ++i) {
      sizes[i] = sz[i];
      strides[i] = st[i];
    }
  }

  const T* data;
  int dims;
  int sizes[MAX_TENSORINFO_DIMS];
  int strides[MAX_TENSORINFO_DIMS];
};


template <typename T>
TensorInfo<T> getTensorInfo(const paddle::Tensor &tensor) {
  int sizes[MAX_TENSORINFO_DIMS];
  int strides[MAX_TENSORINFO_DIMS];

  int dims = tensor.shape().size();
  for (int i = dims - 1; i >= 0; --i) {
    sizes[i] = tensor.shape()[i];
    sizes[i] = tensor.strides()[i];
  }

  return TensorInfo<T>(tensor.data<T>(), dims, sizes,
                              strides);
}

// Uses dynamic (runtime) instead of static (compiletime) dims
template <typename T>
struct IndexToOffset {
  static inline __host__ __device__ int get(
    int linearId,
    const TensorInfo<T>& info) {

      int offset = 0;

      for (int i = info.dims - 1; i > 0; --i) {
        int curDimIndex = linearId % info.sizes[i];
        int curDimOffset = curDimIndex * info.strides[i];
        offset += curDimOffset;
        linearId /= info.sizes[i];
      }

      return offset + linearId * info.strides[0];
  }
};

// We need our own `IndexToOffset` implementation since we do not want to
// access the last element of the `indexptr`.
template <typename T>
struct IndexPtrToOffset {
  static inline __host__ __device__ int get(
    int idx,
    const TensorInfo<T>& info) {

    int offset = idx % (info.sizes[info.dims - 1] - 1);
    offset *= info.strides[info.dims - 1];
    idx /= info.sizes[info.dims - 1] - 1;
    for (int i = info.dims - 2; i >= 0; --i) {
      offset += (idx % info.sizes[i]) * info.strides[i];
      idx /= info.sizes[i];
    }

    return offset;
  }
};

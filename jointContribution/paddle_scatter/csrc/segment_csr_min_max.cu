#include "paddle/extension.h"

#include <vector>
#include <limits>
#include <numeric>

#include "atomics.cuh"
#include "index_info.cuh"
#include "utils.cuh"

#define THREADS 256
#define BLOCKS(TB, N) (TB * N + THREADS - 1) / THREADS
#define FULL_MASK 0xffffffff


enum ReductionType { MIN, MAX };

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"min", MIN},   {"max", MAX}
};

template <typename data_t, typename index_t, int TB>
__global__ void segment_csr_min_max_kernel(const data_t *x_data,
                                           const TensorInfo<index_t> indptr_info,
                                           ReductionType reduce_type,
                                           data_t *out_data,
                                           index_t *arg_out_data,
                                           size_t N,
                                           size_t E) {

  // Each warp processes exactly `32/TB` rows and aggregates all row values
  // via a parallel reduction.

  using MPType = typename MPTypeTrait<data_t>::Type;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / TB;
  int lane_idx = thread_idx & (TB - 1);

  if (row_idx < N) {
    int offset = IndexPtrToOffset<index_t>::get(row_idx, indptr_info);
    index_t row_start = __ldg(indptr_info.data + offset);
    index_t row_end = __ldg(indptr_info.data + offset +
                            indptr_info.strides[indptr_info.dims - 1]);

    data_t val = (reduce_type == MIN) ? static_cast<data_t>(std::numeric_limits<MPType>::max()) : static_cast<data_t>(std::numeric_limits<MPType>::lowest());
    index_t arg, arg_tmp;

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E;
    for (index_t x_idx = row_start + lane_idx; x_idx < row_end;
         x_idx += TB) {
      // update
      auto cmp = x_data[offset + x_idx];
      if ((reduce_type == MIN && cmp < val) || 
          (reduce_type == MAX && cmp > val)) {
        val = cmp;
        arg = x_idx;
      }
    }

#pragma unroll
    for (int i = TB / 2; i > 0; i /= 2) {
      // Parallel reduction inside a single warp.
      arg_tmp = SHFL_DOWN_SYNC(FULL_MASK, arg, i);
      // update
      MPType cmp = SHFL_DOWN_SYNC(FULL_MASK, static_cast<MPType>(val), i);
      if ((reduce_type == MIN && cmp < static_cast<MPType>(val)) || 
          (reduce_type == MAX && cmp > static_cast<MPType>(val))) {
        val = static_cast<data_t>(cmp);
        arg = arg_tmp;
      }
    }

    if (lane_idx == 0) {
      // write
      if (row_end - row_start > 0) {
        out_data[row_idx] = val;
        arg_out_data[row_idx] = arg;
      } else {
        out_data[row_idx] = static_cast<data_t>(0);
      }
    }
  }
}

template <typename data_t, typename index_t>
__global__ void segment_csr_broadcast_min_max_kernel(const data_t *x_data,
                                                     const TensorInfo<index_t> indptr_info,
                                                     ReductionType reduce_type,
                                                     data_t *out_data,
                                                     index_t *arg_out_data,
                                                     size_t N,
                                                     size_t K,
                                                     size_t E) {

  // Each thread processes exactly one row. It turned out that is more
  // efficient than using shared memory due to avoiding synchronization
  // barriers.

  using MPType = typename MPTypeTrait<data_t>::Type;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / K;
  int lane_idx = thread_idx % K;

  if (thread_idx < N * K) {
    int offset = IndexPtrToOffset<index_t>::get(row_idx, indptr_info);
    index_t row_start = __ldg(indptr_info.data + offset);
    index_t row_end = __ldg(indptr_info.data + offset +
                            indptr_info.strides[indptr_info.dims - 1]);

    data_t val = (reduce_type == MIN) ? static_cast<data_t>(std::numeric_limits<MPType>::max()) : static_cast<data_t>(std::numeric_limits<MPType>::lowest());
    index_t arg;

    offset = (row_idx / (indptr_info.sizes[indptr_info.dims - 1] - 1)) * E * K;
    for (index_t x_idx = row_start; x_idx < row_end; x_idx++) {
      // update
      auto cmp = x_data[offset + K * x_idx + lane_idx];
      if ((reduce_type == MIN && cmp < val) || 
          (reduce_type == MAX && cmp > val)) {
        val = cmp;
        arg = x_idx;
      }
    }

    // write
    if (row_end - row_start > 0) {
      out_data[thread_idx] = val;
      arg_out_data[thread_idx] = arg;
    } else {
      out_data[thread_idx] = static_cast<data_t>(0);
    }
  }
}

std::vector<paddle::Tensor> segment_csr_min_max_cuda_forward(const paddle::Tensor& x,
                                                             const paddle::Tensor& indptr,
                                                             const std::vector<int64_t>& return_shape,
                                                             const std::string& reduce) {
  CHECK_CUDA(indptr);

  auto x_dims = x.shape();
  auto indptr_dims = indptr.shape();
  CHECK_INPUT(x_dims.size() >= indptr_dims.size());
  auto dim = indptr_dims.size() - 1;

  // custom op input tensors are already contiguous
  // x = x.contiguous();

  paddle::Tensor out;
  out = paddle::empty(return_shape, x.dtype(), x.place());
  
  paddle::Tensor arg_out;
  arg_out = paddle::experimental::full_like(out, x_dims[dim], indptr.dtype(), indptr.place());

  auto N = return_shape[dim] * (indptr.numel() / indptr_dims[dim]);
  auto K = out.numel() / N;
  auto E = x_dims[dim];

  PD_DISPATCH_FLOATING_AND_INTEGRAL_AND_2_TYPES(
    paddle::DataType::FLOAT16, paddle::DataType::BFLOAT16,
    x.dtype(), "segment_csr_min_max_cuda_forward_kernel", ([&] {

    const data_t* x_data = x.data<data_t>();
    data_t* out_data = out.data<data_t>();
    switch(indptr.dtype()) {
      case paddle::DataType::INT32:
      { 
        auto indptr_info = getTensorInfo<int>(indptr);
        int* arg_out_data = arg_out.data<int>();
        if (K == 1) {
          segment_csr_min_max_kernel<data_t, int, 1>
              <<<BLOCKS(32, N), THREADS, 0, x.stream()>>>(
                  x_data, indptr_info, reduce2REDUCE.at(reduce), 
                  out_data, arg_out_data, N, E);
        } else {
          segment_csr_broadcast_min_max_kernel<data_t, int>
              <<<BLOCKS(1, N * K), THREADS, 0, x.stream()>>>(
                  x_data, indptr_info, reduce2REDUCE.at(reduce), 
                  out_data, arg_out_data, N, K, E);
        }
        break;
      }
      case paddle::DataType::INT64:
      {
        auto indptr_info = getTensorInfo<int64_t>(indptr);
        int64_t* arg_out_data = arg_out.data<int64_t>();
        if (K == 1) {
          segment_csr_min_max_kernel<data_t, int64_t, 1>
              <<<BLOCKS(32, N), THREADS, 0, x.stream()>>>(
                  x_data, indptr_info, reduce2REDUCE.at(reduce), 
                  out_data, arg_out_data, N, E);
        } else {
          segment_csr_broadcast_min_max_kernel<data_t, int64_t>
              <<<BLOCKS(1, N * K), THREADS, 0, x.stream()>>>(
                  x_data, indptr_info, reduce2REDUCE.at(reduce), 
                  out_data, arg_out_data, N, K, E);
        }
        break;
      }
      default:
        PD_THROW(
          "function segment_csr_min_max_cuda_forward_kernel is not implemented for the indptr data type `",
          phi::DataTypeToString(indptr.dtype()), "`");
    }
  }));

  return {out, arg_out};
}

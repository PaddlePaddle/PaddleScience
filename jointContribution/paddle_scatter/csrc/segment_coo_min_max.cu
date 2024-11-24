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

template <typename data_t, typename mp_t, typename index_t, bool HAS_VAL>
__global__ void
segment_coo_min_max_cuda_forward_kernel(const data_t* x_data,
                                        const TensorInfo<index_t> index_info,
                                        ReductionType reduce_type,
                                        mp_t* out_data,
                                        size_t E,
                                        size_t N) {

  // Each thread processes exactly one entry. Within a warp, we perform a
  // parallel reduction across equal indices, and write the intermediate
  // result via atomics.

  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_idx = row_idx & (32 - 1);
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < E) {
    int offset = IndexToOffset<index_t>::get(row_idx, index_info);
    int64_t idx = index_info.data[offset], next_idx;
    int out_idx = (row_idx / D) * N + idx;

    mp_t val = HAS_VAL ? static_cast<mp_t>(x_data[row_idx]) : (mp_t)1, tmp;

#pragma unroll
    for (int i = 1; i < 32; i *= 2) {
      // Parallel reduction inside a single warp.
      tmp = SHFL_UP_SYNC(FULL_MASK, val, i);
      next_idx = SHFL_UP_SYNC(FULL_MASK, idx, i);
      if (lane_idx >= i && row_idx / D == (row_idx - i) / D) {
        assert(idx >= next_idx);
        if (idx == next_idx) {
          if ((reduce_type == MIN && tmp < val) ||
              (reduce_type == MAX && tmp > val))
              val = tmp;
        }
      }
    }

    next_idx = SHFL_DOWN_SYNC(FULL_MASK, idx, 1);
    if (lane_idx == 32 - 1 || row_idx / D != (row_idx + 1) / D ||
        idx != next_idx) {
      switch(reduce_type) {
        case MIN:
          {
        atomMin(out_data + out_idx, val);
          break;
          }
        case MAX:
        atomMax(out_data + out_idx, val);
          break;
      }
    }
  }
}

template <typename data_t, typename mp_t, typename index_t, int TB>
__global__ void segment_coo_min_max_broadcast_cuda_forward_kernel(const data_t *x_data,
                                                                  const TensorInfo<index_t> index_info,
                                                                  ReductionType reduce_type,
                                                                  mp_t *out_data,
                                                                  size_t E,
                                                                  size_t K,
                                                                  size_t N) {

  // Each thread processes a single column and `TB` index entries. Coalesced
  // read and write is performed in column-major order. The intermediate
  // results are written via atomics.

  int D = index_info.sizes[index_info.dims - 1];
  int E_1 = E / D;
  int E_2 = (D - 1) + TB - ((D - 1) % TB);

  int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int col_idx = blockIdx.y * blockDim.x + threadIdx.x;

  int dim_start = (row_idx * TB) / E_2;
  int row_start = (row_idx * TB) % E_2;

  if (dim_start < E_1 && col_idx < K) {

    int offset = IndexToOffset<index_t>::get(dim_start * D + row_start, index_info);
    int idx1 = __ldg(index_info.data + offset), idx2;

    mp_t val = static_cast<mp_t>(x_data[K * (dim_start * D + row_start) + col_idx]);

#pragma unroll
    for (int i = 1; i < TB; i++) {
      if (row_start + i >= D)
        break;

      idx2 = __ldg(index_info.data + offset +
                   i * index_info.strides[index_info.dims - 1]);
      assert(idx1 <= idx2);
      if (idx1 == idx2) {
        mp_t tmp = static_cast<mp_t>(x_data[K * (dim_start * D + row_start + i) + col_idx]);
        if ((reduce_type == MIN && tmp < val) ||
            (reduce_type == MAX && tmp > val))
            val = tmp;
      } else {
        switch(reduce_type) {
          case MIN:
            {
          atomMin(out_data + (dim_start * N + idx1) * K + col_idx, val);
            break;
            }
          case MAX:
          atomMax(out_data + (dim_start * N + idx1) * K + col_idx, val);
            break;
        }
        val = x_data[K * (dim_start * D + row_start + i) + col_idx];
      }

      idx1 = idx2;
    }

    switch(reduce_type) {
      case MIN:
        {
      atomMin(out_data + (dim_start * N + idx1) * K + col_idx, val);
        break;
        }
      case MAX:
      atomMax(out_data + (dim_start * N + idx1) * K + col_idx, val);
        break;
    }
  }
}


template <typename data_t, typename mp_t, typename index_t>
__global__ void segment_coo_arg_kernel(const data_t *x_data,
                                       const TensorInfo<index_t> index_info,
                                       mp_t *out_data,
                                       index_t *arg_out_data,
                                       size_t E,
                                       size_t N) {

  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < E) {
    int offset = IndexToOffset<index_t>::get(row_idx, index_info);
    index_t idx = index_info.data[offset];
    int out_idx = (row_idx / D) * N + idx;

    mp_t val = __ldg(out_data + out_idx);
    if (static_cast<mp_t>(x_data[row_idx]) == val)
      arg_out_data[out_idx] = row_idx % D;
  }
}

template <typename data_t, typename mp_t, typename index_t>
__global__ void segment_coo_arg_broadcast_kernel(const data_t *x_data,
                                                 const TensorInfo<index_t> index_info,
                                                 mp_t *out_data,
                                                 index_t *arg_out_data,
                                                 size_t E,
                                                 size_t K,
                                                 size_t N) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = thread_idx / K;
  int col_idx = thread_idx % K;
  int D = index_info.sizes[index_info.dims - 1];

  if (row_idx < E && col_idx < K) {
    int offset = IndexToOffset<index_t>::get(row_idx, index_info);
    int idx = __ldg(index_info.data + offset);
    int out_idx = ((row_idx / D) * N + idx) * K + col_idx;

    mp_t val = __ldg(out_data + out_idx);
    if (static_cast<mp_t>(x_data[thread_idx]) == val)
      arg_out_data[out_idx] = row_idx % D;
  }
}

template <typename data_t>
__global__ void post_process_kernel(data_t init_val,
                                    int numel,
                                    data_t* out_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid < numel) {
    if (out_data[tid] == init_val)
      out_data[tid] = static_cast<data_t>(0.0);
  }
}

std::vector<paddle::Tensor> segment_coo_min_max_cuda_forward(const paddle::Tensor& x,
                                                             const paddle::Tensor& index,
                                                             const paddle::optional<paddle::Tensor>& init,
                                                             std::vector<int64_t> return_shape,
                                                             std::string reduce) {
  CHECK_CUDA(index);
  if (init)
    CHECK_CUDA(init.get());

  auto x_dims = x.shape();
  auto index_dims = index.shape();
  CHECK_INPUT(x_dims.size() >= index_dims.size());
  for (auto i = 0; i < index_dims.size() - 1; ++i)
    CHECK_INPUT(x_dims[i] >= index_dims[i]);

  // custom op input tensors are already contiguous
  // x = x.contiguous();

  paddle::Tensor out;
  if (init) {
    // paddle::Tensor init_contiguous = init->contiguous();
    // out = paddle::Tensor(init_contiguous);
    out = paddle::Tensor(init.get());
  }
  else {
    out = paddle::empty(return_shape, x.dtype(), x.place());
  }
  
  auto dim = index_dims.size() - 1;
  paddle::Tensor arg_out;
  arg_out = paddle::experimental::full_like(out, x_dims[dim], index.dtype(), index.place());

  auto E = index.numel();
  auto E_2 = index.shape()[dim];
  auto E_1 = index.numel() / E_2;
  auto K = x.numel() / E;
  auto N = out.shape()[dim];
  auto avg_len = (float)E_2 / (float)N;

  PD_DISPATCH_FLOATING_AND_INTEGRAL_AND_2_TYPES(
    paddle::DataType::FLOAT16, paddle::DataType::BFLOAT16,
    x.dtype(), "segment_coo_min_max_cuda_forward_kernel", ([&] {

    using MPType = typename MPTypeTrait<data_t>::Type;
    paddle::Tensor out_mp;
    if (x.dtype() == paddle::DataType::FLOAT16 || x.dtype() == paddle::DataType::BFLOAT16) {
      out_mp = paddle::empty(return_shape, paddle::DataType::FLOAT32, x.place());  
    } else {
      out_mp = out;
    }
    
    if (!init) {
      if (reduce == "min")
        paddle::experimental::fill_(out_mp, std::numeric_limits<MPType>::max());
      else
        paddle::experimental::fill_(out_mp, std::numeric_limits<MPType>::lowest());
    }

    const data_t* x_data = x.data<data_t>();
    MPType* out_data = out_mp.data<MPType>();
    auto out_numel = std::accumulate(return_shape.begin(), return_shape.end(), 1.0, std::multiplies<int64_t>());

    switch(index.dtype()) {
      case paddle::DataType::INT32:
      { 
        auto index_info = getTensorInfo<int>(index);
        int* arg_out_data = arg_out.data<int>();
        if (K == 1)
          segment_coo_min_max_cuda_forward_kernel<data_t, MPType, int, true>
            <<<BLOCKS(1, E), THREADS, 0, x.stream()>>>(
              x_data, index_info, reduce2REDUCE.at(reduce), out_data, E, N);

        else if (avg_len <= 8)
          segment_coo_min_max_broadcast_cuda_forward_kernel<data_t, MPType, int, 4>
              <<<dim3((E_1 * ((E_2 + 3) / 4) + 7) / 8, (K + 31) / 32),
                dim3(32, 8), 0, x.stream()>>>(
              x_data, index_info, reduce2REDUCE.at(reduce), out_data, E, K, N);
        else if (avg_len <= 16)
          segment_coo_min_max_broadcast_cuda_forward_kernel<data_t, MPType, int, 8>
              <<<dim3((E_1 * ((E_2 + 7) / 8) + 7) / 8, (K + 31) / 32),
                dim3(32, 8), 0, x.stream()>>>(
              x_data, index_info, reduce2REDUCE.at(reduce), out_data, E, K, N);
        else if (avg_len <= 32)
          segment_coo_min_max_broadcast_cuda_forward_kernel<data_t, MPType, int, 16>
              <<<dim3((E_1 * ((E_2 + 15) / 16) + 7) / 8, (K + 31) / 32),
                dim3(32, 8), 0, x.stream()>>>(
              x_data, index_info, reduce2REDUCE.at(reduce), out_data, E, K, N);
        else
          segment_coo_min_max_broadcast_cuda_forward_kernel<data_t, MPType, int, 32>
              <<<dim3((E_1 * ((E_2 + 31) / 32) + 7) / 8, (K + 31) / 32),
                dim3(32, 8), 0, x.stream()>>>(
              x_data, index_info, reduce2REDUCE.at(reduce), out_data, E, K, N);

        if (K == 1)
        segment_coo_arg_kernel<data_t, MPType, int>
            <<<BLOCKS(1, E), THREADS, 0, x.stream()>>>(
              x_data, index_info, out_data, arg_out_data, E, N);
        else
          segment_coo_arg_broadcast_kernel<data_t, MPType, int>
            <<<BLOCKS(1, E * K), THREADS, 0, x.stream()>>>(
              x_data, index_info, out_data, arg_out_data, E, K, N);
        break;
      }
      case paddle::DataType::INT64:
      {
        auto index_info = getTensorInfo<int64_t>(index);
        int64_t* arg_out_data = arg_out.data<int64_t>();
        if (K == 1)
          segment_coo_min_max_cuda_forward_kernel<data_t, MPType, int64_t, true>
            <<<BLOCKS(1, E), THREADS, 0, x.stream()>>>(
              x_data, index_info, reduce2REDUCE.at(reduce), out_data, E, N);

        else if (avg_len <= 8)
          segment_coo_min_max_broadcast_cuda_forward_kernel<data_t, MPType, int64_t, 4>
              <<<dim3((E_1 * ((E_2 + 3) / 4) + 7) / 8, (K + 31) / 32),
                dim3(32, 8), 0, x.stream()>>>(
              x_data, index_info, reduce2REDUCE.at(reduce), out_data, E, K, N);
        else if (avg_len <= 16)
          segment_coo_min_max_broadcast_cuda_forward_kernel<data_t, MPType, int64_t, 8>
              <<<dim3((E_1 * ((E_2 + 7) / 8) + 7) / 8, (K + 31) / 32),
                dim3(32, 8), 0, x.stream()>>>(
              x_data, index_info, reduce2REDUCE.at(reduce), out_data, E, K, N);
        else if (avg_len <= 32)
          segment_coo_min_max_broadcast_cuda_forward_kernel<data_t, MPType, int64_t, 16>
              <<<dim3((E_1 * ((E_2 + 15) / 16) + 7) / 8, (K + 31) / 32),
                dim3(32, 8), 0, x.stream()>>>(
              x_data, index_info, reduce2REDUCE.at(reduce), out_data, E, K, N);
        else
          segment_coo_min_max_broadcast_cuda_forward_kernel<data_t, MPType, int64_t, 32>
              <<<dim3((E_1 * ((E_2 + 31) / 32) + 7) / 8, (K + 31) / 32),
                dim3(32, 8), 0, x.stream()>>>(
              x_data, index_info, reduce2REDUCE.at(reduce), out_data, E, K, N);

        if (K == 1)
        segment_coo_arg_kernel<data_t, MPType, int64_t>
            <<<BLOCKS(1, E), THREADS, 0, x.stream()>>>(
              x_data, index_info, out_data, arg_out_data, E, N);
        else
          segment_coo_arg_broadcast_kernel<data_t, MPType, int64_t>
            <<<BLOCKS(1, E * K), THREADS, 0, x.stream()>>>(
              x_data, index_info, out_data, arg_out_data, E, K, N);
        break;
      }
      default:
        PD_THROW(
          "function segment_coo_min_max_cuda_forward_kernel is not implemented for the index data type `",
          phi::DataTypeToString(index.dtype()), "`");
    }

   if (x.dtype() == paddle::DataType::FLOAT16 || x.dtype() == paddle::DataType::BFLOAT16) {
      out = paddle::experimental::cast(out_mp, x.dtype());
   }
   
   if (!init) {
    data_t init_val = static_cast<data_t>((reduce == "min") ? std::numeric_limits<MPType>::max() : std::numeric_limits<MPType>::lowest());
    post_process_kernel<data_t>
    <<<BLOCKS(1, out_numel), THREADS, 0, x.stream()>>>(
      init_val, out_numel, out.data<data_t>()
    );
   }

  }));

  return {out, arg_out};
}

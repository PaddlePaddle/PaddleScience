#include "paddle/extension.h"

#include <vector>
#include <limits>
#include <numeric>

#include "atomics.cuh"
#include "index_info.cuh"
#include "utils.cuh"

#define THREADS 256
#define BLOCKS(N) (N + THREADS - 1) / THREADS


enum ReductionType { MIN, MAX };

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"min", MIN},   {"max", MAX}
};

template <typename data_t, typename mp_t, typename index_t>
__global__ void scatter_min_max_cuda_forward_kernel(const data_t* x_data,
                                                    const TensorInfo<index_t> index_info,
                                                    ReductionType reduce_type,
                                                    int numel,
                                                    int E,
                                                    int K,
                                                    int N,
                                                    mp_t* out_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int b = tid / (E * K);
  int k = tid % K;

  if (tid < numel) {
    int64_t idx = index_info.data[IndexToOffset<index_t>::get(tid, index_info)];

    switch(reduce_type) {
      case MIN:
      {
        atomMin(out_data + b * N * K + idx * K + k,
          static_cast<mp_t>(x_data[tid]));
        break;
      }
      case MAX:
      {
        atomMax(out_data + b * N * K + idx * K + k,
          static_cast<mp_t>(x_data[tid]));
        break;
      }
    }

  }
}

template <typename data_t, typename mp_t, typename index_t>
__global__ void scatter_arg_min_max_cuda_forward_kernel(const data_t* x_data,
                                                        const TensorInfo<index_t> index_info,
                                                        int numel,
                                                        int E,
                                                        int K,
                                                        int N,
                                                        mp_t* out_data,
                                                        index_t *arg_out_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int b = tid / (E * K);
  int e = (tid / K) % E;
  int k = tid % K;

  if (tid < numel) {
    int64_t idx = index_info.data[IndexToOffset<index_t>::get(tid, index_info)];

    if (static_cast<mp_t>(x_data[tid]) == out_data[b * N * K + idx * K + k]) {
      arg_out_data[b * N * K + idx * K + k] = e;
    }
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

std::vector<paddle::Tensor> scatter_min_max_cuda_forward(const paddle::Tensor& x,
                                                         const paddle::Tensor& index,
                                                         const paddle::optional<paddle::Tensor>& init,
                                                         const std::vector<int64_t>& return_shape,
                                                         const std::string& reduce,
                                                         int64_t dim) {
  CHECK_CUDA(index);
  if (init)
    CHECK_CUDA(init.get());

  auto x_dims = x.shape();
  auto index_dims = index.shape();
  CHECK_INPUT(x_dims.size() == index_dims.size());
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

  paddle::Tensor arg_out;
  arg_out = paddle::experimental::full_like(out, x_dims[dim], index.dtype(), index.place());

  auto B = 1;
  for (auto i = 0; i < dim; ++i)
    B *= x_dims[i];
  auto E = x_dims[dim];
  auto K = x.numel() / (B * E);
  auto N = return_shape[dim];

  PD_DISPATCH_FLOATING_AND_INTEGRAL_AND_2_TYPES(
    paddle::DataType::FLOAT16, paddle::DataType::BFLOAT16,
    x.dtype(), "scatter_min_max_cuda_forward_kernel", ([&] {

    using MPType = typename MPTypeTrait<data_t>::Type;
    paddle::Tensor out_mp;
    if (x.dtype() == paddle::DataType::FLOAT16 || x.dtype() == paddle::DataType::BFLOAT16) {
      out_mp = paddle::experimental::cast(out, paddle::DataType::FLOAT32);  
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
        scatter_min_max_cuda_forward_kernel<data_t, MPType, int>
        <<<BLOCKS(x.numel()), THREADS, 0, x.stream()>>>(
          x_data, index_info, reduce2REDUCE.at(reduce), x.numel(),
          E, K, N, out_data);
        
       scatter_arg_min_max_cuda_forward_kernel<data_t, MPType, int>
      <<<BLOCKS(x.numel()), THREADS, 0, x.stream()>>>(
         x_data, index_info, x.numel(),
         E, K, N, out_data, arg_out_data);
        break;
      }
      case paddle::DataType::INT64:
      {
        auto index_info = getTensorInfo<int64_t>(index);
        int64_t* arg_out_data = arg_out.data<int64_t>();
        scatter_min_max_cuda_forward_kernel<data_t, MPType, int64_t>
          <<<BLOCKS(x.numel()), THREADS, 0, x.stream()>>>(
            x_data, index_info, reduce2REDUCE.at(reduce), x.numel(),
            E, K, N, out_data);
        
       scatter_arg_min_max_cuda_forward_kernel<data_t, MPType, int64_t>
       <<<BLOCKS(x.numel()), THREADS, 0, x.stream()>>>(
         x_data, index_info, x.numel(),
         E, K, N, out_data, arg_out_data);
        break;
      }
      default:
        PD_THROW(
          "function scatter_min_max_cuda_forward_kernel is not implemented for the index data type `",
          phi::DataTypeToString(index.dtype()), "`");
    }

   if (x.dtype() == paddle::DataType::FLOAT16 || x.dtype() == paddle::DataType::BFLOAT16) {
      out = paddle::experimental::cast(out_mp, x.dtype());
   }

   if (!init) {
    data_t init_val = static_cast<data_t>((reduce == "min") ? std::numeric_limits<MPType>::max() : std::numeric_limits<MPType>::lowest());
    post_process_kernel<data_t>
    <<<BLOCKS(out_numel), THREADS, 0, x.stream()>>>(
      init_val, out_numel, out.data<data_t>()
    );
   }

  }));

  return {out, arg_out};
}

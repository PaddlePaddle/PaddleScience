#include "paddle/extension.h"

#include <vector>
#include <limits>
#include <numeric>

#include "index_info.h"
#include "utils.h"


template <typename data_t, typename index_t>
void scatter_min_max_cpu_forward_kernel(const data_t* x_data,
                                        const index_t* index_data,
                                        const std::vector<int64_t>& return_shape,
                                        const std::vector<int64_t>& x_dims,
                                        const std::string& reduce,
                                        const TensorInfo<index_t>& index_info,
                                        int64_t x_numel,
                                        int64_t dim,
                                        bool post_process,
                                        data_t* out_data,
                                        index_t* arg_out_data) {
  using MPType = typename MPTypeTrait<data_t>::Type;
  auto B = 1;
  for (auto i = 0; i < dim; ++i)
    B *= x_dims[i];
  auto E = x_dims[dim];
  auto K = x_numel / (B * E);
  auto N = return_shape[dim];

  int64_t i, idx, out_data_idx;
  for (auto b = 0; b < B; b++) {
    for (auto e = 0; e < E; e++) {
      for (auto k = 0; k < K; k++) {
        i = b * E * K + e * K + k;
        idx = index_info.data[IndexToOffset<index_t>::get(i, index_info)];
        out_data_idx = b * N * K + idx * K + k;
        if ((reduce == "min" && x_data[i] < out_data[out_data_idx]) || 
            (reduce == "max" && x_data[i] > out_data[out_data_idx])) {
          out_data[out_data_idx] = x_data[i];
          arg_out_data[out_data_idx] = e;
        }
      }
    }
  }

  if (post_process) {
    auto out_numel = std::accumulate(return_shape.begin(), return_shape.end(), 1.0, std::multiplies<int64_t>());
    data_t init_val = static_cast<data_t>((reduce == "min") ? std::numeric_limits<MPType>::max() : std::numeric_limits<MPType>::lowest());
    for (auto i = 0; i < out_numel; ++i) {
      if (out_data[i] == init_val)
        out_data[i] = static_cast<data_t>(0.0);
    }
  }
}

std::vector<paddle::Tensor> scatter_min_max_cpu_forward(const paddle::Tensor& x,
                                                        const paddle::Tensor& index,
                                                        const paddle::optional<paddle::Tensor>& init,
                                                        const std::vector<int64_t>& return_shape,
                                                        const std::string& reduce,
                                                        int64_t dim) {
  CHECK_CPU(index);
  if (init)
    CHECK_CPU(init.get());

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

  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
    x.dtype(), "scatter_min_max_cpu_forward_kernel", ([&] {

    using MPType = typename MPTypeTrait<data_t>::Type;
    if (!init) {
      if (reduce == "min")
        paddle::experimental::fill_(out, static_cast<data_t>(std::numeric_limits<MPType>::max()));
      else
        paddle::experimental::fill_(out, static_cast<data_t>(std::numeric_limits<MPType>::lowest()));
    }

    bool post_process = (!init) ? true : false;
    switch(index.dtype()) {
      case paddle::DataType::INT32:
      { 
        auto index_info = getTensorInfo<int>(index);
        scatter_min_max_cpu_forward_kernel<data_t, int>(
              x.data<data_t>(),
              index.data<int>(),
              return_shape,
              x_dims,
              reduce,
              index_info,
              x.numel(),
              dim,
              post_process,
              out.data<data_t>(),
              arg_out.data<int>());
        break;
      }
      case paddle::DataType::INT64:
      {
        auto index_info = getTensorInfo<int64_t>(index);
        scatter_min_max_cpu_forward_kernel<data_t, int64_t>(
                x.data<data_t>(),
                index.data<int64_t>(),
                return_shape,
                x_dims,
                reduce,
                index_info,
                x.numel(),
                dim,
                post_process,
                out.data<data_t>(),
                arg_out.data<int64_t>());
        break;
      }
      default:
        PD_THROW(
          "function scatter_min_max_cpu_forward_kernel is not implemented for the index data type `",
          phi::DataTypeToString(index.dtype()), "`");
    }
  }));

  return {out, arg_out};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> scatter_min_max_cuda_forward(const paddle::Tensor& x,
                                                         const paddle::Tensor& index,
                                                         const paddle::optional<paddle::Tensor>& init,
                                                         const std::vector<int64_t>& return_shape,
                                                         const std::string& reduce,
                                                         int64_t dim);
#endif

std::vector<paddle::Tensor> ScatterMinMaxForward(const paddle::Tensor& x,
                                                 const paddle::Tensor& index,
                                                 const paddle::optional<paddle::Tensor>& init,
                                                 std::vector<int64_t> return_shape,
                                                 std::string reduce,
                                                 int64_t dim) {
  if (x.is_cpu()) {
    return scatter_min_max_cpu_forward(x, index, init, return_shape, reduce, dim);
#ifdef PADDLE_WITH_CUDA
  } else if (x.is_gpu()) {
    return scatter_min_max_cuda_forward(x, index, init, return_shape, reduce, dim);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of custom scatter_min_max operator.");
  }
}

std::vector<paddle::Tensor> ScatterMinMaxBackward(const paddle::Tensor& x,
                                                  const paddle::Tensor& arg_out,
                                                  const paddle::Tensor& grad_out, 
                                                  int64_t dim) {
  if (!x.is_cpu() && !x.is_gpu() ) {
    PD_THROW("Unsupported device type for backward function of custom scatter_min_max operator.");
  }
  auto x_shape = x.shape();
  x_shape[dim] += 1;
  auto grad_x = paddle::zeros(x_shape, x.dtype(), x.place());
  paddle::experimental::put_along_axis_(grad_x, arg_out, grad_out, dim);
  grad_x = paddle::experimental::slice(grad_x, {dim}, {0}, {x_shape[dim] - 1}, {1}, {});
  return {grad_x};
}

std::vector<std::vector<int64_t>> ScatterMinMaxFWInferShape(const std::vector<int64_t>& x_shape,
                                                            const std::vector<int64_t>& index_shape,
                                                            const paddle::optional<std::vector<int64_t>>& init_shape,
                                                            std::vector<int64_t> return_shape,
                                                            std::string reduce,
                                                            int64_t dim) {
  return {return_shape, return_shape};
}

std::vector<paddle::DataType> ScatterMinMaxFWInferDtype(const paddle::DataType& x_dtype,
                                                        const paddle::DataType& index_dtype,
                                                        const paddle::optional<paddle::DataType>& init_dtype) {
  return {x_dtype, index_dtype};
}

std::vector<std::vector<int64_t>> ScatterMinMaxBWInferShape(const std::vector<int64_t>& x_shape,
                                                            const std::vector<int64_t>& arg_out_shape,
                                                            const std::vector<int64_t>& grad_out_shape,
                                                            int64_t dim) {
  return {x_shape};
}

std::vector<paddle::DataType> ScatterMinMaxBWInferDtype(const paddle::DataType& x_dtype,
                                                        const paddle::DataType& arg_out_dtype,
                                                        const paddle::DataType& grad_out_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_scatter_min_max)
    .Inputs({"X", "Index", paddle::Optional("Init")})
    .Outputs({"Out", "ArgOut"})
    .Attrs({"return_shape: std::vector<int64_t>",
            "reduce: std::string",
            "dim: int64_t"})
    .SetKernelFn(PD_KERNEL(ScatterMinMaxForward))
    .SetInferShapeFn(PD_INFER_SHAPE(ScatterMinMaxFWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ScatterMinMaxFWInferDtype));

PD_BUILD_GRAD_OP(custom_scatter_min_max)
    .Inputs({"X", "ArgOut", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .Attrs({"dim: int64_t"})
    .SetKernelFn(PD_KERNEL(ScatterMinMaxBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(ScatterMinMaxBWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ScatterMinMaxBWInferDtype));

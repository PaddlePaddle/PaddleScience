#include "paddle/extension.h"

#include <vector>
#include <limits>
#include <numeric>

#include "index_info.h"
#include "utils.h"


template <typename data_t, typename index_t>
void segment_coo_cpu_forward_kernel(const data_t* x_data,
                                    const index_t* index_data,
                                    const std::vector<int64_t>& return_shape,
                                    const std::vector<int64_t>& x_dims,
                                    const std::string& reduce,
                                    const TensorInfo<index_t>& index_info,
                                    int64_t x_numel,
                                    int64_t dim,
                                    bool post_process,
                                    data_t* out_data,
                                    index_t* arg_out_data,
                                    data_t* count_data) {
  using MPType = typename MPTypeTrait<data_t>::Type;
  auto B = 1;
  for (auto i = 0; i < dim; ++i)
    B *= x_dims[i];
  auto E = x_dims[dim];
  auto K = x_numel / (B * E);
  auto N = return_shape[dim];

  auto stride = index_info.strides[index_info.dims - 1];
  std::vector<index_t> args(K);
  std::vector<MPType> vals(K);

  int64_t idx, next_idx, row_start;
  for (auto b = 0; b < B; b++) {
    auto offset = IndexToOffset<index_t>::get(b * E, index_info);
    idx = index_info.data[offset];

    for (auto k = 0; k < K; k++)
      vals[k] = static_cast<MPType>(out_data[b * N * K + k]);

    row_start = 0;
    for (auto e = 0; e < E; e++) {
      // update
      for (auto k = 0; k < K; k++) {
        auto cmp = static_cast<MPType>(x_data[b * E * K + e * K + k]);
        if ((reduce == "min" && cmp < vals[k]) || 
            (reduce == "max" && cmp > vals[k])) {
          vals[k] = cmp;
          args[k] = e;
        } else if (reduce == "sum" || reduce == "mean") {
          vals[k] += cmp;
        }
      }
      //write
      if (e == E - 1) {
        for (auto k = 0; k < K; k++) {
          auto idx_k = b * N * K + idx * K + k;
          auto count = E - row_start;
          if (reduce == "sum") {
            out_data[idx_k] = static_cast<data_t>(vals[k]);
          } else if (reduce == "mean") {
            out_data[idx_k] = static_cast<data_t>(vals[k] / static_cast<MPType>(count > 0 ? count : 1));
          } else if (reduce == "min" || reduce == "max") {
            if (count > 0) {
              out_data[idx_k] = static_cast<data_t>(vals[k]);
              arg_out_data[idx_k] = args[k];
            } else {
              out_data[idx_k] = static_cast<data_t>(0);
            }
          }
        }

        if (reduce == "mean")
          count_data[b * N + idx] = static_cast<data_t>(e + 1 - row_start);
      } else {
        next_idx = index_info.data[offset + (e + 1) * stride];
        assert(idx <= next_idx);

        if (idx != next_idx) {
          //write
          for (auto k = 0; k < K; k++) {
            auto idx_k = b * N * K + idx * K + k;
            auto count = e + 1 - row_start;
            if (reduce == "sum") {
              out_data[idx_k] = static_cast<data_t>(vals[k]);
            } else if (reduce == "mean") {
              out_data[idx_k] = static_cast<data_t>(vals[k] / static_cast<MPType>(count > 0 ? count : 1));
            } else if (reduce == "min" || reduce == "max") {
              if (count > 0) {
                out_data[idx_k] = static_cast<data_t>(vals[k]);
                arg_out_data[idx_k] = args[k];
              } else {
                out_data[idx_k] = static_cast<data_t>(0);
              }
            }

            vals[k] = static_cast<MPType>(out_data[b * N * K + next_idx * K + k]);
          }
          if (reduce == "mean")
            count_data[b * N + idx] = static_cast<data_t>(e + 1 - row_start);
          row_start = e + 1;
        }

        idx = next_idx;
      }
    }
  }

  if (post_process) {
    if (reduce == "min" || reduce == "max") {
      auto out_numel = std::accumulate(return_shape.begin(), return_shape.end(), 1.0, std::multiplies<int64_t>());
      data_t init_val = static_cast<data_t>((reduce == "min") ? std::numeric_limits<MPType>::max() : std::numeric_limits<MPType>::lowest());
      for (auto i = 0; i < out_numel; ++i) {
        if (out_data[i] == init_val)
          out_data[i] = static_cast<data_t>(0.0);
      }
    }
    if (reduce == "mean") {
      auto count_data_numel = sizeof(count_data) / sizeof(data_t);
      for (auto i = 0; i < count_data_numel; ++i) {
        if (count_data[i] < static_cast<data_t>(1.0))
          count_data[i] = static_cast<data_t>(1.0);
      }
    }
  }
}

std::vector<paddle::Tensor> segment_coo_cpu_forward(const paddle::Tensor& x,
                                                    const paddle::Tensor& index,
                                                    const paddle::optional<paddle::Tensor>& init,
                                                    const std::vector<int64_t>& return_shape,
                                                    const std::string& reduce) {
  CHECK_CPU(index);
  if (init)
    CHECK_CPU(init.get());

  auto x_dims = x.shape();
  auto index_dims = index.shape();
  CHECK_INPUT(x_dims.size() >= index_dims.size());
  for (auto i = 0; i < index_dims.size() - 1; ++i)
    CHECK_INPUT(x_dims[i] >= index_dims[i]);

  auto dim = index_dims.size() - 1;
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
  if (reduce == "min" || reduce == "max") {
    arg_out = paddle::experimental::full_like(out, x_dims[dim], index.dtype(), index.place());
  } else if (reduce == "mean") {
    auto sizes = index.shape();
    sizes[dim] = return_shape[dim];
    arg_out = paddle::zeros(sizes, out.dtype(), index.place());
  }

  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
    x.dtype(), "segment_coo_cpu_forward_kernel", ([&] {

    using MPType = typename MPTypeTrait<data_t>::Type;
    if (!init) {
      if (reduce == "min")
        paddle::experimental::fill_(out, static_cast<data_t>(std::numeric_limits<MPType>::max()));
      else if (reduce == "max")
        paddle::experimental::fill_(out, static_cast<data_t>(std::numeric_limits<MPType>::lowest()));
      else if (reduce == "sum" || reduce == "mean")
        paddle::experimental::fill_(out, static_cast<data_t>(0));
    }

    bool post_process = (!init) ? true : false;
    switch(index.dtype()) {
      case paddle::DataType::INT32:
      { 
        auto index_info = getTensorInfo<int>(index);
        segment_coo_cpu_forward_kernel<data_t, int>(
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
              (reduce == "min" || reduce == "max") ? arg_out.data<int>() : nullptr,
              (reduce == "mean") ? arg_out.data<data_t>() : nullptr);
        break;
      }
      case paddle::DataType::INT64:
      {
        auto index_info = getTensorInfo<int64_t>(index);
        segment_coo_cpu_forward_kernel<data_t, int64_t>(
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
                (reduce == "min" || reduce == "max") ? arg_out.data<int64_t>() : nullptr,
                (reduce == "mean") ? arg_out.data<data_t>() : nullptr);
        break;
      }
      default:
        PD_THROW(
          "function segment_coo_cpu_forward_kernel is not implemented for the index data type `",
          phi::DataTypeToString(index.dtype()), "`");
    }
  }));

  return {out, arg_out};
}

template <typename data_t, typename index_t>
void gather_coo_cpu_forward_kernel(const data_t* x_data,
                                  const TensorInfo<index_t>& index_info,
                                  int stride,
                                  int B,
                                  int E,
                                  int K,
                                  int N,
                                  data_t* out_data) {
  std::vector<data_t> vals(K);
  int64_t idx, next_idx;
  for (auto b = 0; b < B; b++) {
    auto offset = IndexToOffset<index_t>::get(b * E, index_info);
    idx = index_info.data[offset];

    for (auto k = 0; k < K; k++)
      vals[k] = x_data[b * N * K + idx * K + k];

    for (auto e = 0; e < E; e++) {
      for (auto k = 0; k < K; k++)
        out_data[b * E * K + e * K + k] = vals[k];

      if (e < E - 1) {
        next_idx = index_info.data[offset + (e + 1) * stride];
        CHECK_INPUT(idx <= next_idx);

        if (idx != next_idx) {
          idx = next_idx;
          for (auto k = 0; k < K; k++)
            vals[k] = x_data[b * N * K + idx * K + k];
        }
      }
    }
  }
}

std::vector<paddle::Tensor> gather_coo_cpu_forward(const paddle::Tensor& x,
                                                    const paddle::Tensor& index,
                                                    const paddle::optional<paddle::Tensor>& init,
                                                    std::vector<int64_t> return_shape) {
  CHECK_CPU(index);
  if (init)
    CHECK_CPU(init.get());

  auto x_dims = x.shape();
  auto index_dims = index.shape();
  CHECK_INPUT(x_dims.size() >= index_dims.size());
  for (auto i = 0; i < index_dims.size() - 1; ++i)
    CHECK_INPUT(x_dims[i] == index_dims[i]);

  auto dim = index_dims.size() - 1;
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

  auto B = index.numel() / return_shape[dim];
  auto E = index_dims[dim];
  auto K = out.numel() / index.numel();
  auto N = x_dims[dim];
  
  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
    x.dtype(), "gather_coo_cpu_forward_kernel", ([&] {
    switch(index.dtype()) {
      case paddle::DataType::INT32:
      { 
        auto index_info = getTensorInfo<int>(index);
        auto stride = index_info.strides[index_info.dims - 1];
        gather_coo_cpu_forward_kernel<data_t, int>(
              x.data<data_t>(), index_info, stride, 
              B, E, K, N, out.data<data_t>());
        break;
      }
      case paddle::DataType::INT64:
      {
        auto index_info = getTensorInfo<int64_t>(index);
        auto stride = index_info.strides[index_info.dims - 1];
        gather_coo_cpu_forward_kernel<data_t, int64_t>(
              x.data<data_t>(), index_info, stride, 
              B, E, K, N, out.data<data_t>());
        break;
      }
      default:
        PD_THROW(
          "function gather_coo_cpu_forward_kernel is not implemented for the index data type `",
          phi::DataTypeToString(index.dtype()), "`");
    }
  }));

  return {out};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> segment_coo_cuda_forward(const paddle::Tensor& x,
                                                    const paddle::Tensor& index,
                                                    const paddle::optional<paddle::Tensor>& init,
                                                    std::vector<int64_t> return_shape,
                                                    std::string reduce);
#endif

std::vector<paddle::Tensor> SegmentCooForward(const paddle::Tensor& x,
                                              const paddle::Tensor& index,
                                              const paddle::optional<paddle::Tensor>& init,
                                              std::vector<int64_t> return_shape,
                                              std::string reduce) {
  if (x.is_cpu()) {
    return segment_coo_cpu_forward(x, index, init, return_shape, reduce);
#ifdef PADDLE_WITH_CUDA
  } else if (x.is_gpu()) {
    return segment_coo_cuda_forward(x, index, init, return_shape, reduce);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of custom segment_coo operator.");
  }
}


#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> gather_coo_cuda_forward(const paddle::Tensor& x,
                                                    const paddle::Tensor& index,
                                                    const paddle::optional<paddle::Tensor>& init,
                                                    std::vector<int64_t> return_shape);
#endif

std::vector<paddle::Tensor> GatherCooForward(const paddle::Tensor& x,
                                            const paddle::Tensor& index,
                                            const paddle::optional<paddle::Tensor>& init,
                                            std::vector<int64_t> return_shape) {
  if (x.is_cpu()) {
    return gather_coo_cpu_forward(x, index, init, return_shape);
#ifdef PADDLE_WITH_CUDA
  } else if (x.is_gpu()) {
    return gather_coo_cuda_forward(x, index, init, return_shape);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of custom gather_coo operator.");
  }
}

std::vector<paddle::Tensor> SegmentCooBackward(const paddle::Tensor& x,
                                               const paddle::Tensor& index,
                                               const paddle::optional<paddle::Tensor>& arg_out,
                                               const paddle::Tensor& grad_out,
                                               std::string reduce) {
  if (!x.is_cpu() && !x.is_gpu() ) {
    PD_THROW("Unsupported device type for backward function of custom segment_coo operator.");
  }
  if (reduce == "min" || reduce == "max") {
    int64_t dim = index.shape().size() - 1;
    auto x_shape = x.shape();
    x_shape[dim] += 1;
    auto grad_x = paddle::zeros(x_shape, x.dtype(), x.place());
    paddle::experimental::put_along_axis_(grad_x, arg_out.get(), grad_out, dim);
    grad_x = paddle::experimental::slice(grad_x, {dim}, {0}, {x_shape[dim] - 1}, {1}, {});
    return {grad_x};
  } else if (reduce == "mean") {
    auto grad_x = paddle::empty(x.shape(), x.dtype(), x.place());
    paddle::Tensor count = arg_out.get();
    paddle::Tensor grad_in = GatherCooForward(grad_out, index, paddle::optional<paddle::Tensor>(grad_x), grad_x.shape())[0];
    auto sizes = arg_out.get().shape();
    int64_t dim = index.shape().size() - 1;
    sizes[dim] = index.shape()[dim];
    count = GatherCooForward(count, index, paddle::optional<paddle::Tensor>(paddle::none), sizes)[0];
    for (auto i = 0; i < grad_out.shape().size() - index.shape().size(); i++)
      count = paddle::experimental::unsqueeze(count, {-1});
    paddle::experimental::divide_(grad_in, count);
    return {grad_in};
  } else if (reduce == "sum") {
    auto grad_x = paddle::empty(x.shape(), x.dtype(), x.place());
    paddle::Tensor grad_in = GatherCooForward(grad_out, index, paddle::optional<paddle::Tensor>(grad_x), grad_x.shape())[0];
    return {grad_in};
  }
}

std::vector<paddle::Tensor> GatherCooBackward(const paddle::Tensor& x,
                                              const paddle::Tensor& index,
                                              const paddle::Tensor& grad_out) {
  if (!x.is_cpu() && !x.is_gpu() ) {
    PD_THROW("Unsupported device type for backward function of custom gather_coo operator.");
  }
  auto x_shape = x.shape();
  auto grad_x = paddle::zeros(x_shape, x.dtype(), x.place());
  paddle::Tensor grad_in = SegmentCooForward(grad_out, index, paddle::optional<paddle::Tensor>(grad_x), grad_x.shape(), "sum")[0];
  return {grad_in};
}

std::vector<std::vector<int64_t>> SegmentCooFWInferShape(const std::vector<int64_t>& x_shape,
                                                        const std::vector<int64_t>& index_shape,
                                                        const paddle::optional<std::vector<int64_t>>& init_shape,
                                                        std::vector<int64_t> return_shape,
                                                        std::string reduce) {
  return {return_shape, return_shape};
}

std::vector<paddle::DataType> SegmentCooFWInferDtype(const paddle::DataType& x_dtype,
                                                    const paddle::DataType& index_dtype,
                                                    const paddle::optional<paddle::DataType>& init_dtype) {
  return {x_dtype, index_dtype};
}

std::vector<std::vector<int64_t>> SegmentCooBWInferShape(const std::vector<int64_t>& x_shape,
                                                        const std::vector<int64_t>& index_shape,
                                                        const paddle::optional<std::vector<int64_t>>& arg_out_shape,
                                                        const std::vector<int64_t>& grad_out_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> SegmentCooBWInferDtype(const paddle::DataType& x_dtype,
                                                    const paddle::DataType& index_dtype,
                                                    const paddle::optional<paddle::DataType>& arg_out_dtype,
                                                    const paddle::DataType& grad_out_dtype) {
  return {x_dtype};
}

std::vector<std::vector<int64_t>> GatherCooFWInferShape(const std::vector<int64_t>& x_shape,
                                                        const std::vector<int64_t>& index_shape,
                                                        const paddle::optional<std::vector<int64_t>>& init_shape,
                                                        std::vector<int64_t> return_shape) {
  return {return_shape};
}

std::vector<paddle::DataType> GatherCooFWInferDtype(const paddle::DataType& x_dtype,
                                                    const paddle::DataType& index_dtype,
                                                    const paddle::optional<paddle::DataType>& init_dtype) {
  return {x_dtype};
}

std::vector<std::vector<int64_t>> GatherCooBWInferShape(const std::vector<int64_t>& x_shape,
                                                        const std::vector<int64_t>& index_shape,
                                                        const std::vector<int64_t>& grad_out_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> GatherCooBWInferDtype(const paddle::DataType& x_dtype,
                                                    const paddle::DataType& index_dtype,
                                                    const paddle::DataType& grad_out_dtype) {
  return {x_dtype};
}


PD_BUILD_OP(custom_segment_coo)
    .Inputs({"X", "Index", paddle::Optional("Init")})
    .Outputs({"Out", paddle::Optional("ArgOut")})
    .Attrs({"return_shape: std::vector<int64_t>",
            "reduce: std::string"})
    .SetKernelFn(PD_KERNEL(SegmentCooForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentCooFWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentCooFWInferDtype));

PD_BUILD_GRAD_OP(custom_segment_coo)
    .Inputs({"X", "Index", paddle::Optional("ArgOut"), paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .Attrs({"reduce: std::string"})
    .SetKernelFn(PD_KERNEL(SegmentCooBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentCooBWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentCooBWInferDtype));

PD_BUILD_OP(custom_gather_coo)
    .Inputs({"X", "Index", paddle::Optional("Init")})
    .Outputs({"Out"})
    .Attrs({"return_shape: std::vector<int64_t>"})
    .SetKernelFn(PD_KERNEL(GatherCooForward))
    .SetInferShapeFn(PD_INFER_SHAPE(GatherCooFWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GatherCooFWInferDtype));

PD_BUILD_GRAD_OP(custom_gather_coo)
    .Inputs({"X", "Index", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(GatherCooBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(GatherCooBWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GatherCooBWInferDtype));

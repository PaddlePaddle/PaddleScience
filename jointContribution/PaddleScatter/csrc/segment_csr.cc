#include "paddle/extension.h"

#include <vector>
#include <limits>
#include <numeric>

#include "index_info.h"
#include "utils.h"


template <typename data_t, typename index_t>
void segment_csr_cpu_forward_kernel(const data_t* x_data,
                                    const std::vector<int64_t>& indptr_shape,
                                    const TensorInfo<index_t>& indptr_info,
                                    const std::string& reduce,
                                    int stride,
                                    int64_t dim,
                                    int N,
                                    int K,
                                    int E,
                                    data_t* out_data,
                                    index_t* arg_out_data) {
  using MPType = typename MPTypeTrait<data_t>::Type;
  std::vector<index_t> args(K);
  std::vector<MPType> vals(K);
  index_t row_start, row_end;
  for (auto n = 0; n < N; n++) {
    auto offset = IndexPtrToOffset<index_t>::get(n, indptr_info);
    row_start = indptr_info.data[offset];
    row_end = indptr_info.data[offset + stride];

    offset = (n / (indptr_shape[dim] - 1)) * E * K;
    for (auto k = 0; k < K; k++) {
      // init
      if (reduce == "min")
        vals[k] = static_cast<data_t>(std::numeric_limits<MPType>::max());
      else if (reduce == "max")
        vals[k] = static_cast<data_t>(std::numeric_limits<MPType>::lowest());
      else if (reduce == "sum" || reduce == "mean")
        vals[k] = static_cast<data_t>(0);
    }

    for (auto e = row_start; e < row_end; e++) {
      for (auto k = 0; k < K; k++) {
        // update
        auto cmp = static_cast<MPType>(x_data[offset + e * K + k]);
        if ((reduce == "min" && cmp < vals[k]) || 
            (reduce == "max" && cmp > vals[k])) {
          vals[k] = cmp;
          args[k] = e;
        } else if (reduce == "sum" || reduce == "mean") {
          vals[k] += cmp;
        }
      }
    }

    for (auto k = 0; k < K; k++) {
      // write
      auto idx = n * K + k;
      auto count = row_end - row_start;
      if (reduce == "sum") {
        out_data[idx] = static_cast<data_t>(vals[k]);
      } else if (reduce == "mean") {
        out_data[idx] = static_cast<data_t>(vals[k] / static_cast<MPType>(count > 0 ? count : 1));
      } else if (reduce == "min" || reduce == "max") {
        if (count > 0) {
          out_data[idx] = static_cast<data_t>(vals[k]);
          arg_out_data[idx] = args[k];
        } else {
          out_data[idx] = static_cast<data_t>(0);
        }
      }
    }
  }
}

std::vector<paddle::Tensor> segment_csr_cpu_forward(const paddle::Tensor& x,
                                                    const paddle::Tensor& indptr,
                                                    const paddle::optional<paddle::Tensor>& init,
                                                    const std::vector<int64_t>& return_shape,
                                                    const std::string& reduce) {
  CHECK_CPU(indptr);
  if (init)
    CHECK_CPU(init.get());

  auto x_dims = x.shape();
  auto indptr_dims = indptr.shape();
  CHECK_INPUT(x_dims.size() >= indptr_dims.size());
  auto dim = indptr_dims.size() - 1;

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
    arg_out = paddle::experimental::full_like(out, x_dims[dim], indptr.dtype(), indptr.place());
  }

  auto N = return_shape[dim] * (indptr.numel() / indptr_dims[dim]);
  auto K = out.numel() / N;
  auto E = x_dims[dim];

  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
    x.dtype(), "segment_csr_cpu_forward_kernel", ([&] {

    switch(indptr.dtype()) {
      case paddle::DataType::INT32:
      {
        auto indptr_info = getTensorInfo<int>(indptr);
        int stride = indptr_info.strides[indptr_info.dims - 1];
        segment_csr_cpu_forward_kernel<data_t, int>(
          x.data<data_t>(), indptr_dims, indptr_info, reduce,
          stride, dim, N, K, E, out.data<data_t>(), 
          (reduce == "min" || reduce == "max") ? arg_out.data<int>() : nullptr);
        break;
      }
      case paddle::DataType::INT64:
      {
        auto indptr_info = getTensorInfo<int64_t>(indptr);
        int stride = indptr_info.strides[indptr_info.dims - 1];
        segment_csr_cpu_forward_kernel<data_t, int64_t>(
          x.data<data_t>(), indptr_dims, indptr_info, reduce,
          stride, dim, N, K, E, out.data<data_t>(),
          (reduce == "min" || reduce == "max") ? arg_out.data<int64_t>() : nullptr);
        break;
      }
      default:
        PD_THROW(
          "function segment_csr_cpu_forward_kernel is not implemented for the indptr data type `",
          phi::DataTypeToString(indptr.dtype()), "`");
    }
  }));

  return {out, arg_out};
}

template <typename data_t, typename index_t>
void gather_csr_cpu_forward_kernel(const data_t* x_data,
                                  const TensorInfo<index_t>& indptr_info,
                                  const std::vector<int64_t>& indptr_shape,
                                  int64_t dim,
                                  int stride,
                                  int N,
                                  int K,
                                  int E,
                                  data_t* out_data) {
  std::vector<data_t> vals(K);
  int64_t row_start, row_end;
  for (auto n = 0; n < N; n++) {
    auto offset = IndexPtrToOffset<index_t>::get(n, indptr_info);
    row_start = indptr_info.data[offset];
    row_end = indptr_info.data[offset + stride];

    for (auto k = 0; k < K; k++)
      vals[k] = x_data[n * K + k];

    offset = (n / (indptr_shape[dim] - 1)) * E * K;
    for (auto e = row_start; e < row_end; e++)
      for (auto k = 0; k < K; k++)
        out_data[offset + e * K + k] = vals[k];
  }
}

std::vector<paddle::Tensor> gather_csr_cpu_forward(const paddle::Tensor& x,
                                                  const paddle::Tensor& indptr,
                                                  const paddle::optional<paddle::Tensor>& init,
                                                  const std::vector<int64_t>& return_shape) {
  CHECK_CPU(indptr);
  if (init)
    CHECK_CPU(init.get());

  auto x_dims = x.shape();
  auto indptr_dims = indptr.shape();
  CHECK_INPUT(x_dims.size() >= indptr_dims.size());
  auto dim = indptr_dims.size() - 1;

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

  auto N = x_dims[dim] * (indptr.numel() / indptr_dims[dim]);
  auto K = x.numel() / N;
  auto E = return_shape[dim];

  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
    x.dtype(), "gather_csr_cpu_forward_kernel", ([&] {

    switch(indptr.dtype()) {
      case paddle::DataType::INT32:
      {
        auto indptr_info = getTensorInfo<int>(indptr);
        int stride = indptr_info.strides[indptr_info.dims - 1];
        gather_csr_cpu_forward_kernel<data_t, int>(
          x.data<data_t>(), indptr_info, indptr_dims, dim,
          stride, N, K, E, out.data<data_t>());
        break;
      }
      case paddle::DataType::INT64:
      {
        auto indptr_info = getTensorInfo<int64_t>(indptr);
        int stride = indptr_info.strides[indptr_info.dims - 1];
        gather_csr_cpu_forward_kernel<data_t, int64_t>(
          x.data<data_t>(), indptr_info, indptr_dims, dim,
          stride, N, K, E, out.data<data_t>());
        break;
      }
      default:
        PD_THROW(
          "function gather_csr_cpu_forward_kernel is not implemented for the indptr data type `",
          phi::DataTypeToString(indptr.dtype()), "`");
    }
  }));

  return {out};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> segment_csr_cuda_forward(const paddle::Tensor& x,
                                                    const paddle::Tensor& indptr,
                                                    const paddle::optional<paddle::Tensor>& init,
                                                    const std::vector<int64_t>& return_shape,
                                                    const std::string& reduce);
#endif

std::vector<paddle::Tensor> SegmentCsrForward(const paddle::Tensor& x,
                                              const paddle::Tensor& indptr,
                                              const paddle::optional<paddle::Tensor>& init,
                                              const std::vector<int64_t>& return_shape,
                                              const std::string& reduce) {
  if (x.is_cpu()) {
    return segment_csr_cpu_forward(x, indptr, init, return_shape, reduce);
#ifdef PADDLE_WITH_CUDA
  } else if (x.is_gpu()) {
    return segment_csr_cuda_forward(x, indptr, init, return_shape, reduce);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of custom segment_csr operator.");
  }
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> gather_csr_cuda_forward(const paddle::Tensor& x,
                                                    const paddle::Tensor& indptr,
                                                    const paddle::optional<paddle::Tensor>& init,
                                                    const std::vector<int64_t>& return_shape);
#endif

std::vector<paddle::Tensor> GatherCsrForward(const paddle::Tensor& x,
                                            const paddle::Tensor& indptr,
                                            const paddle::optional<paddle::Tensor>& init,
                                            const std::vector<int64_t>& return_shape) {
  if (x.is_cpu()) {
    return gather_csr_cpu_forward(x, indptr, init, return_shape);
#ifdef PADDLE_WITH_CUDA
  } else if (x.is_gpu()) {
    return gather_csr_cuda_forward(x, indptr, init, return_shape);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of custom gather_csr operator.");
  }
}

std::vector<paddle::Tensor> SegmentCsrBackward(const paddle::Tensor& x,
                                              const paddle::Tensor& indptr,
                                              const paddle::optional<paddle::Tensor>& arg_out,
                                              const paddle::Tensor& grad_out,
                                              std::string reduce) {
  if (!x.is_cpu() && !x.is_gpu() ) {
    PD_THROW("Unsupported device type for backward function of custom segment_csr operator.");
  }
  if (reduce == "min" || reduce == "max") {
    int64_t dim = indptr.shape().size() - 1;
    auto x_shape = x.shape();
    x_shape[dim] += 1;
    auto grad_x = paddle::zeros(x_shape, x.dtype(), x.place());
    paddle::experimental::put_along_axis_(grad_x, arg_out.get(), grad_out, dim);
    grad_x = paddle::experimental::slice(grad_x, {dim}, {0}, {x_shape[dim] - 1}, {1}, {});
    return {grad_x};
  } else if (reduce == "mean") {
    auto grad_x = paddle::empty(x.shape(), x.dtype(), x.place());
    int64_t dim = indptr.shape().size() - 1;
    if (grad_x.numel() > 0) {
      grad_x = GatherCsrForward(grad_out, indptr, paddle::optional<paddle::Tensor>(grad_x), grad_x.shape())[0];
      auto indptr1 = paddle::experimental::slice(indptr, {dim}, {0}, {indptr.shape()[dim] - 1}, {1}, {});
      auto indptr2 = paddle::experimental::slice(indptr, {dim}, {1}, {indptr.shape()[dim]}, {1}, {});
      auto count = paddle::experimental::cast(indptr2 - indptr1, grad_x.dtype());
      auto sizes = count.shape();
      sizes[dim] = grad_x.shape()[dim];
      // sizes[dim] = *indptr.flatten()[-1].data<int64_t>();
      count = GatherCsrForward(count, indptr, paddle::optional<paddle::Tensor>(paddle::none), sizes)[0];
      for (auto i = 0; i < grad_out.shape().size() - indptr.shape().size(); i++)
        paddle::experimental::unsqueeze_(count, {-1});
      paddle::experimental::divide_(grad_x, count);
    }
    return {grad_x};
  } else if (reduce == "sum") {
    auto grad_x = paddle::empty(x.shape(), x.dtype(), x.place());
    paddle::Tensor grad_in = GatherCsrForward(grad_out, indptr, paddle::optional<paddle::Tensor>(grad_x), grad_x.shape())[0];
    return {grad_in};
  } 
}

std::vector<paddle::Tensor> GatherCsrBackward(const paddle::Tensor& x,
                                              const paddle::Tensor& indptr,
                                              const paddle::Tensor& grad_out) {
  if (!x.is_cpu() && !x.is_gpu() ) {
    PD_THROW("Unsupported device type for backward function of custom gather_csr operator.");
  }
  auto x_shape = x.shape();
  auto grad_x = paddle::empty(x_shape, x.dtype(), x.place());
  paddle::Tensor grad_in = SegmentCsrForward(grad_out, indptr, paddle::optional<paddle::Tensor>(grad_x), grad_x.shape(), "sum")[0];
  return {grad_in};
}

std::vector<std::vector<int64_t>> SegmentCsrFWInferShape(const std::vector<int64_t>& x_shape,
                                                        const std::vector<int64_t>& indptr_shape,
                                                        const paddle::optional<std::vector<int64_t>>& init_shape,
                                                        std::vector<int64_t> return_shape,
                                                        std::string reduce) {
  return {return_shape, return_shape};
}

std::vector<paddle::DataType> SegmentCsrFWInferDtype(const paddle::DataType& x_dtype,
                                                     const paddle::DataType& indptr_dtype,
                                                     const paddle::optional<paddle::DataType>& init_dtype) {
  return {x_dtype, indptr_dtype};
}

std::vector<std::vector<int64_t>> SegmentCsrBWInferShape(const std::vector<int64_t>& x_shape,
                                                        const std::vector<int64_t>& indptr_shape,
                                                        const paddle::optional<std::vector<int64_t>>& arg_out_shape,
                                                        const std::vector<int64_t>& grad_out_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> SegmentCsrBWInferDtype(const paddle::DataType& x_dtype,
                                                    const paddle::DataType& indptr_dtype,
                                                    const paddle::optional<paddle::DataType>& arg_out_dtype,
                                                    const paddle::DataType& grad_out_dtype) {
  return {x_dtype};
}

std::vector<std::vector<int64_t>> GatherCsrFWInferShape(const std::vector<int64_t>& x_shape,
                                                        const std::vector<int64_t>& indptr_shape,
                                                        const paddle::optional<std::vector<int64_t>>& init_shape,
                                                        std::vector<int64_t> return_shape) {
  return {return_shape};
}

std::vector<paddle::DataType> GatherCsrFWInferDtype(const paddle::DataType& x_dtype,
                                                    const paddle::DataType& indptr_dtype,
                                                    const paddle::optional<paddle::DataType>& init_dtype) {
  return {x_dtype};
}

std::vector<std::vector<int64_t>> GatherCsrBWInferShape(const std::vector<int64_t>& x_shape,
                                                        const std::vector<int64_t>& indptr_shape,
                                                        const std::vector<int64_t>& grad_out_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> GatherCsrBWInferDtype(const paddle::DataType& x_dtype,
                                                    const paddle::DataType& indptr_dtype,
                                                    const paddle::DataType& grad_out_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_segment_csr)
    .Inputs({"X", "Indptr", paddle::Optional("Init")})
    .Outputs({"Out", paddle::Optional("ArgOut")})
    .Attrs({"return_shape: std::vector<int64_t>",
            "reduce: std::string"})
    .SetKernelFn(PD_KERNEL(SegmentCsrForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentCsrFWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentCsrFWInferDtype));

PD_BUILD_GRAD_OP(custom_segment_csr)
    .Inputs({"X", "Indptr", paddle::Optional("ArgOut"), paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .Attrs({"reduce: std::string"})
    .SetKernelFn(PD_KERNEL(SegmentCsrBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentCsrBWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentCsrBWInferDtype));

PD_BUILD_OP(custom_gather_csr)
    .Inputs({"X", "Indptr", paddle::Optional("Init")})
    .Outputs({"Out"})
    .Attrs({"return_shape: std::vector<int64_t>"})
    .SetKernelFn(PD_KERNEL(GatherCsrForward))
    .SetInferShapeFn(PD_INFER_SHAPE(GatherCsrFWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GatherCsrFWInferDtype));

PD_BUILD_GRAD_OP(custom_gather_csr)
    .Inputs({"X", "Indptr", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(GatherCsrBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(GatherCsrBWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GatherCsrBWInferDtype));
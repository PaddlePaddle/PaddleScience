#include "paddle/extension.h"

#include <vector>
#include <limits>
#include <numeric>

#include "index_info.h"
#include "utils.h"


template <typename data_t, typename index_t>
void segment_csr_min_max_cpu_forward_kernel(const data_t* x_data,
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
      if (reduce == "min")
        vals[k] = static_cast<data_t>(std::numeric_limits<MPType>::max());
      else
        vals[k] = static_cast<data_t>(std::numeric_limits<MPType>::lowest());
    }

    for (auto e = row_start; e < row_end; e++) {
      for (auto k = 0; k < K; k++) {
        // update
        auto cmp = static_cast<MPType>(x_data[offset + e * K + k]);
        if ((reduce == "min" && cmp < vals[k]) || 
            (reduce == "max" && cmp > vals[k])) {
          vals[k] = cmp;
          args[k] = e;
        }
      }
    }

    for (auto k = 0; k < K; k++) {
      // write
      auto idx = n * K + k;
      if (row_end - row_start > 0) {
        out_data[idx] = static_cast<data_t>(vals[k]);
        arg_out_data[idx] = args[k];
      } else {
        out_data[idx] = static_cast<data_t>(0);
      }
    }
  }
}

std::vector<paddle::Tensor> segment_csr_min_max_cpu_forward(const paddle::Tensor& x,
                                                            const paddle::Tensor& indptr,
                                                            const std::vector<int64_t>& return_shape,
                                                            const std::string& reduce) {
  CHECK_CPU(indptr);

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

  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
    x.dtype(), "segment_csr_min_max_cpu_forward_kernel", ([&] {

    switch(indptr.dtype()) {
      case paddle::DataType::INT32:
      {
        auto indptr_info = getTensorInfo<int>(indptr);
        int stride = indptr_info.strides[indptr_info.dims - 1];
        segment_csr_min_max_cpu_forward_kernel<data_t, int>(
          x.data<data_t>(), indptr_dims, indptr_info, reduce,
          stride, dim, N, K, E, out.data<data_t>(), arg_out.data<int>());
        break;
      }
      case paddle::DataType::INT64:
      {
        auto indptr_info = getTensorInfo<int64_t>(indptr);
        int stride = indptr_info.strides[indptr_info.dims - 1];
        segment_csr_min_max_cpu_forward_kernel<data_t, int64_t>(
          x.data<data_t>(), indptr_dims, indptr_info, reduce,
          stride, dim, N, K, E, out.data<data_t>(), arg_out.data<int64_t>());
        break;
      }
      default:
        PD_THROW(
          "function segment_csr_min_max_cpu_forward_kernel is not implemented for the indptr data type `",
          phi::DataTypeToString(indptr.dtype()), "`");
    }
  }));

  return {out, arg_out};
}


std::vector<paddle::Tensor> segment_csr_min_max_cuda_forward(const paddle::Tensor& x,
                                                             const paddle::Tensor& indptr,
                                                             const std::vector<int64_t>& return_shape,
                                                             const std::string& reduce);

std::vector<paddle::Tensor> SegmentCsrMinMaxForward(const paddle::Tensor& x,
                                                    const paddle::Tensor& indptr,
                                                    const std::vector<int64_t>& return_shape,
                                                    const std::string& reduce) {
  if (x.is_cpu()) {
    return segment_csr_min_max_cpu_forward(x, indptr, return_shape, reduce);
  } else if (x.is_gpu()) {
    return segment_csr_min_max_cuda_forward(x, indptr, return_shape, reduce);
  } else {
    PD_THROW("Unsupported device type for forward function of custom segment_csr_min_max operator.");
  }
}

std::vector<paddle::Tensor> SegmentCsrMinMaxBackward(const paddle::Tensor& x,
                                                     const paddle::Tensor& indptr,
                                                     const paddle::Tensor& arg_out,
                                                     const paddle::Tensor& grad_out) {
  if (!x.is_cpu() && !x.is_gpu() ) {
    PD_THROW("Unsupported device type for backward function of custom segment_csr_min_max operator.");
  }
  int64_t dim = indptr.shape().size() - 1;
  auto x_shape = x.shape();
  x_shape[dim] += 1;
  auto grad_x = paddle::zeros(x_shape, x.dtype(), x.place());
  paddle::experimental::put_along_axis_(grad_x, arg_out, grad_out, dim);
  grad_x = paddle::experimental::slice(grad_x, {dim}, {0}, {x_shape[dim] - 1}, {1}, {});
  return {grad_x};
}

std::vector<std::vector<int64_t>> SegmentCsrMinMaxFWInferShape(const std::vector<int64_t>& x_shape,
                                                               const std::vector<int64_t>& indptr_shape,
                                                               std::vector<int64_t> return_shape,
                                                               std::string reduce) {
  return {return_shape, return_shape};
}

std::vector<paddle::DataType> SegmentCsrMinMaxFWInferDtype(paddle::DataType x_dtype,
                                                           paddle::DataType indptr_dtype) {
  return {x_dtype, indptr_dtype};
}

std::vector<std::vector<int64_t>> SegmentCsrMinMaxBWInferShape(const std::vector<int64_t>& x_shape,
                                                               const std::vector<int64_t>& indptr_shape,
                                                               const std::vector<int64_t>& arg_out_shape,
                                                               const std::vector<int64_t>& grad_out_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> SegmentCsrMinMaxBWInferDtype(paddle::DataType x_dtype,
                                                           paddle::DataType indptr_dtype,
                                                           paddle::DataType arg_out_dtype,
                                                           paddle::DataType grad_out_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_segment_csr_min_max)
    .Inputs({"X", "Indptr"})
    .Outputs({"Out", "ArgOut"})
    .Attrs({"return_shape: std::vector<int64_t>",
            "reduce: std::string"})
    .SetKernelFn(PD_KERNEL(SegmentCsrMinMaxForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentCsrMinMaxFWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentCsrMinMaxFWInferDtype));

PD_BUILD_GRAD_OP(custom_segment_csr_min_max)
    .Inputs({"X", "Indptr", "ArgOut", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(SegmentCsrMinMaxBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentCsrMinMaxBWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentCsrMinMaxBWInferDtype));

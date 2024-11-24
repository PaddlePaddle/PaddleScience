#include "paddle/extension.h"

#include <vector>
#include <limits>
#include <numeric>

#include "index_info.h"
#include "utils.h"


template <typename data_t, typename index_t>
void segment_coo_min_max_cpu_forward_kernel(const data_t* x_data,
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
        }
      }
      //write
      if (e == E - 1) {
        for (auto k = 0; k < K; k++) {
          auto idx_k = b * N * K + idx * K + k;
          if (E - row_start > 0) {
            out_data[idx_k] = static_cast<data_t>(vals[k]);
            arg_out_data[idx_k] = args[k];
          } else {
            out_data[idx_k] = static_cast<data_t>(0);
          }
        }
      } else {
        next_idx = index_info.data[offset + (e + 1) * stride];
        assert(idx <= next_idx);

        if (idx != next_idx) {
          //write
          for (auto k = 0; k < K; k++) {
            auto idx_k = b * N * K + idx * K + k;
            if (e + 1 - row_start > 0) {
              out_data[idx_k] = static_cast<data_t>(vals[k]);
              arg_out_data[idx_k] = args[k];
            } else {
              out_data[idx_k] = static_cast<data_t>(0);
            }
            vals[k] = static_cast<MPType>(out_data[b * N * K + next_idx * K + k]);
          }
          row_start = e + 1;
        }

        idx = next_idx;
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

std::vector<paddle::Tensor> segment_coo_min_max_cpu_forward(const paddle::Tensor& x,
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
  arg_out = paddle::experimental::full_like(out, x_dims[dim], index.dtype(), index.place());

  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
    x.dtype(), "segment_coo_min_max_cpu_forward_kernel", ([&] {

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
        segment_coo_min_max_cpu_forward_kernel<data_t, int>(
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
        segment_coo_min_max_cpu_forward_kernel<data_t, int64_t>(
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
          "function segment_coo_min_max_cpu_forward_kernel is not implemented for the index data type `",
          phi::DataTypeToString(index.dtype()), "`");
    }
  }));

  return {out, arg_out};
}


std::vector<paddle::Tensor> segment_coo_min_max_cuda_forward(const paddle::Tensor& x,
                                                             const paddle::Tensor& index,
                                                             const paddle::optional<paddle::Tensor>& init,
                                                             std::vector<int64_t> return_shape,
                                                             std::string reduce);

std::vector<paddle::Tensor> SegmentCooMinMaxForward(const paddle::Tensor& x,
                                                    const paddle::Tensor& index,
                                                    const paddle::optional<paddle::Tensor>& init,
                                                    std::vector<int64_t> return_shape,
                                                    std::string reduce) {
  if (x.is_cpu()) {
    return segment_coo_min_max_cpu_forward(x, index, init, return_shape, reduce);
  } else if (x.is_gpu()) {
    return segment_coo_min_max_cuda_forward(x, index, init, return_shape, reduce);
  } else {
    PD_THROW("Unsupported device type for forward function of custom segment_coo_min_max operator.");
  }
}

std::vector<paddle::Tensor> SegmentCooMinMaxBackward(const paddle::Tensor& x,
                                                     const paddle::Tensor& index,
                                                     const paddle::Tensor& arg_out,
                                                     const paddle::Tensor& grad_out) {
  if (!x.is_cpu() && !x.is_gpu() ) {
    PD_THROW("Unsupported device type for backward function of custom segment_coo_min_max operator.");
  }
  int64_t dim = index.shape().size() - 1;
  auto x_shape = x.shape();
  x_shape[dim] += 1;
  auto grad_x = paddle::zeros(x_shape, x.dtype(), x.place());
  paddle::experimental::put_along_axis_(grad_x, arg_out, grad_out, dim);
  grad_x = paddle::experimental::slice(grad_x, {dim}, {0}, {x_shape[dim] - 1}, {1}, {});
  return {grad_x};
}

std::vector<std::vector<int64_t>> SegmentCooMinMaxFWInferShape(const std::vector<int64_t>& x_shape,
                                                               const std::vector<int64_t>& index_shape,
                                                               const paddle::optional<std::vector<int64_t>>& init_shape,
                                                               std::vector<int64_t> return_shape,
                                                               std::string reduce) {
  return {return_shape, return_shape};
}

std::vector<paddle::DataType> SegmentCooMinMaxFWInferDtype(const paddle::DataType& x_dtype,
                                                           const paddle::DataType& index_dtype,
                                                           const paddle::optional<paddle::DataType>& init_dtype) {
  return {x_dtype, index_dtype};
}

std::vector<std::vector<int64_t>> SegmentCooMinMaxBWInferShape(const std::vector<int64_t>& x_shape,
                                                               const std::vector<int64_t>& index_shape,
                                                               const std::vector<int64_t>& arg_out_shape,
                                                               const std::vector<int64_t>& grad_out_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> SegmentCooMinMaxBWInferDtype(const paddle::DataType& x_dtype,
                                                           const paddle::DataType& index_dtype,
                                                           const paddle::DataType& arg_out_dtype,
                                                           const paddle::DataType& grad_out_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_segment_coo_min_max)
    .Inputs({"X", "Index", paddle::Optional("Init")})
    .Outputs({"Out", "ArgOut"})
    .Attrs({"return_shape: std::vector<int64_t>",
            "reduce: std::string"})
    .SetKernelFn(PD_KERNEL(SegmentCooMinMaxForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentCooMinMaxFWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentCooMinMaxFWInferDtype));

PD_BUILD_GRAD_OP(custom_segment_coo_min_max)
    .Inputs({"X", "Index", "ArgOut", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(SegmentCooMinMaxBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentCooMinMaxBWInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentCooMinMaxBWInferDtype));

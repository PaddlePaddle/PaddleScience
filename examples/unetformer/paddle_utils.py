import paddle


def _Tensor_view(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        return paddle.reshape(self, shape=list(kwargs.values())[0])


setattr(paddle.Tensor, "view", _Tensor_view)


def _Tensor_reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])


setattr(paddle.Tensor, "reshape", _Tensor_reshape)


def device2int(device):
    if isinstance(device, paddle.fluid.libpaddle.Place):
        if device.is_gpu_place():
            return device.gpu_device_id()
        else:
            return 0
    elif isinstance(device, str):
        device = device.replace("cuda", "gpu")
        device = device.replace("gpu:", "")
        try:
            return int(device)
        except ValueError:
            return 0
    else:
        return 0


def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


class PaddleFlag:
    cudnn_enabled = True
    cudnn_benchmark = False
    matmul_allow_tf32 = False
    cudnn_allow_tf32 = True
    cudnn_deterministic = False


def add_tensor_methods():
    def _Tensor_view(self, *args, **kwargs):
        if args:
            if len(args) == 1 and isinstance(args[0], (tuple, list)):
                return paddle.reshape(self, args[0])
            else:
                return paddle.reshape(self, list(args))
        elif kwargs:
            return paddle.reshape(self, shape=list(kwargs.values())[0])

    setattr(paddle.Tensor, "view", _Tensor_view)

    def _Tensor_reshape(self, *args, **kwargs):
        if args:
            if len(args) == 1 and isinstance(args[0], (tuple, list)):
                return paddle.reshape(self, args[0])
            else:
                return paddle.reshape(self, list(args))
        elif kwargs:
            assert "shape" in kwargs
            return paddle.reshape(self, shape=kwargs["shape"])

    setattr(paddle.Tensor, "reshape", _Tensor_reshape)

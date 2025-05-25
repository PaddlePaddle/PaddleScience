import warnings

import paddle

try:
    import paddle_lightning as pl
    LightningModule = pl.LightningModule
    Callback = pl.Callback
except ImportError:
    pl = object
    LightningModule = paddle.nn.Layer
    Callback = object

    warnings.warn("Please install 'paddle_lightning' via "
                  "'pip install paddle_lightning' in order to use GraphGym")

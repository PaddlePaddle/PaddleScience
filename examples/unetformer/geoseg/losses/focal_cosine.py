import paddle
from paddle_utils import add_tensor_methods

__all__ = ["FocalCosineLoss"]

add_tensor_methods()


class FocalCosineLoss(paddle.nn.Layer):
    """
    Implementation Focal cosine loss from the "Data-Efficient Deep Learning Method for Image Classification
    Using Data Augmentation, Focal Cosine Loss, and Ensemble" (https://arxiv.org/abs/2007.07805).

    Credit: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
    """

    def __init__(
        self, alpha: float = 1, gamma: float = 2, xent: float = 0.1, reduction="mean"
    ):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, input: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        cosine_loss = paddle.nn.functional.cosine_embedding_loss(
            input1=input,
            input2=paddle.nn.functional.one_hot(
                num_classes=input.shape[-1], x=target
            ).astype("int64"),
            label=paddle.to_tensor(data=[1], place=target.place),
            reduction=self.reduction,
        )
        cent_loss = paddle.nn.functional.cross_entropy(
            input=paddle.nn.functional.normalize(x=input),
            label=target,
            reduction="none",
        )
        pt = paddle.exp(x=-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss
        if self.reduction == "mean":
            focal_loss = paddle.mean(x=focal_loss)
        return cosine_loss + self.xent * focal_loss

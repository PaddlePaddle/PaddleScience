import paddle
import utils.paddle_aux as paddle_aux


class Normalizer(paddle.nn.Layer):
    def __init__(self, size, max_accumulations=10**7, epsilon=1e-08, device=None):
        """
        Online normalization module

        size: feature dimension
        max_accumulation: maximum number of batches
        epsilon: std cutoff for constant variable
        device: device
        """
        super(Normalizer, self).__init__()
        self.max_accumulations = max_accumulations
        self.epsilon = epsilon
        self.register_buffer(
            name="acc_count",
            tensor=paddle.to_tensor(
                data=1.0, dtype=paddle.get_default_dtype(), place=device
            ),
        )
        self.register_buffer(
            name="num_accumulations",
            tensor=paddle.to_tensor(
                data=1.0, dtype=paddle.get_default_dtype(), place=device
            ),
        )
        self.register_buffer(
            name="acc_sum",
            tensor=paddle.zeros(shape=size, dtype=paddle.get_default_dtype()),
        )
        self.register_buffer(
            name="acc_sum_squared",
            tensor=paddle.zeros(shape=size, dtype=paddle.get_default_dtype()),
        )

    def forward(self, batched_data, accumulate=True):
        """
        Updates mean/standard deviation and normalizes input data

        batched_data: batch of data
        accumulate: if True, update accumulation statistics
        """
        if accumulate and self.num_accumulations < self.max_accumulations:
            self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std()

    def inverse(self, normalized_batch_data):
        """
        Unnormalizes input data
        """
        return normalized_batch_data * self._std().to(
            normalized_batch_data.place
        ) + self._mean().to(normalized_batch_data.place)

    def _accumulate(self, batched_data):
        """
        Accumulates statistics for mean/standard deviation computation
        """
        count = paddle.to_tensor(data=tuple(batched_data.shape)[0]).astype(
            dtype="float32"
        )
        data_sum = paddle.sum(x=batched_data, axis=0)
        squared_data_sum = paddle.sum(x=batched_data**2, axis=0)
        self.acc_sum += data_sum.to(self.acc_sum.place)
        self.acc_sum_squared += squared_data_sum.to(self.acc_sum_squared.place)
        self.acc_count += count.to(self.acc_count.place)
        self.num_accumulations += 1

    def _mean(self):
        """
        Returns accumulated mean
        """
        safe_count = paddle_aux.max(
            self.acc_count, paddle.to_tensor(data=1.0).astype(dtype="float32")
        )
        return self.acc_sum / safe_count

    def _std(self):
        """
        Returns accumulated standard deviation
        """
        safe_count = paddle_aux.max(
            self.acc_count, paddle.to_tensor(data=1.0).astype(dtype="float32")
        )
        std = paddle.sqrt(x=self.acc_sum_squared / safe_count - self._mean() ** 2)
        std[std < self.epsilon] = 1.0
        return std

"""PhyCRNet for solving spatiotemporal PDEs"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn as nn
import scipy.io as scio
from paddle.nn import utils
from paddle.optimizer import lr

paddle.seed(5)
np.random.seed(5)

# define the high-order finite difference kernels
lapl_op = [
    [
        [
            [0, 0, -1 / 12, 0, 0],
            [0, 0, 4 / 3, 0, 0],
            [-1 / 12, 4 / 3, -5, 4 / 3, -1 / 12],
            [0, 0, 4 / 3, 0, 0],
            [0, 0, -1 / 12, 0, 0],
        ]
    ]
]

partial_y = [
    [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1 / 12, -8 / 12, 0, 8 / 12, -1 / 12],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    ]
]

partial_x = [
    [
        [
            [0, 0, 1 / 12, 0, 0],
            [0, 0, -8 / 12, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 8 / 12, 0, 0],
            [0, 0, -1 / 12, 0, 0],
        ]
    ]
]

# specific parameters for burgers equation
def initialize_weights(module):
    if isinstance(module, nn.Conv2D):
        c = 1.0  # 0.5
        initializer = nn.initializer.Uniform(
            -c * np.sqrt(1 / (3 * 3 * 320)), c * np.sqrt(1 / (3 * 3 * 320))
        )
        initializer(module.weight)
    elif isinstance(module, nn.Linear):
        initializer = nn.initializer.Constant(0.0)
        initializer(module.bias)


class ConvLSTMCell(nn.Layer):
    """Convolutional LSTM"""

    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kernel_size,
        input_stride,
        input_padding,
    ):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_features = 4

        # padding for hidden state
        self.padding = int((self.hidden_kernel_size - 1) / 2)

        self.Wxi = nn.Conv2D(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias_attr=None,
            padding_mode="circular",
        )

        self.Whi = nn.Conv2D(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias_attr=False,
            padding_mode="circular",
        )

        self.Wxf = nn.Conv2D(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias_attr=None,
            padding_mode="circular",
        )

        self.Whf = nn.Conv2D(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias_attr=False,
            padding_mode="circular",
        )

        self.Wxc = nn.Conv2D(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias_attr=None,
            padding_mode="circular",
        )

        self.Whc = nn.Conv2D(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias_attr=False,
            padding_mode="circular",
        )

        self.Wxo = nn.Conv2D(
            self.input_channels,
            self.hidden_channels,
            self.input_kernel_size,
            self.input_stride,
            self.input_padding,
            bias_attr=None,
            padding_mode="circular",
        )

        self.Who = nn.Conv2D(
            self.hidden_channels,
            self.hidden_channels,
            self.hidden_kernel_size,
            1,
            padding=1,
            bias_attr=False,
            padding_mode="circular",
        )

        initializer_0 = paddle.nn.initializer.Constant(0.0)
        initializer_1 = paddle.nn.initializer.Constant(1.0)

        initializer_0(self.Wxi.bias)
        initializer_0(self.Wxf.bias)
        initializer_0(self.Wxc.bias)
        initializer_1(self.Wxo.bias)

    def forward(self, x, h, c):
        ci = paddle.nn.functional.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = paddle.nn.functional.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * paddle.tanh(self.Wxc(x) + self.Whc(h))
        co = paddle.nn.functional.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * paddle.tanh(cc)
        return ch, cc

    def init_hidden_tensor(self, prev_state):
        return ((prev_state[0]).cuda(), (prev_state[1]).cuda())


class encoder_block(nn.Layer):
    """encoder with CNN"""

    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kernel_size,
        input_stride,
        input_padding,
    ):
        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = utils.weight_norm(
            nn.Conv2D(
                self.input_channels,
                self.hidden_channels,
                self.input_kernel_size,
                self.input_stride,
                self.input_padding,
                bias_attr=None,
                padding_mode="circular",
            )
        )

        self.act = nn.ReLU()

        initializer_0 = paddle.nn.initializer.Constant(0.0)
        initializer_0(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))


class PhyCRNet(nn.Layer):
    """physics-informed convolutional-recurrent neural networks"""

    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kernel_size,
        input_stride,
        input_padding,
        dt,
        num_layers,
        upscale_factor,
        step=1,
        effective_step=[1],
    ):
        super(PhyCRNet, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.dt = dt
        self.upscale_factor = upscale_factor

        # number of layers
        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]

        # encoder - downsampling
        for i in range(self.num_encoder):
            name = "encoder{}".format(i)
            cell = encoder_block(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i],
            )

            setattr(self, name, cell)
            self._all_layers.append(cell)

        # ConvLSTM
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            name = "convlstm{}".format(i)
            cell = ConvLSTMCell(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i],
            )

            setattr(self, name, cell)
            self._all_layers.append(cell)

        # output layer
        self.output_layer = nn.Conv2D(
            2, 2, kernel_size=5, stride=1, padding=2, padding_mode="circular"
        )

        # pixelshuffle - upscale
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)

        # initialize weights
        self.apply(initialize_weights)
        initializer_0 = paddle.nn.initializer.Constant(0.0)
        initializer_0(self.output_layer.bias)

    def forward(self, initial_state, x):
        self.initial_state = initial_state
        internal_state = []
        outputs = []
        second_last_state = []

        for step in range(self.step):
            xt = x

            # encoder
            for i in range(self.num_encoder):
                name = "encoder{}".format(i)
                x = getattr(self, name)(x)

            # convlstm
            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                name = "convlstm{}".format(i)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden_tensor(
                        prev_state=self.initial_state[i - self.num_encoder]
                    )
                    internal_state.append((h, c))

                # one-step forward
                (h, c) = internal_state[i - self.num_encoder]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i - self.num_encoder] = (x, new_c)

            # output
            x = self.pixelshuffle(x)
            x = self.output_layer(x)

            # residual connection
            x = xt + self.dt * x

            if step == (self.step - 2):
                second_last_state = internal_state.copy()

            if step in self.effective_step:
                outputs.append(x)

        return outputs, second_last_state


class Conv2DDerivative(nn.Layer):
    def __init__(self, DerFilter, resol, kernel_size=3, name=""):
        super(Conv2DDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2D(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            padding=0,
            bias_attr=False,
        )

        # Fixed gradient operator
        self.filter.weight = self.create_parameter(
            shape=self.filter.weight.shape,
            dtype=self.filter.weight.dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(
                    DerFilter, dtype=paddle.get_default_dtype(), stop_gradient=True
                )
            ),
        )
        self.filter.weight.stop_gradient = True

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1DDerivative(nn.Layer):
    def __init__(self, DerFilter, resol, kernel_size=3, name=""):
        super(Conv1DDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1D(
            self.input_channels,
            self.output_channels,
            self.kernel_size,
            1,
            padding=0,
            bias_attr=False,
        )

        # Fixed gradient operator
        self.filter.weight = self.create_parameter(
            shape=self.filter.weight.shape,
            dtype=self.filter.weight.dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(
                    DerFilter, dtype=paddle.get_default_dtype(), stop_gradient=True
                )
            ),
        )
        self.filter.weight.stop_gradient = True

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Layer):
    """Loss generator for physics loss"""

    def __init__(self, dt=(10.0 / 200), dx=(20.0 / 128)):
        """Construct the derivatives, X = Width, Y = Height"""

        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2DDerivative(
            DerFilter=lapl_op, resol=(dx**2), kernel_size=5, name="laplace_operator"
        )

        self.dx = Conv2DDerivative(
            DerFilter=partial_x, resol=(dx * 1), kernel_size=5, name="dx_operator"
        )

        self.dy = Conv2DDerivative(
            DerFilter=partial_y, resol=(dx * 1), kernel_size=5, name="dy_operator"
        )

        # temporal derivative operator
        self.dt = Conv1DDerivative(
            DerFilter=[[[-1, 0, 1]]], resol=(dt * 2), kernel_size=3, name="partial_t"
        )

    def get_phy_Loss(self, output):
        # spatial derivatives
        laplace_u = self.laplace(output[1:-1, 0:1, :, :])  # [t,c,h,w]
        laplace_v = self.laplace(output[1:-1, 1:2, :, :])

        u_x = self.dx(output[1:-1, 0:1, :, :])
        u_y = self.dy(output[1:-1, 0:1, :, :])
        v_x = self.dx(output[1:-1, 1:2, :, :])
        v_y = self.dy(output[1:-1, 1:2, :, :])

        # temporal derivative - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_Conv1D = u.transpose((2, 3, 1, 0))  # [height(Y), width(X), c, step]
        u_Conv1D = u_Conv1D.reshape((lenx * leny, 1, lent))
        u_t = self.dt(u_Conv1D)  # lent-2 due to no-padding
        u_t = u_t.reshape((leny, lenx, 1, lent - 2))
        u_t = u_t.transpose((3, 2, 0, 1))  # [step-2, c, height(Y), width(X)]

        # temporal derivative - v
        v = output[:, 1:2, 2:-2, 2:-2]
        v_Conv1D = v.transpose((2, 3, 1, 0))  # [height(Y), width(X), c, step]
        v_Conv1D = v_Conv1D.reshape((lenx * leny, 1, lent))
        v_t = self.dt(v_Conv1D)  # lent-2 due to no-padding
        v_t = v_t.reshape((leny, lenx, 1, lent - 2))
        v_t = v_t.transpose((3, 2, 0, 1))  # [step-2, c, height(Y), width(X)]

        u = output[1:-1, 0:1, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]
        v = output[1:-1, 1:2, 2:-2, 2:-2]  # [t, c, height(Y), width(X)]

        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        R = 200.0

        # 2D burgers eqn
        f_u = u_t + u * u_x + v * u_y - (1 / R) * laplace_u
        f_v = v_t + u * v_x + v * v_y - (1 / R) * laplace_v

        return f_u, f_v


def compute_loss(output, loss_func):
    """calculate the physics loss"""

    # Padding x axis due to periodic boundary condition
    # shape: [t, c, h, w]
    output = paddle.concat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), axis=3)

    # Padding y axis due to periodic boundary condition
    # shape: [t, c, h, w]
    output = paddle.concat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), axis=2)

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_Loss(output)
    loss = mse_loss(f_u, paddle.zeros_like(f_u).cuda()) + mse_loss(
        f_v, paddle.zeros_like(f_v).cuda()
    )

    return loss


def train(
    model,
    input,
    initial_state,
    n_iters,
    time_batch_size,
    learning_rate,
    dt,
    dx,
    save_path,
    pre_model_save_path,
    num_time_batch,
):
    train_loss_list = []
    second_last_state = []
    prev_output = []

    batch_loss = 0.0
    best_loss = 1e4

    # load previous model
    scheduler = lr.StepDecay(step_size=100, gamma=0.97, learning_rate=learning_rate)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=scheduler
    )
    model, optimizer, scheduler = load_checkpoint(
        model, optimizer, scheduler, pre_model_save_path
    )

    print(f"learning_rate: {optimizer.get_lr()}")

    loss_func = loss_generator(dt, dx)

    for epoch in range(n_iters):
        # input: [t,b,c,h,w]
        optimizer.clear_grad()
        batch_loss = 0

        state_detached = []
        for time_batch_id in range(num_time_batch):
            # update the first input for each time batch
            if time_batch_id == 0:
                hidden_state = initial_state
                u0 = input
            else:
                hidden_state = state_detached
                u0 = prev_output[-2:-1].detach()  # second last output

            # output is a list
            output, second_last_state = model(hidden_state, u0)

            # [t, c, height (Y), width (X)]
            output = paddle.concat(tuple(output), axis=0)

            # concatenate the initial state to the output for central diff
            output = paddle.concat((u0.cuda(), output), axis=0)

            # get loss
            loss = compute_loss(output, loss_func)
            loss.backward(retain_graph=True)
            batch_loss += loss.item()

            # update the state and output for next batch
            prev_output = output
            state_detached = []
            for i in range(len(second_last_state)):
                (h, c) = second_last_state[i]
                state_detached.append((h.detach(), c.detach()))  # hidden state

        optimizer.step()
        scheduler.step()

        # print loss in each epoch
        print(
            "[%d/%d %d%%] loss: %.10f"
            % ((epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), batch_loss)
        )
        train_loss_list.append(batch_loss)

        # save model
        if batch_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, save_path)
            best_loss = batch_loss

    return train_loss_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.stop_gradient)


def post_process(output, true, axis_lim, uv_lim, num, fig_save_path):
    """
    axis_lim: [xmin, xmax, ymin, ymax]
    uv_lim: [u_min, u_max, v_min, v_max]
    num: Number of time step
    """

    # get the limit
    xmin, xmax, ymin, ymax = axis_lim
    u_min, u_max, v_min, v_max = uv_lim

    # grid
    x = np.linspace(xmin, xmax, 128 + 1)
    # Source code has, run with error
    # x = x[:-1]
    x_star, y_star = np.meshgrid(x, x)

    u_star = true[num, 0, 1:-1, 1:-1]
    u_pred = output[num, 0, 1:-1, 1:-1].detach().cpu().numpy()

    v_star = true[num, 1, 1:-1, 1:-1]
    v_pred = output[num, 1, 1:-1, 1:-1].detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0, 0].scatter(
        x_star,
        y_star,
        c=u_pred,
        alpha=0.9,
        edgecolors="none",
        cmap="RdYlBu",
        marker="s",
        s=4,
        vmin=u_min,
        vmax=u_max,
    )
    ax[0, 0].axis("square")
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_title("u-RCNN")
    fig.colorbar(cf, ax=ax[0, 0])

    cf = ax[0, 1].scatter(
        x_star,
        y_star,
        c=u_star,
        alpha=0.9,
        edgecolors="none",
        cmap="RdYlBu",
        marker="s",
        s=4,
        vmin=u_min,
        vmax=u_max,
    )
    ax[0, 1].axis("square")
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_title("u-Ref.")
    fig.colorbar(cf, ax=ax[0, 1])

    cf = ax[1, 0].scatter(
        x_star,
        y_star,
        c=v_pred,
        alpha=0.9,
        edgecolors="none",
        cmap="RdYlBu",
        marker="s",
        s=4,
        vmin=v_min,
        vmax=v_max,
    )
    ax[1, 0].axis("square")
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    cf.cmap.set_under("whitesmoke")
    cf.cmap.set_over("black")
    ax[1, 0].set_title("v-RCNN")
    fig.colorbar(cf, ax=ax[1, 0])

    cf = ax[1, 1].scatter(
        x_star,
        y_star,
        c=v_star,
        alpha=0.9,
        edgecolors="none",
        cmap="RdYlBu",
        marker="s",
        s=4,
        vmin=v_min,
        vmax=v_max,
    )
    ax[1, 1].axis("square")
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    cf.cmap.set_under("whitesmoke")
    cf.cmap.set_over("black")
    ax[1, 1].set_title("v-Ref.")
    fig.colorbar(cf, ax=ax[1, 1])

    # plt.draw()
    plt.savefig(fig_save_path + "uv_comparison_" + str(num).zfill(3) + ".png")
    plt.close("all")

    return u_star, u_pred, v_star, v_pred


def save_checkpoint(model, optimizer, scheduler, save_dir):
    """save model and optimizer"""

    paddle.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        save_dir,
    )


def load_checkpoint(model, optimizer, scheduler, save_dir):
    """load model and optimizer"""

    if not os.path.exists(save_dir):
        return model, optimizer, scheduler

    checkpoint = paddle.load(save_dir)
    model.set_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.set_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.set_state_dict(checkpoint["scheduler_state_dict"])

    print("Pretrained model loaded!")

    return model, optimizer, scheduler


def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)


def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor**2))


if __name__ == "__main__":

    ######### download the ground truth data ############
    data_dir = "./output/data/2dBurgers/burgers_1501x2x128x128.mat"
    data = scio.loadmat(data_dir)
    uv = data["uv"]  # [t,c,h,w]

    # initial condition
    uv0 = uv[0:1, ...]
    input = paddle.to_tensor(uv0, dtype=paddle.get_default_dtype())

    # set initial states for convlstm
    num_convlstm = 1
    (h0, c0) = (paddle.randn((1, 128, 16, 16)), paddle.randn((1, 128, 16, 16)))
    initial_state = []
    for i in range(num_convlstm):
        initial_state.append((h0, c0))

    # grid parameters
    time_steps = 1001
    dt = 0.002
    dx = 1.0 / 128

    ################# build the model #####################
    time_batch_size = 1000
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    # previous iter
    pre_iter = 0
    n_iters_adam = 200
    # save iter
    save_iter = pre_iter + n_iters_adam
    lr_adam = 1e-4  # 1e-3
    pre_model_save_path = f"./output/checkpoint{pre_iter}.pt"
    model_save_path = f"./output/checkpoint{save_iter}.pt"
    fig_save_path = "./output/figures/"

    model = PhyCRNet(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=dt,
        num_layers=[3, 1],
        upscale_factor=8,
        step=steps,
        effective_step=effective_step,
    )

    start = time.time()
    train_loss = train(
        model,
        input,
        initial_state,
        n_iters_adam,
        time_batch_size,
        lr_adam,
        dt,
        dx,
        model_save_path,
        pre_model_save_path,
        num_time_batch,
    )
    end = time.time()

    np.save("./output/train_loss", train_loss)
    print("The training time is: ", (end - start))

    ########### model inference ##################
    time_batch_size_load = 1000
    steps_load = time_batch_size_load + 1
    num_time_batch = int(time_steps / time_batch_size_load)
    effective_step = list(range(0, steps_load))

    model = PhyCRNet(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=dt,
        num_layers=[3, 1],
        upscale_factor=8,
        step=steps_load,
        effective_step=effective_step,
    )

    model, _, _ = load_checkpoint(
        model, optimizer=None, scheduler=None, save_dir=model_save_path
    )
    output, _ = model(initial_state, input)

    # shape: [t, c, h, w]
    output = paddle.concat(tuple(output), axis=0)
    output = paddle.concat((input.cuda(), output), axis=0)

    # Padding x and y axis due to periodic boundary condition
    output = paddle.concat((output[:, :, :, -1:], output, output[:, :, :, 0:2]), axis=3)
    output = paddle.concat((output[:, :, -1:, :], output, output[:, :, 0:2, :]), axis=2)

    # [t, c, h, w]
    truth = uv[0:1001, :, :, :]

    # [101, 2, 131, 131]
    truth = np.concatenate((truth[:, :, :, -1:], truth, truth[:, :, :, 0:2]), axis=3)
    truth = np.concatenate((truth[:, :, -1:, :], truth, truth[:, :, 0:2, :]), axis=2)

    # post-process
    ten_true = []
    ten_pred = []
    for i in range(0, 50):
        u_star, u_pred, v_star, v_pred = post_process(
            output,
            truth,
            [0, 1, 0, 1],
            [-0.7, 0.7, -1.0, 1.0],
            num=20 * i,
            fig_save_path=fig_save_path,
        )

        ten_true.append([u_star, v_star])
        ten_pred.append([u_pred, v_pred])

    # compute the error
    error = frobenius_norm(np.array(ten_pred) - np.array(ten_true)) / frobenius_norm(
        np.array(ten_true)
    )

    print("The predicted error is: ", error)

    u_pred = output[:-1, 0, :, :].detach().cpu().numpy()
    u_pred = np.swapaxes(u_pred, 1, 2)  # [h,w] = [y,x]
    u_true = truth[:, 0, :, :]

    t_true = np.linspace(0, 2, 1001)
    t_pred = np.linspace(0, 2, time_steps)

    plt.plot(t_pred, u_pred[:, 32, 32], label="x=32, y=32, CRL")
    plt.plot(t_true, u_true[:, 32, 32], "--", label="x=32, y=32, Ref.")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.xlim(0, 2)
    plt.legend()
    plt.savefig(fig_save_path + "x=32,y=32.png")
    plt.close("all")

    # plot train loss
    plt.figure()
    plt.plot(train_loss, label="train loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(fig_save_path + "train loss.png", dpi=300)

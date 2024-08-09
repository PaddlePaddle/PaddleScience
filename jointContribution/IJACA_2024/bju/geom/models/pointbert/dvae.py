import paddle
import utils.paddle_aux  # NOQA
from geom.models.pointbert import misc
from geom.models.pointbert.knn import knn


class DGCNN(paddle.nn.Layer):
    def __init__(self, encoder_channel, output_channel):
        super().__init__()
        """
        K has to be 16
        """
        self.input_trans = paddle.nn.Conv1D(
            in_channels=encoder_channel, out_channels=128, kernel_size=1
        )
        self.layer1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=256, out_channels=256, kernel_size=1, bias_attr=False
            ),
            paddle.nn.GroupNorm(num_groups=4, num_channels=256),
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )
        self.layer2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=512, out_channels=512, kernel_size=1, bias_attr=False
            ),
            paddle.nn.GroupNorm(num_groups=4, num_channels=512),
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )
        self.layer3 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=1024, out_channels=512, kernel_size=1, bias_attr=False
            ),
            paddle.nn.GroupNorm(num_groups=4, num_channels=512),
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )
        self.layer4 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=1024, out_channels=1024, kernel_size=1, bias_attr=False
            ),
            paddle.nn.GroupNorm(num_groups=4, num_channels=1024),
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )
        self.layer5 = paddle.nn.Sequential(
            paddle.nn.Conv1D(
                in_channels=2304,
                out_channels=output_channel,
                kernel_size=1,
                bias_attr=False,
            ),
            paddle.nn.GroupNorm(num_groups=4, num_channels=output_channel),
            paddle.nn.LeakyReLU(negative_slope=0.2),
        )

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        k = 4
        batch_size = x_k.shape[0]
        num_points_k = x_k.shape[2]
        num_points_q = x_q.shape[2]
        with paddle.no_grad():
            _, idx = knn(coor_k, coor_q, k=4)
            assert tuple(idx.shape)[1] == k
            idx_base = (
                paddle.arange(start=0, end=batch_size).view(-1, 1, 1) * num_points_k
            )
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.shape[1]
        x = x_k
        perm_0 = list(range(x.ndim))
        perm_0[2] = 1
        perm_0[1] = 2
        x_k = x.transpose(perm=perm_0)
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).transpose(
            perm=[0, 3, 2, 1]
        )
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(
            shape=[-1, -1, -1, k]
        )
        feature = paddle.concat(x=(feature - x_q, x_q), axis=1)
        return feature

    def forward(self, f, coor):
        feature_list = []
        x = coor
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        coor = x.transpose(perm=perm_1)
        x = f
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        f = x.transpose(perm=perm_2)
        f = self.input_trans(f)
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]
        feature_list.append(f)
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        feature_list.append(f)
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]
        feature_list.append(f)
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        feature_list.append(f)
        f = paddle.concat(x=feature_list, axis=1)
        f = self.layer5(f)
        x = f
        perm_3 = list(range(x.ndim))
        perm_3[-1] = -2
        perm_3[-2] = -1
        f = x.transpose(perm=perm_3)
        return f


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = paddle.topk(
        k=nsample, largest=False, sorted=False, x=sqrdists, axis=-1
    )
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = tuple(src.shape)
    _, M, _ = tuple(dst.shape)
    dist = -2 * paddle.matmul(x=src, y=dst.transpose(perm=[0, 2, 1]))
    dist += paddle.sum(x=src**2, axis=-1).view(B, N, 1)
    dist += paddle.sum(x=dst**2, axis=-1).view(B, 1, M)
    return dist


class Group(paddle.nn.Layer):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        """
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
        """
        batch_size, num_points, _ = tuple(xyz.shape)
        center = misc.fps(xyz, self.num_group)
        idx = knn_point(self.group_size, xyz, center)
        assert idx.shape[1] == self.num_group
        assert idx.shape[2] == self.group_size
        idx_base = paddle.arange(start=0, end=batch_size).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3)
        neighborhood = neighborhood - center.unsqueeze(axis=2)
        return neighborhood, center


class Encoder(paddle.nn.Layer):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = paddle.nn.Sequential(
            paddle.nn.Conv1D(in_channels=3, out_channels=128, kernel_size=1),
            paddle.nn.BatchNorm1D(num_features=128),
            paddle.nn.ReLU(),
            paddle.nn.Conv1D(in_channels=128, out_channels=256, kernel_size=1),
        )
        self.second_conv = paddle.nn.Sequential(
            paddle.nn.Conv1D(in_channels=512, out_channels=512, kernel_size=1),
            paddle.nn.BatchNorm1D(num_features=512),
            paddle.nn.ReLU(),
            paddle.nn.Conv1D(
                in_channels=512, out_channels=self.encoder_channel, kernel_size=1
            ),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, _ = tuple(point_groups.shape)
        point_groups = point_groups.reshape(bs * g, n, 3)
        x = point_groups
        perm_4 = list(range(x.ndim))
        perm_4[2] = 1
        perm_4[1] = 2
        feature = self.first_conv(x.transpose(perm=perm_4))
        feature_global = (
            paddle.max(x=feature, axis=2, keepdim=True),
            paddle.argmax(x=feature, axis=2, keepdim=True),
        )[0]
        feature = paddle.concat(
            x=[feature_global.expand(shape=[-1, -1, n]), feature], axis=1
        )
        feature = self.second_conv(feature)
        feature_global = (
            paddle.max(x=feature, axis=2, keepdim=False),
            paddle.argmax(x=feature, axis=2, keepdim=False),
        )[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Decoder(paddle.nn.Layer):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = self.num_fine // 4
        assert num_fine % 4 == 0
        self.mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=encoder_channel, out_features=1024),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=1024, out_features=1024),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=1024, out_features=3 * self.num_coarse),
        )
        self.final_conv = paddle.nn.Sequential(
            paddle.nn.Conv1D(
                in_channels=encoder_channel + 3 + 2, out_channels=512, kernel_size=1
            ),
            paddle.nn.BatchNorm1D(num_features=512),
            paddle.nn.ReLU(),
            paddle.nn.Conv1D(in_channels=512, out_channels=512, kernel_size=1),
            paddle.nn.BatchNorm1D(num_features=512),
            paddle.nn.ReLU(),
            paddle.nn.Conv1D(in_channels=512, out_channels=3, kernel_size=1),
        )
        a = (
            paddle.linspace(start=-0.05, stop=0.05, num=self.grid_size, dtype="float32")
            .view(1, self.grid_size)
            .expand(shape=[self.grid_size, self.grid_size])
            .reshape(1, -1)
        )
        b = (
            paddle.linspace(start=-0.05, stop=0.05, num=self.grid_size, dtype="float32")
            .view(self.grid_size, 1)
            .expand(shape=[self.grid_size, self.grid_size])
            .reshape(1, -1)
        )
        self.folding_seed = paddle.concat(x=[a, b], axis=0).view(
            1, 2, self.grid_size**2
        )

    def forward(self, feature_global):
        """
        feature_global : B G C
        -------
        coarse : B G M 3
        fine : B G N 3

        """
        bs, g, c = tuple(feature_global.shape)
        feature_global = feature_global.reshape(bs * g, c)
        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3)
        point_feat = coarse.unsqueeze(axis=2).expand(
            shape=[-1, -1, self.grid_size**2, -1]
        )
        x = point_feat.reshape(bs * g, self.num_fine, 3)
        perm_5 = list(range(x.ndim))
        perm_5[2] = 1
        perm_5[1] = 2
        point_feat = x.transpose(perm=perm_5)
        seed = self.folding_seed.unsqueeze(axis=2).expand(
            shape=[bs * g, -1, self.num_coarse, -1]
        )
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.place)
        feature_global = feature_global.unsqueeze(axis=2).expand(
            shape=[-1, -1, self.num_fine]
        )
        feat = paddle.concat(x=[feature_global, seed, point_feat], axis=1)
        center = coarse.unsqueeze(axis=2).expand(
            shape=[-1, -1, self.grid_size**2, -1]
        )
        x = center.reshape(bs * g, self.num_fine, 3)
        perm_6 = list(range(x.ndim))
        perm_6[2] = 1
        perm_6[1] = 2
        center = x.transpose(perm=perm_6)
        fine = self.final_conv(feat) + center
        x = fine.reshape(bs, g, 3, self.num_fine)
        perm_7 = list(range(x.ndim))
        perm_7[-1] = -2
        perm_7[-2] = -1
        fine = x.transpose(perm=perm_7)
        coarse = coarse.reshape(bs, g, self.num_coarse, 3)
        return coarse, fine


class DiscreteVAE(paddle.nn.Layer):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.tokens_dims = config.tokens_dims
        self.decoder_dims = config.decoder_dims
        self.num_tokens = config.num_tokens
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.dgcnn_1 = DGCNN(
            encoder_channel=self.encoder_dims, output_channel=self.num_tokens
        )
        out_5 = paddle.create_parameter(
            shape=paddle.randn(shape=[self.num_tokens, self.tokens_dims]).shape,
            dtype=paddle.randn(shape=[self.num_tokens, self.tokens_dims]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.randn(shape=[self.num_tokens, self.tokens_dims])
            ),
        )
        out_5.stop_gradient = not True
        self.codebook = out_5
        self.dgcnn_2 = DGCNN(
            encoder_channel=self.tokens_dims, output_channel=self.decoder_dims
        )
        self.decoder = Decoder(
            encoder_channel=self.decoder_dims, num_fine=self.group_size
        )

    def recon_loss(self, ret, gt):
        whole_coarse, whole_fine, coarse, fine, group_gt, _ = ret
        bs, g, _, _ = tuple(coarse.shape)
        coarse = coarse.reshape(bs * g, -1, 3)
        fine = fine.reshape(bs * g, -1, 3)
        group_gt = group_gt.reshape(bs * g, -1, 3)
        loss_coarse_block = self.loss_func_cdl1(coarse, group_gt)
        loss_fine_block = self.loss_func_cdl1(fine, group_gt)
        loss_recon = loss_coarse_block + loss_fine_block
        return loss_recon

    def get_loss(self, ret, gt):
        loss_recon = self.recon_loss(ret, gt)
        logits = ret[-1]
        softmax = paddle.nn.functional.softmax(x=logits, axis=-1)
        mean_softmax = softmax.mean(axis=1)
        log_qy = paddle.log(x=mean_softmax)
        log_uniform = paddle.log(
            x=paddle.to_tensor(data=[1.0 / self.num_tokens], place=gt.place)
        )
        loss_klv = paddle.nn.functional.kl_div(
            log_qy,
            log_uniform.expand(shape=[log_qy.shape[0], log_qy.shape[1]]),
            None,
            None,
            "batchmean",
            log_target=True,
        )
        return loss_recon, loss_klv

    def forward(self, inp, temperature=1.0, hard=False, **kwargs):
        neighborhood, center = self.group_divider(inp)
        logits = self.encoder(neighborhood)
        logits = self.dgcnn_1(logits, center)
        soft_one_hot = paddle.nn.functional.gumbel_softmax(
            x=logits, temperature=temperature, axis=2, hard=hard
        )
        sampled = paddle.einsum("b g n, n c -> b g c", soft_one_hot, self.codebook)
        feature = self.dgcnn_2(sampled, center)
        coarse, fine = self.decoder(feature)
        with paddle.no_grad():
            whole_fine = (fine + center.unsqueeze(axis=2)).reshape(inp.shape[0], -1, 3)
            whole_coarse = (coarse + center.unsqueeze(axis=2)).reshape(
                inp.shape[0], -1, 3
            )
        assert fine.shape[2] == self.group_size
        ret = whole_coarse, whole_fine, coarse, fine, neighborhood, logits
        return ret

import functools


class CuboidSelfAttentionPatterns:
    def __init__(self):
        super().__init__()
        self.patterns = {}
        self.patterns = {"full": self.full_attention,
                         "axial": self.axial,
                         "divided_st": self.divided_space_time}
        for p in [1, 2, 4, 8, 10]:
            for m in [1, 2, 4, 8, 16, 32]:
                key = f"video_swin_{p}x{m}"
                self.patterns[key] = functools.partial(self.video_swin, P=p, M=m)

        for m in [1, 2, 4, 8, 16, 32]:
            key = f"spatial_lg_{m}"
            self.patterns[key] = functools.partial(self.spatial_lg_v1, M=m)

        for k in [2, 4, 8]:
            key = f"axial_space_dilate_{k}"
            self.patterns[key] = functools.partial(self.axial_space_dilate_K, K=k)

    def get(self, pattern_name):
        return self.patterns[pattern_name]

    def full_attention(self, input_shape):
        T, H, W, _ = input_shape
        cuboid_size = [(T, H, W)]
        strategy = [("l", "l", "l")]
        shift_size = [(0, 0, 0)]
        return cuboid_size, strategy, shift_size

    def axial(self, input_shape):
        """Axial attention proposed in https://arxiv.org/abs/1912.12180

        Parameters
        ----------
        input_shape
            T, H, W

        Returns
        -------
        cuboid_size
        strategy
        shift_size
        """
        T, H, W, _ = input_shape
        cuboid_size = [(T, 1, 1), (1, H, 1), (1, 1, W)]
        strategy = [("l", "l", "l"), ("l", "l", "l"), ("l", "l", "l")]
        shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        return cuboid_size, strategy, shift_size

    def divided_space_time(self, input_shape):
        T, H, W, _ = input_shape
        cuboid_size = [(T, 1, 1), (1, H, W)]
        strategy = [("l", "l", "l"), ("l", "l", "l")]
        shift_size = [(0, 0, 0), (0, 0, 0)]
        return cuboid_size, strategy, shift_size

    def video_swin(self, input_shape, P=2, M=4):
        """Adopt the strategy in Video SwinTransformer https://arxiv.org/pdf/2106.13230.pdf"""
        T, H, W, _ = input_shape
        P = min(P, T)
        M = min(M, H, W)
        cuboid_size = [(P, M, M), (P, M, M)]
        strategy = [("l", "l", "l"), ("l", "l", "l")]
        shift_size = [(0, 0, 0), (P // 2, M // 2, M // 2)]
        return cuboid_size, strategy, shift_size

    def spatial_lg_v1(self, input_shape, M=4):
        T, H, W, _ = input_shape
        if H <= M and W <= M:
            cuboid_size = [(T, 1, 1), (1, H, W)]
            strategy = [("l", "l", "l"), ("l", "l", "l")]
            shift_size = [(0, 0, 0), (0, 0, 0)]
        else:
            cuboid_size = [(T, 1, 1), (1, M, M), (1, M, M)]
            strategy = [("l", "l", "l"), ("l", "l", "l"), ("d", "d", "d")]
            shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        return cuboid_size, strategy, shift_size

    def axial_space_dilate_K(self, input_shape, K=2):
        T, H, W, _ = input_shape
        K = min(K, H, W)
        cuboid_size = [
            (T, 1, 1),
            (1, H // K, 1),
            (1, H // K, 1),
            (1, 1, W // K),
            (1, 1, W // K),
        ]
        strategy = [
            ("l", "l", "l"),
            ("d", "d", "d"),
            ("l", "l", "l"),
            ("d", "d", "d"),
            ("l", "l", "l"),
        ]
        shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
        return cuboid_size, strategy, shift_size


class CuboidCrossAttentionPatterns:
    def __init__(self):
        super().__init__()
        self.patterns = {}
        for k in [1, 2, 4, 8]:
            key1 = f"cross_{k}x{k}"
            key2 = f"cross_{k}x{k}_lg"
            key3 = f"cross_{k}x{k}_heter"
            self.patterns[key1] = functools.partial(self.cross_KxK, K=k)
            self.patterns[key2] = functools.partial(self.cross_KxK_lg, K=k)
            self.patterns[key3] = functools.partial(self.cross_KxK_heter, K=k)

    def get(self, pattern_name):
        return self.patterns[pattern_name]

    def cross_KxK(self, mem_shape, K):
        """

        Parameters
        ----------
        mem_shape
        K

        Returns
        -------
        cuboid_hw
        shift_hw
        strategy
        n_temporal
        """
        T_mem, H, W, _ = mem_shape
        K = min(K, H, W)
        cuboid_hw = [(K, K)]
        shift_hw = [(0, 0)]
        strategy = [("l", "l", "l")]
        n_temporal = [1]
        return cuboid_hw, shift_hw, strategy, n_temporal

    def cross_KxK_lg(self, mem_shape, K):
        """

        Parameters
        ----------
        mem_shape
        K

        Returns
        -------
        cuboid_hw
        shift_hw
        strategy
        n_temporal
        """
        T_mem, H, W, _ = mem_shape
        K = min(K, H, W)
        cuboid_hw = [(K, K), (K, K)]
        shift_hw = [(0, 0), (0, 0)]
        strategy = [("l", "l", "l"), ("d", "d", "d")]
        n_temporal = [1, 1]
        return cuboid_hw, shift_hw, strategy, n_temporal

    def cross_KxK_heter(self, mem_shape, K):
        """

        Parameters
        ----------
        mem_shape
        K

        Returns
        -------
        cuboid_hw
        shift_hw
        strategy
        n_temporal
        """
        T_mem, H, W, _ = mem_shape
        K = min(K, H, W)
        cuboid_hw = [(K, K), (K, K), (K, K)]
        shift_hw = [(0, 0), (0, 0), (K // 2, K // 2)]
        strategy = [("l", "l", "l"), ("d", "d", "d"), ("l", "l", "l")]
        n_temporal = [1, 1, 1]
        return cuboid_hw, shift_hw, strategy, n_temporal


CuboidSelfAttentionPatterns = CuboidSelfAttentionPatterns()
CuboidCrossAttentionPatterns = CuboidCrossAttentionPatterns()
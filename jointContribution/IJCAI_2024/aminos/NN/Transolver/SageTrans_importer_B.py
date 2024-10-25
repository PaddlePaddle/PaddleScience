import paddle

from .FVGNAttUnet import Model


class FVGN(paddle.nn.Layer):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        space_dim = params.cell_input_size
        self.nn = Model(
            space_dim=space_dim,
            n_hidden=params.hidden_size,
            n_layers=3,
            fun_dim=0,
            n_head=4,
            mlp_ratio=2,
            out_dim=1,
            slice_num=32,
            unified_pos=0,
            params=params,
        )

    def forward(
        self,
        graph_cell=None,
        graph_edge=None,
        graph_node=None,
        params=None,
        is_training=True,
    ):
        x = graph_cell.x
        z = x
        output = self.nn(
            z,
            graph_node=graph_node,
            graph_edge=graph_edge,
            graph_cell=graph_cell,
            params=params,
        )
        return output

    def load_checkpoint(
        self, optimizer=None, scheduler=None, ckpdir=None, device=None, is_training=True
    ):
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = paddle.load(path=ckpdir)
        self.set_state_dict(state_dict=dicts["model"])
        keys = list(dicts.keys())
        keys.remove("model")
        if optimizer is not None:
            if type(optimizer) is not list:
                optimizer = [optimizer]
            for i, o in enumerate(optimizer):
                o.set_state_dict(state_dict=dicts["optimizer{}".format(i)])
                keys.remove("optimizer{}".format(i))
        if scheduler is not None:
            if type(scheduler) is not list:
                scheduler = [scheduler]
            for i, s in enumerate(scheduler):
                s.set_state_dict(state_dict=dicts["scheduler{}".format(i)])
                keys.remove("scheduler{}".format(i))
        if not is_training:
            for key in keys.copy():
                if key.find("optimizer") >= 0:
                    keys.remove(key)
                elif key.find("scheduler") >= 0:
                    keys.remove(key)
        print("Simulator model and optimizer/scheduler loaded checkpoint %s" % ckpdir)

    def save_checkpoint(self, path=None, optimizer=None, scheduler=None):
        if path is None:
            path = self.model_dir
        model = self.state_dict()
        to_save = {"model": model}
        if type(optimizer) is not list:
            optimizer = [optimizer]
        for i, o in enumerate(optimizer):
            to_save.update({"optimizer{}".format(i): o.state_dict()})
        if type(scheduler) is not list:
            scheduler = [scheduler]
        for i, s in enumerate(scheduler):
            to_save.update({"scheduler{}".format(i): s.state_dict()})
        paddle.save(obj=to_save, path=path)
        print("Simulator model saved at %s" % path)

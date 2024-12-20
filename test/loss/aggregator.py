import pytest

import ppsci
from ppsci import arch
from ppsci.loss import mtl

__all__ = []


class AggregatorTest:
    def __init__(self):
        self.model = arch.MLP(
            ("x", "y"),
            ("u", "v"),
            3,
            16,
        )

    def _check_agg_state_dict(self, agg):
        model_state = self.model.state_dict()
        agg_state = agg.state_dict()
        for k in agg_state:
            assert k not in model_state

    def test_AGDA(self):
        aggregator = mtl.AGDA(self.model)
        assert aggregator.should_persist is False

    def test_GradNorm(self):
        aggregator = mtl.GradNorm(self.model)
        assert aggregator.should_persist is True
        self._check_agg_state_dict(aggregator)

    def test_LossAggregator(self):
        aggregator = mtl.AGDA(self.model)
        assert aggregator.should_persist is False

    def test_PCGrad(self):
        aggregator = mtl.PCGrad(self.model)
        assert aggregator.should_persist is False

    def test_Relobralo(self):
        aggregator = mtl.Relobralo(self.model)
        assert aggregator.should_persist is True
        self._check_agg_state_dict(aggregator)

    def test_Sum(self):
        aggregator = mtl.Sum(self.model)
        assert aggregator.should_persist is False

    def test_NTK(self):
        aggregator = mtl.NTK(self.model)
        assert aggregator.should_persist is True
        self._check_agg_state_dict(aggregator)

    def test_restore_aggregator(self):
        model = ppsci.arch.MLP(
            ["x", "y"],
            ["u"],
            2,
            16,
        )
        opt = ppsci.optimizer.Adam(1e-3)(model)
        equation = ppsci.equation.Laplace(2)
        geom = ppsci.geometry.Rectangle([0, 0], [1, 1])
        BC = ppsci.constraint.BoundaryConstraint(
            equation.equations,
            {"laplace": 0.0},
            geom,
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": 10,
                "batch_size": 16,
            },
            loss=ppsci.loss.MSELoss(),
        )
        solver = ppsci.solver.Solver(
            model,
            {"bound": BC},
            optimizer=opt,
            output_dir="./tmp",
            iters_per_epoch=10,
            epochs=2,
        )
        solver.train()
        solver = ppsci.solver.Solver(
            model,
            {"bound": BC},
            optimizer=opt,
            output_dir="./tmp",
            iters_per_epoch=10,
            epochs=2,
            checkpoint_path="./tmp/checkpoints/latest",
        )


if __name__ == "__main__":
    pytest.main()

import paddle
from paddle import nn

from ppsci.arch import base


class SuzukiMiyauraModel(base.Arch):
    def __init__(
        self, input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4, output_dim
    ):
        super().__init__()

        self.r1_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
        )

        self.r2_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
        )

        self.ligand_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
        )

        self.base_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
        )

        self.solvent_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
        )

        self.weights = paddle.create_parameter(
            shape=[5],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor([0.2, 0.2, 0.2, 0.2, 0.2])
            ),
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.ReLU(),
            nn.Linear(hidden_dim4, output_dim),
        )

    def weighted_average(self, features, weights):

        weights = weights.clone().detach()

        weighted_sum = sum(f * w for f, w in zip(features, weights))

        total_weight = weights.sum()

        return weighted_sum / total_weight

    def forward(self, x):
        x = self.concat_to_tensor(x, ("v"), axis=-1)

        input_splits = paddle.split(x, num_or_sections=5, axis=1)

        r1_input, r2_input, ligand_input, base_input, solvent_input = input_splits

        r1_features = self.r1_fc(r1_input)

        r2_features = self.r2_fc(r2_input)

        ligand_features = self.ligand_fc(ligand_input)

        base_features = self.base_fc(base_input)

        solvent_features = self.solvent_fc(solvent_input)

        features = [
            r1_features,
            r2_features,
            ligand_features,
            base_features,
            solvent_features,
        ]

        combined_features = self.weighted_average(features, self.weights)

        output = self.fc_combined(combined_features)
        output = self.split_to_dict(output, ("u"), axis=-1)
        return output

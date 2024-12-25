## Model 0.2.0

This is the pretrained weights published with CHGNet Nature Machine Intelligence paper.
All the experiments and results shown in the paper were performed with this version of weights.

Date: 2/24/2023

Author: Bowen Deng

## Model Parameters

```python
model = CHGNet(
    atom_fea_dim=64,
    bond_fea_dim=64,
    angle_fea_dim=64,
    composition_model="MPtrj",
    num_radial=9,
    num_angular=9,
    n_conv=4,
    atom_conv_hidden_dim=64,
    update_bond=True,
    bond_conv_hidden_dim=64,
    update_angle=True,
    angle_layer_hidden_dim=0,
    conv_dropout=0,
    read_out="ave",
    mlp_hidden_dims=[64, 64],
    mlp_first=True,
    is_intensive=True,
    non_linearity="silu",
    atom_graph_cutoff=5,
    bond_graph_cutoff=3,
    graph_converter_algorithm="fast",
    cutoff_coeff=5,
    learnable_rbf=True,
    mlp_out_bias=True,
)
```

## Dataset Used

MPtrj dataset with 8-1-1 train-val-test splitting

## Trainer

```python
trainer = Trainer(
    model=model,
    targets='efsm',
    energy_loss_ratio=1,
    force_loss_ratio=1,
    stress_loss_ratio=0.1,
    mag_loss_ratio=0.1,
    optimizer='Adam',
    weight_decay=0,
    scheduler='CosLR',
    criterion='Huber',
    delta=0.1,
    epochs=20,
    starting_epoch=0,
    learning_rate=1e-3,
    use_device='cuda',
    print_freq=1000
)
```

## Mean Absolute Error (MAE) logs

| partition  | Energy (meV/atom) | Force (meV/A) | stress (GPa) | magmom (muB) |
| ---------- | ----------------- | ------------- | ------------ | ------------ |
| Train      | 22                | 59            | 0.246        | 0.030        |
| Validation | 30                | 75            | 0.350        | 0.033        |
| Test       | 30                | 77            | 0.348        | 0.032        |

## Model 0.3.0

Major changes:

1. Increased AtomGraph cutoff to 6A
2. Resolved discontinuity issue when no BondGraph presents
3. Added some normalization layers
4. Slight improvements on energy, force, stress accuracies

Date: 10/22/2023

Author: Bowen Deng

## Model Parameters

```python
model = CHGNet(
    atom_fea_dim=64,
    bond_fea_dim=64,
    angle_fea_dim=64,
    composition_model="MPtrj",
    num_radial=31,
    num_angular=31,
    n_conv=4,
    atom_conv_hidden_dim=64,
    update_bond=True,
    bond_conv_hidden_dim=64,
    update_angle=True,
    angle_layer_hidden_dim=0,
    conv_dropout=0,
    read_out="ave",
    gMLP_norm='layer',
    readout_norm='layer',
    mlp_hidden_dims=[64, 64, 64],
    mlp_first=True,
    is_intensive=True,
    non_linearity="silu",
    atom_graph_cutoff=6,
    bond_graph_cutoff=3,
    graph_converter_algorithm="fast",
    cutoff_coeff=8,
    learnable_rbf=True,
)
```

## Dataset Used

MPtrj dataset with 9-0.5-0.5 train-val-test splitting

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
    scheduler_params={'decay_fraction': 0.5e-2},
    criterion='Huber',
    delta=0.1,
    epochs=30,
    starting_epoch=0,
    learning_rate=5e-3,
    use_device='cuda',
    print_freq=1000
)
```

## Mean Absolute Error (MAE) logs

| partition  | Energy (meV/atom) | Force (meV/A) | stress (GPa) | magmom (muB) |
| ---------- | ----------------- | ------------- | ------------ | ------------ |
| Train      | 26                | 60            | 0.266        | 0.037        |
| Validation | 29                | 70            | 0.308        | 0.037        |
| Test       | 29                | 68            | 0.314        | 0.037        |

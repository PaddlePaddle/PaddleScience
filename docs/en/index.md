# PaddleScience

--8<--
./README.md:status
--8<--

--8<--
./README.md:announcement
--8<--

<style>
    .container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
    }
    .card {
        font-family: 'Noto Serif SC', sans-serif;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        color: black;
        font-weight: bold;
        height: 100px;
        padding: 20px;
        width: 170px;
        text-align: center;
        transition: border-color 0.1s; /* 边框颜色变化的过渡效果 */
        border: 2px solid transparent; /* 默认透明边框，用于悬浮时边框的平滑过渡 */
        /* text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); */
    }
    .card:hover {
        border-color: #7793FF; /* Dark blue border on hover */
    }
    .card-deepxde {
        background-color: #A6CAFE; /* Light blue background */
    }
    .card-deepmd {
        background-color: #A6CAFE; /* Light blue background */
    }
    .card-modulus {
        background-color: #A6CAFE; /* Light blue background */
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #666;
    }
    .text-large {
        font-size: 14px;
    }
    .text-decoration {
        text-decoration: underline;
    }
</style>

## 👀Description

--8<--
./README.md:description
--8<--

## ✨Feature

--8<--
./README.md:feature
--8<--

--8<--
./docs/en/overview.md:panorama
--8<--

## 📝Case List

<style>
    table  th{
        background: #C1E6FE;
    }
</style>

=== "Mathematics"

    | Problem Type | Case Name | Optimization Method | Model Type | Training Method | Dataset | References |
    |-----|---------|-----|---------|----|---------|---------|
    | Helmholtz Equation | [SPINN(Helmholtz3D)](./examples/spinn.md) | Physics-driven | SPINN | Unsupervised Learning | - | [Paper](https://arxiv.org/pdf/2306.15969) |
    | Phase Field Equation | [Allen-Cahn](./examples/allen_cahn.md) | Physics-driven | MLP | Unsupervised Learning | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat) | [Paper](https://arxiv.org/pdf/2402.00326) |
    | Differential Equation | [Laplace Equation](./examples/laplace2d.md) | Physics-driven | MLP | Unsupervised Learning | -        | - |
    | Differential Equation | [Burgers Equation](./examples/deephpms.md) | Physics-driven | MLP | Unsupervised Learning | [Data](https://github.com/maziarraissi/DeepHPMs/tree/master/Data) | [Paper](https://arxiv.org/pdf/1801.06637.pdf) |
    | Differential Equation | [Nonlinear PDE](./examples/pirbn.md) | Physics-driven | PIRBN | Unsupervised Learning | - | [Paper](https://arxiv.org/abs/2304.06234) |
    | Differential Equation | [Lorenz Equation](./examples/lorenz.md) | Data-driven | Transformer-Physx | Supervised Learning | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957) |
    | Differential Equation | [Rössler Equation](./examples/rossler.md) | Data-driven | Transformer-Physx | Supervised Learning | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957) |
    | Operator Learning | [DeepONet](./examples/deeponet.md) | Data-driven | MLP | Supervised Learning | [Data](https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html) | [Paper](https://export.arxiv.org/pdf/1910.03193.pdf) |
    | Differential Equation | [Gradient-Enhanced Physics-Informed PDE Solving](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/gpinn/poisson_1d.py) | Physics-driven | gPINN | Unsupervised Learning | - |  [Paper](https://doi.org/10.1016/j.cma.2022.114823) |
    | Integral Equation | [Volterra Integral Equation](./examples/volterra_ide.md) | Physics-driven | MLP | Unsupervised Learning | - | [Project](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py) |
    | Differential Equation | [Fractional Differential Equation](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/fpde/fractional_poisson_2d.py) | Physics-driven | MLP | Unsupervised Learning | - | - |
    | Optical Soliton | [Optical soliton](./examples/nlsmb.md) | Physics-driven | MLP | Unsupervised Learning | - | [Paper](https://doi.org/10.1007/s11071-023-08824-w)|
    | Optical Rogue Wave | [Optical rogue wave](./examples/nlsmb.md) | Physics-driven | MLP | Unsupervised Learning | - | [Paper](https://doi.org/10.1007/s11071-023-08824-w)|
    | Domain Decomposition | [XPINN](./examples/xpinns.md) | Physics-driven | MLP | Unsupervised Learning | - | [Paper](https://doi.org/10.4208/cicp.OA-2020-0164)|
    | Brusselator Diffusion System | [3D-Brusselator](./examples/brusselator3d.md) | Data-driven | LNO | Supervised Learning | - | [Paper](https://arxiv.org/abs/2303.10528)|
    | Symbolic Regression | [Transformer4SR](./examples/transformer4sr.md) | Data-driven | Transformer | Supervised Learning | - | [Paper](https://arxiv.org/abs/2312.04070)|
    | Operator Learning | [Latent Neural Operator LNO](./examples/latent_no.md) | Data-driven | Transformer | Supervised Learning | - | [Paper](https://arxiv.org/abs/2406.03923)|

=== "Engineering Science"

    | Problem Type | Case Name | Optimization Method | Model Type | Training Method | Dataset | References |
    |-----|---------|-----|---------|----|---------|---------|
    | Vehicle Surface Drag Prediction | [Transolver](./examples/transolver.md) | Data-driven | Transolver | Supervised Learning | [Data](http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip) | [Paper](https://arxiv.org/abs/2402.02366) |
    | Vehicle Surface Drag Prediction | [DrivAerNet](./examples/drivaernet.md) | Data-driven | RegDGCNN | Supervised Learning | [Data](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/data.tar) | [Paper](https://www.researchgate.net/publication/378937154_DrivAerNet_A_Parametric_Car_Dataset_for_Data-Driven_Aerodynamic_Design_and_Graph-Based_Drag_Prediction) |
    | 1D Linear Convection Problem | [1D Linear Convection](./examples/adv_cvit.md) | Data-driven | ViT | Supervised Learning | [Data](https://github.com/Zhengyu-Huang/Operator-Learning/tree/main/data) | [Paper](https://arxiv.org/abs/2405.13998) |
    | Unsteady Incompressible Flow | [2D Buoyancy-Driven Cavity Flow](./examples/ns_cvit.md) | Data-driven | ViT | Supervised Learning | [Data](https://huggingface.co/datasets/pdearena/NavierStokes-2D) | [Paper](https://arxiv.org/abs/2405.13998) |
    | Steady Incompressible Flow | [Re3200 2D Steady Cavity Flow](./examples/ldc2d_steady.md) | Physics-driven | MLP | Unsupervised Learning | - |  |
    | Steady Incompressible Flow | [2D Darcy Flow](./examples/darcy2d.md) | Physics-driven | MLP | Unsupervised Learning | - |   |
    | Steady Incompressible Flow | [2D Pipe Flow](./examples/labelfree_DNN_surrogate.md) | Physics-driven | MLP | Unsupervised Learning | - | [Paper](https://arxiv.org/abs/1906.02382) |
    | Steady Incompressible Flow | [3D Intracranial Aneurysm](./examples/aneurysm.md) | Physics-driven | MLP | Unsupervised Learning | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar) | [Project](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html)|
    | Steady Incompressible Flow | [Arbitrary 2D Geometry Flow](./examples/deepcfd.md) | Data-driven | DeepCFD | Supervised Learning | - | [Paper](https://arxiv.org/abs/2004.08826)|
    | Unsteady Incompressible Flow | [2D Unsteady Cavity Flow](./examples/ldc2d_unsteady.md) | Physics-driven | MLP | Unsupervised Learning | - | -|
    | Unsteady Incompressible Flow | [Re100 2D Cylinder Flow](./examples/cylinder2d_unsteady.md) | Physics-driven | MLP | Semi-supervised Learning | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/cylinder2d_unsteady_Re100/cylinder2d_unsteady_Re100_dataset.tar) | [Paper](https://arxiv.org/abs/2004.08826)|
    | Unsteady Incompressible Flow | [Re100~750 2D Cylinder Flow](./examples/cylinder2d_unsteady_transformer_physx.md) | Data-driven | Transformer-Physx | Supervised Learning | [Data](https://github.com/zabaras/transformer-physx) | [Paper](https://arxiv.org/abs/2010.03957)|
    | Compressible Flow | [2D Air Shock Wave](./examples/shock_wave.md) | Physics-driven | PINN-WE | Unsupervised Learning | [Data](https://aistudio.baidu.com/datasetdetail/167250) | -|
    | Aircraft Design | [MeshGraphNets](https://aistudio.baidu.com/projectdetail/5322713) | Data-driven | GNN | Supervised Learning | [Data](https://aistudio.baidu.com/datasetdetail/184320) | [Paper](https://arxiv.org/abs/2010.03409)|
    | Aircraft Design | [Rocket Engine Vacuum Plume](https://aistudio.baidu.com/projectdetail/4486133) | Data-driven | CNN | Supervised Learning | [Data](https://aistudio.baidu.com/datasetdetail/167250) | - |
    | Aircraft Design | [Deep-Flow-Prediction](https://aistudio.baidu.com/projectdetail/5671596) | Data-driven | TurbNetG | Supervised Learning | [Data](https://aistudio.baidu.com/datasetdetail/197778) | [Paper](https://arxiv.org/abs/1810.08217) |
    | General Flow Simulation | [Aerodynamic Shape Design](./examples/amgnet.md) | Data-driven | AMGNet | Supervised Learning | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip) | [Paper](https://arxiv.org/abs/1810.08217) |
    | Fluid-Structure Interaction | [Vortex-Induced Vibration](./examples/viv.md) | Physics-driven | MLP | Semi-supervised Learning | [Data](https://github.com/PaddlePaddle/PaddleScience/blob/develop/examples/fsi/VIV_Training_Neta100.mat) | [Paper](https://arxiv.org/abs/2206.03864)|
    | Multiphase Flow | [Gas-Liquid Two-Phase Flow](./examples/bubble.md) | Physics-driven | BubbleNet | Semi-supervised Learning | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat) | [Paper](https://pubs.aip.org/aip/adv/article/12/3/035153/2819394/Predicting-micro-bubble-dynamics-with-semi-physics)|
    | Multiphase Flow | [twophasePINN](https://aistudio.baidu.com/projectdetail/5379212) | Physics-driven | MLP | Unsupervised Learning | - | [Paper](https://doi.org/10.1016/j.mlwa.2021.100029)|
    | High-Resolution Flow Field Reconstruction | [2D Turbulent Flow Field Reconstruction](./examples/tempoGAN.md) | Data-driven | tempoGAN | Supervised Learning | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat) | [Paper](https://dl.acm.org/doi/10.1145/3197517.3201304)|
    | High-Resolution Flow Field Reconstruction | [2D Turbulent Flow Field Reconstruction](https://aistudio.baidu.com/projectdetail/4493261?contributionType=1) | Data-driven | cycleGAN | Supervised Learning | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat) | [Paper](https://arxiv.org/abs/2007.15324)|
    | High-Resolution Flow Field Reconstruction | [Global Field Reconstruction from Sparse Sensors via Voronoi Embedding-Assisted Deep Learning](https://aistudio.baidu.com/projectdetail/5807904) | Data-driven | CNN | Supervised Learning | [Data1](https://drive.google.com/drive/folders/1K7upSyHAIVtsyNAqe6P8TY1nS5WpxJ2c)<br>[Data2](https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu)<br>[Data3](https://drive.google.com/drive/folders/1xIY_jIu-hNcRY-TTf4oYX1Xg4_fx8ZvD) | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |
    | Flow Field Prediction | [Catheter](https://aistudio.baidu.com/projectdetail/5379212) | Data-driven | FNO | Supervised Learning | [Data](https://aistudio.baidu.com/datasetdetail/291940) | [Paper](https://www.science.org/doi/pdf/10.1126/sciadv.adj1741) |
    | Solver Coupling | [CFD-GCN](./examples/cfdgcn.md) | Data-driven | GCN | Supervised Learning | [Data](https://aistudio.baidu.com/aistudio/datasetdetail/184778)<br>[Mesh](https://paddle-org.bj.bcebos.com/paddlescience/datasets/CFDGCN/meshes.tar) | [Paper](https://arxiv.org/abs/2007.04439)|
    | Force Analysis | [1D Euler Beam Deformation](./examples/euler_beam.md) | Physics-driven | MLP | Unsupervised Learning | - | - |
    | Force Analysis | [2D Plate Deformation](./examples/biharmonic2d.md) | Physics-driven | MLP | Unsupervised Learning | - | [Paper](https://arxiv.org/abs/2108.07243) |
    | Force Analysis | [3D Bracket Deformation](./examples/bracket.md) | Physics-driven | MLP | Unsupervised Learning | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar) | [Tutorial](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html) |
    | Force Analysis | [Structural Vibration Simulation](./examples/phylstm.md) | Physics-driven | PhyLSTM | Supervised Learning | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyLSTM/data_boucwen.mat) | [Paper](https://arxiv.org/abs/2002.10253) |
    | Force Analysis | [2D Elastoplastic Structure](./examples/epnn.md) | Physics-driven | EPNN | Unsupervised Learning | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat) | [Paper](https://arxiv.org/abs/2204.12088) |
    | Force Analysis and Inverse Problem | [3D Vehicle Control Arm Deformation](./examples/control_arm.md) | Physics-driven | MLP | Unsupervised Learning | - | - |
    | Force Analysis and Inverse Problem | [3D Heart Simulation](./examples/heart.md) | Physics-Data Fusion | PINN | Supervised Learning | - | - |
    | Topology Optimization | [2D Topology Optimization](./examples/topopt.md) | Data-driven | TopOptNN | Supervised Learning | [Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5) | [Paper](https://arxiv.org/pdf/1709.09578) |
    | Topology Optimization | [2/3D Topology Optimization](./examples/ntopo.md) | Physics-driven | DenseSIRENModel | Unsupervised Learning | - | [Paper](https://arxiv.org/abs/2102.10782) |
    | Thermal Simulation | [1D Heat Exchanger Thermal Simulation](./examples/heat_exchanger.md) | Physics-driven | PI-DeepONet | Unsupervised Learning | - | - |
    | Thermal Simulation | [2D Thermal Simulation](./examples/heat_pinn.md) | Physics-driven | PINN | Unsupervised Learning | - | [Paper](https://arxiv.org/abs/1711.10561)|
    | Thermal Simulation | [2D Chip Thermal Simulation](./examples/chip_heat.md) | Physics-driven | PI-DeepONet | Unsupervised Learning | - | [Paper](https://doi.org/10.1063/5.0194245)|

=== "Materials Science"

    | Problem Type | Case Name | Optimization Method | Model Type | Training Method | Dataset | References |
    |-----|---------|-----|---------|----|---------|---------|
    | Material Design | [Diffuser Design (Inverse Problem)](./examples/hpinns.md) | Physics-driven | Transformer | Unsupervised Learning | [Train Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat)<br>[Eval Data](https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat) | [Paper](https://arxiv.org/pdf/2102.04626.pdf) |
    | Crystal Material Property Prediction | [CGCNN](./examples/cgcnn.md) | Data-driven | GNN | Supervised Learning | [MP](https://next-gen.materialsproject.org/) / [Perovskite](https://cmr.fysik.dtu.dk/cubic_perovskites/cubic_perovskites.html) / [C2DB](https://cmr.fysik.dtu.dk/c2db/c2db.html) / [test](https://paddle-org.bj.bcebos.com/paddlescience/datasets/cgcnn/cgcnn-test.zip) | [Paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) |
    | 2D Material Generation and Database | [ML2DDB](./examples/ml2ddb.md) | Data-driven | GNN/Diffusion | Supervised Learning | Coming Soon | [Paper](https://arxiv.org/pdf/2507.00584) |

=== "Earth Sciences"

    | Problem Type | Case Name | Optimization Method | Model Type | Training Method | Dataset | References |
    |-----|---------|-----|---------|----|---------|---------|
    | Meteorological Downscaling | [KMCast](./examples/kmcast.md) | Data-driven | Diffusion | Supervised Learning | [GFS](https://rda.ucar.edu/datasets/d084006/) | - |
    | Weather Forecasting | [Extformer-MoE Weather Forecasting](./examples/extformer_moe.md) | Data-driven | Transformer | Supervised Learning | [enso](https://tianchi.aliyun.com/dataset/98942) | - |
    | Weather Forecasting | [FourCastNet Weather Forecasting](./examples/fourcastnet.md) | Data-driven | AFNO | Supervised Learning | [ERA5](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F) | [Paper](https://arxiv.org/pdf/2202.11214.pdf) |
    | Weather Forecasting | [NowCastNet Weather Forecasting](./examples/nowcastnet.md) | Data-driven | GAN | Supervised Learning | [MRMS](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F) | [Paper](https://www.nature.com/articles/s41586-023-06184-4) |
    | Weather Forecasting | [GraphCast Weather Forecasting](./examples/graphcast.md) | Data-driven | GNN | Supervised Learning | - | [Paper](https://arxiv.org/abs/2212.12794) |
    | Weather Forecasting | [GenCast Weather Forecasting](./examples/gencast.md) | Data-driven | Diffusion+Graph transformer | Supervised Learning | [Gencast](https://console.cloud.google.com/storage/browser/dm_graphcast) | [Paper](https://arxiv.org/abs/2312.15796) |
    | Weather Forecasting | [Fuxi Weather Forecasting](./examples/fuxi.md) | Data-driven | Transformer | Supervised Learning | - | [Paper](https://arxiv.org/abs/2306.12873) |
    | Weather Forecasting | [FengWu Weather Forecasting](./examples/fengwu.md) | Data-driven | Transformer | Supervised Learning | - | [Paper](https://arxiv.org/pdf/2304.02948) |
    | Weather Forecasting | [Pangu-Weather Weather Forecasting](./examples/pangu_weather.md) | Data-driven | Transformer | Supervised Learning | - | [Paper](https://arxiv.org/pdf/2211.02556) |
    | Atmospheric Pollutants | [UNet Pollutant Diffusion](https://aistudio.baidu.com/projectdetail/5663515?channel=0&channelType=0&sUid=438690&shared=1&ts=1698221963752) | Data-driven | UNet | Supervised Learning | [Data](https://aistudio.baidu.com/datasetdetail/198102) | - |
    | Atmospheric Pollutants | [STAFNet Pollutant Concentration Prediction](./examples/stafnet.md) | Data-driven | STAFNet | Supervised Learning | [Data](https://quotsoft.net/air) | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-78186-5_22) |
    | Weather Forecasting | [DGMR Weather Forecasting](./examples/dgmr.md) | Data-driven | GAN | Supervised Learning | [UK dataset](https://huggingface.co/datasets/openclimatefix/nimrod-uk-1km) | [Paper](https://arxiv.org/pdf/2104.00954.pdf) |
    | Seismic Waveform Inversion | [VelocityGAN Seismic Waveform Inversion](./examples/velocity_gan.md) | Data-driven | VelocityGAN | Supervised Learning | [OpenFWI](https://openfwi-lanl.github.io/docs/data.html#vel) | [Paper](https://arxiv.org/abs/1809.10262v6) |
    | Remote Sensing Image Segmentation | [UNetFormer Image Segmentation](./examples/unetformer.md) | Data-driven | UNetformer | Supervised Learning | [Vaihingen](https://paperswithcode.com/dataset/isprs-vaihingen) | [Paper](https://github.com/WangLibo1995/GeoSeg) |
    | Traffic Prediction | [TGCN Traffic Flow Prediction](./examples/tgcn.md) | Data-driven | GCN & CNN | Supervised Learning | [PEMSD4 & PEMSD8](https://paddle-org.bj.bcebos.com/paddlescience/datasets/tgcn/tgcn_data.zip) | - |
    | Weather Forecasting | [Meteoformer Multi-Meteorological Element Prediction](./examples/meteoformer.md) | Data-driven | Transformer | Supervised Learning | [ERA5](https://https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download) | - |
    | Weather Forecasting | [Preformer Short-Term Precipitation Prediction](./examples/preformer.md) | Data-driven | Transformer | Supervised Learning | [ERA5](https://https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download) | [Paper](https://ieeexplore.ieee.org/document/10288072) |
    | Weather Forecasting | [Climateformer Climate Prediction](./examples/climateformer.md) | Data-driven | Transformer | Supervised Learning | [ERA5](https://https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download) | - |
    | Generative Model | [Gradient Penalty Application in Image Generation](./examples/wgan_gp.md) | Data-driven | WGAN GP | Supervised Learning | [Data1](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)<br>[Data2](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz) | [Paper](https://github.com/igul222/improved_wgan_training) |
    | Remote Sensing Image Segmentation | [UTAE Remote Sensing Time Series Semantic/Panoptic Segmentation](./examples/UTAE.md) | Data-driven | UTAE | Supervised Learning | [PASTIS](https://zenodo.org/records/5012942) | [Paper](https://arxiv.org/abs/2107.07933) |

=== "Chemical Sciences"

    | Problem Type | Case Name | Optimization Method | Model Type | Training Method | Dataset | References |
    |-----|---------|-----|---------|----|---------|---------|
    | Chemical Molecule Generation | [Moflow](./examples/moflow.md) | Data-driven | moflow | Supervised Learning | qm9/ zink250k | [MoFlow: An Invertible Flow Model for Generating Molecular Graphs](https://arxiv.org/abs/2006.10137v1) |
    | Chemical Reaction Prediction | [IFM](./examples/ifm.md) | Data-driven | IFM-MLP | Supervised Learning | tox21/sider/hiv/bace/bbbp | [Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions](https://openreview.net/pdf?id=NLFqlDeuzt) |

## 🚀Quick Installation

=== "Method 1: Source Code Installation [Recommended]"

    --8<--
    ./README.md:git_install
    --8<--

=== "Method 2: pip Installation"

    ``` sh
    python -m pip install -U paddlesci
    ```

**Complete Installation Guide**: [Installation and Setup](./install_setup.md)

## 🕘Recent Updates

--8<--
./README.md:update
--8<--

## 🎈Ecosystem Tools

--8<--
./README.md:adaptation
--8<--

## 💬Support

--8<--
./README.md:support
--8<--

## 👫Contribution

--8<--
./README.md:contribution
--8<--

## 🎯Collaboration

--8<--
./README.md:collaboration
--8<--

## ❤️Thanks

--8<--
./README.md:thanks
--8<--

- Part of PaddleScience's code is contributed by the following outstanding developers (sorted by [Contributors](https://github.com/PaddlePaddle/PaddleScience/graphs/contributors)):

    <style>
        .avatar {
            height: 64px;
            width: 64px;
            border: 2px solid rgba(128, 128, 128, 0.308);
            border-radius: 50%;
        }

        .avatar:hover {
            box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.4);
            transition: 0.4s;
            transform:translateY(-10px);
        }
    </style>
    <div id="contributors"></div>

## 🤝Partner Organizations

![cooperation](./images/overview/cooperation.png)

## 📜License

--8<--
./README.md:license
--8<--

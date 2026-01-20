# DGMR (Deep Generative Models of Radar)

=== "Model Training Command"

    None

=== "Model Evaluation Command"

    ``` sh
    # Download data from Huggingface
    mkdir openclimatefix/nimrod-uk-1km/20200718/valid/subsampled_tiles_256_20min_stride
    cd openclimatefix/nimrod-uk-1km/20200718/valid/subsampled_tiles_256_20min_stride
    git lfs install
    git lfs pull --include="seq-24-*-of-00033.tfrecord.gz"

    python dgmr.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/dgmr/dgmr_pretrained.pdparams
    ```

| Pretrained Model | Metrics |
|:--| :--|
| [dgmr_pretrained.pdparams](https://paddle-org.bj.bcebos.com/paddlescience/models/dgmr/dgmr_pretrained.pdparams) | d_loss: 127.041<br>g_loss: 59.2409<br>grid_loss: 1.7699 |

## 1. Background Introduction

Precipitation nowcasting is a high-resolution forecast of precipitation within the next two hours, supporting many practical socio-economic needs that rely on weather decisions. State-of-the-art operational nowcasting methods typically use radar-based wind estimates to advect precipitation fields, but often struggle to capture important nonlinear events such as convective initiation. Recently introduced deep learning methods use radar to directly predict future rainfall rates, freeing themselves from physical constraints. While they are able to accurately predict low-intensity rainfall, due to the lack of constraints, they produce blurry nowcasts at longer lead times, resulting in poor performance on medium to heavy rain events and limiting their operational utility. To address these challenges, we propose a deep generative model for radar-based probabilistic precipitation nowcasting. Our model is capable of generating realistic and spatiotemporally consistent predictions at lead times of 5-90 minutes over a region of 1536 km × 1280 km. Systematic evaluation by more than fifty expert forecasters from the Met Office showed that our generative model ranked first in accuracy and utility in 88% of cases, outperforming two competing methods, demonstrating its ability in decision-making value and providing physical insight to real-world experts. In terms of quantitative verification, these nowcasts have good skill without blurring. We show that generative nowcasting can provide probabilistic forecasts that improve forecast value and support operational utility, especially where alternative methods struggle in terms of resolution and lead time.

Precipitation nowcasting is important in engineering and science in many ways, mainly in the following aspects:

- Social impact: Precipitation nowcasting has a direct social impact on practical decisions in all walks of life. For example, agriculture, water resources management, urban flood control and transportation all require accurate precipitation forecasts to take corresponding countermeasures to reduce potential losses and risks.
- Security guarantee: Precipitation nowcasting is essential to ensure public safety. For example, early warning systems can notify people in advance of possible heavy rain, floods, debris flows and other disasters based on precipitation nowcasting, so as to take timely evacuation measures to reduce casualties and property losses.
- Ecological environment protection: Accurate forecasting of precipitation helps to protect and manage the ecological environment. For example, predicting precipitation can help decision-makers adjust the operation of water conservancy projects in time, ensure the healthy operation of ecosystems, and provide necessary water resources for vegetation growth.
- Scientific research: Precipitation nowcasting is also an important part of meteorological scientific research. Through the study and prediction of precipitation processes, we can better understand the water cycle process in the atmospheric environment, and provide important data and support for research in meteorology, climatology and other related fields.
- Engineering planning and design: In the fields of urban planning, civil engineering, agricultural irrigation, etc., accurate precipitation nowcasting is essential for engineering planning and design. For example, in urban drainage system design, it is necessary to consider the precipitation that may occur in the future short period of time to ensure the normal operation of the drainage system and the flood control capacity of the city.

In general, the importance of precipitation nowcasting in engineering and science is reflected in ensuring social security, promoting scientific research, supporting ecological environment protection and promoting engineering development, which is of great significance for the development of various industries and the sustainable development of society.

## 2. Model Principle

### 2.1 Model Structure

DGMR is built within the algorithm framework of conditional generative adversarial networks. Based on past radar data, detailed and credible predictions are made for future radar. That is, at a given time point $T$, using radar-based surface precipitation estimates $X_T$, future $N$ radar fields are predicted based on past $M$ radar fields. i.e.:

$$
P\left(X_{M+1: M+N} \mid X_{1: M}\right)=\int P\left(X_{M+1: M+N} \mid \mathrm{Z}, X_{1: M}, \boldsymbol{\theta}\right) P\left(\mathrm{Z} \mid X_{1: M}\right) d \mathrm{Z}.
$$

Where $Z$ is a random vector and $\theta$ is the parameter of the generative model.

- The left side of the equation is the conditional probability, given the radar precipitation at the past $M$ moments, predicting the radar precipitation at the next $N$ moments.
- The right side writes the probability as an integral form of ensemble forecasting:
  - Given random sampling $Z$ and generation network parameter $\theta$, predict the radar precipitation at the next $N$ moments under the constraint of radar precipitation at the past $M$ moments;
  - Calculate the conditional probability of random sampling $Z$ under the constraint of radar precipitation at the past $M$ moments;
  - Multiply the two to be the probability of occurrence of the result, and integrate to obtain the ensemble forecast under multiple samplings.

Integration over the random vector $Z$ ensures that the predictions produced by the model are spatially correlated. DGMR is specifically used for precipitation prediction problems. Four consecutive radar observations (previous 20 minutes) are used as input to the generator, which allows sampling of multiple realizations of future precipitation, each containing 18 frames (90 minutes). The schematic diagram of the model architecture is shown in the figure.

<figure markdown>
  ![domain_chip.pdf](https://paddle-org.bj.bcebos.com/paddlescience/docs/DGMR/dgmr.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> Schematic diagram of model architecture. </figcaption>
</figure>

DGMR is a generator trained using two discriminators and an additional regularization term. The figure below shows a detailed schematic of the generative model and discriminators:

<figure markdown>
  ![domain_chip.pdf](https://paddle-org.bj.bcebos.com/paddlescience/docs/DGMR/g_d.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> a, Generator architecture. b, Generator temporal discriminator architecture (top left), spatial discriminator (middle left), and latent conditioning stack (bottom left). On the right are the architectures of the G block (top), D and 3D blocks (middle), and L block (right). </figcaption>
</figure>

### 2.2 Objective Function

The generator is trained by the loss of two discriminators and a grid cell regularization term (denoted as $\mathcal{L}_R(\theta)$). The spatial discriminator $D\phi$ has parameters $\phi$, the temporal discriminator $T_\psi$ has parameters $\psi$, and the generator $G_\theta$ has parameters $\theta$. We use the notation $\{X ; G\}$ to denote the concatenation of two fields. The maximized generator loss is as follows:

$$
\begin{gathered}
\mathcal{L}_G(\theta)=\mathbb{E}_{X_{1: M+N}}\left[\mathbb{E}_Z\left[D\left(G_\theta\left(Z ; X_{1: M}\right)\right)+T\left(\left\{X_{1: M} ; G_\theta\left(Z ; X_{1: M}\right)\right\}\right)\right]-\lambda \mathcal{L}_R(\theta)\right] ; \\
\mathcal{L}_R(\theta)=\frac{1}{H W N}\left\|\left(\mathbb{E}_Z\left[G_\theta\left(Z ; X_{1: M}\right)\right]-X_{M+1: M+N}\right) \odot w\left(X_{M+1: M+N}\right)\right\|_1 .
\end{gathered}
$$

We use Monte Carlo estimation for the expectation of the latent variable $\mathrm{Z}$ in the above formula. These estimates are calculated using six samples for each input $X_{1: M}$, which includes $M=4$ radar observations. The grid cell regularization term ensures that the average prediction remains close to the true value and is averaged over all grid cells on the height $H$, width $W$, and lead time $N$ axes. It weights to higher rainfall targets via the function $w(y)=\max (y+1,24)$, which operates element-wise on the input vector and truncates at 24 to improve robustness to anomalously large values in radar. The GAN spatial discriminator loss $\mathcal{L}_D(\phi)$ and temporal discriminator loss $\mathcal{L}_T(\psi)$ are minimized with respect to parameters $\phi$ and $\psi$ respectively. The discriminator loss uses the hinge loss formula:

$$
\begin{aligned}
& \mathcal{L}_D(\phi)=\mathbb{E}_{X_{1: M+N}, Z}\left[\operatorname{ReLU}\left(1-D_\phi\left(X_{M+1: M+N}\right)\right)+\operatorname{ReLU}\left(1+D_\phi\left(G\left(Z ; X_{1: M}\right)\right)\right)\right], \\
& \mathcal{L}_T(\psi)=\mathbb{E}_{X_{1: M+N}, Z}\left[\operatorname{ReLU}\left(1-T_\psi\left(X_{1: M+N}\right)\right)+\operatorname{ReLU}\left(1+T_\psi\left(\left\{X_{1: M} ; G\left(Z ; X_{1: M}\right)\right\}\right)\right)\right],
\end{aligned}
$$

Where $\operatorname{ReLU} = \max(0,x)$. For more detailed theoretical derivation, please refer to [Skillful Precipitation Nowcasting using Deep Generative Models of Radar](https://arxiv.org/pdf/2104.00954.pdf).

## 3. Problem Solving

Next, we will explain how to convert this problem into PaddleScience code step by step and use DGMR to predict precipitation nowcasting. In order to quickly understand PaddleScience, only key steps such as model construction and constraint construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

To train and evaluate the UK nowcasting model, DGMR uses radar composite data from the Met Office RadarNet4 network. Radar data collected every five minutes between January 1, 2016 and December 31, 2019 are used. We use the following data splits for model development. Fields from the first day of each month from 2016 to 2018 are assigned to the validation set. All other dates from 2016 to 2018 are assigned to the training set. Finally, data from 2019 is used as a test set to prevent data leakage and distribution generalization testing. The open source UK training dataset has been mirrored to [HuggingFace Dataset](https://huggingface.co/datasets/openclimatefix/nimrod-uk-1km), and users can download and use it themselves.

In the PaddleScience code of this model, we can call `ppsci.data.dataset.DGMRDataset` to load the dataset, the code is as follows:

``` py linenums="199"
--8<--
examples/dgmr/dgmr.py:199:201
--8<--
```

### 3.2 Model Construction

In the DGMR model, the past four radar field data are input to predict 18 future radar fields (the next 90 minutes). The DGMR network can be represented as a mapping function $f$ from $X_{1:4}$ to output $X_{5:22}$, i.e.:

$$
X_{5:22} = f(X_{1:4}),\\
$$

In the above formula, $f$ represents the DGMR model. We define the DGMR model class built in PaddleScience and call it, expressed in PaddleScience code as follows

``` py linenums="197"
--8<--
examples/dgmr/dgmr.py:197:198
--8<--
```

In this way, we have instantiated a DGMR model. For model parameter settings, please refer to the document [Skillful Precipitation Nowcasting using Deep Generative Models of Radar](https://arxiv.org/pdf/2104.00954.pdf).

### 3.3 Model Evaluation and Visualization

After the model is built and data is loaded, pass the instantiated objects to `ppsci.solver.Solver` in order, and then start evaluation and visualization. First, we initialize the model:

``` py linenums="202"
--8<--
examples/dgmr/dgmr.py:202:207
--8<--
```

Then customize the evaluation method and loss function, the code is as follows:

``` py linenums="89"
--8<--
examples/dgmr/dgmr.py:89:189
--8<--
```

Finally, traverse and evaluate each batch in the data, and visualize the prediction results.

``` py linenums="209"
--8<--
examples/dgmr/dgmr.py:209:232
--8<--
```

## 4. Complete Code

``` py linenums="1" title="dgmr.py"
--8<--
examples/dgmr/dgmr.py
--8<--
```

## 5. Result Display

The figure shows the meteorological precipitation forecast at $T+5, T+10, \cdots, T+45$ minutes respectively. Compared with the real precipitation situation, it can be seen that the model can provide relatively good precipitation nowcasting.

<figure markdown>
  ![chip.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/DGMR/Generated_Image_Frame.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> Precipitation situation predicted by the model</figcaption>
</figure>

<figure markdown>
  ![chip.png](https://paddle-org.bj.bcebos.com/paddlescience/docs/DGMR/Target_Image_Frame.png){ loading=lazy style="height:80%;width:80%" align="center" }
  <figcaption> Real precipitation situation</figcaption>
</figure>

## 6. References

References: [Skillful Precipitation Nowcasting using Deep Generative Models of Radar](https://arxiv.org/pdf/2104.00954.pdf)

Reference Code: [DGMR](https://github.com/openclimatefix/skillful_nowcasting)

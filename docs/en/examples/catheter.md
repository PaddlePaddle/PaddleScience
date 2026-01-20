# AI-aided geometric design of anti-infection catheters

Distributed under a creative commons Attribution license 4.0 (CC BY).

## 1. Background Introduction
### 1.1 Paper Information
| Year           | Journal            | Authors                                                                                             | Citations | Paper PDF                                                                                                                                                                                                                                                                                                                                                                 |
| -------------- | --------------- | ------------------------------------------------------------------------------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3 January 2024 | Science Advance | Tingtao Zhou, X Wan, DZ Huang, Zongyi Li, Z Peng, A Anandkumar, JF Brady, PW Sternberg, C Daraio | 15     | [Paper](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/2024 AI-aided geometric design of anti-infection catheters.pdf), [Supplementary PDF 1](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/sciadv.adj1741_sm.pdf) |

### 1.2 Author Introduction

- First Author: Tingtao Zhou, California Institute of Technology <br> Research Areas: Statistical Physics, Fluid Mechanics, Active Matter, Disordered Materials <br> ![alt text](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheter5.png)

- Corresponding Author: Chiara Daraio (Cited 21038), Division of Engineering and Applied Science, California Institute of Technology <br> Faculty Homepage: https://www.eas.caltech.edu/people/daraio <br> Research Areas: Mechanics, Materials, Nonlinear Dynamics, Soft Matter, Biomaterials <br> ![alt text](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheter6.png)

- Corresponding Author: Paul W. Sternberg (Cited 56555), Division of Biology and Biological Engineering, California Institute of Technology <br> Faculty Homepage: https://www.bbe.caltech.edu/people/paul-w-sternberg <br> Research Areas: Systems biology of C. elegans development; Neural circuits underlying sex and sleep; Nematode functional genomics and chemical ecology; Text mining. <br> ![alt text](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheter7.png)

- Other Authors' Affiliations <br> California Institute of Technology, Division of Engineering and Applied Science\Division of Chemistry and Chemical Engineering\Division of Biology and Biological Engineering <br> Peking University, Beijing International Center for Mathematical Research <br> Meta Platforms Inc. (formerly Facebook), Reality Labs

### 1.3 Model & Reproduction Code

| Problem Type             | Online Run                                                                                                                   | Neural Network           | Pretrained Model                                                                                                                                                              | Metrics              |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| Operator Neural Network Predicting Flow Field | [AI-aided geometric design of anti-infection catheters](https://aistudio.baidu.com/projectdetail/8252779?sUid=1952564&shared=1&ts=172724369783) | Geo-FNO | [GeoFNO_pretrained.pdparams](<https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/result_GeoFNO.pdparams>) | loss(MAE): 0.0664 |


=== "Model Training Command"

    ``` sh
    # linux
    wget -c 'https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/data.zip'
    # windows
    # curl 'https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/data.zip' -o data.zip
    unzip data.zip
    python catheter.py
    ```

=== "Pretrained Model Fast Evaluation"

    ``` sh
    python catheter.py mode=eval EVAL.pretrained_model=https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/result_GeoFNO.pdparams
    ```

In the fluid environment within narrow tubes, bacteria can migrate upstream with the help of hydrodynamic interactions, posing a serious threat of urinary tract infection to patients using indwelling catheters. Although coatings and structured surfaces have been proposed to inhibit bacterial growth in catheters, unfortunately, no surface structure or coating technology has fundamentally solved the contamination problem so far. In view of this, based on the physical principle of upstream swimming, we innovatively proposed a geometric design scheme and predicted and optimized bacterial inflow dynamics through an AI model. Compared with traditional simulation methods, the adopted Fourier Neural Operator AI technology achieved significant speed improvement.

In quasi-2D microfluidic experiments, we used E. coli as the object to verify the anti-infection mechanism of this design, and evaluated its effectiveness through 3D printed catheter prototypes at clinically relevant flow rates. The experimental results show that our catheter design achieved an improvement of 1-2 orders of magnitude in suppressing bacterial contamination at the upstream end of the catheter, which is expected to significantly prolong the safe indwelling time of the catheter and reduce the risk of catheter-associated urinary tract infection as a whole.

## 2. Problem Definition

Catheter-associated urinary tract infections (CAUTIs) (1-5) are one of the most common infections in hospitalized patients, causing a loss of about $30 million annually (6). From a material/device perspective, previous methods to prevent such infections include impregnating catheters with antibacterial silver nanoparticles (7) or using antibiotic lock solutions, anti-adhesive or antibacterial materials (8, 9). However, the effects of these methods have not surpassed strict nursing procedures, and the current focus of preventing CAUTI in clinical practice is to reduce the indwelling time of catheters to prevent infection. Designing a catheter that can reduce bacterial motility in the presence of fluid will bring significant improvements to current CAUTI management.

Such a design requires us to understand the movement patterns of microorganisms in fluid flow under restricted conditions. Typical microbial trajectories alternate between running (straight propulsion) and tumbling (random change of direction) to explore the environment (10-13). Hydrodynamic interactions and quorum sensing lead to more complex dynamic behaviors, such as enhanced attraction to surfaces (14, 15) and collective swarming motion (16-19). In shear flow, microscopic run-and-tumble (RTP) motion can lead to macroscopic upstream swimming (20-27). Typically, passive particles are convected downstream in addition to diffusion (28). However, the self-propulsion of microorganisms leads to a qualitative difference in their macroscopic transport: bacterial bodies are rotated by fluid vorticity when traversing the tube, causing them to swim against the flow direction. Both biological microswimmers and synthetic active particles exhibit upstream motility. For biological microswimmers, such as E. coli and mammalian sperm, their fore-aft body asymmetry and the resulting hydrodynamic interactions with walls are often used to explain their upstream swimming behavior (20, 21, 25, 29-31).

On the other hand, for point-like active particles of negligible size, upstream swimming phenomenon still exists (24, 27). Consider the case of a point-like active particle approaching a wall: its front end must point to the wall. Near the wall, the vorticity of Poiseuille flow (at its maximum) always reorients the particles to the upstream direction (see also Materials and Methods section) (27), and then they swim upstream along the wall (Figure 1, A and B). Many other factors, such as body asymmetry, flagellar chirality, and hydrodynamic interactions between bacteria and boundaries, also affect upstream swimming behavior. Recent experiments (32) have demonstrated the super-contamination phenomenon of E. coli in microfluidic channels, highlighting the importance of its power-law run-time distribution, which significantly enhances the tendency of bacteria to swim upstream, allowing bacteria to continuously swim against the flow direction.

Mainstream strategies for preventing bacterial contamination include:

- (i) Physical barriers, such as filters or membranes (33-38);

- (ii) Antimicrobial agents, such as antibiotics (36, 37);

- (iii) Surface modification of medical devices to reduce bacterial adhesion and biofilm formation (38-44);

- (iv) Control of physical/chemical environment, such as high/low temperature, low oxygen levels or use of disinfectants to inhibit bacterial growth and survival (45-48);

- (v) Strict disinfection procedures, such as wearing gloves and gowns (49-51);

- (vi) Regular monitoring of patient conditions for early detection and treatment of bacterial contamination (52-54).

Although various surface modifications or coatings have been proposed to reduce bacterial adhesion, no studies have shown that they effectively prevent upstream swimming or catheter contamination (38-40). Other passive antibacterial methods, such as membranes or filtration, may be difficult to apply directly to patients with indwelling catheters.

Compared with antibiotics or other chemical methods, controlling microbial distribution through geometry is safer in terms of antibiotic resistance (55-58). In other cases, specific shapes have been used to confine and trap unwanted bacteria (59). Due to the "dry" geometric rectification effect, asymmetric shapes can also affect the partitioning of motile bacteria (60, 61), and extruded boundary shapes can locally enhance the vorticity of Poiseuille flow, the enhancement being proportional to the extrusion curvature.

We are committed to designing catheters that prevent bacteria from swimming upstream and minimize contamination. To optimize the catheter geometry, we limit the design space to placing triangular obstacles on the inner wall of the catheter. We captured the simplest upstream swimming physical mechanism exhibited by self-propelled spheres (27) and performed fluid and particle dynamics simulations to identify geometric design principles (Figure 1C). We modeled bacterial distribution by coupling fluid dynamics and geometric rectification effects into a stochastic partial differential equation (SPDE). Then, we trained an artificial intelligence (AI) model based on geometric Fourier Neural Operator (Geo-FNO) using simulation data (62, 63) to learn the solution of SPDE, and used the trained model to optimize the catheter geometry (Figure 1D). Based on the optimized design, we fabricated quasi-two-dimensional (2D) microfluidic devices (Figure 1E) and 3D printed prototype catheters (Figure 1F) to evaluate the effectiveness of our concept. Experimental results show that compared with our standard catheters, bacterial super-contamination inhibition is improved by up to two orders of magnitude, providing a new way for the management of catheter-associated urinary tract infections (CAUTI).

![Figure 1. Schematic of proposed CAUTI mechanism and anti-infection design process](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheter.png)

**Figure 1. Schematic of proposed CAUTI mechanism and anti-infection design process**

- **(A) Proposed CAUTI mechanism**: When urine flows out from the patient's bladder through the catheter, bacteria can swim against the urine flow direction (i.e., upstream), which may invade the patient's body and cause infection.
- **(B) Run-and-tumble motion and upstream swimming mechanism of bacteria**: Bacteria achieve upstream swimming in fluid environments through a unique run-and-tumble motion pattern.
- **(C) Simulation exploration of catheter shapes**: Using simulation technology to explore the impact of different catheter shapes on bacterial upstream swimming, in order to find catheter designs that can inhibit bacterial upstream swimming.
- **(D) AI-aided optimization**: Using Geo-FNO framework for AI-aided optimization to further refine catheter design parameters and improve its inhibitory effect on bacterial upstream swimming.
- **(E) 2D channel microfluidic experiment**: Experimental verification of the optimized catheter design in 2D microfluidic channels to evaluate its anti-infection performance in real fluid environments.
- **(F) 3D experimental verification**: Using designed actual size catheters for 3D experiments to further verify its anti-infection effect under clinical use conditions.

We are committed to designing catheters that prevent bacteria from moving upstream and minimize contamination. To optimize the catheter geometry, we limited the design space to arranging triangular obstacles on the inner wall of the catheter. We captured the simplest upstream swimming physical mechanism exhibited by self-propelled spheres (27) and conducted fluid and particle dynamics simulations to find geometric design principles (Figure 1C). By coupling fluid dynamics and geometric rectification effects into a stochastic partial differential equation (SPDE), we modeled the bacterial distribution. Subsequently, we used simulation data to train an artificial intelligence (AI) model based on Geometric Fourier Neural Operator (Geo-FNO) (62, 63) to learn the solution of SPDE, and utilized the trained model to optimize the catheter geometry (Figure 1D). Based on the optimized design, we fabricated quasi-two-dimensional (2D) microfluidic devices (Figure 1E) and 3D printed prototype catheters (Figure 1F) to evaluate the effectiveness of our design concept. Experimental results show that compared with standard catheters, our design improves the inhibition of bacterial over-contamination by up to two orders of magnitude, providing a new avenue for the management of catheter-associated urinary tract infections (CAUTI).

### 2.1 Exploring Microscopic Mechanisms

We adopted a simple model (27) to describe the dynamic behavior of bacteria in shear flow. In this model, bacteria are approximated as spheres of negligible size, and their orientation $q$ is given by the following equation:
Based on the physical mechanism of bacterial upstream swimming, a corresponding mathematical model is established, usually represented by the ABP model:

$$
\frac{d\vec{q}}{dt} = \frac{1}{2} \vec{\omega} + \frac{2}{\tau_R} \eta(t) \times \vec{q}
$$

This model considers hydrodynamic interactions between bacteria and catheter walls, as well as factors such as bacterial shape, size, and surface properties.
Where

- $dt(q)$ represents the rate of change of bacterial orientation
- $ω$ represents local fluid vorticity
- $η(t)$ represents Gaussian noise, satisfying $<η(t)>=0$ and $<η(0)η(t)>=\delta(t)I$
- $\vec{q}$ represents bacterial orientation vector
- $\tau_R$ represents average run time (more details in supplementary materials)

We first investigated the role of traditional surface modification methods, such as antimicrobial nanoparticle coatings (36, 42), engineered roughness, or hydrophobic treatments (65, 66), in inhibiting bacterial upstream swimming through numerical simulations. These modified surfaces prevent bacteria from getting too close to the wall. To simulate the presence of these surfaces, we assumed that they cause bacteria to detach from the surface and remain at least 3 μm away from the surface, a distance exceeding the typical body length of E. coli (1 to 2 μm). Although surface modifications may also affect hydrodynamic interactions between bacteria and walls, this was ignored in our simple general model based on point-like spheres.

We found that within the tested flow rate range, surface repulsion had little effect on bacterial upstream swimming behavior. By comparing simulated trajectories of continuously swimming bacteria in smooth channels (Figure 2D) and surface-modified channels (Figure 2E), we found their upstream swimming behaviors to be similar.

We used two population statistical indicators to quantify the effectiveness of inhibiting bacterial upstream swimming:

- (i) Average upstream swimming distance $x_{up}=-\int_{0}^{-\infty}\rho(x)xdx$, calculated by the weighted average of bacterial distribution function $ρ(x)$, where $x$ is bacterial location;

- (ii) Distance reached by the top $1\%$ furthest upstream swimming bacteria $x_{1\%}$. Simulation results show that surface modification only slightly reduced $x_{up}$ at medium flow rates, but had almost no effect on $x_{1\%}$ (blue and pink lines in Figure 2F). This result of poor surface modification effect is consistent with experimental observations in several recent papers (39, 40).

Subsequently, we explored the role of catheter surface geometry by adding physical obstacles. We found that both symmetric and asymmetric obstacles significantly inhibited bacterial upstream swimming (as shown by black and green lines in Figure 2F). We identified two synergistic effects: First, the slope of the obstacle changes the swimming direction of bacteria when they depart from the top of the obstacle, thereby interrupting their continuous climbing along the wall surface. Asymmetric shapes bias bacterial motion downstream (as shown in Figure 2A), which is reflected in simulated trajectories at zero flow rate (Supplementary Materials and Figure S1) and differences in upstream swimming statistics at low flow rates (black and green lines in Figure 2F). Second, at finite flow rates, the flow field is different from Poiseuille flow in smooth channels (as shown in Figure 2B). In Poiseuille flow, vorticity turns bacteria downstream. However, near obstacles, vorticity is enhanced, causing bacteria to turn upstream (as shown in Figure 2C and Supplementary Figure S2), thereby reinforcing the bacterial turning mechanism. Combining these two effects, we expect significant reduction in bacterial upstream swimming in channels with optimized obstacle geometry.

The parameter space for design optimization is characterized by four parameters: obstacle base length $L$, height $h$, tip position $s$, and obstacle spacing $d$; we denote channel width by W (Figure 2G). To optimize this space, we set two constraints. First, if adjacent obstacles are too close, the vortices at their tips begin to overlap. Due to this overlap, the maximum effective vortex strength (right at the obstacle tip; mathematical definition of effective vortex see supplementary materials) and the effective size of the vortex decrease. In addition, larger boundary layers and stagnation zones are formed (Figure S2, A and B). Therefore, we constrain the obstacle spacing to $d > 0.5W$ (Figure S2G). Second, with other parameters fixed, as h increases, the effective vortex strength at the obstacle tip also increases (Figure S2, C to H), which facilitates the vortex reorientation effect. However, when $h = W/2$, the pipe clearly becomes clogged. This trend of increased clogging with increasing $h$ is reflected in the continuous increase in pressure drop required to maintain the same effective flow rate (Figure S2I). To avoid clogging, we constrain the height to $h < 0.3W$.

![Figure 2. Physical mechanisms of obstacle inhibition of upstream swimming and geometric optimization](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheter2.png)

**Figure 2. Physical mechanisms of obstacle inhibition of upstream swimming and geometric optimization**

- **(A) Geometric rectification effect without flow**: Describes the effect of geometry on bacterial swimming direction in the absence of fluid flow.

- **(B) Poiseuille flow in smooth channel**: Colored background shows relative magnitude of flow vorticity, darker color indicates larger vorticity. In smooth channels, vorticity generated by Poiseuille flow rotates bacterial head downstream.

- **(C) Flow in channel with symmetric obstacles**: In channels with symmetric obstacles, flow velocity and vorticity near the top of obstacles are enhanced, leading to stronger torque acting on bacteria, redirecting them downstream.

- **(D) and (E) Simulated bacterial trajectories under different conditions**:
  - - **(D) Smooth channel**: In a 2D smooth channel with width 50μm, simulated trajectories of bacteria show their continuous swimming state.
  - - **(E) Bacteria-repelling surface-modified channel**: In channels with surfaces modified to repel bacteria, bacterial swimming trajectories are significantly affected.

- **(F) Population statistics of upstream swimming**:
  - - Solid line (left y-axis) represents average upstream distance, reflecting the average swimming distance of bacterial population in the upstream direction.
  - - Dashed line (right y-axis) represents upstream distance of top 1% swimmers in the population, revealing the performance of a few efficient swimming bacteria.
  - - Different colored lines represent different channel conditions: blue for smooth channel, orange for surface-modified channel, black for symmetric obstacle channel, green for asymmetric obstacle channel.

- **(G) AI operator neural network model and results**:
  - - Geo-FnO model aims to learn the relationship between catheter geometry and bacterial distribution, implemented through a series of neural operator layers.
  - - The model first maps irregular channel geometry to unit segment [0,1], then applies Fourier-based kernels in latent space for prediction.
  - - Finally, predicted bacterial distribution is transformed from latent space back to physical space.
  - - Right figure shows comparison between random initial conditions (black) and optimized design (pink), as well as Geo-FnO prediction results verified by fluid and particle dynamics simulations (green dashed line).

![Figure 3. Microfluidic experiments](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheter3.png)

**Figure 3. Microfluidic experiments**

- **(A) Schematic of microfluidic experiment**: One end of the microfluidic channel is connected to a syringe containing imaging solution, and the other end is connected to a reservoir containing E. coli. Long arrow indicates flow direction.

- **(B) Bacterial accumulation at sharp corners**: Due to flow stagnation, bacteria accumulate at sharp corners of the channel.

- **(C) Bright-field image of microfluidic channel**: Shows actual structure of the channel.

- **(D) Typical events of bacteria detaching from channel walls**:
  - - Trajectories of bacteria (white dots) in past 5s shown as yellow lines.
  - - Upper panel shows a type 1 trajectory where bacteria detach from obstacle tip.
  - - Lower panel shows a typical type 2 trajectory where bacteria detach from smooth part of channel.
  - - Left column is experimental images, right column is simulated images.

- **(E) Statistics of detachment events**: Provides statistical data on bacterial detachment events.

![Figure 4. Experiments with 3D printed catheter prototypes](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheter4.png)

**Fig. 4. Experiments with 3D printed catheter prototypes**

- **(A) Experimental setup**: Downstream end of catheter connected to E. coli reservoir, upstream end connected to syringe filled with culture medium controlled by syringe pump. After 1 hour, catheter cut into equal length segments, and internal liquid extracted for 24-hour culture. Number of E. coli colonies counted under microscope to reflect bacterial count in each segment.

- **(B) E. coli super-contamination in smooth catheter**: Shows contamination of E. coli in smooth catheter.

- **(C) Comparison of designed catheter vs smooth catheter**: Compares differences in bacterial contamination between designed catheter and smooth catheter. Inset shows same data plotted on logarithmic scale.

### 2.2 AI-aided Optimized Geometric Design

In recent years, Artificial Intelligence (AI) based models, such as Neural Operators, have been used to learn surrogates for forward simulation or observation models in fluid dynamics and other fields. Since these models are differentiable, they can be directly used for inverse design, i.e., we can use gradients to optimize directly in the design space. This makes generating previously unstudied designs much more concise and efficient. We used an AI model to optimize the channel shape characterized by the four parameters and two constraints described above (Figure 2G). This method first maps the irregular channel geometry to a function in latent space (a unit catheter segment length $[0,1]$), then applies a Fourier Neural Operator (FNO) model in latent space, and finally transforms the bacterial distribution back to physical space (Figure 2G). We then use this trained surrogate model for inverse design optimization to determine the optimal channel shape. To evaluate the effectiveness of each design, we measured the average ⟨$x_{up}$⟩ value at $T=500$s under three flow rates ($5, 10$ and $15μm/s$). Our AI-aided shape design based on geometry-aware Fourier Neural Operator improved weighted bacterial distribution by about $20\%$ compared to given shapes in training data. The entire design optimization process was very fast: generating 1000 training instances in parallel (running 10 hours on 50 GPUs), each instance taking 30 minutes; training the model on 1 GPU taking 20 minutes; while our trained AI model generated the optimal design on 1 GPU in just 15s. The optimization process yielded the following optimal structural parameters: $d=62.26μm, h=30.0μm, s=-19.56μm, L=12.27μm$, for channel width $W=100μm$. According to the mechanism described above, this structure provides strong geometric rectification and vortex reorientation effects to inhibit bacterial upstream swimming.

### 2.3 Microfluidic Experiments

To evaluate the effectiveness of the optimized structure, we fabricated quasi-2D microfluidic channels with width $W=100μm$ (wall-to-wall distance) and vertical depth of 20μm for microscopic observation of bacterial movement (Figure 3A). We selected a subset of upstream swimming bacteria and classified them based on where they detached from the wall. If bacteria detached from the top of an obstacle, their trajectory was labeled "Type 1" (Figure 3D, top); if bacteria detached from a smooth part of the wall, it was labeled "Type 2" (Figure 3D, bottom). Type 1 trajectories are affected by both geometric rectification and enhanced hydrodynamic rotation disruption effects. Type 2 trajectories are not affected by geometric rectification effects and only slightly by vortex reorientation effects, as vorticity enhancement is strongest at obstacle tips. For flow rates $U_0<100μm/s$, 70% to 80% of upstream swimming trajectories belonged to Type 1 (Figure 3E). We also noted that all observed upstream swimming trajectories in these experiments were redirected downstream (Figure 3E, red line). Bacterial accumulation was observed near sharp corners (Figure 3B), possibly due to the presence of stagnation zones (Figure 2C and Supplementary Figure S1, white areas near corners). To prevent bacterial accumulation at corners, we rounded the geometry with arcs of radius $r=h/2$ (Figure 3C).

### 2.4 Macro-scale Catheter Experiments

The mechanisms and design principles demonstrated above are easily extended to catheters. In 3D pipes, bacteria can traverse the pipe through any cut line of the cross-section (Supplementary Figure S2J). Due to the same mechanisms as above (Figure 2, A, B, F to I, and Supplementary Figure S1), only dimensionless shear rate near walls matters (27), so bacteria moving close to boundaries (Trajectory 1 in Supplementary Figure S2J) can still swim upstream. Super-contaminating bacteria can swim distances exceeding 1mm (32), comparable to rescaled obstacle dimensions, and rectification effects are expected to persist at these scales (61). Order-of-magnitude estimates suggest that Reynolds number descent using adjoint methods can also be employed (71). We note that geometric design cannot completely eliminate bacterial upstream swimming, especially near zero flow rates. However, it drastically reduces the amount of super-contamination and may significantly extend catheter indwelling time. Using our designed catheters is not expected to require changes to routine clinical protocols or retraining of medical staff. Furthermore, our solution introduces no chemicals into the catheter, making it safe and requiring no additional maintenance. Our geometric design approach is expected to be compatible with other procedural measures, antimicrobial surface modifications, and environmental control methods.


![S1. Microfluidic experiments](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheterS1.png)

**Figure S1. Example simulated trajectories of active Brownian particles**

Trajectories in (A)(C) channels with symmetric obstacles and (B)(D) channels with asymmetric obstacles. (A)(B) No fluid flow. (C)(D) With fluid flow. Color indicates local normalized vorticity.

![S2. Example simulated trajectories of active Brownian particles](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheterS2-1.png)
![S2. Example simulated trajectories of active Brownian particles](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheterS2-2.png)

**Figure S2. Considerations for geometric optimization constraints and scaling**

- (A-H) Normalized vorticity as a function of obstacle height ℎ and obstacle spacing 𝑑.
- (I) Normalized pressure drop over one period along the channel as a function of normalized obstacle height ℎ/𝑑.
- (J) Cross-sectional view of bacterial movement inside a macroscopic cylindrical tube. Compared to trajectory 2, bacteria in trajectory 2 experience strong downstream flow near tube center, while another bacterium taking trajectory 1 is close to tube wall. Therefore, flow conditions experienced by bacterium 1 are more similar to those in our considered microfluidic channels. Its upstream swimming behavior and geometric inhibition mechanisms will be similar to microfluidic conditions, with only quantitative differences.

### 2.5 Fluid and Particle Dynamics Simulation

We used COMSOL software (72) to simulate Stokes flow in channels with no-slip boundary conditions. Subsequently, the obtained velocity and vorticity fields were coupled into particle dynamics simulations, while ignoring feedback of particle motion on fluid dynamics in the limit of dilute suspensions and small particle sizes. Particle dynamics are described by Active Brownian Particle (ABP) model with Gaussian statistics and Run-and-Tumble Particle (RTP) model with power-law (Levy) statistics. Simulations were performed using our in-house developed GPU Julia code with time step 10^-4s. In ABP model, single particle dynamics are integrated according to overdamped Langevin equation.

$$0=-\zeta(U-u)+\zeta U_0q(t)+\sqrt{2D_T\xi(t)}$$

$$d{q}/dt=\left[1/2{\omega}+B{q}\times(\mathbf{E}\cdot{q})+\sqrt{2/\tau_R{\eta}({t})}\right]\times{q}$$

Where $ζ$ is viscous drag coefficient, $U$ is particle velocity, ${q}$ is particle orientation vector, $u$ is local flow velocity, $ω$ is vorticity vector of local flow field, $E$ is local strain rate tensor of flow field. $B$ is a geometric coefficient (3, 74), equal to 1 for infinitely thin rods and 0 for spheres. Since value of B does not significantly affect upstream swimming statistics (27), we set $B=0$ in results presented here. $ξ(t)$ is Gaussian random noise satisfying $⟨ξ(t)⟩=0$ and $⟨ξ(0)ξ(t)⟩=δ(t)I$. Since bacteria are μm-scale particles, their Brownian motion is relatively weak, so we set translational diffusion coefficient DT to $0.1 μm²/s$ in simulations. As long as this value remains small, its variation has little effect on results.

η is Gaussian noise satisfying $⟨η(t)⟩=0$ and $⟨η(0)η(t)⟩=δ(t)I$, τR is average run time. In RTP (Run-and-Tumble) model, a single particle displaces with $η(t)=0$ (i.e. "run" phase) for time $0<t<τ_R$. Then, ${q}$ instantaneously changes to a random new direction ${q'}$ (i.e. "tumble"), and repeats this process with new run time $τ_R$. For Lévy swimming particles, run times are sampled from Pareto distribution of form:

$$ϕ(τ)=\frac{(ατ₀^α)}{(τ+τ₀)^{(α+1)}}$$

where parameter $1<α<2$ controls power-law exponent (75).

To simplify, we treat bacterial shape as negligible size spheres. In mechanism demonstration of Figure 2J, we simulated 1,000,000 particles in a 50μm wide 2D channel, each particle having continuous run time $τ_R$ of 2s, total simulation time 200s. Periodic boundary conditions applied to both flow field and particle dynamics along channel direction. Thus, channel is effectively infinite in length, and obstacles repeat every 100μm. Particles released at $x=$ of computational domain, initially uniformly distributed and randomly oriented in channel.

For designed channels, except for surface coating case, slip boundary condition for particle dynamics and no-slip boundary condition for hydrodynamics applied on geometric boundaries of walls. In surface coating case, no-slip boundary located at wall, while particle slip boundary condition set at distance $3μm$ from wall.

### 2.6 Geo-Fno Model and Machine Learning Configuration

Catheter design problem is an optimization problem constrained by Stochastic Partial Differential Equation (SPDE), where objective function

$$⟨x_{up}⟩ = - ∫^{-∞}_{0} ρ(x)xdx ≈ - \frac{1}{N} ∑_{i=1}^{N} xi$$

depends on SPDE solution of fluid and particle dynamics problem. Here, ρ(x) is empirical bacterial distribution function at $T=500 s$, approximated by $N$ bacteria. Traditional optimization methods require repeated evaluation of this computationally expensive model, and also require an adjoint solver when applying gradient-based optimization. To overcome these computational challenges, we trained a Geometric Fourier Neural Operator (Geo-FNO) G as a surrogate model for forward fluid and particle dynamics simulation, mapping channel geometry to bacterial population function $G: c → ρ$. In contrast, previous works using AI methods to solve various design problems only selected a few parameters as input to traditional solvers of SPDE (76, 77). The full model consists of five Fourier neural layers with Gaussian Error Linear Unit (GeLU) activation layers, having fast quasi-linear time complexity.

We performed fluid and particle dynamics simulations at three maximum flow rates (5, 10 and 15μm/s) using ABP and Levy RTP models to generate training and test data for Geo-FNO. For training data, we generated 1000 simulations in parallel on 50 GPUs, taking 10 hours, with designs in each simulation randomly selected from following parameter space:

- Obstacles of height $20μm < h < 30μm$ placed periodically on channel walls
- Distance between obstacles $60μm < d < 250μm$
- Base length satisfies $15μm < L < \frac{d}{4}$, and tip position satisfies $-\frac{d}{4} < s < \frac{d}{4}$.

Constraints on these parameters are to satisfy manufacturing limitations and physical conditions of vortex generation mechanism (Figure 2, B and C; more discussion in supplementary text and Figure S2). Dataset stored for reuse in future tasks. We use relative empirical mean squared error as loss function. Model trained on 1 GPU using Adam optimizer for 20 minutes. It achieved about $4\%$ relative error on 100 test data points.

### 2.7 Gradient-based Fast Inverse Design Optimization

Our AI method has significant speed advantage over traditional solvers, and its differentiability allows fast application of gradient-based methods for geometric design optimization. On GPU, each evaluation takes only 0.005s, while using GPU-based fluid and particle dynamics simulation takes 10 minutes, thus making thousands of evaluations during optimization feasible. Furthermore, we utilize automatic differentiation tools in deep learning packages to efficiently calculate gradients with respect to design variables, enabling application of gradient-based design optimization methods.

In optimization process, we start from initial design parameters ($d = 100 μm, h = 25 μm, s = 10 μm, L = 20 μm$) and use Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm to minimize objective function $⟨x_{up}⟩$ post-processed from bacterial population predicted by Geo-FNO. When optimization gets stuck in local minima, we generate new initial condition by adding random Gaussian noise sampled from $N(0, I)$ to recorded global minimum point, and restart optimization process.

This method not only improves optimization efficiency, but also enhances flexibility and accuracy of design optimization. Through fast iteration and precise gradient calculation, we can find design parameters meeting specific performance requirements faster, providing powerful solutions for complex engineering problems such as catheter design.

Our proposed randomized BFGS algorithm ensures monotonic decrease of recorded global minimum value. Optimization loss trajectory shown in Figure S1, loss value from $L = 6.68 × 10^5$. AI-based optimization method reached optimal design after about 1500 iterations. Entire pipeline, from data generation (parallel on 50 GPUs for 1000 instances, each taking 30 minutes, total 10 hours) to model training (20 minutes on 1 GPU), to design optimization (15s on 1 GPU) and final verification (10 minutes on 1 GPU), took less than 1 day in total.

Under given parameter constraints, objective function ⟨$x_{up}$⟩ is neither convex nor monotonic with respect to these design variables, but generally decreases with increasing h, decreasing d and increasing s (see Figure S3). Final optimized design parameters are: $d = 62.26 μm, h = 30.0 μm, s = −19.56 μm$, and $L = 15.27 μm$.

![S3. Optimization details](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheterS3.png)

**Figure S3. Optimization details**

- A. Training and test error of Geo-FNO: Both training and test errors of Geo-FNO converge without overfitting. This indicates model learned well on training set while maintaining good generalization on test set.

- B. Optimization loss from Geo-FNO surrogate accelerated by randomized BFGS: Randomized BFGS algorithm accelerated by Geo-FNO surrogate reached recorded global minimum loss after about 1500 iterations. This shows efficiency and accuracy of algorithm in finding optimal solution.

- C. Visualization of loss landscape around optimized design in $ℎ-𝑑$ cross-section for Geo-FNO

- D. Visualization of loss landscape around optimized design in $L-s$ cross-section for Geo-FNO

- E. Visualization of loss landscape around optimized design in $d-s$ cross-section for Geo-FNO

### 2.8 Bacterial Strains, Culture Conditions, Materials and Chemicals

For 3D catheter long-term experiments, we used wild-type $BW25113$ E. coli with kanamycin resistance; for microfluidic experiments, we used $BW25113$ E. coli expressing mScarlet red fluorescent protein and having kanamycin resistance.

A single colony of target bacteria was picked from fresh streak plate and suspended in LB medium to prepare bacterial inoculum. Starter culture was grown overnight in LB medium at $37°C$ until reaching final concentration of approx $OD_{600} (optical density at 600nm) = 0.4$.

For microfluidic experiments, 300 microliters of starter culture transferred to new flask containing 100 ml LB medium, and cultured at $16°C$ until OD600 reached 0.1 to 0.2. Subsequently, bacteria washed twice by centrifugation (2300g, 15 minutes) and suspended in motility imaging medium composed of $10mM$ potassium phosphate ($pH=7.0$), $0.1mM$ K-EDTA, $34mM$ K-acetate, $20mM$ sodium lactate and $0.005\%$ polyvinylpyrrolidone(34). This medium maintains bacterial motility while inhibiting cell division. Final bacterial concentration in reservoir was $OD_{600} = 0.02$.

For 3D catheter long-term experiments, 3 ml of starter culture transferred to new flask containing 500 ml LB medium, and cultured at $16°C$ until $OD_{600}$ reached 0.4. Then these bacteria used directly and injected into bacterial reservoir.

Kanamycin added to all media and LB plates. 10 minutes before experiment start, bacterial motility checked using fluorescence microscope (Differential Interference Contrast (DIC) optics for $BW25113$, red channel epifluorescence for $BW25113$ mScarlet strain).

### 2.9 Microfluidic Experiments

To demonstrate design mechanisms and test effectiveness of optimized structures, researchers fabricated quasi-2D microfluidic channels for microscopic observation of bacterial movement. These microfluidic devices fabricated by photolithography and polydimethylsiloxane (PDMS) soft lithography. As shown in Figure 3A, one end of microfluidic channel connected to syringe with imaging solution, other end connected to reservoir with E. coli. Flow rate controlled by adjusting height of syringe relative to downstream outlet. Fluorescent beads injected into imaging solution as passive tracers for real-time flow monitoring.

Experiments used Olympus BX51WI microscope equipped with two Photometrics Prime95B cameras connected via Hamamatsu W-View Gemini-2 optical splitter. Olympus 20× dry objective used. Time-lapse images acquired at 12.4 fps, $488nm$ laser intensity set to $20\%$. Microscope focal plane fixed at middle of channel depth $z$ to avoid recording bacteria crawling on top and bottom of channel.

Experiments conducted over three days, using independent E. coli culture batches each day, five 15-minute recordings per day. Video post-processing using ImageJ software (Fiji) to extract bacterial trajectories. Filtered by forward progress linearity of trajectories to eliminate fast downstream moving trajectories and visually highlight upstream swimming trajectories. Researchers estimated upstream swimming time interval as $10s$ before bacterial detachment. Maximum flow rate defined as highest flow velocity along channel centerline. Instantaneous maximum flow rate estimated by averaging fastest speeds of bacteria and fluorescent beads along centerline during upstream detachment intervals. Several video recordings provided in supplementary materials.

<video width="640"  controls>
    <source src="https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/adj1741_Movie_S1.mp4" type="video/mp4">
</video>

<video width="640"  controls>
    <source src="https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/adj1741_Movie_S2.mp4" type="video/mp4">
</video>

**Video S1-S2. Recordings of bacteria detaching from walls under different flow conditions**

<video width="640"  controls>
    <source src="https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/adj1741_Movie_S3.mp4" type="video/mp4">
</video>

**Video S3. Real-time optimized design**

### 2.10 3D Catheter Long-term Experiments

Prototype catheter tubes (including geometrically designed and smooth versions) printed using Connex-Triplex 3D printer. Internal structure of tubes with obstacles similar to quasi-2D structure but scaled up and rotated around channel centerline, making obstacles extruded rings on inner wall. Considering available 3D printing precision and typical catheter dimensions, these prototypes have inner diameter of 1.6cm. For tubes with obstacles, spacing between extruded rings is 1mm. To facilitate removal of support material from 3D printing, each tube printed in two halves with mortise shape on long side, assembled into complete tube after support material removal.

As shown in Figure 4A, upper end of tube connected to syringe controlled by mechanical pump to maintain constant flow rate. Lower end of tube connected to 80mm diameter Petri dish acting as E. coli reservoir. After 1 hour, tube cut into $2cm$ long segments, liquid in each segment transferred to culture plate, discarding most upstream and downstream segments. After incubating plates at room temperature for 24 hours, bacterial colonies on each plate counted to reflect contamination in corresponding tube section.

To count colonies, four circular, equidistant, 8mm diameter areas selected on culture plate (see Figure S5). Total colony count on entire plate estimated by counting total colonies in these four areas and multiplying by ratio of total plate area to these four areas (i.e., 25 times). When colonies on plate too numerous, crowded or overlapping to count precisely, we recorded total colony count on plate as 30,000.

### 2.11 Discussion

In this study, we introduced an effective geometric design for inner surface of medical catheters aimed at inhibiting bacterial upstream swimming and excessive contamination. Our design approach is based on physical mechanisms hindering bacterial upstream swimming, while considering general model of rheotaxis of spherical particles with power-law dynamics. Since infectious microorganisms vary in shape, flagellar characteristics and hydrodynamic interactions, to simplify design and improve generality, simplified model adopted in this study ignores details of bacterial motion such as flagellar helicity (29) and hydrodynamic interactions with boundaries (20). Simulation results used to guide experimental design rather than specifically predict E. coli experimental results. Future research can employ more complex models considering details of specific microbial species.

We found that effective vorticity is generated near obstacle tips due to interaction of overlapping vortices, for which we determined lower bound for spacing between obstacles (Figure S2 and supplementary material). Constraint on obstacle height is a trade-off between enhancing effective vorticity and avoiding pipe clogging (Figure S2). Although we chose to use this AI framework to optimize catheter geometry, other methods such as genetic algorithms combined with numerical solvers (70) or gradient descent combined with adjoint methods (71) could also be employed.

We note that geometric design cannot completely eliminate bacterial upstream swimming, especially at flow rates close to zero. However, it significantly reduces amount of excessive contamination and may substantially extend catheter indwelling time. Use of our designed catheters is not expected to require changes to routine clinical protocols or retraining of medical staff. Furthermore, our solution introduces no chemicals into catheter, thus is safe and requires no additional maintenance. We expect this geometric design approach to be compatible with other procedural measures, antimicrobial surface modifications and environmental control methods.

## 3. Problem Solving

The paper uses Geometric Fourier Neural Operator (Geo-FNO) to build an AI model. This model can learn and solve stochastic partial differential equations (SPDE) related to geometric shapes, thereby optimizing catheter geometry. Through microfluidic experiments and 3D printing technology, catheter prototypes with different geometries were fabricated and tested for their effect on inhibiting bacterial upstream swimming.
Next, we will explain how to convert the problem into PaddleScience code step by step and solve the problem using deep learning methods. In order to quickly understand PaddleScience, only key steps such as model construction and computational domain construction are described below, while other details please refer to [API Documentation](../api/arch.md).

### 3.1 Dataset Introduction

Data file description is as follows:

|      `./data.zip/training/`       |                    |      `./data.zip/test/`       |                   |
| :-------------------------------: | :----------------: | :---------------------------: | :---------------: |
|              Filename             |        Description        |            Filename             |       Description        |
| training/x_1d_structured_mesh.npy | Shape (2001, 3003) | test/x_1d_structured_mesh.npy | Shape (2001, 300) |
| training/y_1d_structured_mesh.npy | Shape (2001, 3003) | test/y_1d_structured_mesh.npy | Shape (2001, 300) |
|      training/data_info.npy       |  Shape (7, 3003)   |      test/data_info.npy       |  Shape (7, 300)   |
|   training/density_1d_data.npy    | Shape (2001, 3003) |   test/density_1d_data.npy    | Shape (2001, 300) |

After loading data, x and y need to be merged, and merged training data reshaped to `(1000, 2001, 2)` format. Specific code is as follows:

```py
--8<--
examples/catheter/catheter.py:31:75
--8<--
```

### 3.2 GeoFNO Model

GeoFNO is a machine learning model based on **Geometric Fourier Neural Operator (Geo-FNO)**, which transforms geometric shapes into Fourier space to better capture shape features, and utilizes invertibility of Fourier transform to transform results back to physical space.

In the paper, this model can learn and solve partial differential equations (SPDE) related to geometric shapes, thereby optimizing catheter geometry. Code representation is as follows:

```py
--8<--
ppsci/arch/geofno.py:95:205
--8<--
```

To accurately and quickly access values of specific variables during calculation, we specify network model input variable name as `("input",)` and output variable name as `("output",)`, consistent with subsequent code.

Then by specifying number of layers, feature channels, neurons of FNO1d, and loading initialization weight model mentioned above, we instantiate a neural network model `model`.

### 3.3 Model Training and Evaluation

After completing above settings, just pass instantiated objects to `ppsci.solver.Solver` in order, then start training and evaluation.

```python
--8<--
examples/catheter/catheter.py:162:177
--8<--
```

## 4. Result Display

=== "Training, Inference loss"

Below shows first and last prediction results of model on test data after training.

=== "First Prediction Result"

    ![1725427977357](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheter10.png)

=== "Last Prediction Result"

    ![1725428017615](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheter9.png)

=== "Training Test Loss"

    ![1725894134717](https://dataset.bj.bcebos.com/PaddleScience/2024 AI-aided geometric design of anti-infection catheters/catheter8.png)

It can be seen that model prediction results are basically consistent with real results. Optimized catheter has specific geometric shape, such as obstacle distribution and spacing, which can significantly affect hydrodynamic interactions, thereby inhibiting bacterial upstream swimming behavior.

## 6. References

Reference Code: /zongyi-li/Geo-FNO-catheter

Reference List

1. J. W. Warren, the catheter and urinary tract infection. Med. Clin. North Am. 75, 481–493
(1991).
2. l. e. nicolle, catheter- related urinary tract infection. Drugs Aging 22, 627–639 (2005).
3. e. K. Shuman, c. e. chenoweth, Urinary catheter- associated infections. Infect. Dis. Clin.
North Am. 32, 885–897 (2018).
4. n. Buetti, A. tabah, J. F. timsit, W. Zingg, What is new in catheter use and catheter
infection prevention in the icU. Curr. Opin. Crit. Care 26, 459–465 (2020).
5. l. chuang, P. A. tambyah, catheter-associated urinary tract infection. J. Infect. Chemother.
27, 1400–1406 (2021).
6. e. Zimlichman, d. henderson, O. tamir, c. Franz, P. Song, c. K. Yamin, c. Keohane,
c. R. denham, d. W. Bates, health care–associated infections. JAMA Intern. Med. 173,
2039–2046 (2013).
7. U. Samuel, J. Guggenbichler, Prevention of catheter-related infections: the potential of a
new nano-silver impregnated catheter. Int. J. Antimicrob. Agents 23, 75–78 (2004).
8. W. Kohnen, B. Jansen, Polymer materials for the prevention of catheter-related infections.
Zentralbl. Bakteriol. 283, 175–186 (1995).
9. A. hameed, F. chinegwundoh, A. thwaini, Prevention of catheter-related urinary tract
infections. Med. Hypotheses 71, 148–152 (2010).
10. h. c. Berg, d. A. Brown, chemotaxis in Escherichia coli analysed by three-dimensional
tracking. Nature 239, 500–504 (1972).
11. h. c. Berg, the rotary motor of bacterial flagella. Annu. Rev. Biochem. 72, 19–54 (2003).
12. h. c. Berg, E. coli in Motion (Springer, 2004).
13. M. Polin, i. tuval, K. drescher, J. P. Gollub, R. e. Goldstein, chlamydomonas swims with two
“gears” in a eukaryotic version of run-and- tumble locomotion. Science 325, 487–490
(2009).
14. A. P. Berke, l. turner, h. c. Berg, e. lauga, hydrodynamic attraction of swimming
microorganisms by surfaces. Phys. Rev. Lett. 101, 038102 (2008).
15. e. lauga, t. R. Powers, the hydrodynamics of swimming microorganisms. Rep. Prog. Phys.
72, 096601 (2009).
16. d. Kaiser, Bacterial swarming: A re-examination of cell-movement patterns. Curr. Biol. 17,
R561–R570 (2007).
17. n. verstraeten, K. Braeken, B. debkumari, M. Fauvart, J. Fransaer, J. vermant, J. Michiels,
living on a surface: Swarming and biofilm formation. Trends Microbiol. 16, 496–506
(2008).
18. d. B. Kearns, A field guide to bacterial swarming motility. Nat. Rev. Microbiol. 8, 634–644
(2010).
19. d. Ghosh, X. cheng, to cross or not to cross: collective swimming of Escherichia coli under
two-dimensional confinement. Phys. Rev. Res. 4, 023105 (2022).
20. J. hill, O. Kalkanci, J. l. McMurry, h. Koser, hydrodynamic surface interactions enable
Escherichia coli to seek efficient routes to swim upstream. Phys. Rev. Lett. 98, 068101 (2007).
21. t. Kaya, h. Koser, direct upstream motility in Escherichia coli. Biophys. J. 102, 1514–1523
(2012).
22. Marcos, h. c. Fu, t. R. Powers, R. Stocker, Bacterial rheotaxis. Proc. Natl. Acad. Sci. U.S.A.
109, 4780–4785 (2012).
23. Y. Shen, A. Siryaporn, S. lecuyer, Z. Gitai, h. A. Stone, Flow directs surface-attached
bacteria to twitch upstream. Biophys. J. 103, 146–151 (2012).
24. A. Zöttl, h. Stark, nonlinear dynamics of a microswimmer in Poiseuille flow. Phys. Rev. Lett.
108, 218104 (2012).
25. c.-K. tung, F. Ardon, A. Roy, d. l. Koch, S. S. Suarez, M. Wu, emergence of upstream
swimming via a hydrodynamic transition. Phys. Rev. Lett. 114, 108102 (2015).
26. A. J. Mathijssen, t. n. Shendruk, J. M. Yeomans, A. doostmohammadi, Upstream
swimming in microbiological flows. Phys. Rev. Lett. 116, 028104 (2016).
27. Z. Peng, J. F. Brady, Upstream swimming and taylor dispersion of active Brownian
particles. Phys. Rev. Fluids 5, 073102 (2020).
28. G. i. taylor, dispersion of soluble matter in solvent flowing slowly through a tube.
Proc. R. Soc. A-Math. Phys. Eng. Sci. 219, 186–203 (1953).
29. t. Kaya, h. Koser, characterization of hydrodynamic surface interactions of Escherichia coli
cell bodies in shear flow. Phys. Rev. Lett. 103, 138103 (2009).
30. v. Kantsler, J. dunkel, M. Blayney, R. e. Goldstein, Rheotaxis facilitates upstream
navigation of mammalian sperm cells. eLife 3, e02403 (2014).
31. t. Omori, t. ishikawa, Upward swimming of a sperm cell in shear flow. Phys. Rev. E. 93,
032402 (2016).
32. n. Figueroa-Morales, A. Rivera, R. Soto, A. lindner, e. Altshuler, É. clément, E. coli
“super-contaminates” narrow ducts fostered by broad run-time distribution. Sci. Adv. 6,
eaay0155 (2020).
33. B. e. logan, t. A. hilbert, R. G. Arnold, Removal of bacteria in laboratory filters: Models and
experiments. Water Res. 27, 955–962 (1993).
34. W. dzik, Use of leukodepletion filters for the removal of bacteria. Immunol. Invest. 24,
95–115 (1995).
35. l. Fernandez Garcia, S. Alvarez Blanco, F. A. Riera Rodriguez, Microfiltration applied to
dairy streams: Removal of bacteria. J. Sci. Food Agric. 93, 187–196 (2013).
36. G. Franci, A. Falanga, S. Galdiero, l. Palomba, M. Rai, G. Morelli, M. Galdiero, Silver
nanoparticles as potential antibacterial agents. Molecules 20, 8856–8874 (2015).
37. M. i. hutchings, A. W. truman, B. Wilkinson, Antibiotics: Past, present and future. Curr.
Opin. Microbiol. 51, 72–80 (2019).
38. J. W. costerton, h. M. lappin-Scott, introduction to microbial biofilms, in Microbial Biofilms,
h. M. lappin-Scott, J. W. costerton, eds. (cambridge Univ. Press, 1995), pp. 1–11.
39. W.- h. Sheng, W.-J. Ko, J.-t. Wang, S.-c. chang, P.-R. hsueh, K.-t. luh, evaluation of
antiseptic-impregnated central venous catheters for prevention of catheter-related
infection in intensive care unit patients. Diagn. Microbiol. Infect. Dis. 38, 1–5 (2000).
40. W. M. dunne Jr., Bacterial adhesion: Seen any good biofilms lately? Clin. Microbiol. Rev. 15,
155–166 (2002).
41. R. P. Allaker, the use of nanoparticles to control oral biofilm formation. J. Dent. Res. 89,
1175–1186 (2010).
42. M. l. Knetsch, l. h. Koole, new strategies in the development of antimicrobial coatings:
the example of increasing usage of silver and silver nanoparticles. Polymers 3, 340–366
(2011).
43. M. Birkett, l. dover, c. cherian lukose, A. Wasy Zia, M. M. tambuwala, Á. Serrano-Aroca,
Recent advances in metal-based antimicrobial coatings for high-touch surfaces. Int. J.
Mol. Sci. 23, 1162 (2022).
44. J. R. lex, R. Koucheki, n. A. Stavropoulos, J. di Michele, J. S. toor, K. tsoi, P. c. Ferguson,
R. e. turcotte, P. J. Papagelopoulos, Megaprosthesis anti-bacterial coatings: A
comprehensive translational review. Acta Biomater. 140, 136–148 (2022).
45. J. Monod, the growth of bacterial cultures. Annu. Rev. Microbiol. 3, 371–394 (1949).
46. M. hecker, W. Schumann, U. völker, heat-shock and general stress response in Bacillus
subtilis. Mol. Microbiol. 19, 417–428 (1996).
47. P. Setlow, Spores of Bacillus subtilis: their resistance to and killing by radiation, heat and
chemicals. J. Appl. Microbiol. 101, 514–525 (2006).
48. M. Falagas, P. thomaidis, i. Kotsantis, K. Sgouros, G. Samonis, d. Karageorgopoulos,
Airborne hydrogen peroxide for disinfection of the hospital environment and infection
control: A systematic review. J. Hosp. Infect. 78, 171–177 (2011).
49. W. A. Rutala, d. J. Weber, disinfection and sterilization in health care facilities: What
clinicians need to know. Clin. Infect. Dis. 39, 702–709 (2004).
50. W. A. Rutala, d. J. Weber, disinfection and sterilization: An overview. Am. J. Infect. Control
41, S2–S5 (2013).
51. n. P. tipnis, d. J. Burgess, Sterilization of implantable polymer-based medical devices: A
review. Int. J. Pharm. 544, 455–460 (2018).
52. M. Berger, R. Shiau, J. M. Weintraub, Review of syndromic surveillance: implications for
waterborne disease detection. J. Epidemiol. Community Health 60, 543–550 (2006).
53. M. v. Storey, B. van der Gaag, B. P. Burns, Advances in on-line drinking water quality
monitoring and early warning systems. Water Res. 45, 741–747 (2011).
54. S. hyllestad, e. Amato, K. nygård, l. vold, P. Aavitsland, the effectiveness of syndromic
surveillance for the early detection of waterborne outbreaks: A systematic review.
BMC Infect. Dis. 21, 696 (2021).
55. F. Baquero, J.- l. Martínez, R. cantón, Antibiotics and antibiotic resistance in water
environments. Curr. Opin. Biotechnol. 19, 260–265 (2008).
56. R. i. Aminov, the role of antibiotics and antibiotic resistance in nature. Environ. Microbiol.
11, 2970–2988 (2009).
57. J. M. Munita, c. A. Arias, Mechanisms of antibiotic resistance. Microbiol Spectr, (2016).
58. U. theuretzbacher, K. Bush, S. harbarth, M. Paul, J. h. Rex, e. tacconelli, G. e. thwaites,
critical analysis of antibacterial agents in clinical development. Nat. Rev. Microbiol. 18,
286–298 (2020).
59. R. di Giacomo, S. Krödel, B. Maresca, P. Benzoni, R. Rusconi, R. Stocker, c. daraio,
deployable micro- traps to sequester motile bacteria. Sci. Rep. 7, 45897 (2017).
60. P. Galajda, J. Keymer, P. chaikin, R. Austin, A wall of funnels concentrates swimming
bacteria. J. Bacteriol. 189, 8704–8707 (2007).
61. c. M. Kjeldbjerg, J. F. Brady, theory for the casimir effect and the partitioning of active
matter. Soft Matter 17, 523–530 (2021).
62. Z. li, n. Kovachki, K. Azizzadenesheli, B. liu, K. Bhattacharya, A. Stuart, A. Anandkumar.
Fourier neural operator for parametric partial differential equations. arXiv:2010.08895
[cs.lG] (2020).
63. Z. li, d. Z. huang, B. liu, A. Anandkumar. Fourier neural operator with learned
deformations for pdes on general geometries. arXiv:2207.05209 [cs.lG] (2022).
64. A. J. Mathijssen, n. Figueroa-Morales, G. Junot, É. clément, A. lindner, A. Zöttl, Oscillatory
surface rheotaxis of swimming E. coli bacteria. Nat. Commun. 10, 3434 (2019).
65. S. B. Goodman, Z. Yao, M. Keeney, F. Yang, the future of biologic coatings for orthopaedic
implants. Biomaterials 34, 3174–3183 (2013).
66. A. Jaggessar, h. Shahali, A. Mathew, P. K. Yarlagadda, Bio-mimicking nano and
micro-structured surface fabrication for antibacterial properties in medical implants.
J. Nanobiotechnol. 15, 64 (2017).
67. e. Macedo, R. Malhotra, R. claure- del Granado, P. Fedullo, R. l. Mehta, defining urine
output criterion for acute kidney injury in critically ill patients. Nephrol Dial Transplant 26,
509–515 (2011).
68. K. B. chenitz, M. B. lane-Fall, decreased urine output and acute kidney injury in the
postanesthesia care unit. Anesthesiol. Clin. 30, 513–526 (2012).
69. J. A. Kellum, F. e. Sileanu, R. Murugan, n. lucko, A. d. Shaw, G. clermont, classifying AKi by
urine output versus serum creatinine level. J. Am. Soc. Nephrol. 26, 2231–2238 (2015).
70. S. Mirjalili, Genetic algorithm, in Evolutionary Algorithms and Neural Networks: Theory and
Applications (Springer, 2019), pp. 43–55.
71. S. Ruder, An overview of gradient descent optimization algorithms. arXiv:1609.04747
[cs.lG] (2016).
72. c. Multiphysics, introduction to cOMSOl multiphysics. cOMSOl Multiphysics, Burlington,
MA, accessed 2018 Feb 9: 32 (1998).
73. G. B. Jeffery, the motion of ellipsoidal particles immersed in a viscous fluid. Proc. R. soc. Lond.
Ser. A-Contain. Pap. Math. Phys. Character 102, 161–179 (1922).
74. F. P. Bretherton, the motion of rigid particles in a shear flow at low Reynolds number.
J. Fluid Mech. 14, 284–304 (1962).
75. t. Zhou, Z. Peng, M. Gulian, J. F. Brady, distribution and pressure of active lévy swimmers
under confinement. J. Phys. A Math. Theor. 54, 275002 (2021).
76. P. i. Frazier, J. Wang, Bayesian optimization for materials design, in Information Science for
Materials Discovery and Design (Springer, 2016), pp. 45–75.
77. Y. Zhang, d. W. Apley, W. chen, Bayesian optimization for materials design with mixed
quantitative and qualitative variables. Sci. Rep. 10, 4924 (2020).
78. J. Schindelin, i. Arganda-carreras, e. Frise, v. Kaynig, M. longair, t. Pietzsch, S. Preibisch,
c. Rueden, S. Saalfeld, B. Schmid, Fiji: An open-source platform for biological-image
analysis. Nat. Methods 9, 676–682 (2012).
79. d. ershov, M.-S. Phan, J. W. Pylvänäinen, S. U. Rigaud, l. le Blanc, A. charles- Orszag,
J. R. conway, R. F. laine, n. h. Roy, d. Bonazzi, Bringing trackMate into the era of
machine-learning and deep-learning. bioRxiv 458852 [Preprint] (2021). https://doi.
org/10.1101/2021.09.03.458852.

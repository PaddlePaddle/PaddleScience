# ML2DDB

[Monolayer Two-dimensional Materials Database (ML2DDB) and Applications](https://arxiv.org/pdf/2507.00584)

Zhongwei Liu<sup>a, b, #</sup>,
Zhimin Zhang<sup>c, #</sup>,
Xuwei Liu<sup>c, #</sup>,
Mingjia Yao<sup>b</sup>,
Xin He<sup>a</sup>,
Yuanhui Sun<sup>b, *</sup>,
Xin Chen<sup>b, *</sup>,
Lijun Zhang<sup>a, b, *</sup>

<sup>a</sup>
State Key Laboratory of Integrated Optoelectronics, Key Laboratory of Automobile Materials of MOE and College of Materials Science and Engineering, Jilin University, Changchun 130012, China

<sup>b</sup> Suzhou Laboratory, Suzhou, 215123, China

<sup>c</sup> Baidu Inc., Beijing, P.R. China.

<sup>#</sup> These authors contributed equally to this work.

E-mail: sunyh@szlab.ac.cn; chenx01@szlab.ac.cn; lijun_zhang@jlu.edu.cn

## Abstract

The discovery of two-dimensional (2D) materials with tailored properties is critical to meet the increasing demands of high-performance applications across flexible electronics, optoelectronics, catalysis, and energy storage. However, current 2D material databases are constrained by limited scale and compositional diversity. In this study, we introduce a scalable active learning workflow that integrates deep neural networks with density functional theory (DFT) calculations to efficiently explore a vast set of candidate structures. These structures are generated through physics-informed elemental substitution strategies, enabling broad and systematic discovery of stable 2D materials. Through six iterative screening cycles, we established the creation of the Monolayer 2D Materials Database (ML2DDB), which contains 242,546 DFT-validated stable structures—an order-of-magnitude increase over the largest known 2D materials databases. In particular, the number of ternary and quaternary compounds showed the most significant increase. Combining this database with a generative diffusion model, we demonstrated effective structure generation under specified chemistry and symmetry constraints. This work accomplished an organically interconnected loop of 2D material data expansion and application, which provides a new paradigm for the discovery of new materials.

![ML2DDB](https://paddle-org.bj.bcebos.com/paddlescience/docs/ML2DDB/ml2ddb.png)

## Dataset of 2D materials

We developed ML2DDB, a large-scale 2D material database containing >242k DFT-validated monolayer structures (𝐸<sub>hull</sub><sup>𝐷𝐹𝑇</sup> <50 meV/atom), representing a 10× increase over existing datasets. Key features:

- Broad elemental coverage: 81 elements across the periodic table (excluding radioactive/noble gases).
- Enhanced diversity: Significantly more compounds with 3–4 distinct elements compared to prior work.
- Structural richness: Diverse prototypes and cation-anion combinations.
- Extended resource: >1M candidate structures (𝐸<sub>hull</sub><sup>MLIP</sup> <200 meV/atom) for future studies.

![dataset](https://paddle-org.bj.bcebos.com/paddlescience/docs/ML2DDB/ml2ddb_dataset.png)

## Diffusion model generation of S.U.N. materials

The capability to generate S.U.N. (stable, unique, new) 2D materials are prerequisites for diffusion models. We considered a generated structure as stable with 𝐸<sub>hull</sub><sup>𝐷𝐹𝑇</sup> < 100 meV/atom with respect to ML2DDB. The unique is specified whether a generated structure matches any other structure generated in the same batch or not, and the new is whether it is identical to any of the structures in ML2DDB. As shown in Figure 5b, we performed DFT structure optimization on 1024 structures to evaluate the stable attribute. The results show that 74.8% of them are considered stable (𝐸<sub>hull</sub><sup>𝐷𝐹𝑇</sup> < 100 meV/atom), which is comparable to the success rate of 3D stable structure generation of MatterGen. When the constraint is set to 𝐸<sub>hull</sub><sup>𝐷𝐹𝑇</sup> < 0 meV/atom, our method achieved a success rate of 59.6%, which is significantly higher than that of  MatterGen (~13%). In addition, the Root-mean-square displacement (RMSD) of the generated structure is lower than 0.26 Å compared to the DFT relaxation structure, which is still less than the radius of the hydrogen atom (0.53 Å). For the generation of unique structures, the success rate accounts for 100% when generating one thousand structures. The rate only decreases 4.4% when generating ten thousand structures. For the generation of new structures, the rate decreases from 100% to 73.5% when the generated structures grow from one thousand to two thousand. This indicates that our model has a relatively excellent ability to generate completely new stable structures.

![dataset](https://paddle-org.bj.bcebos.com/paddlescience/docs/ML2DDB/gen_2d.png)

## Conclusion

This study establishes a novel framework integrating active learning workflows with conditional diffusion-based structural generation, achieving unprecedented expansion of 2D materials databases. Key contributions include:

1. **Dataset Advancement**
   - Created ML2DDB containing >242,546 thermodynamically stable 2D materials (E_hull^DFT <50 meV/atom), exceeding existing databases by ≥10x
   - Achieved 1100% and 960% growth in ternary/quaternary compounds respectively
   - Generated >1 million candidate structures (𝐸<sub>hull</sub><sup>MLIP</sup> <200 meV/atom)
2. **Methodological Innovation**
   - Developed MLIP model with 92.36% accuracy in stability classification
   - Enabled phase diagram generation and space group-specific design through diffusion model integration
   - Demonstrated applicability to nonlinear optical and ferroelectric materials discovery

# Downloaded Papers

1. [A Simple Weight Decay Can Improve Generalization](krogh_hertz_1991_simple_weight_decay.pdf)
   - Authors: Anders Krogh, John A. Hertz
   - Year: 1991
   - Source: NeurIPS 1991
   - Why relevant: Classic paper establishing the core argument for L2/weight decay as a bias toward smaller-norm solutions and improved generalization under noisy targets.

2. [Decoupled Weight Decay Regularization](loshchilov_hutter_2019_decoupled_weight_decay.pdf)
   - Authors: Ilya Loshchilov, Frank Hutter
   - Year: 2019
   - Source: ICLR 2019 / arXiv:1711.05101
   - Why relevant: Important implementation detail for experiments using Adam-family optimizers; clarifies that Adam plus L2 is not the same as true weight decay.

3. [Understanding Deep Learning Requires Rethinking Generalization](zhang_et_al_2017_rethinking_generalization.pdf)
   - Authors: Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals
   - Year: 2017
   - Source: ICLR 2017 / arXiv:1611.03530
   - Why relevant: Strong counterpoint showing explicit regularization alone does not explain modern generalization behavior.

4. [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](power_et_al_2022_grokking.pdf)
   - Authors: Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra
   - Year: 2022
   - Source: arXiv:2201.02177
   - Why relevant: Directly studies very small-data regimes and reports that weight decay materially improves data efficiency and eventual generalization.

5. [Why Do We Need Weight Decay in Modern Deep Learning?](xie_et_al_2024_why_weight_decay.pdf)
   - Authors: Francesco D'Angelo, Maksym Andriushchenko, Aditya Varre, Nicolas Flammarion
   - Year: 2024
   - Source: NeurIPS 2024
   - Why relevant: Modern view of weight decay as optimization-dynamics control rather than purely classical capacity control.

6. [Time Matters in Regularizing Deep Networks: Weight Decay and Data Augmentation Affect Early Learning Dynamics, Matter Little Near Convergence](golatkar_achille_soatto_2019_time_matters_regularizing.pdf)
   - Authors: Aditya Golatkar, Alessandro Achille, Stefano Soatto
   - Year: 2019
   - Source: NeurIPS 2019 / arXiv:1905.13277
   - Why relevant: Shows that regularization timing matters, which is useful when designing short small-data training runs.

Chunked reading artifacts were created for the three papers that were read more deeply:
- `papers/pages_krogh/`
- `papers/pages_grokking/`
- `papers/pages_xie/`

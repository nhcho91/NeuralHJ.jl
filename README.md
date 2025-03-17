# NeuralHJ.jl
Julia Codes for Hamilton-Jacobi Reachability Analysis with Physics-Informed Neural Networks, a.k.a., DeepReach.


## Implementations
Currently, three different pipelines are developed to implement DeepReach for Dubins3D example.

- NestedAD-Enzyme: Custom implementation using nested automatic differentiation with [Lux](https://github.com/LuxDL/Lux.jl) + [Reactant](https://github.com/EnzymeAD/Reactant.jl) + [Enzyme](https://github.com/EnzymeAD/Enzyme.jl)
- NestedAD-Zygote: Custom implementation using nested automatic differentiation with [Lux](https://github.com/LuxDL/Lux.jl) + [LuxCUDA](https://github.com/LuxDL/LuxCUDA.jl) + [Zygote](https://github.com/FluxML/Zygote.jl)
- SciML-NeuralPDE: Direct application of SciML functionalities with [Lux](https://github.com/LuxDL/Lux.jl) + [LuxCUDA](https://github.com/LuxDL/LuxCUDA.jl) + [NeuralPDE](https://github.com/SciML/NeuralPDE.jl)

The codes inside the directory `NestedAD` provides a convenient single entry point in testing NestedAD-Enzyme and NestedAD-Zygote for both single and multiple GPU training.
- `main_NeuralHJ_NestedAD.jl`: CPU or single-GPU-accelerated training
- `main/runvis_NeuralHJ_NestedAD_multiGPU.jl`: multi-GPU-accelerated [distributed data-parallel training](https://lux.csail.mit.edu/dev/manual/distributed_utils) (intended for running on HPC, e.g., HILDA at Cranfield University)

### Details
- [ExactBC](https://arxiv.org/abs/2404.00814) and pretraining are implemented.
- [Curriculum learning](https://doi.org/10.1109/ICRA48506.2021.9561949) is not implemented yet.
- [HardNet](https://arxiv.org/abs/2410.10807)-Reach framework is currently under development.

### Comparison
NestedAD-Enzyme are tested on both macbook pro with CPU and Linux server with GPU (Reactant fails to be precompiled on Windows.)

NestedAD-Zygote and SciML-NeuralPDE are tested on both macbook pro with CPU and Windows desktop with GPU.

- Training Speed: NestedAD-Enzyme > NestedAD-Zygote > SciML-NeuralPDE
- Reliability: NestedAD-Zygote >= NestedAD-Enzyme > SciML-NeuralPDE

### Training Stats
NestedAD
- using Lux.jl + Reactant.jl + Enzyme.jl
- configured to use single GPU (NVIDIA A100-PCIE-40GB) on Linux server
- evaluating loss function at around 65000 datapoints without mini-batching
- using 3 hidden layers with 512 nodes per each layer (around 790000 parameters in total)
- for Dubins3D example
=> achieves around 190.3 seconds / 10000 epochs (except compilation time)

## References
- Somil Bansal - [DeepReach: A Deep Learning Approach to High-Dimensional Reachability](https://doi.org/10.1109/ICRA48506.2021.9561949)
- Somil Bansal - [DeepReach-public_release](https://github.com/smlbansal/deepreach/tree/public_release)
- William Sharpless - [DeepReach-hopf_exact_bc](https://github.com/willsharpless/deepreach/tree/hopf_exact_bc)
- William Sharpless - [HopfReachability](https://github.com/UCSD-SASLab/HopfReachability)

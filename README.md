Kolmogorov-Arnold Networks (KANs) are a recent neural network architecture offering an alternative to Multilayer Perceptrons (MLPs) with improved explainability and expressibility. However, KANs are significantly slower than MLPs due to the recursive nature of B-spline function computations, limiting their application. This work addresses these issues by proposing a novel base-spline Linear-Time B-splines Kolmogorov-Arnold Network (LTBs-KAN) with linear complexity. Unlike previous methods that rely on the Boor-Mansfield-Cox spline algorithm or other computationally intensive mathematical functions, our approach significantly reduces the computational burden. Additionally, we further reduce model's parameter through product-of-sums matrix factorization in the forward pass without sacrificing performance. Experiments on MNIST, Fashion-MNIST and CIFAR-10 demonstrate that LTBs-KAN achieves good time complexity and parameter reduction,
when used as building architectural blocks, compared to other KAN implementations.


If you use this work, please cite:

```bibtex
@misc{merinmartinez2026ltbskanlineartimebsplineskolmogorovarnold,
  title={LTBs-KAN: Linear-Time B-splines Kolmogorov-Arnold Networks}, 
  author={Eduardo Said Merin-Martinez and Andres Mendez-Vazquez and Eduardo Rodriguez-Tello},
  year={2026},
  eprint={2604.22034},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2604.22034}
}
```

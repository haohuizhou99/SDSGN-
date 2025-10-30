This is the implementation of SDSGN

To address the 'representation bottleneck' where existing GNNs struggle to balance global structural perception (prone to over-smoothing) and local neighborhood flexibility (susceptible to noise interference), SDSGN proposes a functionally-decoupled dual-stream architecture.

1、Static Stream: Employs a deep graph network based on Ordered Gated Convolution (ONGNNConv) to robustly capture global topological consistency and effectively alleviate over-smoothing.

2、Dynamic Stream: Utilizes Reinforcement Learning (TD3-Agent) to dynamically learn the optimal local aggregation strategy for each node, achieving adaptivity and avoiding noise.

3、Symmetric Fusion: Adopts an efficient, parameter-free symmetric averaging operation to fuse the dual-stream representations, preserving their orthogonal advantages.

Main Results
1、SOTA Performance: On the node classification task across 12 public benchmark datasets (covering both homophilous and heterophilous graphs), SDSGN achieved SOTA performance on 11 of them.

2、Advantage on Heterophilous Graphs: Showed particularly outstanding performance on heterophilous graphs, with an average improvement of up to 3.76%.

Installation requirements
python3.8.18 numpy==1.21.4 torch-cluster==1.6.3+pt22cu121 torch-scatter==2.1.2+pt22cu121 torch-sparse==0.6.18+pt22cu121 torch_spline_conv==1.2.2+pt22cu121 torch-geometric==2.6.1 torch==2.2.1

Example
python train

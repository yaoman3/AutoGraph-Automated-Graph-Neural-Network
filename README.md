# AutoGraph: Automated Graph Neural Network

This repository contains an implementation of our paper [AutoGraph: Automated Graph Neural Network](https://arxiv.org/abs/2011.11288) presented in ICONIP 2020.

## Requirements

- Python 3
- PyTorch 1.3.1
- torch-scatter 1.4.0
- torch-sparse 0.4.3
- torch-cluster 1.4.5
- torch-geometric 1.3.2

## Usage

Search for the GNN architecture:
```
python main.py --dataset Cora 
```

Evaluate the architecture:
```
python main.py --dataset Cora --mode test --test_structure  "['gcn','sum','relu',4,256,0,'gcn','sum','relu',4,256,1,'gcn','sum','relu',4,256,0,'const','sum','leaky_relu',4,7,0]"
```

## Reference

```
@inproceedings{li2020autograph,
  title={AutoGraph: Automated Graph Neural Network},
  author={Li, Yaoman and King, Irwin},
  booktitle={International Conference on Neural Information Processing},
  pages={189--201},
  year={2020},
  organization={Springer}
}
```

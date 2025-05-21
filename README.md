# Pruning CNNs using Reinforcement Learning
Implementation of the paper [Learning to Prune Filters in Convolutional Neural Networks](https://arxiv.org/pdf/1801.07365.pdf)

## Installation

```
> git clone https://github.com/saadmanrafat/pruning-cnn-using-rl
> cd into the directory 
> pip install -r requirements.txt
```

## Benchmarks


Performance comparison on VGG16 for CIFAR-10:

| Method | Pruning Ratio | FLOPs Reduction | Accuracy Drop |
|--------|---------------|-----------------|---------------|
| Original | 0% | 0% | 0% |
| b=0.5 | ~80% | ~45% | <0.5% |
| b=1.0 | ~83% | ~55% | <1.0% |
| b=2.0 | ~87% | ~65% | <2.0% |

## Citation

```bibtex
@article{huang2018learning,
  title={Learning to prune filters in convolutional neural networks},
  author={Huang, Qiangui and Zhou, Kevin and You, Suya and Neumann, Ulrich},
  journal={arXiv preprint arXiv:1801.07365},
  year={2018}
}
```

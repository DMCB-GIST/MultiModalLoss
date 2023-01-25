# Deep Metric Loss for Multimodal Learning

Implementation of XXXX 2023 paper [Deep Metric Loss for Multimodal Learning](https://github.com/idstcv/SoftTriple).


<img width="50%" src="https://user-images.githubusercontent.com/37695581/214497492-51ae08b2-7407-4731-88c1-aea138c52473.png"/>

This repository contains the code and the synthetic data to reproduce the result from the paper:

## MultiModal Loss
```{r}
MultiModalLoss(num_classes, num_modalities, proxies_per_class=20, gamma=0.1)
```
**Parameters:**
* **num_classes:** The number of classes.
* **num_modalities:** The number of modalities.
* **proxies_per_class:** The number of proxies per class. The papaer uses 20.
* **gamma:** Scaling factor.




## Prerequisites
* Python (3.7.9)
* PyTorch (1.9.0)
* pytorch_metric_learning (0.9.99)


## Acknowledgments
This code is inspired by [SoftTriple Loss](https://github.com/idstcv/SoftTriple) and [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

## Citation
If you find the MultiModal Loss is userful, please cite the above paper:
```{r}
@article{moon2022moma,
  title={MOMA: a multi-task attention learning algorithm for multi-omics data interpretation and classification},
  author={Moon, Sehwan and Lee, Hyunju},
  journal={Bioinformatics},
  volume={38},
  number={8},
  pages={2287--2296},
  year={2022},
  publisher={Oxford University Press}
}
```

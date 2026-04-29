# Weakly Supervised Segmentation and Classification of Alpha-Synuclein Aggregates in Brightfield Midbrain Images


This repository contains the official implementation of the paper "Weakly Supervised Segmentation and Classification of Alpha-Synuclein Aggregates in Brightfield Midbrain Images" presented at the International Symposium on Biomedical Imaging 2026

https://arxiv.org/abs/2511.16268

Erwan Dereure, Robin Louiset, Laura Parkkinen, David A Menassa, David Holcman


---


## Installation

First, clone the repository.

Then, install the dependencies:


```
pip3 install -e .[umap,h5]
```

## Experiments
### 1. Pretrain backbone (self-supervised learning)

```
python3 main_pretrain.py \
    --config-path scripts/pretrain/custom/ \
    --config-name simclr.yaml
```

### 2. Train linear classifier
```
python3 main_linear.py \
    --config-path scripts/linear/custom/ \
    --config-name vanilla.yaml
```

### 3. Image retrieval algorithm


```
python3 main_save_images_to_label.py
```

### 4. Visualization and analysis

See the notebook:
```
experiments.ipynb
```

This notebook includes:

- Segmentation visualization
- Classification results

## Inference

Pretrained weights are available at:

https://huggingface.co/edereure/synuclein_processing/tree/main

## Acknowledgments

This project builds upon the following repositories:

- https://github.com/vturrisi/solo-learn
- https://github.com/holcman-lab/DeepCellMap

We thank the authors for releasing their code.

## License

This project is licensed under the MIT License.

## Disclaimer

This code is provided "as is", without any warranty, expressed or implied.

## Citation

If you use this work, please cite:

```
@article{dereure2025weakly,
  title={Weakly Supervised Segmentation and Classification of Alpha-Synuclein Aggregates in Brightfield Midbrain Images},
  author={Dereure, Erwan and Louiset, Robin and Parkkinen, Laura and Menassa, David A and Holcman, David},
  journal={International Symposium on Biomedical Imaging},
  year={2026}
}
```

## Contact

For questions or issues, please open a GitHub issue.
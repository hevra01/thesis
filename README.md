# Diffusion Model Training & LID Estimation

This repository provides code for training diffusion models (e.g., on MNIST) and estimating Local Intrinsic Dimensionality (LID) using a Fokker-Planck-based estimator.

## Training for LID estimation

To train a model (e.g., on MNIST), run:

```sh
python train.py experiment=train_mnist
```

- This uses Hydra for configuration management.
- The experiment config file is located at `conf/experiment/train_mnist.yaml`.
- Training logs and checkpoints will be saved in the output directory specified in the config.

## LID Estimation

To estimate the Local Intrinsic Dimensionality (LID) using a trained model, run:

```sh
python estimate_LID.py experiment=estimate_lid_imageNet.yaml
```


## Notes

- Make sure to train the model and have a valid checkpoint before running LID estimation.
- you can also used pretrained diffusion model's as score functions. E.g. for imagenet, I used guided diffusion from OPENAI.
- For new datasets, update the dataset section in the config accordingly.
---
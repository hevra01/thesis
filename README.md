# Diffusion Model Training & LID Estimation

This repository provides code for training diffusion models (e.g., on MNIST) and estimating Local Intrinsic Dimensionality (LID) using a Fokker-Planck-based estimator.

## Training

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
python LID/estimate_lid.py --config conf/LID/LID_estimator_config.json
```

- The config file specifies the model architecture, checkpoint path, device, and dataset details.
- You can modify `conf/LID/LID_estimator_config.json` to change the dataset, batch size, or other parameters.

## Notes

- Make sure to train the model and have a valid checkpoint before running LID estimation.
- For new datasets, update the dataset section in the config accordingly.
---
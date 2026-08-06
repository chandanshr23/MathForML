# MNIST Training Project

This folder contains a small NumPy-based multilayer perceptron for training a simple MNIST classifier.

## Requirements

Make sure you are in the MNIST folder and install the dependencies:

```bash
pip install -r requirement.txt
```

If you are using a virtual environment, activate it first.

## Files

- `train.py` – trains the model
- `test_model.py` – runs a small forward/backward smoke test
- `gradient_checking.py` – checks the gradients numerically
- `model.py` – neural network implementation
- `mnist_data.npz` – dataset file used by the training script

## Run the training script

From the MNIST folder:

```bash
python train.py
```

This will:
- load the dataset from `mnist_data.npz`
- train the model for 20 epochs
- print training and test accuracy each epoch
- save weights to `mnist_weights.npz`

## Run a quick model test

```bash
python test_model.py
```

This performs a basic forward/backward pass and prints the output shapes of the gradients.

## Run gradient checking

```bash
python gradient_checking.py
```

This numerically checks the gradients of the model parameters and prints whether each gradient is passing.

## Notes

- The project uses NumPy only.
- Training may take a little time depending on your machine.
- If you want to retrain from scratch, delete `mnist_weights.npz` first.

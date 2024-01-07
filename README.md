# A simple project to parctice with a TensorFlow object detection model

## Trained on

- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist/tree/master) dataset

Fashion-MNIST is a dataset of Zalando's article images consisting of a training
set of 60,000 examples and a test set of 10,000 examples. Each example is a
28x28 grayscale image, associated with a label from 10 classes.

Classification labels are integers in the range 0-9.
Designation labels are strings in the set: 

`{"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"}`

It's comes preloaded in Keras.

## Model Architecture

- [standard convolutional neural network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network)

## Requirements

- MacOS 12 or higher (`tensorflow-macos`, `tensorflow-metal` are used)
- Python 3.11 or higher

## Create venv

```shell
python -m venv fashion-mnist
```

## Activate venv

```shell
source fashion-mnist/bin/activate
```

## Install requirements

```shell
pip install -r requirements.txt
```

## Run

1. Set a .png file from the `data` directory as the input image in the `main.py` file.
2. Run the `main.py` file from and IDE or terminal to test a model prediction.

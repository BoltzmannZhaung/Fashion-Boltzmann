# Fashion-Boltzmann
## Bonjour à tous!

![Image crashed](https://raw.githubusercontent.com/BoltzmannZhaung/Fashion-Boltzmann/master/img/pic.jpg)


## Date Jul 22, 2018
## What I have done
This is a practice of Dataset Fashion MNIST.At present, I just finished two files.

- **load.py :** to load dataset that you can download from web https://www.kaggle.com/c/ml1718-fashion-mnist/data.<br>
And Fashion-MNIST is consist of four .gz package.
```latex
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```
- **draw.py :** As we know,each example is a 28 by 28 grayscale image.
So you can draw everyone if you want.<br>
The pictures you drew are saved in './pictures'.

- **conv_net.py:** It will be a Convolutional Neural Network model.
I expect to accomplish it just use low-level TensorFlow APIs.<br>
GOD BLESS ME!

## What's the preparation
Before running,you should make sure that:<br>
**numpy**       is installed<br>
**matplotlib**  is installed<br>
**tensorflow**  is installed<br>

## Date Jul 28, 2018
## A little Interlude
It will be to your benefit to understand how a **filter** or **kernel**(aka **feature detector**) slide over an image and compute the dot product.
In my job,an image given like this:<br>
![Image crashed](https://raw.githubusercontent.com/BoltzmannZhaung/Fashion-Boltzmann/master/img/pic2.png)<br>
To form a matrix(also a image) is called the **Convolved Feature** or **Activation Map** or the **Feature Map** like this:<br>
![Image crashed](https://raw.githubusercontent.com/BoltzmannZhaung/Fashion-Boltzmann/master/img/pic3.png)<br>
ENJOY IT !

## Merci~!
# Fashion-Boltzmann
![Image crashed](https://raw.githubusercontent.com/BoltzmannZhaung/Fashion-Boltzmann/master/img/logo5.png)
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
In my job,give an image where the left half of the image is 0 and right half is 255,so we can see it will be like this:<br>
![Image crashed](https://raw.githubusercontent.com/BoltzmannZhaung/Fashion-Boltzmann/master/img/pic2.png)<br>
Obviously,there is a clearly very strong vertical edge right down the middle of the image.So we convolve this image with
the filter or kernel to form a matrix(also an image) is called the **Convolved Feature** or **Activation Map** or the **Feature Map** 
as follow where there is a light region right down in the middle:<br>
![Image crashed](https://raw.githubusercontent.com/BoltzmannZhaung/Fashion-Boltzmann/master/img/pic3.png)<br>
Actually,this light region corresponds to having detected the vertical edge.
ENJOY IT<br>
RUN **convolve.py** 

## Date Jul 30, 2018
Run **conv_demo.py** to train a CNN.
## Merci~!
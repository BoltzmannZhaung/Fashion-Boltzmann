# Fashion-Boltzmann
Bonjour à tous!

![Image crashed]https://raw.githubusercontent.com/BoltzmannZhaung/Fashion-Boltzmann/master/img/pic.jpg

This is a practice of Dataset Fashion MNIST.At present, I just finished two files.
  load.py : to load dataset that you can download from web www.kaggle.com.
            And Fashion-MNIST is consist of four .gz package.
            train-images-idx3-ubyte.gz
            train-labels-idx1-ubyte.gz
            t10k-images-idx3-ubyte.gz
            t10k-labels-idx1-ubyte.gz

  draw.py : As we know,each example is a 28 by 28 grayscale image.
            So you can draw everyone if you want.
            The pictures you drew are saved in './pictures'.

  conv_net.py: It will be a Convolutional Neural Network model.
               I expect to accomplish it just use low-level TensorFlow APIs.
               later,the BLAS(Basic Linear Algebra Subprograms) if I could.
               GOD BLESS ME!

Before running,you should make sure that:
numpy       is installed
matplotlib  is installed
tensorflow  is installed

Merci~!
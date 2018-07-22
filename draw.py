import os
import load
import numpy as np
import matplotlib.pyplot as plt

#Load DataSet,however Do NOT one_hot labels
mnist_fashion = load.read_data_sets(data_dir='./data',one_hot= False)
images_train = mnist_fashion.train.images
labels_train = mnist_fashion.train.labels

images_validation = mnist_fashion.validation.images
labels_validation = mnist_fashion.validation.labels

images_test = mnist_fashion.test.images
labels_test = mnist_fashion.test.labels

"""
To creat pictures location
"""
path_train = './pictures/train'
path_validation = './pictures/validation'
path_test = './pictures/test'

if not os.path.exists(path_train):
    os.makedirs(path_train);
if not os.path.exists(path_validation):
    os.makedirs(path_validation);
if not os.path.exists(path_test):
    os.makedirs(path_test);



"""
Each training and test example is assigned to one of the following labels:
Label	Description
0	T-shirt
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
"""
label_names = {0:'T-shirt',
               1:'Trouser',
               2:'Pullover',
               3:'Dress',
               4:'Coat',
               5:'Sandal',
               6:'Shirt',
               7:'Sneaker',
               8:'Bag',
               9:'Ankle-boot'}


"""
To draw every sample in training Set of fashion mnist
Args -----> images: image data.
            labels: label data.
            path: pictures' location
            num: How many samples do you wannna draw? Default is TEN.
Returns --> PNG image in ./pictures
Raises ---> ValueError: view the Error_INFO.
"""
def Draw_images(images,labels,path,num=10):
    Error_INFO=['The quantitys of images and labels are NOT Equal,Please check them.',
                'Invalid num %d is bigger than the quantity of images ,The maxima num is %d.'
                %(num,len(images))]
    if len(images) != len(labels):
        raise ValueError(Error_INFO[0])
    if num > len(images):
        raise ValueError(Error_INFO[1])

    for i in np.arange(num):
        image= images[i]
        label= labels[i]
        image = image.reshape((28, 28))
        plt.title('{label}'.format(label=label_names[label]))
        plt.imshow(image, cmap='gray')
        plt.savefig(path+'/num_{}.png'.format(i))
    print('Drawing Done','\n')

#Drawing train
print('Drawing train')
Draw_images(images_train,labels_train,path=path_train)

#Drawing validation
print('Drawing validation')
Draw_images(images_validation,labels_validation,path=path_validation)

#Drawing test
print('Drawing test')
Draw_images(images_test,labels_test,path=path_test)
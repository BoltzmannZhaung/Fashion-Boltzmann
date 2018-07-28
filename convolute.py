import os
import numpy as np
import matplotlib.pyplot as plt

path = './conved_results/'
if not os.path.exists(path):
	os.mkdir(path)

matrix = np.arange(784).reshape(28,28)
matrix[:,:14] = 0
matrix[:,14:28] = 255
print('A dimensions of {} has been created !'.format(matrix.shape))
np.savetxt(path + 'image.', matrix,fmt='%3d', delimiter=' ')

conv_kernel = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
print('A dimensions of {} has been created !'.format(conv_kernel.shape))

def convolute(x, kernel):
	row, col = x.shape
	#print(row,col)
	out = []
	for i in range(row-2):
		for j in range(col-2):
			temp = x[i:i+3,i:i+3]*(kernel.T)
			summation = -np.sum(temp)
			#print(summation)
			out.append(summation)
	return np.array(out).reshape(26,26).T


def painter(matrix,conved):
	fig = plt.figure()
	a = fig.add_subplot(1, 2, 1)
	imgplot = plt.imshow(matrix, cmap='gray')
	a.set_title('Given image')
	a = fig.add_subplot(1, 2, 2)
	imgplot = plt.imshow(conved, cmap='gray')
	a.set_title('After convolution')
	plt.savefig(path+'Result.png')
	#plt.show()

conved = convolute(matrix,conv_kernel)
np.savetxt(path + 'conved', conved,fmt='%3d', delimiter=' ')
painter(matrix,conved)

print('Mission Completed!')
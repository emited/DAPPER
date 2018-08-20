import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/home/debezenac/packages/manta/tensorflow/tools")
import uniio

# root = 'uni/from_init/simSimple_1000'
root = 'uni/simSimple_1000'
# root = 'uni/no_pressure_step_at_init/simSimple_1000'
den_fn = os.path.join(root, 'density_%04d.uni')
vel_fn = os.path.join(root, 'vel_%04d.uni')

densities = []
velocities = []
for i in range(0, 500, 30):
	header1, content1 = uniio.readUni(den_fn % i) # returns [Z,Y,X,C] np array
	header2, content2 = uniio.readUni(vel_fn % i) # returns [Z,Y,X,C] np array
	h1 = header1['dimX']
	w1 = header1['dimY']
	h2 = header2['dimX']
	w2 = header2['dimY']
	arr = content1[:, : :-1, :, :] # reverse order of Y axis
	arr = np.reshape(arr, [w1, h1, 1]) # discard Z
	densities.append(arr)
	arr = content2[:, : :-1, :, :] # reverse order of Y axis
	arr = np.reshape(arr, [w2, h2, 3]) # discard Z
	velocities.append(arr)

	plt.subplot(1, 2, 1)
	plt.title('{}, velocities {}'.format(root, i))
	plt.imshow(velocities[-1][:, :, 1])
	plt.subplot(1, 2, 2)
	plt.title('{} densities {}'.format(root, i))
	plt.imshow(densities[-1].squeeze())
	plt.show()

	# print(arr.)
velocities = np.array(velocities)
densities = np.array(densities)
print('vel', velocities.shape, 'den', densities.shape)
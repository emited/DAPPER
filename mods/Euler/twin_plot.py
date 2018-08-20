import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/home/debezenac/packages/manta/tensorflow/tools")
import uniio

interval = 5

# root_1 = 'uni/simSimple_1000'
# root_2 = 'uni/from_init/simSimple_1000'

# root_1 = '/home/debezenac/projects/DAPPER/mods/Euler/uni/from_init/with_pressuresimSimple_1000'
# root_2 = '/home/debezenac/projects/DAPPER/mods/Euler/uni/from_init/with_pressure_reloadsimSimple_1000'

# root_1 = '/home/debezenac/projects/DAPPER/mods/Euler/uni/from_init/with_pressuresimSimple_1000'
# root_2 = '/home/debezenac/projects/DAPPER/mods/Euler/uni/from_init/no_pressure_reloadsimSimple_1000'

root_1 = '/home/debezenac/projects/DAPPER/mods/Euler/uni/from_init/with_pressuresimSimple_1000'
root_2 = '/tmp/tmpdz9t3bpr/'#'/home/debezenac/projects/DAPPER/mods/Euler/uni/test2'

# root = 'uni/no_pressure_step_at_init/simSimple_1000'
den_fn_1 = os.path.join(root_1, 'density_%04d.uni')
vel_fn_1 = os.path.join(root_1, 'vel_%04d.uni')

den_fn_2 = os.path.join(root_2, 'density_%04d.uni')
vel_fn_2 = os.path.join(root_2, 'vel_%04d.uni')

densities = []
velocities = []
for i in range(0, 500, interval):
	header1, content1 = uniio.readUni(den_fn_1 % (i + 1)) # returns [Z,Y,X,C] np array
	header2, content2 = uniio.readUni(vel_fn_1 % (i + 1)) # returns [Z,Y,X,C] np array
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

	plt.subplot(2, 2, 1)
	plt.title('{}, velocities {}'.format(root_1, i))
	plt.imshow(velocities[-1][:, :, 1])
	plt.subplot(2, 2, 2)
	plt.title('{} densities {}'.format(root_1, i))
	plt.imshow(densities[-1].squeeze())


	header1, content1 = uniio.readUni(den_fn_2 % i) # returns [Z,Y,X,C] np array
	header2, content2 = uniio.readUni(vel_fn_2 % i) # returns [Z,Y,X,C] np array
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
	plt.subplot(2, 2, 3)
	plt.title('{}, velocities {}'.format(root_2, i))
	plt.imshow(velocities[-1][:, :, 1])
	plt.subplot(2, 2, 4)
	plt.title('{} densities {}'.format(root_2, i))
	plt.imshow(densities[-1].squeeze())


	plt.show()
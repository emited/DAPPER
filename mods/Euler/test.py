import os
from manta import *
import sys
sys.path.append("/home/debezenac/packages/manta/tensorflow/tools")
import uniio

root_2 = '/home/debezenac/projects/DAPPER/mods/Euler/uni/test2'
vel_fn_2 = os.path.join(root_2, 'vel_%04d.uni')
den_fn_2 = os.path.join(root_2, 'density_%04d.uni')

i =0
header1, content1 = uniio.readUni(den_fn_2 % (i + 1)) # returns [Z,Y,X,C] np array
header2, content2 = uniio.readUni(vel_fn_2 % (i + 1)) # returns [Z,Y,X,C] np array

import numpy as np
print('content', content1.shape, content2.shape)
print('n', np.sum(content2[:, :, :, 0]==0))
print('n', np.sum(content2[:, :, :, 1]==0))
print('n', np.sum(content2[:, :, :, 2]==0))
print(64**2)

from collections import OrderedDict
import time

def generate_header(content):
	'''be careful, the order in the header counts!'''

	header = OrderedDict([
		('dimX', 64),
		('dimY', 64),
		('dimZ', 1),
	])

	# check if density or velocity
	if content.shape[3] == 1: # density
		header['gridType'] = 1
		header['elementType'] = 1
		header['bytesPerElement'] = 4
		header['info'] = b'mantaflow 0.12 64bit fp1 omp commit 15eaf4aa72da62e174df6c01f85ccd66fde20acc from Aug 14 2018, 15:47:38\x00`\xb6a]0\x7f\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x90\xb1PI0\x7f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@\na]0\x7f\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00 \x00\x00\x000\x00\x00\x00\x90\xb2PI0\x7f\x00\x00\xd0\xb1PI0\x7f\x00\x00\x10\xb6PI0\x7f\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x000\xbd\x86I0\x7f\x00\x00\x10\xb6PI0\x7f\x00\x00\xf0\xb5PI0\x7f\x00\x00\xf0\xb5PI0\x7f\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00 \xb6PI'
	elif content.shape[3] == 3: # velocity
		header['gridType'] = 12
		header['elementType'] = 2
		header['bytesPerElement'] = 12
		header['info'] = b'mantaflow 0.12 64bit fp1 omp commit 15eaf4aa72da62e174df6c01f85ccd66fde20acc from Aug 14 2018, 15:47:38\x00\x90\xb2PI0\x7f\x00\x00\xf8\x8e\x0eP0\x7f\x00\x00\n\x00\x00\x00\x00\x00\x00\x00f\x8b:]0\x7f\x00\x00@\xb6PI0\x7f\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\xb0\xb1PI0\x7f\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00 \x00\x00\x000\x00\x00\x00\x90\xb2PI0\x7f\x00\x00\xd0\xb1PI0\x7f\x00\x00\x10\xb6PI0\x7f\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00\x068*0\x7f\x00\x00\x10\xb6PI0\x7f\x00\x00\xf0\xb5PI0\x7f\x00\x00\xf0\xb5PI0\x7f\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00 \xb6PI'
	else:
		raise NotImplementedError(
			'Does not correspond to density or velocity'
		)
		
	header['dimT'] = 0
	header['timestamp'] = int(time.time())


	return header

# sanity check
uniio.writeUni('test.uni', generate_header(content1), content1)
_, content1_new = uniio.readUni('test.uni')
print((content1 != content1_new).sum())

uniio.writeUni('test.uni', generate_header(content2), content2)
_, content2_new = uniio.readUni('test.uni')
print((content2 != content2_new).sum())



# print(dir(uniio))
exit()
res    = 64
dim    = 2 
offset = 20
interval = 1

scaleFactor = 4

gs = vec3(res,res, 1 if dim==2 else res )
buoy  = vec3(0,-1e-3,0)

# wlt Turbulence input fluid
sm = Solver(name='zero_border', gridSize = gs, dim=dim)
sm.timestep = 0.5

timings = Timings()

# Simulation Grids  -------------------------------------------------------------------#
flags    = sm.create(FlagGrid)
vel      = sm.create(MACGrid)
density  = sm.create(RealGrid)
pressure = sm.create(RealGrid)

print(type(flags), dir(flags))
import argparse
from manta import *
import os, shutil, math, sys, time
import numpy as np

from mods.Euler.utils import generate_header

def load_state(root, step=0):
	"""loads velocities and densities, and writes them inplace"""
	den_init_fn = os.path.join(root, 'density_%04d.uni' % step)
	vel_init_fn = os.path.join(root, 'vel_%04d.uni' % step)
	density.load(den_init_fn)
	vel.load(vel_init_fn)
	return vel, density


def random_init(offset=20):
	""" Initialize initial conditions randomly"""

	# setOpenBound(flags,    bWidth,'yY',FlagOutflow|FlagEmpty) 

	# inflow sources ----------------------------------------------------------------------#
	# Main params  ----------------------------------------------------------------------#
	npSeed = -1
	t = 0

	if(npSeed>0): np.random.seed(npSeed)

	# init random density
	noise    = []
	sources  = []

	noiseN = 12
	nseeds = np.random.randint(10000,size=noiseN)

	cpos = vec3(0.5,0.3,0.5)

	randoms = np.random.rand(noiseN, 8)
	for nI in range(noiseN):
		noise.append( sm.create(NoiseField, fixedSeed= int(nseeds[nI]), loadFromFile=True) )
		noise[nI].posScale = vec3( res * 0.1 * (randoms[nI][7] + 1) )
		noise[nI].clamp = True
		noise[nI].clampNeg = 0
		noise[nI].clampPos = 1.0
		noise[nI].valScale = 1.0
		noise[nI].valOffset = -0.01 # some gap
		noise[nI].timeAnim = 0.3
		noise[nI].posOffset = vec3(1.5)
		
		# random offsets
		coff = vec3(0.4) * (vec3( randoms[nI][0], randoms[nI][1], randoms[nI][2] ) - vec3(0.5))
		radius_rand = 0.035 + 0.035 * randoms[nI][3]
		upz = vec3(0.95)+ vec3(0.1) * vec3( randoms[nI][4], randoms[nI][5], randoms[nI][6] )
		if(dim == 2): 
			coff.z = 0.0
			upz.z = 1.0
		sources.append(sm.create(Sphere, center=gs*(cpos+coff), radius=gs.x*radius_rand, scale=upz)) 
		densityInflow( flags=flags, density=density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
		print (nI, "centre", gs*(cpos+coff), "radius", gs.x*radius_rand, "other", upz ) 

	# init random velocity
	Vrandom = np.random.rand(3)
	v1pos = vec3(0.7 + 0.4 *(Vrandom[0] - 0.5) ) #range(0.5,0.9) 
	v2pos = vec3(0.3 + 0.4 *(Vrandom[1] - 0.5) ) #range(0.1,0.5)
	vtheta = Vrandom[2] * math.pi * 0.5
	velInflow = 0.04 * vec3(math.sin(vtheta), math.cos(vtheta), 0)

	if(dim == 2):
		v1pos.z = v2pos.z = 0.5
		sourcV1 = sm.create(Sphere, center=gs*v1pos, radius=gs.x*0.1, scale=vec3(1))
		sourcV2 = sm.create(Sphere, center=gs*v2pos, radius=gs.x*0.1, scale=vec3(1))
		sourcV1.applyToGrid( grid=vel , value=(-velInflow*float(gs.x)) )
		sourcV2.applyToGrid( grid=vel , value=( velInflow*float(gs.x)) )
	elif(dim == 3):
		VrandomMore = np.random.rand(3)
		vtheta2 = VrandomMore[0] * math.pi * 0.5
		vtheta3 = VrandomMore[1] * math.pi * 0.5
		vtheta4 = VrandomMore[2] * math.pi * 0.5
		for dz in range(1,10,1):
			v1pos.z = v2pos.z = (0.1*dz)
			vtheta_xy = vtheta *(1.0 - 0.1*dz ) + vtheta2 * (0.1*dz)
			vtheta_z  = vtheta3 *(1.0 - 0.1*dz ) + vtheta4 * (0.1*dz)
			velInflow = 0.04 * vec3( math.cos(vtheta_z) * math.sin(vtheta_xy), math.cos(vtheta_z) * math.cos(vtheta_xy),  math.sin(vtheta_z))
			sourcV1 = sm.create(Sphere, center=gs*v1pos, radius=gs.x*0.1, scale=vec3(1))
			sourcV2 = sm.create(Sphere, center=gs*v2pos, radius=gs.x*0.1, scale=vec3(1))
			sourcV1.applyToGrid( grid=vel , value=(-velInflow*float(gs.x)) )
			sourcV2.applyToGrid( grid=vel , value=( velInflow*float(gs.x)) )

	print('solver', dir(sm))
	while t < offset:
		curt = t * sm.timestep
		mantaMsg( "Current time t: " + str(curt) +" \n" )
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, openBounds=True, boundaryWidth=bWidth)
		advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2, openBounds=True, boundaryWidth=bWidth)
		setWallBcs(flags=flags, vel=vel)
		addBuoyancy(density=density, vel=vel, gravity=buoy , flags=flags)
		vorticityConfinement( vel=vel, flags=flags, strength=0.05 )

		# if savedata and t>=offset and (t-offset)%interval==0:
		# 	tf = (t-offset)/interval
		# 	vel.save(simPath + 'vel_no_pressure_%04d.uni' % (tf))

		solvePressure(flags=flags, vel=vel, pressure=pressure ,  cgMaxIterFac=10.0, cgAccuracy=0.0001 )
		setWallBcs(flags=flags, vel=vel)

		sm.step()
		t = t+1			


def euler_step():
	advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, openBounds=True, boundaryWidth=bWidth)
	advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2, openBounds=True, boundaryWidth=bWidth)
	setWallBcs(flags=flags, vel=vel)
	addBuoyancy(density=density, vel=vel, gravity=buoy , flags=flags)
	solvePressure(flags=flags, vel=vel, pressure=pressure ,  cgMaxIterFac=10.0, cgAccuracy=0.0001 )
	setWallBcs(flags=flags, vel=vel)
	sm.step()


def navier_stokes_step():
	raise NotImplementedError()


parser = argparse.ArgumentParser(description='')
parser.add_argument('--init', metavar='DIR', default='')
parser.add_argument('--save', metavar='DIR', default='')
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--init-step', type=int, default=0)
parser.add_argument('--gui', dest='show_gui', action='store_true')


# Scene settings  ---------------------------------------------------------------------#
setDebugLevel(1)

# Solver params  ----------------------------------------------------------------------#
res    = 64
dim    = 2 
# interval = 1

scaleFactor = 4

gs = vec3(res,res, 1 if dim==2 else res )
buoy  = vec3(0,-1e-3,0)

# wlt Turbulence input fluid
sm = Solver(name='zero_border', gridSize=gs, dim=dim)
sm.timestep = 0.5

timings = Timings()

# Simulation Grids  -------------------------------------------------------------------#
flags    = sm.create(FlagGrid)
vel      = sm.create(MACGrid)
density  = sm.create(RealGrid)
pressure = sm.create(RealGrid)


# open boundaries
bWidth=1
flags.initDomain(boundaryWidth=bWidth)
flags.fillGrid()

if __name__ == "__main__":

	args = parser.parse_args()

	if args.init:
		print('loading Initial conditions from ' + args.init)
		load_state(args.init)
	else:
		print('Random Initial conditions')
		random_init()

	# Setup UI ---------------------------------------------------------------------#
	if (args.show_gui and GUI):
		gui=Gui()
		gui.show()
		gui.pause()

	t = 0
	resetN = 20

	if args.save:
		os.makedirs(args.save)
		print("using output dir" + args.save)


	# main loop --------------------------------------------------------------------#
	while t < args.steps:
		curt = t * sm.timestep
		mantaMsg( "Current time t: " + str(curt) +" \n" )
		
		step()

		# save data
		if args.save and t%args.interval==0:
			tf = t/args.interval
			density.save(os.path.join(args.save, 'density_%04d.uni' % (tf)))
			vel.save(os.path.join(args.save, 'vel_%04d.uni' % (tf)))
			print('saving to ' + args.save)

		t = t+1


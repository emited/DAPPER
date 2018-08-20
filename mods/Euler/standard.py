import os
import tempfile
import sys
from functools import partial


sys.path.append("/home/debezenac/packages/manta/tensorflow/tools")
import uniio


from common import *

from mods.Euler.core import vel, density, euler_step, sm
from mods.Euler.utils import generate_header


sim_dir = tempfile.mkdtemp()
print('simulation dir: ' + sim_dir)


def load_initial_conditions():
	root0 = '/home/debezenac/projects/DAPPER/mods/Euler/uni/simSimple_1000'
	vel0_fn = os.path.join(root0, 'vel_%04d.uni')
	density0_fn = os.path.join(root0, 'density_%04d.uni')
	_, np_density0 = uniio.readUni(density0_fn % 0)
	_, np_vel0 = uniio.readUni(vel0_fn % 0)
	x0 = zeros((1, 64, 64, 3))
	x0[:, :, :, [0]] = np_density0
	x0[:, :, :, 1:] = np_vel0[:, :, :, :2] # manta takes 3d velocities
	x0 = x0.reshape(3 * 64 * 64)
	return x0

def step_1(x, t, dt):

	# from numpy array to uni file
	vel_fn = os.path.join(sim_dir, 'vel_%04d.uni')
	density_fn = os.path.join(sim_dir, 'density_%04d.uni')

	xr = x.reshape(1, 64, 64, 3)
	np_density = xr[:, :, :, [0]]
	np_vel = zeros((1, 64, 64, 3)) # manta takes 3d velocities
	np_vel[:, :, :, :2] = xr[:, :, :, 1:]

	uniio.writeUni(density_fn % (t / dt), generate_header(np_density), np_density)
	uniio.writeUni(vel_fn % (t / dt), generate_header(np_vel), np_vel)

	# loading written data to state
	density.load(density_fn % (t / dt))
	vel.load(vel_fn % (t / dt))

	# applying model
	euler_step()

	# writing to uni
	density.save(density_fn % ((t + dt) / dt))
	vel.save(vel_fn % ((t + dt) / dt))

	# loading back as numpy array
	_, np_density = uniio.readUni(density_fn % ((t + dt) / dt))
	_, np_vel = uniio.readUni(vel_fn % ((t + dt) / dt))

	xn = zeros((xr.shape[0], 64, 64, 3))
	xn[:, :, :, [0]] = np_density
	xn[:, :, :, 1:] = np_vel[:, :, :, :2] # manta takes 3d velocities
	xn = xn.reshape(1, 3 * 64 * 64)

	return xn


from tools.utils import multiproc_map
def step(E, t, dt_):
  """Vector and 2D-array (ens) input, with multiproc for ens case."""
  if E.ndim==1:
    return step_1(E,t,dt_)
  if E.ndim==2:
    # Parallelized:
    # E = np.array(multiproc_map(step_1, E, t=t, dt=dt_))
    # Non-parallelized:
    for n,x in enumerate(E):
    	E[n] = step_1(x,t,dt_)
    return E



m = 3 * (64 ** 2)
p = 64 ** 2
t = Chronology(0.5,
	dkObs=1, # number of model steps between consecutive obs?
	T=10,
	BurnIn=-1
)


f = {
	'm': m,
	'model': step,
	'noise': 0,
}

# loading intial conditions


# mu0 = zeros((m))
mu0 = load_initial_conditions()
# mu0.reshape(64, 64, 3)[:, :, 1:] = 0

X0 = GaussRV(C=0.1, mu=mu0)
# X0.sample(1)


from tools.localization import inds_and_coeffs, unravel
xIJ = unravel(arange(m), (64,64, 3)) # 2-by-m
def locf(radius,direction,t,tag=None):
  """
  Prepare function:
  inds, coeffs = locf_at(state_or_obs_index)
  """
  yIJ = xIJ[:,obs_inds(t)] # 2-by-p
  def ic(cntr, domain):
    return inds_and_coeffs(cntr, domain, (64,64), radius, tag=tag)
  if direction is 'x2y':
    # Could pre-compute ic() for all xIJ,
    # but speed gain is not worth it.
    def locf_at(ind):
      return ic(xIJ[:,ind], yIJ)
  elif direction is 'y2x':
    def locf_at(ind):
      return ic(yIJ[:,ind], xIJ)
  else: raise KeyError
  return locf_at



jj = arange(0, 3 * 64 * 64, 3) # only take density dimension
h = partial_direct_obs_setup(m, jj)
h['noise'] = .01
h['loc_f'] = locf
setup = TwinSetup(f, h, t, X0)



# Test
# root = '/home/debezenac/projects/DAPPER/mods/Euler/uni/from_init/with_pressuresimSimple_1000'
# den_fn = os.path.join(root	, 'density_%04d.uni')
# vel_fn = os.path.join(root, 'vel_%04d.uni')
# h_density, c_density = uniio.readUni(den_fn % 0) # returns [Z,Y,X,C] np array
# h_vel, c_vel = uniio.readUni(vel_fn % 0) # returns [Z,Y,X,C] np array

# mu0 = zeros((1, 64, 64, 3))
# mu0[:, :, :, [0]] = c_density
# mu0[:, :, :, 1:] = c_vel[:, :, :, :2]
# for t in range(200):
# 	mu0 = step_1(mu0, t*0.5, 0.5)
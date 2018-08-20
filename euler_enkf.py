"""launch script with:
 /home/debezenac/packages/manta2/manta/build/manta manta.py
 """

from common import *

# Load "twin experiment" setup
from mods.Euler.standard import *


from os.path import join, dirname
import sys
sys.path.append(join(dirname(__file__), '/home/debezenac/projects/flow/flow/submodules/OpticalFlowToolkit/lib'))
import flowlib

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)



def plot_vel(x):
	plt.imshow(flowlib.flow_to_image(x.reshape(64, 64, 3)[:, :, 1:]))

def plot_density(x, min=None, max=None):
	plt.imshow(x.reshape(64, 64, 3)[:, :, 0], vmin=min, vmax=max)

def simulate_without_sample(setup,desc='Truth & Obs'):
  """Generate synthetic truth and observations."""
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0

  # Init
  xx    = zeros((chrono.K+1,f.m))

  xx[0] = X0.mu#sample(1)
  yy    = zeros((chrono.KObs+1,h.m))

  # Loop
  for k,kObs,t,dt in progbar(chrono.forecast_range,desc):
    xx[k] = f(xx[k-1],t-dt,dt) + sqrt(dt)*f.noise.sample(1)
    if kObs is not None:
      yy[kObs] = h(xx[k],t) + h.noise.sample(1)

  return xx,yy



# Specify a DA method configuration
setup.t.T = 10	
config = EnKF('Sqrt', N=25, infl=1.02, rot=True, liveplotting=True)
# config = LETKF(N=25,rot=True,infl=1.04,loc_rad=10,taper='Gauss') # 0.6

# Simulate synthetic truth (xx) and noisy obs (yy)
xx,yy = simulate_without_sample(setup)
xxn, yyn = simulate(setup)

# Assimilate yy (knowing the twin setup). Assess vis-a-vis xx.
stats = config.assimilate(setup,xx,yy)

for i, (mu, x, xn) in enumerate(zip(stats.mu.a, xx[1:], xxn[1:])):

	d = mu.reshape(64, 64, 3)[:, :, 0]

	mx = np.max([x.reshape(64, 64, 3)[:, :, 0],
		mu.reshape(64, 64, 3)[:, :, 0],
		xn.reshape(64, 64, 3)[:, :, 0]]
	)
	
	mn = np.min([x.reshape(64, 64, 3)[:, :, 0],
		mu.reshape(64, 64, 3)[:, :, 0],
		xn.reshape(64, 64, 3)[:, :, 0]]
	)

	plt.subplot(2, 3, 1)
	plt.title('mu ' + str(i))
	plot_density(mu, min=mn, max=mx)
	plt.subplot(2, 3, 2)
	plt.title('x ' + str(i))
	plot_density(x, min=mn, max=mx)
	plt.subplot(2, 3, 3)
	plt.title('xn')
	plot_density(xn, min=mn, max=mx)

	plt.subplot(2, 3, 4)
	plot_vel(mu)
	plt.subplot(2, 3, 5)
	plot_vel(x)
	plt.subplot(2, 3, 6)
	plot_vel(xn)

	print('saving' + 'mods/Euler/images/euler_letkf_{}.png'.format(i))
	plt.savefig('mods/Euler/images/euler_letkf_{}.png'.format(i))

# np.sum(stats.mu.a - yy)
# Average stats time series
# avrgs = stats.average_in_time()

# Print averages
# print_averages(config,avrgs,[],['rmse_a','rmv_a'])

# Plot some diagnostics 
# plot_time_series(stats)

# "Explore" objects individually
#print(setup)
#print(config)
#print(stats)
#print(avrgs)




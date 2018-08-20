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



def plot_vel(xd):
	plt.axis('off')
	plt.imshow(flowlib.flow_to_image(xd.reshape(64, 64, 3)[:, :, 1:]))

def plot_density(xd, min=None, max=None):
	plt.axis('off')
	plt.imshow(xd.reshape(64, 64, 3)[:, :, 0], vmin=min, vmax=max)

def simulate_without_sample(setup,desc='Truth & Obs'):
  """Generate synthetic truth and observations."""
  f,h,chrono,X0 = setup.f, setup.h, setup.t, setup.X0

  # Init
  xxd    = zeros((chrono.K+1,f.m))

  xxd[0] = X0.mu#sample(1)
  yyd    = zeros((chrono.KObs+1,h.m))

  # Loop
  for k,kObs,t,dt in progbar(chrono.forecast_range,desc):
    xxd[k] = f(xxd[k-1],t-dt,dt) + sqrt(dt)*f.noise.sample(1)
    if kObs is not None:
      yyd[kObs] = h(xxd[k],t) + h.noise.sample(1)

  return xxd,yyd


# Specify a DA method configuration
setup.t.T = 10	
config = EnKF('Sqrt', N=5, infl=1.02, rot=True, liveplotting=True)
# config = LETKF(N=25,rot=True,infl=1.04,loc_rad=10,taper='Gauss') # 0.6

# Simulate synthetic truth (xxd) and noisy obs (yyd)
xxd, yyd = simulate_without_sample(setup)
xxs, yys = simulate(setup)

# Assimilate yyd (knowing the twin setup). Assess vis-a-vis xxd.
stats = config.assimilate(setup,xxd,yys)

print('\n\n', stats.mu.__dict__ ,'\n\n')


plt.subplot(2, 2, 1)
plt.title('xd')
plot_density(xxd[0])
plt.subplot(2, 2, 3)
plot_vel(xxd[0])
plt.subplot(2, 2, 2)
plt.title('xs')
plot_density(xxs[0])
plt.subplot(2, 2, 4)
plot_vel(xxs[0])
save_fn = 'mods/Euler/images/euler_letkf_{}.png'.format(0)
print('saveing initial conditions to ' + save_fn)
plt.savefig(save_fn)

for t, (muf, mua, xd, xs) in enumerate(zip(stats.mu.f, stats.mu.a, xxd[1:], xxs[1:])):

	ims = {
		'mu.f': muf,
		'mu.a': mua,
		'xd': xd,
		'xs': xs,
	}

	mn = np.min([im.reshape(64, 64, 3)[:, :, 0] for im in ims.values()])
	mx = np.max([im.reshape(64, 64, 3)[:, :, 0] for im in ims.values()])

	for i, (k, v) in enumerate(ims.items()):
		plt.subplot(2, len(ims), i + 1)
		plt.title('{} {}'.format(k, t + 1))
		plot_density(v, min=mn, max=mx)
		plt.subplot(2, len(ims), i + len(ims) + 1)
		plot_vel(v)

	save_fn = 'mods/Euler/images/euler_letkf_{}.png'.format(t + 1)
	print('saving to ' + save_fn)
	plt.savefig(save_fn)


# np.sum(stats.mu.a - yyd)
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




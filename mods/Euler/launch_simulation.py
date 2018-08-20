import os

manta_bin_fn = '/home/debezenac/packages/manta/build/manta'
manta_gen_fn = 'to_launch.py'
manta_gen_fn += ' --init /home/debezenac/projects/DAPPER/mods/Euler/uni/from_init/with_pressuresimSimple_1000'
manta_gen_fn += ' --steps 30'
manta_gen_fn += ' --save /home/debezenac/projects/DAPPER/mods/Euler/uni/from_init/test'
manta_gen_fn += ' --init-step 0'


print(manta_bin_fn + ' ' + manta_gen_fn)
os.system(manta_bin_fn + ' ' + manta_gen_fn)
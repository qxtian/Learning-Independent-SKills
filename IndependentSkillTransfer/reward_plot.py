import matplotlib.pyplot as plt
import numpy as np
import re
import visdom



# progress_folder = '/home/weikaichen/hegsns/Python Proj/high-level-skill/network_reserve/half_cheetah_hurdle_v0_model_1210/HalfCheetah_hurdle-v0/HalfCheetah_hurdle-v0_s0'
# progress_folder = './network_reserve/sac_gpu/HalfCheetah_hurdle-v0/s0'
# progress_folder = './network_reserve/model_2323_diversity_trans_matrix_gpu/HalfCheetah_hurdle-v0/s0/'
# progress_folder = './network_reserve/model_1210/HalfCheetah_hurdle-v0/s0'
# progress_folder = './network_reserve/model_2323_diversity_trans_matrix_gpu/HalfCheetah_hurdle-v2/s0'
# progress_folder = './network_reserve/sac_gpu/HalfCheetah_hurdle-v2/s0'
progress_folder = './network_reserve/model_1211/HalfCheetah_hurdle-v2/s0'
# progress_folder = './network_reserve/model_2323_diversity_combine/HalfCheetah_hurdle-v0/s0'
# progress_folder = './network_reserve/model_1128_diversity_combine/HalfCheetah_hurdle-v0/s0'

fname = '/progress-1-2.8.txt'

progress = open(progress_folder + fname)
raw_data = progress.read()

pattern = re.compile(r'\d+\t[-+]?\d+.\d+\t\d+\n')
raw_data2 = pattern.findall(raw_data)
print(raw_data2)
eplisode_buf = []
reward_buf = []
success_buf = [0]
for idx, value in enumerate(raw_data2):
    # if idx >= 4:
    eplisode_str, reward_str, success_str = value.split('\t')
    eplisode_buf.append(int(eplisode_str))
    reward_buf.append(float(reward_str[0:-2]))
    success_buf.append(success_buf[-1] + int(success_str))

vis = visdom.Visdom()
win = vis.line(
          X=np.array(eplisode_buf),
          Y=np.array(success_buf[1:]),
          opts=dict(
            # xtickmin=-2,
            # xtickmax=2,
            # xtickstep=1,
            # ytickmin=-1,
            # ytickmax=5,
            # ytickstep=1,
            title=(fname),
            markersymbol='dot',
            markersize=5,
        ),
        # update="new",
        name="1",
      )
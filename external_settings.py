vid_input = "_91sarahclose.mp4"
# 97 on stablewarp images_out has a good seed
init_image = "/notebooks/images_out/_93archtrimlonger/_93archtrimlonger(115)_000001.png"

width =  1024
height = 576 

michael_mode = True

force_vid_extract = False

sd_model_path = "/storage/deforumsd/models/sd-v1-4.ckpt"  # @param {'type':'string'}

import math
import numpy as np
import random

styled_seconds = 1
ramp_seconds = 5
psych_poster_seed = 245114
seed = random.random() * 1000000
steps_schedule = [150]
strength_schedule = [.2] * 24 + list(np.linspace(0.2, 0.05, 36))
flow_blend_schedule = list(np.linspace(0.99, 0.6, 60))
#     np.linspace(0.4, 0, 24 * 1)
# )
turbo_steps_schedule = [5]
# turbo_steps_schedule = [
#     math.ceil(n) for n in list(np.linspace(5, 2.01, 24 * (styled_seconds + ramp_seconds - 1)))
# ] + [
#     1
# ]  # Could also turn on turbo colormatch here

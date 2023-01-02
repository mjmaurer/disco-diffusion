vid_input = "_89timdance.mp4"
# 97 on stablewarp images_out has a good seed
init_image = "/notebooks/deforum-output/deforum-texture/2022-12-31-15-13/20221231151334_01976.png"

width = 1024 # 960 # 832 # 1024
height = 576 # 512      # 576 

michael_mode = True

force_vid_extract = False

sd_model_path = "/storage/deforumsd/models/sd-v1-4.ckpt"  # @param {'type':'string'}

import math
import numpy as np
import random

styled_seconds = 1
ramp_seconds = 5
strength_schedule = (
    [0.3] 
)
psych_poster_seed = 245114
seed = -1 # 1261233236 #660008352 #925432632 # 245114 # random.random() * 1000000
flow_blend_schedule = (
    [0.9]
    # [0.9] * (24 * styled_seconds) # .999 before
    # + list(np.linspace(0.9, 0.55, 24 * (ramp_seconds - 1)))
    # + list(np.linspace(0.55, 0, 24))
    # # + list(np.linspace(0.8, 0, 24))
)
steps_schedule = [150]
# flow_blend_schedule = list(np.linspace(0.999, 0.4, 24 * (seconds - 1))) + list(
#     np.linspace(0.4, 0, 24 * 1)
# )
turbo_steps_schedule = [2]
# turbo_steps_schedule = [
#     math.ceil(n) for n in list(np.linspace(5, 2.01, 24 * (styled_seconds + ramp_seconds - 1)))
# ] + [
#     1
# ]  # Could also turn on turbo colormatch here

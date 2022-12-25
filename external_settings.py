vid_input = "_93archtrimlonger.mp4"
# 97 on stablewarp images_out has a good seed
init_image = "output/_93archtrimlonger/_990_12_25__00_09/_93archtrimlonger(0)_0000.png"

width = 1024
height = 576

michael_mode = True

force_vid_extract = False

sd_model_path = "/storage/deforumsd/models/sd-v1-4.ckpt"  # @param {'type':'string'}

import math
import numpy as np
import random

styled_seconds = 1
ramp_seconds = 5
turbo_steps = 5
strength_schedule = (
    [0.75] + [0.35] * (24 * styled_seconds) + list(np.linspace(0.35, 0.01, 24 * ramp_seconds))
)
frames_skip_steps_schedule_vid_input = (
    [0.25] + [0.65] * (24 * styled_seconds) + list(np.linspace(0.65, 0.99, 24 * ramp_seconds))
)
frames_skip_steps_schedule_3d = list(np.linspace(0.75, 0.35, turbo_steps * 3))
frames_skip_steps_schedule_3d = frames_skip_steps_schedule_3d * 20
frames_skip_steps_schedule = frames_skip_steps_schedule_3d
frames_skip_steps_schedule = (
    [0.75] + frames_skip_steps_schedule + [frames_skip_steps_schedule[-1]] * 500
)
psych_poster_seed = 245114
seed = 798329237  # random.random() * 1000000
flow_blend_schedule = (
    [0.9] * (24 * styled_seconds)  # .999 before
    + list(np.linspace(0.9, 0.55, 24 * (ramp_seconds - 1)))
    + list(np.linspace(0.55, 0, 24))
    # + list(np.linspace(0.8, 0, 24))
)
flow_blend_schedule = flow_blend_schedule + [flow_blend_schedule[-1]] * 500
steps_schedule = [150]
# flow_blend_schedule = list(np.linspace(0.999, 0.4, 24 * (seconds - 1))) + list(
#     np.linspace(0.4, 0, 24 * 1)
# )
# turbo_steps_schedule = [
#     math.ceil(n) for n in list(np.linspace(5, 2.01, 24 * (styled_seconds + ramp_seconds - 1)))
# ] + [
#     1
# ]  # Could also turn on turbo colormatch here

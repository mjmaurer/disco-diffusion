vid_input = "_93archtrimlonger.mp4"
# 97 on stablewarp images_out has a good seed
init_image = "/notebooks/images_out/_93archtrimlonger/_93archtrimlonger(99)_000001.png"

width = 1024
height = 576

michael_mode = True

force_vid_extract = False

sd_model_path = "/storage/deforumsd/models/sd-v1-4.ckpt"  # @param {'type':'string'}

import math
import numpy as np

styled_seconds = 4
ramp_seconds = 5
strength_schedule = (
    [0.75] + [0.35] * (24 * styled_seconds) + list(np.linspace(0.35, 0, 24 * ramp_seconds))
)
psych_poster_seed = 785922
seed = random.random() * 1000000
flow_blend_schedule = (
    [0.999] * (24 * styled_seconds)
    + list(np.linspace(0.999, 0.55, 24 * (ramp_seconds - 1)))
    + [0]
    # + list(np.linspace(0.8, 0, 24))
)
steps_schedule = [150]
# flow_blend_schedule = list(np.linspace(0.999, 0.4, 24 * (seconds - 1))) + list(
#     np.linspace(0.4, 0, 24 * 1)
# )
turbo_steps_schedule = [5] * (24 * (styled_seconds + ramp_seconds)) + [1]
# turbo_steps_schedule = [
#     math.ceil(n) for n in list(np.linspace(5, 2.01, 24 * (styled_seconds + ramp_seconds - 1)))
# ] + [
#     1
# ]  # Could also turn on turbo colormatch here

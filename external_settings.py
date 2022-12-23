vid_input = "_93archtrimlonger.mp4"
init_image = "/notebooks/images_out/_93archtrimlonger/_93archtrimlonger(70)_000002.png"

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
    [0.55] + [0.25] * (24 * styled_seconds) + list(np.linspace(0.25, 0, 24 * ramp_seconds))
)
flow_blend_schedule = (
    [0.999] * (24 * styled_seconds)
    + list(np.linspace(0.999, 0.55, 24 * (ramp_seconds - 1)))
    + [0]
    # + list(np.linspace(0.8, 0, 24))
)
steps_schedule = [180]
# flow_blend_schedule = list(np.linspace(0.999, 0.4, 24 * (seconds - 1))) + list(
#     np.linspace(0.4, 0, 24 * 1)
# )
turbo_steps_schedule = [
    math.ceil(n) for n in list(np.linspace(5, 2.01, 24 * (styled_seconds + ramp_seconds - 1)))
] + [
    1
]  # Could also turn on turbo colormatch here

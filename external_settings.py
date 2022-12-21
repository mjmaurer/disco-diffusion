vid_input = "_93archtrimlonger.mp4"

michael_mode = True

force_vid_extract = False

sd_model_path = "/storage/deforumsd/models/sd-v1-4.ckpt"  # @param {'type':'string'}

import math
import numpy as np

styled_seconds = 4
ramp_seconds = 4
strength_schedule = [0.6] + [.27] * (24 * styled_seconds) + list(np.linspace(0.27, 0, 24 * ramp_seconds))
flow_blend_schedule = [0.999] * (24 * styled_seconds) + list(
    np.linspace(0.999, 0.75, 24 * ramp_seconds)
)
steps_schedule = [180]
# flow_blend_schedule = list(np.linspace(0.999, 0.4, 24 * (seconds - 1))) + list(
#     np.linspace(0.4, 0, 24 * 1)
# )
turbo_steps_schedule = [
    math.ceil(n) for n in list(np.linspace(5, 2.01, 24 * (styled_seconds + ramp_seconds - 1)))
]

vid_input = "_94archtrim.mp4"

michael_mode = True

force_vid_extract = False

sd_model_path = "/storage/deforumsd/models/sd-v1-4.ckpt"  # @param {'type':'string'}

import math
import numpy as np

seconds = 7
strength_schedule = [0.52] + list(np.linspace(0.3, 0, 24 * seconds))
steps_schedule = [180]
flow_blend_schedule = list(np.linspace(0.999, 0.4, 24 * (seconds - 1))) + list(
    np.linspace(0.4, 0, 24 * 1)
)
turbo_steps_schedule = map(math.ceil, list(np.linspace(5, 0.01, 24 * (seconds - 1))))

vid_input = "_94archtrim.mp4"

michael_mode = True

force_vid_extract = False

sd_model_path = "/storage/deforumsd/models/sd-v1-4.ckpt"  # @param {'type':'string'}

import numpy as np

strength_schedule = [0.35] + list(np.linspace(0.2, 0, 24 * 3))
steps_schedule = [160]
flow_blend_schedule = list(np.linspace(0.999, 0.2, 24 * 3))

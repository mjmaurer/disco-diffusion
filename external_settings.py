vid_input = "_94archtrim.mp4"

michael_mode = True

force_vid_extract = False

sd_model_path = "/storage/deforumsd/models/sd-v1-4.ckpt"  # @param {'type':'string'}

import numpy as np
strength_schedule = list(np.linspace(0.8, .1, 24 * 5))
steps_schedule = [160] 
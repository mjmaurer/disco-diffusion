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
turbo_steps = 6
strength_schedule = (
    [0.75] + [0.35] * (24 * styled_seconds) + list(np.linspace(0.35, 0.01, 24 * ramp_seconds))
)
frames_skip_steps_schedule_vid_input = (
    [0.25] + [0.65] * (24 * styled_seconds) + list(np.linspace(0.65, 0.99, 24 * ramp_seconds))
)
frames_skip_steps_schedule_3d = [.75] * (turbo_steps * 2 - 2) + [.2] * 2 # list(np.linspace(0.75, 0.25, turbo_steps * 2))
frames_skip_steps_schedule_3d = frames_skip_steps_schedule_3d * 20
frames_skip_steps_schedule = frames_skip_steps_schedule_3d
frames_skip_steps_schedule = (
    [0.75] + frames_skip_steps_schedule + [frames_skip_steps_schedule[-1]] * 500
)
psych_poster_seed = 245114
seed = random.random() * 1000000
flow_blend_schedule = (
    [0.9] * (24 * styled_seconds)  # .999 before
    + list(np.linspace(0.9, 0.55, 24 * (ramp_seconds - 1)))
    + list(np.linspace(0.55, 0, 24))
    # + list(np.linspace(0.8, 0, 24))
)
flow_blend_schedule = flow_blend_schedule + [flow_blend_schedule[-1]] * 500
translation_x = "0:(2 * sin(3.14*t/100))"  # @param {type:"string"}
translation_y = "0: (0)"  # @param {type:"string"}
translation_z = "0: (2.5)"  # @param {type:"string"}
rotation_3d_x = "0:(.5 * sin(3.14*t/150))"  # @param {type:"string"}
rotation_3d_y = "0:(0), 12:(-1.15 * sin(3.14*t/110))"  # @param {type:"string"}
rotation_3d_z = "0: (.5 * cos(3.14*t/90))"  # @param {type:"string"}
rotation_y_trigger = list(np.linspace(0, 1, 10)) + list(np.linspace(1, 0, 10))
trig_length = turbo_steps * 2
trig_space = turbo_steps * 5 # space between triggers
neg_rotation_y_trigger = list(np.linspace(0, -1, trig_length)) + list(np.linspace(-1, 0, trig_length))
rotation_3d_y = rotation_y_trigger + [0] * trig_space + neg_rotation_y_trigger + [0] * trig_space
rotation_3d_y = [0] + rotation_3d_y * 30
translation_z_trigger = list(np.linspace(1.5, 4.5, 10)) + list(np.linspace(4.5, 1.5, 10))
translation_z = translation_z_trigger + [1.5] * trig_space
translation_z = []
for i in range(60):
    translation_z.append(i * .15)
translation_z = translation_z + [1.5] * 60
translation_z = [1.5] + translation_z * 30
steps_schedule = [150]
rotation_3d_x = "0: (0), 24: (0.25*cos(1.2*3.141*(t + 5)/108))"
rotation_3d_y = "0: (0.22*sin(3.141*t/108))"
rotation_3d_z = "0: (8*(sin(3.141*t/216)**75)+0.25)"
# translation_z = "0: (0.2*(t%60))"
# flow_blend_schedule = list(np.linspace(0.999, 0.4, 24 * (seconds - 1))) + list(
#     np.linspace(0.4, 0, 24 * 1)
# )
# turbo_steps_schedule = [
#     math.ceil(n) for n in list(np.linspace(5, 2.01, 24 * (styled_seconds + ramp_seconds - 1)))
# ] + [
#     1
# ]  # Could also turn on turbo colormatch here
from disco_utils import parse_key_frames, get_inbetweens

try:
    if isinstance(translation_x, list):
        translation_x_series = translation_x
    else:
        translation_x_series = get_inbetweens(parse_key_frames(translation_x))
except RuntimeError as e:
    print(
        "WARNING: You have selected to use key frames, but you have not "
        "formatted `translation_x` correctly for key frames.\n"
        "Attempting to interpret `translation_x` as "
        f'"0: ({translation_x})"\n'
        "Please read the instructions to find out how to use key frames "
        "correctly.\n"
    )
    translation_x = f"0: ({translation_x})"
    translation_x_series = get_inbetweens(parse_key_frames(translation_x))

try:
    if isinstance(translation_y, list):
        translation_y_series = translation_y
    else:
        translation_y_series = get_inbetweens(parse_key_frames(translation_y))
except RuntimeError as e:
    print(
        "WARNING: You have selected to use key frames, but you have not "
        "formatted `translation_y` correctly for key frames.\n"
        "Attempting to interpret `translation_y` as "
        f'"0: ({translation_y})"\n'
        "Please read the instructions to find out how to use key frames "
        "correctly.\n"
    )
    translation_y = f"0: ({translation_y})"
    translation_y_series = get_inbetweens(parse_key_frames(translation_y))

try:
    if isinstance(translation_z, list):
        translation_z_series = translation_z
    else:
        translation_z_series = get_inbetweens(parse_key_frames(translation_z))
except RuntimeError as e:
    print(
        "WARNING: You have selected to use key frames, but you have not "
        "formatted `translation_z` correctly for key frames.\n"
        "Attempting to interpret `translation_z` as "
        f'"0: ({translation_z})"\n'
        "Please read the instructions to find out how to use key frames "
        "correctly.\n"
    )
    translation_z = f"0: ({translation_z})"
    translation_z_series = get_inbetweens(parse_key_frames(translation_z))

try:
    if isinstance(rotation_3d_x, list):
        rotation_3d_x_series = rotation_3d_x
    else:
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x))
except RuntimeError as e:
    print(
        "WARNING: You have selected to use key frames, but you have not "
        "formatted `rotation_3d_x` correctly for key frames.\n"
        "Attempting to interpret `rotation_3d_x` as "
        f'"0: ({rotation_3d_x})"\n'
        "Please read the instructions to find out how to use key frames "
        "correctly.\n"
    )
    rotation_3d_x = f"0: ({rotation_3d_x})"
    rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x))

try:
    if isinstance(rotation_3d_y, list):
        rotation_3d_y_series = rotation_3d_y
    else:
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y))
except RuntimeError as e:
    print(
        "WARNING: You have selected to use key frames, but you have not "
        "formatted `rotation_3d_y` correctly for key frames.\n"
        "Attempting to interpret `rotation_3d_y` as "
        f'"0: ({rotation_3d_y})"\n'
        "Please read the instructions to find out how to use key frames "
        "correctly.\n"
    )
    rotation_3d_y = f"0: ({rotation_3d_y})"
    rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y))

try:
    if isinstance(rotation_3d_z, list):
        rotation_3d_z_series = rotation_3d_z
    else:
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z))
except RuntimeError as e:
    print(
        "WARNING: You have selected to use key frames, but you have not "
        "formatted `rotation_3d_z` correctly for key frames.\n"
        "Attempting to interpret `rotation_3d_z` as "
        f'"0: ({rotation_3d_z})"\n'
        "Please read the instructions to find out how to use key frames "
        "correctly.\n"
    )
    rotation_3d_z = f"0: ({rotation_3d_z})"
    rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z))

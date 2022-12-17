import subprocess
from importlib import util as importlibutil
from glob import glob
import numpy as np
from tqdm.notebook import trange
import argparse, PIL, cv2
from PIL import Image
from pathlib import Path



def module_exists(module_name):
    return importlibutil.find_spec(module_name)


def gitclone(url, targetdir=None):
    if targetdir:
        res = subprocess.run(
            ["git", "clone", url, targetdir], stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
    else:
        res = subprocess.run(["git", "clone", url], stdout=subprocess.PIPE).stdout.decode("utf-8")
    print(res)


def pipi(modulestr):
    res = subprocess.run(["pip", "install", modulestr], stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    print(res)


def pipie(modulestr):
    res = subprocess.run(["git", "install", "-e", modulestr], stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    print(res)


def wget(url, outputdir):
    res = subprocess.run(["wget", url, "-P", f"{outputdir}"], stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    print(res)


def parse_key_frames(string, prompt_parser=None):
    """Given a string representing frame numbers paired with parameter values at that frame,
    return a dictionary with the frame numbers as keys and the parameter values as the values.

    Parameters
    ----------
    string: string
        Frame numbers paired with parameter values at that frame number, in the format
        'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
    prompt_parser: function or None, optional
        If provided, prompt_parser will be applied to each string of parameter values.

    Returns
    -------
    dict
        Frame numbers as keys, parameter values at that frame number as values

    Raises
    ------
    RuntimeError
        If the input string does not match the expected format.

    Examples
    --------
    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
    {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
    {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
    """
    import re

    pattern = r"((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])"
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()["frame"])
        param = match_object.groupdict()["param"]
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param

    if frames == {} and len(string) != 0:
        raise RuntimeError("Key Frame string not correctly formatted")
    return frames


def warp_flow(img, flow, mul=1.0):
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    flow *= mul  # new
    res = cv2.remap(img, flow, None, cv2.INTER_LANCZOS4)
    return res


def warp(
    frame1,
    frame2,
    flo_path,
    blend=0.5,
    weights_path=None,
    forward_clip=0.0,
    pad_pct=0.1,
    padding_mode="reflect",
    inpaint_blend=0.0,
    video_mode=False,
    warp_mul=1.0,
    warp_interp = PIL.Image.LANCZOS
):
    flow21 = np.load(flo_path)
    pad = int(max(flow21.shape) * pad_pct)  # new
    flow21 = np.pad(flow21, pad_width=((pad, pad), (pad, pad), (0, 0)), mode="constant")  # new

    frame1pil = np.array(frame1.convert("RGB"))  # .resize((flow21.shape[1], flow21.shape[0])))
    frame1pil = np.pad(
        frame1pil, pad_width=((pad, pad), (pad, pad), (0, 0)), mode=padding_mode
    )  # new
    if video_mode:  # new
        warp_mul = 1.0
    frame1_warped21 = warp_flow(frame1pil, flow21, warp_mul)
    frame1_warped21 = frame1_warped21[
        pad : frame1_warped21.shape[0] - pad, pad : frame1_warped21.shape[1] - pad, :
    ]
    # frame2pil = frame1pil
    # frame2pil = np.array(frame2.convert("RGB").resize((flow21.shape[1], flow21.shape[0])))
    frame2pil = np.array(
        frame2.convert("RGB").resize(
            (flow21.shape[1] - pad * 2, flow21.shape[0] - pad * 2), warp_interp
        )
    )

    if weights_path:
        # TBD
        forward_weights = load_cc(weights_path, blur=consistency_blur)
        # print('forward_weights')
        # print(forward_weights.shape)
        if not video_mode:
            frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)

        forward_weights = forward_weights.clip(forward_clip, 1.0)
        blended_w = frame2pil * (1 - blend) + blend * (
            frame1_warped21 * forward_weights + frame2pil * (1 - forward_weights)
        )
    else:
        # if not video_mode: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
        blended_w = frame2pil * (1 - blend) + frame1_warped21 * (blend)

    blended_w = PIL.Image.fromarray(blended_w.round().astype("uint8"))
    # if not video_mode:
    #     if enable_adjust_brightness: blended_w = adjust_brightness(blended_w)
    return blended_w


# folder=batchFolder, batchNo=batchNum, animMode=animation_mode, blendMode=video_init_blend_mode,
# blendSeries = args.flow_blend_series
def make_video(
    folder,
    floFolder,
    flowBlendSeries,
    fps=24,
    batchNo=0,
    animMode="Video Input",
    blendMode="optical flow",
    paddingRatio=0.2,
    paddingMode="reflect",
):
    # import subprocess in case this cell is run without the above cells
    import subprocess, os, shutil
    from base64 import b64encode

    latest_run = batchNo

    batchName = os.path.basename(folder)
    projectName = os.path.basename(Path(folder).parent)

    run = latest_run  # @param
    final_frame = "final_frame"

    init_frame = 1  # @param {type:"number"} This is the frame where the video will start
    last_frame = final_frame  # @param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
    # view_video_in_cell = True #@param {type: 'boolean'}

    frames = []
    # tqdm.write('Generating video...')

    if last_frame == "final_frame":
        last_frame = len(glob(folder + f"/{projectName}({run})_*.png"))
        print(f"Total frames: {last_frame}")

    image_path = f"{folder}/{projectName}({run})_%04d.png"
    filepath = f"{folder}/{projectName}({run}).mp4"

    if (blendMode == "optical flow") and (animMode == "Video Input"):
        image_path = f"{folder}/flow/{projectName}({run})_%04d.png"
        filepath = f"{folder}/{projectName}({run})_flow.mp4"
        if last_frame == "final_frame":
            last_frame = len(glob(folder + f"/flow/{projectName}({run})_*.png"))
        flo_out = folder + f"/flow"
        os.makedirs(flo_out, exist_ok=True)
        frames_in = sorted(glob(folder + f"/{projectName}({run})_*.png"))
        shutil.copy(frames_in[0], flo_out)
        for i in trange(init_frame, min(len(frames_in), last_frame)):
            frame1_path = frames_in[i - 1]
            frame2_path = frames_in[i]

            frame1 = PIL.Image.open(frame1_path)
            frame2 = PIL.Image.open(frame2_path)
            frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4])+1):04}.jpg"
            flo_path = f"/{floFolder}/{frame1_stem}.npy"
            weights_path = None
            # if video_init_check_consistency:
            #     # TBD
            #     pass

            cur_flow_blend = flowBlendSeries[i]
            print(flowBlendSeries)
            warp(
                frame1,
                frame2,
                flo_path,
                blend=cur_flow_blend,
                weights_path=weights_path,
                pad_pct=paddingRatio,
                padding_mode=paddingMode,
                inpaint_blend=0,
                video_mode=True,
            ).save(folder + f"/flow/{projectName}({run})_{i:04}.png")
    if blendMode == "linear":
        image_path = f"{folder}/blend/{projectName}({run})_%04d.png"
        filepath = f"{folder}/{projectName}({run})_blend.mp4"
        if last_frame == "final_frame":
            last_frame = len(glob(folder + f"/blend/{projectName}({run})_*.png"))
        blend_out = folder + f"/blend"
        os.makedirs(blend_out, exist_ok=True)
        frames_in = glob(folder + f"/{projectName}({run})_*.png")
        shutil.copy(frames_in[0], blend_out)
        blend = 0.5
        for i in trange(1, len(frames_in)):
            frame1_path = frames_in[i - 1]
            frame2_path = frames_in[i]

            frame1 = PIL.Image.open(frame1_path)
            frame2 = PIL.Image.open(frame2_path)

            frame = PIL.Image.fromarray(
                (np.array(frame1) * (1 - blend) + np.array(frame2) * (blend)).astype("uint8")
            ).save(folder + f"/blend/{projectName}({run})_{i:04}.png")

    cmd = [
        "ffmpeg",
        "-y",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-start_number",
        str(init_frame),
        "-i",
        image_path,
        "-frames:v",
        str(last_frame + 1),
        "-c:v",
        "libx264",
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "17",
        "-preset",
        "veryslow",
        filepath,
    ]

    process = subprocess.Popen(cmd, cwd=f"{folder}", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    else:
        print("The video is ready and saved to the images folder")

    # if view_video_in_cell:
    #     mp4 = open(filepath,'rb').read()
    #     data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    #     display.HTML(f'<video width=400 controls><source src="{data_url}" type="video/mp4"></video>')

import argparse
import os
from pathlib import Path
import json
from disco_utils import make_video, parse_key_frames, get_inbetweens
import numpy as np
from PIL import Image, ImageColor


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) in enumerate(
        zip(start_list, stop_list, is_horizontal_list)
    ):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--vid", help="Make video")
    parser.add_argument("--f", nargs="+", help="Colorize first frame")
    args = parser.parse_args()

    if args.vid:
        batchFolder = args.folder
        batchName = os.path.basename(args.folder)
        projectFolder = Path(args.folder).parent.absolute()
        projectName = os.path.basename(projectFolder)
        vidFolder = os.path.join(projectFolder, "videoFrames")
        floFolder = os.path.join(vidFolder, "out_flo_fwd")
        settingsFile = os.path.join(batchFolder, f"{projectName}(0)__settings.txt")
        print(settingsFile)
        settings = {}
        with open(settingsFile, "r", encoding="utf-8") as fp:
            settings = json.loads(fp.read(), strict=False)
        make_video(
            folder=batchFolder,
            floFolder=floFolder,
            flowBlendSeries=get_inbetweens(parse_key_frames(settings["flow_blend"])),
            blendMode=settings["video_init_blend_mode"],
            paddingRatio=settings["padding_ratio"],
            paddingMode=settings["flow_padding_mode"],
        )
    if args.f:  # Takes path to vid folder
        had_orig = False
        if os.path.exists(os.path.join(args.folder, "000001_orig.jpg")):
            had_orig = True
            orig = Image.open(os.path.join(args.folder, "000001_orig.jpg"))
        else:
            orig = Image.open(os.path.join(args.folder, "000001.jpg"))
        width, height = orig.size
        array = get_gradient_3d(
            width,
            height,
            ImageColor.getcolor(args.f[0], "RGB"),
            ImageColor.getcolor(args.f[1], "RGB"),
            (True, False, False),
        )
        if not had_orig:
            orig.save(os.path.join(args.folder, "000001_orig.jpg"), quality=95)

        # Higher number favors second image
        modded = Image.blend(orig, Image.fromarray(np.uint8(array)), 0.4)
        modded.save(os.path.join(args.folder, "000001.jpg"), quality=95)

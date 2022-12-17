import argparse
import os
from pathlib import Path
import json
from disco_utils import make_video, parse_key_frames


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--vid", help="Make video")
    args = parser.parse_args()

    batchFolder = Path(args.folder)
    batchName = os.path.basename(args.folder)
    projectFolder = Path(args.folder).parent.absolute()
    vidFolder = os.path.join(projectFolder, "videoFrames")
    settingsFile = os.path.join(batchFolder, f"{batchName}(0)__settings.txt")
    print(settingsFile)
    if args.vid:
        settings = {}
        with open(settingsFile, "r") as fp:
            settings = json.loads(fp.read())
        make_video(
            folder=batchFolder,
            flowBlendSeries=parse_key_frames(settings["flow_blend_series"]),
            blendMode=settings["video_init_blend_mode"],
            paddingRatio=settings["padding_ratio"],
            paddingMode=settings["flow_padding_mode"]
        )
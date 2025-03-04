import argparse
import os
from pathlib import Path
import json
from disco_utils import make_video, parse_key_frames, get_inbetweens


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--vid", help="Make video")
    args = parser.parse_args()

    batchFolder = args.folder 

    batchName = os.path.basename(args.folder)
    projectFolder = Path(args.folder).parent.absolute()
    projectName = os.path.basename(projectFolder)
    vidFolder = os.path.join(projectFolder, "videoFrames")
    floFolder = os.path.join(vidFolder, "out_flo_fwd")
    settingsFile = os.path.join(batchFolder, f"{projectName}(0)__settings.txt")
    print(settingsFile)
    if args.vid:
        settings = {}
        with open(settingsFile, "r", encoding="utf-8") as fp:
            settings = json.loads(fp.read(), strict=False)
        make_video(
            folder=batchFolder,
            floFolder=floFolder,
            flowBlendSeries=get_inbetweens(parse_key_frames(settings["flow_blend"])),
            blendMode=settings["video_init_blend_mode"],
            paddingRatio=settings["padding_ratio"],
            paddingMode=settings["flow_padding_mode"]
        )
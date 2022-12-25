import argparse
import os
from pathlib import Path
import json
from disco_utils import make_video, parse_key_frames, get_inbetweens
import numpy as np
import requests
from io import BytesIO
from PIL import Image, ImageColor
from glob import glob


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
    parser.add_argument("--finit")
    args = parser.parse_args()
    imgFile = os.path.basename(Path(args.folder))
    timestr = imgFile.split("_")[0]
    batchFolder = str(Path(args.folder).parent.absolute())
    print(batchFolder)
    print(timestr)

    if args.vid:
        skip_video_for_run_all = True  # @param {type: 'boolean'}
        fps = int(args.vid) if args.vid.isdigit() else 24  # @param {type:"number"}
        # @markdown **Manual Settings**
        use_manual_settings = False  # @param {type:"boolean"}
        image_path = (  # @param {type:"string"}
            "/content/drive/MyDrive/AI/StableDiffusion/2022-09/20220903000939_%05d.png"
        )
        mp4_path = (  # @param {type:"string"}
            "/content/drive/MyDrive/AI/StableDiffusion/2022-09/20220903000939.mp4"
        )
        render_steps = False  # @param {type: 'boolean'}
        path_name_modifier = "x0_pred"  # @param ["x0_pred","x"]
        make_gif = False

        import os
        import subprocess
        from base64 import b64encode

        print(f"{image_path} -> {mp4_path}")

        if use_manual_settings:
            max_frames = "200"  # @param {type:"string"}
        else:
            if render_steps:  # render steps from a single image
                fname = f"{path_name_modifier}_%05d.png"
                all_step_dirs = [
                    os.path.join(args.outdir, d)
                    for d in os.listdir(args.outdir)
                    if os.path.isdir(os.path.join(args.outdir, d))
                ]
                newest_dir = max(all_step_dirs, key=os.path.getmtime)
                image_path = os.path.join(newest_dir, fname)
                print(f"Reading images from {image_path}")
                mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
                max_frames = str(args.steps)
            else:  # render images for a video
                last_frame = len(glob(batchFolder + f"/{timestr}_*.png"))
                image_path = os.path.join(batchFolder, f"{timestr}_%05d.png")
                mp4_path = os.path.join(batchFolder, f"_vid.mp4")
                max_frames = str(last_frame - 1)

        # make video
        cmd = [
            "ffmpeg",
            "-y",
            "-vcodec",
            "png",
            "-r",
            str(fps),
            "-start_number",
            str(0),
            "-i",
            image_path,
            "-frames:v",
            max_frames,
            "-c:v",
            "libx264",
            "-vf",
            f"fps={fps}",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            "-preset",
            "veryfast",
            "-pattern_type",
            "sequence",
            mp4_path,
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)

    if args.f or args.finit:  # Takes path to vid folder
        had_orig = False
        if os.path.exists(os.path.join(args.folder, "000001_orig.jpg")):
            had_orig = True
            orig = Image.open(os.path.join(args.folder, "000001_orig.jpg"))
        else:
            orig = Image.open(os.path.join(args.folder, "000001.jpg"))
        width, height = orig.size
        if args.f:
            array = get_gradient_3d(
                width,
                height,
                ImageColor.getcolor(args.f[0], "RGB"),
                ImageColor.getcolor(args.f[1], "RGB"),
                (True, False, False),
            )
            modded = Image.blend(orig, Image.fromarray(np.uint8(array)), 0.4)
        elif args.finit:
            response = requests.get(args.finit)
            modded = Image.open(BytesIO(response.content))
        if not had_orig:
            orig.save(os.path.join(args.folder, "000001_orig.jpg"), quality=95)

        # Higher number favors second image
        modded.save(os.path.join(args.folder, "000001.jpg"), quality=95)

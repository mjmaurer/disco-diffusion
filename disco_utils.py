import subprocess
from importlib import util as importlibutil


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


# folder=batchFolder, batchNo=batchNum, animMode=animation_mode, blendMode=video_init_blend_mode,
# blendSeries = args.flow_blend_series
def make_video(
    folder,
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
    import PIL

    latest_run = batchNo

    batchName = os.path.basename(folder)

    run = latest_run  # @param
    final_frame = "final_frame"

    init_frame = 1  # @param {type:"number"} This is the frame where the video will start
    last_frame = final_frame  # @param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
    # view_video_in_cell = True #@param {type: 'boolean'}

    frames = []
    # tqdm.write('Generating video...')

    if last_frame == "final_frame":
        last_frame = len(glob(folder + f"/{batchName}({run})_*.png"))
        print(f"Total frames: {last_frame}")

    image_path = f"{folder}/{batchName}({run})_%04d.png"
    filepath = f"{folder}/{batchName}({run}).mp4"

    if (blendMode == "optical flow") and (animMode == "Video Input"):
        image_path = f"{folder}/flow/{batchName}({run})_%04d.png"
        filepath = f"{folder}/{batchName}({run})_flow.mp4"
        if last_frame == "final_frame":
            last_frame = len(glob(folder + f"/flow/{batchName}({run})_*.png"))
        flo_out = folder + f"/flow"
        createPath(flo_out)
        frames_in = sorted(glob(folder + f"/{batchName}({run})_*.png"))
        shutil.copy(frames_in[0], flo_out)
        for i in trange(init_frame, min(len(frames_in), last_frame)):
            frame1_path = frames_in[i - 1]
            frame2_path = frames_in[i]

            frame1 = PIL.Image.open(frame1_path)
            frame2 = PIL.Image.open(frame2_path)
            frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4])+1):04}.jpg"
            flo_path = f"/{flo_folder}/{frame1_stem}.npy"
            weights_path = None
            # if video_init_check_consistency:
            #     # TBD
            #     pass

            cur_flow_blend = flowBlendSeries[i]
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
            ).save(folder + f"/flow/{batchName}({run})_{i:04}.png")
    if blendMode == "linear":
        image_path = f"{folder}/blend/{batchName}({run})_%04d.png"
        filepath = f"{folder}/{batchName}({run})_blend.mp4"
        if last_frame == "final_frame":
            last_frame = len(glob(folder + f"/blend/{batchName}({run})_*.png"))
        blend_out = folder + f"/blend"
        createPath(blend_out)
        frames_in = glob(folder + f"/{batchName}({run})_*.png")
        shutil.copy(frames_in[0], blend_out)
        for i in trange(1, len(frames_in)):
            frame1_path = frames_in[i - 1]
            frame2_path = frames_in[i]

            frame1 = PIL.Image.open(frame1_path)
            frame2 = PIL.Image.open(frame2_path)

            frame = PIL.Image.fromarray(
                (np.array(frame1) * (1 - blend) + np.array(frame2) * (blend)).astype("uint8")
            ).save(folder + f"/blend/{batchName}({run})_{i:04}.png")

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

# %% [markdown]
#
#
# # This is a beta of WarpFusion.
# #### May produce meh results and not be very stable.
# Kudos to my [patreon](https://www.patreon.com/sxela)  XL tier supporters: \
# **Ced Pakusevskij**, **John Haugeland**, **Luc Schurgers**, **Fernando Magalhaes**, **Zlata Ponirovskaya**, **Inverse Alien**, **Andrew Farr**, **Francisco Bucknor**, **Seth Pyrzynski**,  **Territory Technical**, **FearTheDev**, **Nik**, **Nora Al Angari**
#
#
#  and all my patreon supporters for their endless support and constructive feedback!\
# Here's the current [public warp](https://colab.research.google.com/github/Sxela/DiscoDiffusion-Warp/blob/main/Disco_Diffusion_v5_2_Warp.ipynb) for videos with openai diffusion model
#
# # WarpFusion v0.5.21 by [Alex Spirin](https://twitter.com/devdef)
# ![visitors](https://visitor-badge.glitch.me/badge?page_id=sxela_ddwarp_colab)
#
# This version improves video init. You can now generate optical flow maps from input videos, and use those to:
# - warp init frames for consistent style
# - warp processed frames for less noise in final video
#
#
#
# ##Init warping
# The feature works like this: we take the 1st frame, diffuse it as usual as an image input with fixed skip steps. Then we warp in with its flow map into the 2nd frame and blend it with the original raw video 2nd frame. This way we get the style from heavily stylized 1st frame (warped accordingly) and content from 2nd frame (to reduce warping artifacts and prevent overexposure)
#
# --------------------------------------
#
# This is a variation of the awesome [DiscoDiffusion colab](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb#scrollTo=Changelog)
#
# If you like what I'm doing you can
# - follow me on [twitter](https://twitter.com/devdef)
# - tip me on [patreon](https://www.patreon.com/sxela)
#
#
# Thank you for being awesome!
#
# --------------------------------------
#
# ### Settings:
# (Located in animation settings tab)
#
# Video Optical Flow Settings:
# - flow_warp - check to warp
# - flow_blend: 0 - you get raw input, 1 - you get warped diffused previous frame
# - check_consistency: check forward-backward flow consistency (uncheck unless getting too many warping artifacts)
#
# ##Output warping
# This feature is plain simple - we just take any frame, warp in to the next frame, blend with real next frame, get smooth noise-free result.
#
# ### Settings:
# (located in create video tab)
# blend_mode:
# - none: just mash frames together in a video
# - optical flow: take frame, warp, blend with the next frame
# - check_consistency: use consistency maps (may prevent warping artfacts)
# - blend: 0 - you get raw 2nd frame, 1 - you get warped 1st frame
#
# #Comprehensive explanation of every cell and setting
#
# Don't forget to check this comprehensive guide created by users for users [here](https://docs.google.com/document/d/11xxHyvkCBBUwT73lWHQx-T_FU_rC7HWzNylftyCExcE) (a backup copy),\
# and [here](https://docs.google.com/document/d/1JrAvp6xtw0mmxqbPmOCRcjeosDDdsmZLE6zk7rtmKFg) (the live one).
#
#
# --------------------------------------
#
# This notebook was based on DiscoDiffusion (though it's not much like it anymore)\
# To learn more about DiscoDiffusion, join the [Disco Diffusion Discord](https://discord.gg/msEZBy4HxA) or message us on twitter at [@somnai_dreams](https://twitter.com/somnai_dreams) or [@gandamu](https://twitter.com/gandamu_ml)

# %% [markdown]
# # Changelog, credits & license

# %% [markdown]
# ### Changelog

# %% [markdown]
# 23.11.2022
# - fix writefile for non-colab interface
# - add xformers install for linux/windows
#
# 20.11.2022
# - add patchmatch inpainting for inconsistent areas
# - add warp towards init (thanks to [Zippika](https://twitter.com/AlexanderRedde3) from [deforum](https://github.com/deforum/stable-diffusion) team
# - add grad with respect to denoised latent, not input (4x faster) (thanks to EnzymeZoo from [deforum](https://github.com/deforum/stable-diffusion) team
# - add init/frame scale towards real frame option (thanks to [Zippika](https://twitter.com/AlexanderRedde3) from [deforum](https://github.com/deforum/stable-diffusion) team
# - add json schedules
# - add settings comparison (thanks to brbbbq)
# - save output videos to a separate video folder (thanks to Colton)
# - fix xformers not loading until restart
#
# 14.11.2022
# - add xformers for colab
# - add latent init blending
# - fix init scale loss to use 1/2 sized images
# - add verbose mode
# - fix frame correction for non-existent reference frames
# - fix user-defined latent stats to support 4 channels (4d)
# - fix start code to use 4d norm
# - track latent stats across all frames
# - print latent norm average stats
#
# 11.11.2022
# - add latent warp mode
# - add consistency support for latent warp mode
# - add masking support for latent warp mode
# - add normalize_latent modes: init_frame, init_frame_offset, stylized_frame, stylized_frame_offset
# - add normalize latent offset setting
#
# 4.11.2022
# - add normalize_latent modes: off, first_latent, user_defined
# - add normalize_latent user preset std and mean settings
# - add latent_norm_4d setting for per-channel latent normalization (was off in legacy colabs)
# - add colormatch_frame modes: off, init_frame, init_frame_offset, stylized_frame, stylized_frame_offset
# - add color match algorithm selection: LAB, PDF, mean (LAB was the default in legacy colabs)
# - add color match offset setting
# - add color match regrain flag
# - add color match strength
#
# 30.10.2022
# - add cfg_scale schedule
# - add option to apply schedule templates to peak difference frames only
# - add flow multiplier (for glitches)
# - add flow remapping (for even more glitches)
# - add inverse mask
# - fix masking in turbo mode (hopefully)
# - fix deleting videoframes not working in some cases
#
# 26.10.2022
# - add negative prompts
# - move google drive init cell higher
#
# 22.10.2022
# - add background mask support
# - add background mask extraction from video (using https://github.com/Sxela/RobustVideoMattingCLI)
# - add separate mask options during render and video creation
#
# 21.10.2022
# - add match first frame color toggle
# - add match first frame latent option
# - add karras noise + ramp up options
#
# 11.10.2022
# - add frame difference analysis
# - make preview persistent
# - fix some bugs with images not being sent
#
# 9.10.2022
# - add steps scheduling
# - add init_scale scheduling
# - add init_latent_scale scheduling
#
# 8.10.2022
# - add skip steps scheduling
# - add flow_blend scheduling
#
# 2.10.2022
# - add auto session shutdown after run
# - add awesome user-generated guide
#
# 23.09.2022
# - add channel mixing for consistency masks
# - add multilayer consistency masks
# - add jpeg-only consistency masks (weight less)
# - add save as pickle option (model weight less, loads faster, uses less CPU RAM)
#
# 18.09.2022
# - add clip guidance (ViT-H/14, ViT-L/14, ViT-B/32)
# - fix progress bar
# - change output dir name to StableWarpFusion
#
# 15.08.2022
# - remove unnecessary inage resizes, that caused a feedback loop in a few frames, kudos to everyoneishappy#5351 @ Discord
#
# 7.08.2022
# - added vram usage fix, now supports up to 1536x1536 images on 16gig gpus (with init_scales and sat_scale off)
# - added frame range (start-end frame) for video inits
# - added pseudo-inpainting by diffusing only inconsistent areas
# - fixed changing width height not working correctly
# - removed LAMA inpainting to reduce load and installation bugs
# - hiden intermediate saves (unusable for now)
# - fixed multiple image operations being applied during intermediate previews (even though the previews were not shown)
# - moved Stable model loading to a later stage to allow processings optical flow for larger frame sizes
# - fixed videoframes being saved correctly without google drive\locally
# - fixed PIL module error for colab to work without restarting
# - fix RAFT models download error x2
#
# 2.09.2022
# - Add Create a video from the init image
# - Add Fixed start code toggle \ blend setting
# - Add Latent frame scale
# - Fix prompt scheduling
# - Return init scale \ frames scale back to its original menus
# - Hide all unused settings
#
# 30.08.2022
# - Add fixes to run locally
#
# 25.08.2022
# - use existing color matching to keep frames from going deep purple
# - temporarily hide non-active stuff
# - fix match_color_var typo
# - fix model path interface
#
# - brought back LAMA inpainting
# - fixed PIL error
#
# 23.08.2022
# - Add Stable Diffusion
#
# 1.08.2022
# - Add color matching from https://github.com/pengbo-learn/python-color-transfer (kudos to authors!)
# - Add automatic brightness correction (thanks to @lowfuel and his awesome https://github.com/lowfuel/progrockdiffusion)
# - Add early stopping
# - Bump 4 leading zeros in frame names to 6. Fixes error for videos with more than 9999 frames
# - Move LAMA and RAFT models to the models' folder
#
# 09.07.2022
# - Add inpainting from https://github.com/saic-mdal/lama
# - Add init image for video mode
# - Add separate video init for optical flows
# - Fix leftover padding_mode definition
#
# 28.06.2022
# - Add input padding
# - Add traceback print
# - Add a (hopefully) self-explanatory warp settings form
#
# 21.06.2022
# - Add pythonic consistency check wrapper from [flow_tools](https://github.com/Sxela/flow_tools)
#
# 15.06.2022
# - Fix default prompt prohibiting prompt animation
#
# 8.06.2022
# - Fix path issue (thanks to Michael Carychao#0700)
# - Add turbo-smooth settings to settings.txt
# - Soften consistency clamping
#
# 7.06.2022
# - Add turbo-smooth
# - Add consistency clipping for normal and turbo frames
# - Add turbo frames skip steps
# - Add disable consistency for turbo frames
#
# 22.05.2022:
# - Add saving frames and flow to google drive (suggested by Chris the Wizard#8082
# )
# - Add back a more stable version of consistency checking
#
#
# 11.05.2022:
# - Add custom diffusion model support (more on training it [here](https://www.patreon.com/posts/generating-faces-66246423))
#
# 16.04.2022:
# - Use width_height size instead of input video size
# - Bring back adabins and 2d/3d anim modes
# - Install RAFT only when video input animation mode is selected
# - Generate optical flow maps only for video input animation mode even with flow_warp unchecked, so you can still save an obtical flow blended video later
# - Install AdaBins for 3d mode only (should do the same for midas)
# - Add animation mode check to create video tab
# 15.04.2022: Init

# %% [markdown]
# ### Credits ⬇️

# %% [markdown]
# #### Credits
#
# This notebook uses:
#
# [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis & StabilityAI\
# [K-diffusion wrapper](https://github.com/crowsonkb/k-diffusion) by Katherine Crowson\
# RAFT model by princeton-vl\
# Consistency Checking from maua\
# Color correction from\
# Auto brightness adjustment from [progrockdiffusion](https://github.com/lowfuel/progrockdiffusion)
#
#
# Original notebook by [Somnai](https://twitter.com/Somnai_dreams), [Adam Letts](https://twitter.com/gandamu_ml) and lots of other awesome people!
#
# Turbo feature by [Chris Allen](https://twitter.com/zippy731)
#
# Improvements to ability to run on local systems, Windows support, and dependency installation by [HostsServer](https://twitter.com/HostsServer)
#
# Warp and custom model support by [Alex Spirin](https://twitter.com/devdef)

# %% [markdown]
# ### License

# %% [markdown]
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors
#
# CreativeML Open RAIL-M
# dated August 22, 2022
#
# Section I: PREAMBLE
#
# Multimodal generative models are being widely adopted and used, and have the potential to transform the way artists, among other individuals, conceive and benefit from AI or ML technologies as a tool for content creation.
#
# Notwithstanding the current and potential benefits that these artifacts can bring to society at large, there are also concerns about potential misuses of them, either due to their technical limitations or ethical considerations.
#
# In short, this license strives for both the open and responsible downstream use of the accompanying model. When it comes to the open character, we took inspiration from open source permissive licenses regarding the grant of IP rights. Referring to the downstream responsible use, we added use-based restrictions not permitting the use of the Model in very specific scenarios, in order for the licensor to be able to enforce the license in case potential misuses of the Model may occur. At the same time, we strive to promote open and responsible research on generative models for art and content generation.
#
# Even though downstream derivative versions of the model could be released under different licensing terms, the latter will always have to include - at minimum - the same use-based restrictions as the ones in the original license (this license). We believe in the intersection between open and responsible AI development; thus, this License aims to strike a balance between both in order to enable responsible open-science in the field of AI.
#
# This License governs the use of the model (and its derivatives) and is informed by the model card associated with the model.
#
# NOW THEREFORE, You and Licensor agree as follows:
#
# 1. Definitions
#
# - "License" means the terms and conditions for use, reproduction, and Distribution as defined in this document.
# - "Data" means a collection of information and/or content extracted from the dataset used with the Model, including to train, pretrain, or otherwise evaluate the Model. The Data is not licensed under this License.
# - "Output" means the results of operating a Model as embodied in informational content resulting therefrom.
# - "Model" means any accompanying machine-learning based assemblies (including checkpoints), consisting of learnt weights, parameters (including optimizer states), corresponding to the model architecture as embodied in the Complementary Material, that have been trained or tuned, in whole or in part on the Data, using the Complementary Material.
# - "Derivatives of the Model" means all modifications to the Model, works based on the Model, or any other model which is created or initialized by transfer of patterns of the weights, parameters, activations or output of the Model, to the other model, in order to cause the other model to perform similarly to the Model, including - but not limited to - distillation methods entailing the use of intermediate data representations or methods based on the generation of synthetic data by the Model for training the other model.
# - "Complementary Material" means the accompanying source code and scripts used to define, run, load, benchmark or evaluate the Model, and used to prepare data for training or evaluation, if any. This includes any accompanying documentation, tutorials, examples, etc, if any.
# - "Distribution" means any transmission, reproduction, publication or other sharing of the Model or Derivatives of the Model to a third party, including providing the Model as a hosted service made available by electronic or other remote means - e.g. API-based or web access.
# - "Licensor" means the copyright owner or entity authorized by the copyright owner that is granting the License, including the persons or entities that may have rights in the Model and/or distributing the Model.
# - "You" (or "Your") means an individual or Legal Entity exercising permissions granted by this License and/or making use of the Model for whichever purpose and in any field of use, including usage of the Model in an end-use application - e.g. chatbot, translator, image generator.
# - "Third Parties" means individuals or legal entities that are not under common control with Licensor or You.
# - "Contribution" means any work of authorship, including the original version of the Model and any modifications or additions to that Model or Derivatives of the Model thereof, that is intentionally submitted to Licensor for inclusion in the Model by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Model, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."
# - "Contributor" means Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Model.
#
# Section II: INTELLECTUAL PROPERTY RIGHTS
#
# Both copyright and patent grants apply to the Model, Derivatives of the Model and Complementary Material. The Model and Derivatives of the Model are subject to additional terms as described in Section III.
#
# 2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare, publicly display, publicly perform, sublicense, and distribute the Complementary Material, the Model, and Derivatives of the Model.
# 3. Grant of Patent License. Subject to the terms and conditions of this License and where and as applicable, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this paragraph) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Model and the Complementary Material, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Model to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Model and/or Complementary Material or a Contribution incorporated within the Model and/or Complementary Material constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for the Model and/or Work shall terminate as of the date such litigation is asserted or filed.
#
# Section III: CONDITIONS OF USAGE, DISTRIBUTION AND REDISTRIBUTION
#
# 4. Distribution and Redistribution. You may host for Third Party remote access purposes (e.g. software-as-a-service), reproduce and distribute copies of the Model or Derivatives of the Model thereof in any medium, with or without modifications, provided that You meet the following conditions:
# Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement (e.g. a license) governing the use and/or distribution of the Model or Derivatives of the Model, and You shall give notice to subsequent users You Distribute to, that the Model or Derivatives of the Model are subject to paragraph 5. This provision does not apply to the use of Complementary Material.
# You must give any Third Party recipients of the Model or Derivatives of the Model a copy of this License;
# You must cause any modified files to carry prominent notices stating that You changed the files;
# You must retain all copyright, patent, trademark, and attribution notices excluding those notices that do not pertain to any part of the Model, Derivatives of the Model.
# You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions - respecting paragraph 4.a. - for use, reproduction, or Distribution of Your modifications, or for any such Derivatives of the Model as a whole, provided Your use, reproduction, and Distribution of the Model otherwise complies with the conditions stated in this License.
# 5. Use-based restrictions. The restrictions set forth in Attachment A are considered Use-based restrictions. Therefore You cannot use the Model and the Derivatives of the Model for the specified restricted uses. You may use the Model subject to this License, including only for lawful purposes and in accordance with the License. Use may include creating any content with, finetuning, updating, running, training, evaluating and/or reparametrizing the Model. You shall require all of Your users who use the Model or a Derivative of the Model to comply with the terms of this paragraph (paragraph 5).
# 6. The Output You Generate. Except as set forth herein, Licensor claims no rights in the Output You generate using the Model. You are accountable for the Output you generate and its subsequent uses. No use of the output can contravene any provision as stated in the License.
#
# Section IV: OTHER PROVISIONS
#
# 7. Updates and Runtime Restrictions. To the maximum extent permitted by law, Licensor reserves the right to restrict (remotely or otherwise) usage of the Model in violation of this License, update the Model through electronic means, or modify the Output of the Model based on updates. You shall undertake reasonable efforts to use the latest version of the Model.
# 8. Trademarks and related. Nothing in this License permits You to make use of Licensors’ trademarks, trade names, logos or to otherwise suggest endorsement or misrepresent the relationship between the parties; and any rights not expressly granted herein are reserved by the Licensors.
# 9. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Model and the Complementary Material (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Model, Derivatives of the Model, and the Complementary Material and assume any risks associated with Your exercise of permissions under this License.
# 10. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Model and the Complementary Material (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.
# 11. Accepting Warranty or Additional Liability. While redistributing the Model, Derivatives of the Model and the Complementary Material thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.
# 12. If any provision of this License is held to be invalid, illegal or unenforceable, the remaining provisions shall be unaffected thereby and remain valid as if such provision had not been set forth herein.
#
# END OF TERMS AND CONDITIONS
#
#
#
#
# Attachment A
#
# Use Restrictions
#
# You agree not to use the Model or Derivatives of the Model:
# - In any way that violates any applicable national, federal, state, local or international law or regulation;
# - For the purpose of exploiting, harming or attempting to exploit or harm minors in any way;
# - To generate or disseminate verifiably false information and/or content with the purpose of harming others;
# - To generate or disseminate personal identifiable information that can be used to harm an individual;
# - To defame, disparage or otherwise harass others;
# - For fully automated decision making that adversely impacts an individual’s legal rights or otherwise creates or modifies a binding, enforceable obligation;
# - For any use intended to or which has the effect of discriminating against or harming individuals or groups based on online or offline social behavior or known or predicted personal or personality characteristics;
# - To exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm;
# - For any use intended to or which has the effect of discriminating against individuals or groups based on legally protected characteristics or categories;
# - To provide medical advice and medical results interpretation;
# - To generate or disseminate information for the purpose to be used for administration of justice, law enforcement, immigration or asylum processes, such as predicting an individual will commit fraud/crime commitment (e.g. by text profiling, drawing causal relationships between assertions made in documents, indiscriminate and arbitrarily-targeted use).
#
# Licensed under the MIT License
#
# Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab)
#
# Copyright (c) 2021 Maxwell Ingham
#
# Copyright (c) 2022 Adam Letts
#
# Copyright (c) 2022 Alex Spirin
#
# Copyright (c) 2022 lowfuel
#
# Copyright (c) 2021-2022 Katherine Crowson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# %% [markdown]
# # 1. Set Up

# %%
# @title 1.1 Prepare Folders
import subprocess, os, sys, ipykernel
from external_settings import michael_mode, vid_input, force_vid_extract


def gitclone(url):
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


try:
    from google.colab import drive

    print("Google Colab detected. Using Google Drive.")
    is_colab = True
    # @markdown If you connect your Google Drive, you can save the final image of each run on your drive.
    google_drive = True  # @param {type:"boolean"}
    # @markdown Click here if you'd like to save the diffusion model checkpoint file to (and/or load from) your Google Drive:
    save_models_to_google_drive = True  # @param {type:"boolean"}
except:
    is_colab = False
    google_drive = False
    save_models_to_google_drive = False
    print("Google Colab not detected.")

if is_colab:
    if google_drive is True:
        drive.mount("/content/drive")
        root_path = "/content/drive/MyDrive/AI/StableWarpFusion"
    else:
        root_path = "/content"
else:
    root_path = os.getcwd()

import os


def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)


initDirPath = f"{root_path}/init_images"
createPath(initDirPath)
outDirPath = f"{root_path}/images_out"
createPath(outDirPath)
root_dir = os.getcwd()

if is_colab:
    root_dir = "/content/"
    if google_drive and not save_models_to_google_drive or not google_drive:
        model_path = "/content/models"
        createPath(model_path)
    if google_drive and save_models_to_google_drive:
        model_path = f"{root_path}/models"
        createPath(model_path)
else:
    model_path = f"{root_path}/models"
    createPath(model_path)

# libraries = f'{root_path}/libraries'
# createPath(libraries)

# %%
# @title writefile ./stable-diffusion/ldm/modules/attention.py
if not os.path.exists("stable-diffusion"):
    print("Cloning Stable Diffusion")
    gitclone("https://github.com/Doggettx/stable-diffusion")
sys.path.append("./stable-diffusion")
file_path = "./stable-diffusion/ldm/modules/attention.py"
file_content = """#https://github.com/TheLastBen/fast-stable-diffusion/blob/main/precompiled/attention.py
import gc
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
try:
  import xformers
  import xformers.ops
  print('using xformers')
except: print('xformers import failed')

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):

    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.attention_op = None
        try:
          import xformers 
          self.use_xformers = True
          
          #print('Using xformers')
        except: 
          self.use_xformers = False
          #print('Disabling xformers')

    def _maybe_init(self, x):

          if self.attention_op is not None:
              return

          _, M, K = x.shape
          try:
              self.attention_op = xformers.ops.AttentionOpDispatch(
                  dtype=x.dtype,
                  device=x.device,
                  k=K,
                  attn_bias_type=type(None),
                  has_dropout=False,
                  kv_len=M,
                  q_len=M,
              ).op

          except NotImplementedError as err:
              raise NotImplementedError(f"Please install xformers with the flash attention / cutlass components.{err}")

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = default(context, x)
        k_in = self.to_k(context)
        v_in = self.to_v(context)
        del context, x
        b, _, _ = q_in.shape
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        try: 
          self._maybe_init(q)
          self.use_xformers = True
        except:
          self.use_xformers = False

        if self.use_xformers:
          # init the attention op, if required, using the proper dimensions


          # actually compute the attention, what we cannot get enough of
          out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
          out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
          )
          return self.to_out(out)

        if not self.use_xformers:
          r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)

          stats = torch.cuda.memory_stats(q.device)
          mem_active = stats['active_bytes.all.current']
          mem_reserved = stats['reserved_bytes.all.current']
          mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
          mem_free_torch = mem_reserved - mem_active
          mem_free_total = mem_free_cuda + mem_free_torch

          gb = 1024 ** 3
          tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
          modifier = 3 if q.element_size() == 2 else 2.5
          mem_required = tensor_size * modifier
          steps = 1


          if mem_required > mem_free_total:
              steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))
              # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
              #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

          if steps > 64:
              max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
              raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                                f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')

          slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
          for i in range(0, q.shape[1], slice_size):
              end = i + slice_size
              s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * self.scale

              s2 = s1.softmax(dim=-1, dtype=q.dtype)
              del s1

              r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
              del s2

          del q, k, v

          r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
          del r1

          return self.to_out(r2)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in"""

with open(file_path, "w") as f:
    f.write(file_content)

# %%
# @title 1.2 Install SD Dependencies
import os, platform

if platform.system() != "Linux":
    print(
        "Trying to install local xformers on Windows. Works only with pytorch 1.12.* and python"
        " 3.10."
    )
    # !pip install https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
if is_colab or (platform.system() == "Linux"):
    print("Installing xformers.")
    if not os.path.exists("triton"):
        gitclone("https://github.com/openai/triton.git")
    try:
        import xformers 
    except:
        pip_res = subprocess.run(
            ["pip", "install", "-e", "./triton/python"], stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
        print(pip_res)

    from subprocess import getoutput
    from IPython.display import HTML
    from IPython.display import clear_output
    import time

    # https://github.com/TheLastBen/fast-stable-diffusion
    gpu = ""
    s = getoutput("nvidia-smi")
    if "T4" in s:
        gpu = "T4"
    elif "P100" in s:
        gpu = "P100"
    elif "V100" in s:
        gpu = "V100"
    elif "A100" in s:
        gpu = "A100"

    while True:
        try:
            gpu == "T4" or gpu == "P100" or gpu == "V100" or gpu == "A100"
            break
        except:
            pass
        print(" it seems that your GPU is not supported at the moment")
        time.sleep(5)

    gpu_wheel = ""
    if gpu == "T4":
        gpu_wheel = "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/T4/xformers-0.0.13.dev0-py3-none-any.whl"

    elif gpu == "P100":
        gpu_wheel = "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/P100/xformers-0.0.13.dev0-py3-none-any.whl"

    elif gpu == "V100":
        gpu_wheel = "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/V100/xformers-0.0.13.dev0-py3-none-any.whl"

    elif gpu == "A100":
        gpu_wheel = "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/xformers-0.0.13.dev0-py3-none-any.whl"

    print(
        subprocess.run(["pip", "install", gpu_wheel], stdout=subprocess.PIPE).stdout.decode("utf-8")
    )
    clear_output()
    print(" DONE !")

try:
    os.chdir(f"./k-diffusion")
    import k_diffusion as K

    os.chdir(f"../")
except:
    print("Installing k-diffusion")
    print(
        subprocess.run(
            ["pip", "install", "--ignore-installed", "Pillow==9.0.0"], stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
    )
    # !git clone https://github.com/CompVis/stable-diffusion
    if not os.path.exists("stable-diffusion"):
        gitclone("https://github.com/Doggettx/stable-diffusion")

    try:
        import ldm
    except:
        print("Installing stable-diffusion")
        print(
            subprocess.run(
                ["pip", "install", "-e", "./stable-diffusion"], stdout=subprocess.PIPE
            ).stdout.decode("utf-8")
        )

    sys.path.append(f"{root_dir}/stable-diffusion")

    multipip_res = subprocess.run(
        [
            "pip",
            "install",
            "ipywidgets==7.7.1",
            "transformers==4.19.2",
            "omegaconf",
            "einops",
            "pytorch_lightning>1.4.1,<=1.7.7",
            "scikit-image",
            "opencv-python",
            "ai-tools",
            "cognitive-face",
            "zprint",
            "kornia==0.5.0",
        ],
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    print(multipip_res)

    try:
        import taming
    except:
        print("Installing taming")
        print(
            subprocess.run(
                [
                    "pip",
                    "install",
                    "-e",
                    "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers",
                ],
                stdout=subprocess.PIPE,
            ).stdout.decode("utf-8")
        )
    sys.path.append(f"{root_dir}/src/taming-transformers")

    try:
        import clip
    except:
        print("Installing clip")
        print(
            subprocess.run(
                ["pip", "install", "-e", "git+https://github.com/openai/CLIP.git@main#egg=clip"],
                stdout=subprocess.PIPE,
            ).stdout.decode("utf-8")
        )
    sys.path.append(f"{root_dir}/src/clip")

    multipip_res = subprocess.run(
        [
            "pip",
            "install",
            "lpips",
            "keras",
        ],
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    print(multipip_res)

    if not os.path.exists("k-diffusion"):
        gitclone("https://github.com/crowsonkb/k-diffusion/")
    os.chdir(f"./k-diffusion")
    print(
        subprocess.run(["pip", "install", "-e", "."], stdout=subprocess.PIPE).stdout.decode("utf-8")
    )
    os.chdir(f"../")
    import sys

    sys.path.append("./k-diffusion")

multipip_res = subprocess.run(
    [
        "pip",
        "install",
        "wget",
        "webdataset",
    ],
    stdout=subprocess.PIPE,
).stdout.decode("utf-8")
print(multipip_res)

try:
    import open_clip
except:
    print(
        subprocess.run(["pip", "install", "open_clip_torch"], stdout=subprocess.PIPE).stdout.decode(
            "utf-8"
        )
    )
    import open_clip

# %%
# @title 1.3 Check GPU Status
import subprocess

simple_nvidia_smi_display = True  # @param {type:"boolean"}
if simple_nvidia_smi_display:
    #!nvidia-smi
    nvidiasmi_output = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    print(nvidiasmi_output)
else:
    #!nvidia-smi -i 0 -e 0
    nvidiasmi_output = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode("utf-8")
    print(nvidiasmi_output)
    nvidiasmi_ecc_note = subprocess.run(
        ["nvidia-smi", "-i", "0", "-e", "0"], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    print(nvidiasmi_ecc_note)

# %%
# @title ### 1.4 Install and import dependencies

multipip_res = subprocess.run(
    [
        "pip",
        "install",
        "opencv-python==4.5.5.64",
        "pandas",
        "matplotlib",
    ],
    stdout=subprocess.PIPE,
).stdout.decode("utf-8")
print(multipip_res)
multipip_res = subprocess.run(
    [
        "pip",
        "uninstall",
        "torchtext",
        "-y",
    ],
    stdout=subprocess.PIPE,
).stdout.decode("utf-8")
print(multipip_res)

if is_colab:
    # !git clone https://github.com/vacancy/PyPatchMatch --recursive
    # !make ./PyPatchMatch
    # from PyPatchMatch import patch_match
    pass

import pathlib, shutil, os, sys

if not is_colab:
    # If running locally, there's a good chance your env will need this in order to not crash upon np.matmul() or similar operations.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_DIR = os.path.abspath(os.getcwd())
USE_ADABINS = False

if is_colab:
    if google_drive is not True:
        root_path = f"/content"
        model_path = "/content/models"
else:
    root_path = os.getcwd()
    model_path = f"{root_path}/models"


multipip_res = subprocess.run(
    [
        "pip",
        "install",
        "lpips",
        "datetime",
        "timm",
        "ftfy",
        "einops",
        "pytorch-lightning",
        "omegaconf",
    ],
    stdout=subprocess.PIPE,
).stdout.decode("utf-8")
print(multipip_res)

if is_colab:
    subprocess.run(["apt", "install", "imagemagick"], stdout=subprocess.PIPE).stdout.decode("utf-8")

# try:
#   from CLIP import clip
# except:
#   if not os.path.exists("CLIP"):
#     gitclone("https://github.com/openai/CLIP")
#   sys.path.append(f'{PROJECT_DIR}/CLIP')

try:
    from guided_diffusion.script_util import create_model_and_diffusion
except:
    if not os.path.exists("guided-diffusion"):
        gitclone("https://github.com/crowsonkb/guided-diffusion")
    sys.path.append(f"{PROJECT_DIR}/guided-diffusion")

try:
    from resize_right import resize
except:
    if not os.path.exists("ResizeRight"):
        gitclone("https://github.com/assafshocher/ResizeRight.git")
    sys.path.append(f"{PROJECT_DIR}/ResizeRight")


import torch
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import gc
import io
import math
import timm
from IPython import display
import lpips
from PIL import Image, ImageOps
import requests
from glob import glob
import json
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm

# from CLIP import clip
from resize_right import resize
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import Output
import hashlib
from functools import partial

if is_colab:
    os.chdir("/content")
    from google.colab import files
else:
    os.chdir(f"{PROJECT_DIR}")
from IPython.display import Image as ipyimg
from numpy import asarray
from einops import rearrange, repeat
import torch, torchvision
import time
from omegaconf import OmegaConf
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
device = DEVICE  # At least one of the modules expects this name..

if torch.cuda.get_device_capability(DEVICE) == (8, 0):  ## A100 fix thanks to Emad
    print("Disabling CUDNN for A100 gpu", file=sys.stderr)
    torch.backends.cudnn.enabled = False

# %%
# @title 1.5 Define necessary functions

# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
import PIL


def interp(t):
    return 3 * t**2 - 2 * t**3


def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert("RGB")
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out


def regen_perlin():
    if perlin_mode == "color":
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == "gray":
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)


def fetch(url_or_path):
    if str(url_or_path).startswith("http://") or str(url_or_path).startswith("https://"):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, "rb")


def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)


def parse_prompt(prompt):
    if prompt.startswith("http://") or prompt.startswith("https://"):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomPerspective(distortion_scale=0.4, p=0.7),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.15),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        )

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(
                    max_size
                    * torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(float(self.cut_size / max_size), 1.0)
                )
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


cutout_debug = False
padargs = {}


class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size, Overview=4, InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        if args.animation_mode == "None":
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomAffine(
                        degrees=10,
                        translate=(0.05, 0.05),
                        interpolation=T.InterpolationMode.BILINEAR,
                    ),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomGrayscale(p=0.1),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ]
            )
        elif args.animation_mode == "Video Input Legacy":
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomPerspective(distortion_scale=0.4, p=0.7),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomGrayscale(p=0.15),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ]
            )
        elif args.animation_mode == "2D" or args.animation_mode == "Video Input":
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.4),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomAffine(
                        degrees=10,
                        translate=(0.05, 0.05),
                        interpolation=T.InterpolationMode.BILINEAR,
                    ),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomGrayscale(p=0.1),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
                ]
            )

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        output_shape_2 = [1, 3, self.cut_size + 2, self.cut_size + 2]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2,
                (sideY - max_size) // 2,
                (sideX - max_size) // 2,
                (sideX - max_size) // 2,
            ),
            **padargs,
        )
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                if is_colab:
                    TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save(
                        "/content/cutout_overview0.jpg", quality=99
                    )
                else:
                    TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save(
                        "cutout_overview0.jpg", quality=99
                    )

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                if is_colab:
                    TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save(
                        "/content/cutout_InnerCrop.jpg", quality=99
                    )
                else:
                    TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save(
                        "cutout_InnerCrop.jpg", quality=99
                    )
        cutouts = torch.cat(cutouts)
        if skip_augs is not True:
            cutouts = self.augs(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def get_image_from_lat(lat):
    img = sd_model.decode_first_stage(lat.cuda().half())[0]
    return TF.to_pil_image(img.add(1).div(2).clamp(0, 1))


def get_lat_from_pil(frame):
    print(frame.shape, "frame2pil.shape")
    frame = np.array(frame)
    frame = (frame / 255.0)[None, ...].transpose(0, 3, 1, 2)
    frame = 2 * torch.from_numpy(frame).cuda().half() - 1.0
    return sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame)).half()


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0 / 200.0


def get_sched_from_json(frame_num, sched_json, blend=False):
    keys = sorted(list(sched_json.keys()))
    # print(keys)
    frame_num = int(frame_num)
    frame_num = min(max(frame_num, 0), max(keys))  # clamp frame num to 0:max(keys) range
    # print('clamped frame num ', frame_num)
    if frame_num in keys:
        return sched_json[frame_num]
        # print('frame in keys')
    if frame_num not in keys:
        for i in range(len(keys) - 1):
            k1 = keys[i]
            k2 = keys[i + 1]
            if frame_num > k1 and frame_num < k2:
                if not blend:
                    print("frame between keys, no blend")
                    return sched_json[k1]
                if blend:
                    total_dist = k2 - k1
                    dist_from_k1 = frame_num - k1
                    return sched_json[k1] * (1 - dist_from_k1 / total_dist) + sched_json[k2] * (
                        dist_from_k1 / total_dist
                    )
            # else: print(f'frame {frame_num} not in {k1} {k2}')


def get_scheduled_arg(frame_num, schedule):
    if isinstance(schedule, list):
        return schedule[frame_num] if frame_num < len(schedule) else schedule[-1]
    if isinstance(schedule, dict):
        return get_sched_from_json(frame_num, schedule, blend=blend_json_schedules)


def img2tensor(img, size=None):
    img = img.convert("RGB")
    if size:
        img = img.resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...].cuda()


def warp_towards_init_fn(sample_pil, init_image):
    print("sample, init", type(sample_pil), type(init_image))
    size = sample_pil.size
    sample = img2tensor(sample_pil)
    init_image = img2tensor(init_image, size)
    flo = get_flow(init_image, sample, raft_model, half=flow_lq)
    # flo = get_flow(sample, init_image, raft_model, half=flow_lq)
    warped = warp(
        sample_pil,
        sample_pil,
        flo_path=flo,
        blend=1,
        weights_path=None,
        forward_clip=0,
        pad_pct=padding_ratio,
        padding_mode=padding_mode,
        inpaint_blend=inpaint_blend,
        warp_mul=warp_strength,
    )
    return warped


def do_3d_step(img_filepath, frame_num, forward_clip):
    global warp_mode
    if warp_mode == "use_image":
        prev = PIL.Image.open(img_filepath)
    # if warp_mode == 'use_latent':
    #   prev = torch.load(img_filepath[:-4]+'_lat.pt')

    frame1_path = f"{videoFramesFolder}/{frame_num:06}.jpg"
    frame2 = PIL.Image.open(f"{videoFramesFolder}/{frame_num+1:06}.jpg")

    flo_path = f"{flo_folder}/{frame1_path.split('/')[-1]}.npy"

    if flow_override_map not in [[], "", None]:
        mapped_frame_num = get_scheduled_arg(frame_num, flow_override_map)
        frame_override_path = f"{videoFramesFolder}/{mapped_frame_num:06}.jpg"
        flo_path = f"{flo_folder}/{frame_override_path.split('/')[-1]}.npy"

    if use_background_mask and not apply_mask_after_warp:
        # if turbo_mode & (frame_num % int(turbo_steps) != 0):
        #   print('disabling mask for turbo step, will be applied during turbo blend')
        # else:
        if VERBOSE:
            print("creating bg mask for frame ", frame_num)
        frame2 = apply_mask(frame2, frame_num, background, background_source, invert_mask)
        # frame2.save(f'frame2_{frame_num}.jpg')
    # init_image = 'warped.png'
    flow_blend = get_scheduled_arg(frame_num, flow_blend_schedule)
    printf(
        "flow_blend: ",
        flow_blend,
        "frame_num:",
        frame_num,
        "len(flow_blend_schedule):",
        len(flow_blend_schedule),
    )
    weights_path = None
    forward_clip = forward_weights_clip
    if check_consistency:
        if reverse_cc_order:
            weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
        else:
            weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"

    if turbo_mode & (frame_num % int(turbo_steps) != 0):
        if forward_weights_clip_turbo_step:
            forward_clip = forward_weights_clip_turbo_step
        if disable_cc_for_turbo_frames:
            if VERBOSE:
                print("disabling cc for turbo frames")
            weights_path = None
    if warp_mode == "use_image":
        prev = PIL.Image.open(img_filepath)
        warped = warp(
            prev,
            frame2,
            flo_path,
            blend=flow_blend,
            weights_path=weights_path,
            forward_clip=forward_clip,
            pad_pct=padding_ratio,
            padding_mode=padding_mode,
            inpaint_blend=inpaint_blend,
            warp_mul=warp_strength,
        )
    if warp_mode == "use_latent":
        prev = torch.load(img_filepath[:-4] + "_lat.pt")
        warped = warp_lat(
            prev,
            frame2,
            flo_path,
            blend=flow_blend,
            weights_path=weights_path,
            forward_clip=forward_clip,
            pad_pct=padding_ratio,
            padding_mode=padding_mode,
            inpaint_blend=inpaint_blend,
            warp_mul=warp_strength,
        )
    # warped = warped.resize((side_x,side_y), warp_interp)

    if use_background_mask and apply_mask_after_warp:
        # if turbo_mode & (frame_num % int(turbo_steps) != 0):
        #   print('disabling mask for turbo step, will be applied during turbo blend')
        #   return warped
        if VERBOSE:
            print("creating bg mask for frame ", frame_num)
        if warp_mode == "use_latent":
            warped = apply_mask(
                warped, frame_num, background, background_source, invert_mask, warp_mode
            )
        else:
            warped = apply_mask(
                warped, frame_num, background, background_source, invert_mask, warp_mode
            )
        # warped.save(f'warped_{frame_num}.jpg')

    return warped


from tqdm.notebook import trange


def apply_mask(
    init_image, frame_num, background, background_source, invert_mask=False, warp_mode="use_image"
):

    if warp_mode == "use_image":
        size = init_image.size
    if warp_mode == "use_latent":
        print(init_image.shape)
        size = init_image.shape[-1], init_image.shape[-2]
        size = [o * 8 for o in size]
        print("size", size)
    init_image_alpha = (
        PIL.Image.open(f"{videoFramesAlpha}/{frame_num+1:06}.jpg").resize(size).convert("L")
    )
    if invert_mask:
        init_image_alpha = PIL.ImageOps.invert(init_image_alpha)
    if background == "color":
        bg = PIL.Image.new("RGB", size, background_source)
    if background == "image":
        bg = PIL.Image.open(background_source).convert("RGB").resize(size)
    if background == "init_video":
        bg = PIL.Image.open(f"{videoFramesFolder}/{frame_num+1:06}.jpg").resize(size)
    # init_image.putalpha(init_image_alpha)
    if warp_mode == "use_image":
        bg.paste(init_image, (0, 0), init_image_alpha)
    if warp_mode == "use_latent":
        # convert bg to latent

        bg = np.array(bg)
        bg = (bg / 255.0)[None, ...].transpose(0, 3, 1, 2)
        bg = 2 * torch.from_numpy(bg).cuda().half() - 1.0
        bg = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(bg)).half()
        bg = bg.cpu().numpy()  # [0].transpose(1,2,0)
        init_image_alpha = np.array(init_image_alpha)[::8, ::8][None, None, ...]
        init_image_alpha = np.repeat(init_image_alpha, 4, axis=1) / 255
        print(
            bg.shape,
            init_image.shape,
            init_image_alpha.shape,
            init_image_alpha.max(),
            init_image_alpha.min(),
        )
        bg = init_image * init_image_alpha + bg * (1 - init_image_alpha)
    return bg


def do_run():
    seed = args.seed
    print(range(args.start_frame, args.max_frames))
    if args.animation_mode != "None":
        batchBar = tqdm(total=args.max_frames, desc="Frames")

    # if (args.animation_mode == 'Video Input') and (args.midas_weight > 0.0):
    # midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model)
    for frame_num in range(args.start_frame, args.max_frames):
        if stop_on_next_loop:
            break

        # display.clear_output(wait=True)

        # Print Frame progress if animation mode is on
        if args.animation_mode != "None":
            display.display(batchBar.container)
            batchBar.n = frame_num
            batchBar.update(1)
            batchBar.refresh()
            # display.display(batchBar.container)

        # Inits if not video frames
        if args.animation_mode != "Video Input Legacy":
            if args.init_image == "":
                init_image = None
            else:
                init_image = args.init_image
            init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
            # init_scale = args.init_scale
            steps = int(get_scheduled_arg(frame_num, steps_schedule))
            style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
            skip_steps = int(steps - steps * style_strength)
            # skip_steps = args.skip_steps

        if args.animation_mode == "Video Input":
            if frame_num == 0:
                steps = int(get_scheduled_arg(frame_num, steps_schedule))
                style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
                skip_steps = int(steps - steps * style_strength)
                # skip_steps = args.skip_steps

                # init_scale = args.init_scale
                init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
                # init_latent_scale = args.init_latent_scale
                init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)
                init_image = f"{videoFramesFolder}/{frame_num+1:06}.jpg"
                if use_background_mask:
                    init_image_pil = PIL.Image.open(init_image)
                    init_image_pil = apply_mask(
                        init_image_pil, frame_num, background, background_source, invert_mask
                    )
                    init_image_pil.save(f"init_alpha_{frame_num}.png")
                    init_image = f"init_alpha_{frame_num}.png"
                if (args.init_image != "") and args.init_image is not None:
                    init_image = args.init_image
                    if use_background_mask:
                        init_image_pil = PIL.Image.open(init_image)
                        init_image_pil = apply_mask(
                            init_image_pil, frame_num, background, background_source, invert_mask
                        )
                        init_image_pil.save(f"init_alpha_{frame_num}.png")
                        init_image = f"init_alpha_{frame_num}.png"
                if VERBOSE:
                    print("init image", args.init_image)
            if frame_num > 0:
                # print(frame_num)
                first_frame = PIL.Image.open(batchFolder + f"/{batch_name}({batchNum})_{0:06}.png")
                first_frame_source = batchFolder + f"/{batch_name}({batchNum})_{0:06}.png"
                seed += 1
                if resume_run and frame_num == start_frame:
                    print("if resume_run and frame_num == start_frame")
                    img_filepath = batchFolder + f"/{batch_name}({batchNum})_{start_frame-1:06}.png"
                    if turbo_mode and frame_num > turbo_preroll:
                        shutil.copyfile(img_filepath, "oldFrameScaled.png")
                    else:
                        shutil.copyfile(img_filepath, "prevFrame.png")
                else:
                    # img_filepath = '/content/prevFrame.png' if is_colab else 'prevFrame.png'
                    img_filepath = "prevFrame.png"

                next_step_pil = do_3d_step(
                    img_filepath, frame_num, forward_clip=forward_weights_clip
                )
                if warp_mode == "use_image":
                    next_step_pil.save("prevFrameScaled.png")
                else:
                    # init_image = 'prevFrameScaled_lat.pt'
                    # next_step_pil.save('prevFrameScaled.png')
                    torch.save(next_step_pil, "prevFrameScaled_lat.pt")

                steps = int(get_scheduled_arg(frame_num, steps_schedule))
                style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
                skip_steps = int(steps - steps * style_strength)
                # skip_steps = args.calc_frames_skip_steps

                ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
                if turbo_mode:
                    if frame_num == turbo_preroll:  # start tracking oldframe
                        if warp_mode == "use_image":
                            next_step_pil.save("oldFrameScaled.png")  # stash for later blending
                        if warp_mode == "use_latent":
                            # lat_from_img = get_lat/_from_pil(next_step_pil)
                            torch.save(next_step_pil, "oldFrameScaled_lat.pt")
                    elif frame_num > turbo_preroll:
                        # set up 2 warped image sequences, old & new, to blend toward new diff image
                        if warp_mode == "use_image":
                            old_frame = do_3d_step(
                                "oldFrameScaled.png",
                                frame_num,
                                forward_clip=forward_weights_clip_turbo_step,
                            )
                            old_frame.save("oldFrameScaled.png")
                        if warp_mode == "use_latent":
                            old_frame = do_3d_step(
                                "oldFrameScaled.png",
                                frame_num,
                                forward_clip=forward_weights_clip_turbo_step,
                            )

                            # lat_from_img = get_lat_from_pil(old_frame)
                            torch.save(old_frame, "oldFrameScaled_lat.pt")
                        if frame_num % int(turbo_steps) != 0:
                            print("turbo skip this frame: skipping clip diffusion steps")
                            filename = f"{args.batch_name}({args.batchNum})_{frame_num:06}.png"
                            blend_factor = ((frame_num % int(turbo_steps)) + 1) / int(turbo_steps)
                            print(
                                "turbo skip this frame: skipping clip diffusion steps and saving"
                                " blended frame"
                            )
                            if warp_mode == "use_image":
                                newWarpedImg = cv2.imread(
                                    "prevFrameScaled.png"
                                )  # this is already updated..
                                oldWarpedImg = cv2.imread("oldFrameScaled.png")
                                blendedImage = cv2.addWeighted(
                                    newWarpedImg, blend_factor, oldWarpedImg, 1 - blend_factor, 0.0
                                )
                                cv2.imwrite(f"{batchFolder}/{filename}", blendedImage)
                                next_step_pil.save(
                                    f"{img_filepath}"
                                )  # save it also as prev_frame to feed next iteration
                            if warp_mode == "use_latent":
                                newWarpedImg = torch.load(
                                    "prevFrameScaled_lat.pt"
                                )  # this is already updated..
                                oldWarpedImg = torch.load("oldFrameScaled_lat.pt")
                                blendedImage = newWarpedImg * (blend_factor) + oldWarpedImg * (
                                    1 - blend_factor
                                )
                                blendedImage = get_image_from_lat(blendedImage).save(
                                    f"{batchFolder}/{filename}"
                                )
                                torch.save(next_step_pil, f"{img_filepath[:-4]}_lat.pt")

                            if turbo_frame_skips_steps is not None:
                                if warp_mode == "use_image":
                                    oldWarpedImg = cv2.imread("prevFrameScaled.png")
                                    cv2.imwrite(
                                        f"oldFrameScaled.png", oldWarpedImg
                                    )  # swap in for blending later
                                print("clip/diff this frame - generate clip diff image")
                                if warp_mode == "use_latent":
                                    oldWarpedImg = torch.load("prevFrameScaled_lat.pt")
                                    torch.save(
                                        oldWarpedImg,
                                        f"oldFrameScaled_lat.pt",
                                    )  # swap in for blending later
                                skip_steps = math.floor(steps * turbo_frame_skips_steps)
                            else:
                                continue
                        else:
                            # if not a skip frame, will run diffusion and need to blend.
                            if warp_mode == "use_image":
                                oldWarpedImg = cv2.imread("prevFrameScaled.png")
                                cv2.imwrite(
                                    f"oldFrameScaled.png", oldWarpedImg
                                )  # swap in for blending later
                            print("clip/diff this frame - generate clip diff image")
                            if warp_mode == "use_latent":
                                oldWarpedImg = torch.load("prevFrameScaled_lat.pt")
                                torch.save(
                                    oldWarpedImg,
                                    f"oldFrameScaled_lat.pt",
                                )  # swap in for blending later
                            # oldWarpedImg = cv2.imread('prevFrameScaled.png')
                            # cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later
                            print("clip/diff this frame - generate clip diff image")
                if warp_mode == "use_image":
                    init_image = "prevFrameScaled.png"
                else:
                    init_image = "prevFrameScaled_lat.pt"
                if use_background_mask:
                    if warp_mode == "use_latent":
                        # pass
                        latent = apply_mask(
                            latent.cpu(),
                            frame_num,
                            background,
                            background_source,
                            invert_mask,
                            warp_mode,
                        )  # .save(init_image)

                    if warp_mode == "use_image":
                        apply_mask(
                            PIL.Image.open(init_image),
                            frame_num,
                            background,
                            background_source,
                            invert_mask,
                        ).save(init_image)
                # init_scale = args.frames_scale
                init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
                # init_latent_scale = args.frames_latent_scale
                init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)

        if args.animation_mode == "Video Input Legacy":
            init_scale = args.frames_scale
            init_latent_scale = args.frames_latent_scale
            steps = int(get_scheduled_arg(frame_num, steps_schedule))
            frame_skip_steps = args.frames_skip_steps_series[frame_num]
            calc_frames_skip_steps = math.floor(steps * frame_skip_steps)
            if steps <= calc_frames_skip_steps:
                sys.exit("ERROR: You can't skip more steps than your total steps")

            skip_steps = calc_frames_skip_steps
            if not video_init_seed_continuity:
                seed += 1
            if flow_warp:
                if frame_num == 0:
                    skip_steps = args.skip_steps
                    init_image = f"{videoFramesFolder}/{frame_num+1:06}.jpg"

                if frame_num > 0:
                    first_frame = PIL.Image.open(
                        batchFolder + f"/{batch_name}({batchNum})_{0:06}.png"
                    )
                    first_frame_source = batchFolder + f"/{batch_name}({batchNum})_{0:06}.png"
                    prev = PIL.Image.open(
                        batchFolder + f"/{batch_name}({batchNum})_{frame_num-1:06}.png"
                    )

                    frame1_path = f"{videoFramesFolder}/{frame_num:06}.jpg"
                    frame2 = PIL.Image.open(f"{videoFramesFolder}/{frame_num+1:06}.jpg")
                    flo_path = f"{flo_folder}/{frame1_path.split('/')[-1]}.npy"

                    init_image = "warped.png"
                    print(flow_blend)
                    weights_path = None
                    forward_clip = forward_weights_clip
                    if check_consistency:
                        weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                    if turbo_mode & (frame_num % int(turbo_steps) != 0):
                        if forward_weights_clip_turbo_step:
                            forward_clip = forward_weights_clip_turbo_step
                        if disable_cc_for_turbo_frames:
                            print("disabling cc for turbo frames")
                            weights_path = None
                    warped = warp(
                        prev,
                        frame2,
                        flo_path,
                        blend=flow_blend,
                        weights_path=weights_path,
                        forward_clip=forward_clip,
                        pad_pct=padding_ratio,
                        padding_mode=padding_mode,
                        inpaint_blend=inpaint_blend,
                        warp_mul=warp_strength,
                    )
                    if turbo_mode:
                        if frame_num % int(turbo_steps) != 0:
                            if not turbo_frame_skips_steps:
                                print("turbo skip this frame: skipping clip diffusion steps")
                                warped = warped.resize((side_x, side_y), warp_interp)
                                warped.save(
                                    batchFolder + f"/{batch_name}({batchNum})_{frame_num:06}.png"
                                )
                                continue
                            if turbo_frame_skips_steps:
                                skip_steps = turbo_frame_skips_steps
                    warped.save(init_image)

            else:
                init_image = f"{videoFramesFolder}/{frame_num+1:06}.jpg"

        loss_values = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        target_embeds, weights = [], []

        if args.prompts_series is not None and frame_num >= len(args.prompts_series):
            frame_prompt = args.prompts_series[-1]
        elif args.prompts_series is not None:
            frame_prompt = args.prompts_series[frame_num]
        else:
            frame_prompt = []

        if VERBOSE:
            print(args.image_prompts_series)
        if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
            image_prompt = args.image_prompts_series[-1]
        elif args.image_prompts_series is not None:
            image_prompt = args.image_prompts_series[frame_num]
        else:
            image_prompt = []

        if VERBOSE:
            print(f"Frame {frame_num} Prompt: {frame_prompt}")

        model_stats = []
        for clip_model in clip_models:
            cutn = 16
            model_stat = {
                "clip_model": None,
                "target_embeds": [],
                "make_cutouts": None,
                "weights": [],
            }
            model_stat["clip_model"] = clip_model

            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append(
                            (txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0, 1)
                        )
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)

            if image_prompt:
                model_stat["make_cutouts"] = MakeCutouts(
                    clip_model.visual.input_resolution, cutn, skip_augs=skip_augs
                )
                for prompt in image_prompt:
                    path, weight = parse_prompt(prompt)
                    img = Image.open(fetch(path)).convert("RGB")
                    img = TF.resize(
                        img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS
                    )
                    batch = model_stat["make_cutouts"](
                        TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1)
                    )
                    embed = clip_model.encode_image(normalize(batch)).float()
                    if fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append(
                                (embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0, 1)
                            )
                            weights.extend([weight / cutn] * cutn)
                    else:
                        model_stat["target_embeds"].append(embed)
                        model_stat["weights"].extend([weight / cutn] * cutn)

            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError("The weights must not sum to 0.")
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)

        init = None
        # if init_image is not None:
        #   if isinstance(init_image, str):
        #     init = Image.open(fetch(init_image)).convert('RGB')
        #     init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
        #     init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        if args.perlin_init:
            if args.perlin_mode == "color":
                init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False)
                init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, False)
            elif args.perlin_mode == "gray":
                init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, True)
                init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True)
            else:
                init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False)
                init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True)
            # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
            init = (
                TF.to_tensor(init)
                .add(TF.to_tensor(init2))
                .div(2)
                .to(device)
                .unsqueeze(0)
                .mul(2)
                .sub(1)
            )
            del init2

        cur_t = None

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if use_secondary_model is True:
                    alpha = torch.tensor(
                        diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32
                    )
                    sigma = torch.tensor(
                        diffusion.sqrt_one_minus_alphas_cumprod[cur_t],
                        device=device,
                        dtype=torch.float32,
                    )
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                    out = diffusion.p_mean_variance(
                        model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
                    )
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out["pred_xstart"] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                for model_stat in model_stats:
                    for i in range(args.cutn_batches):
                        t_int = (
                            int(t.item()) + 1
                        )  # errors on last step without +1, need to find source
                        # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                        try:
                            input_resolution = model_stat["clip_model"].visual.input_resolution
                        except:
                            input_resolution = 224

                        cuts = MakeCutoutsDango(
                            input_resolution,
                            Overview=args.cut_overview[1000 - t_int],
                            InnerCrop=args.cut_innercut[1000 - t_int],
                            IC_Size_Pow=args.cut_ic_pow,
                            IC_Grey_P=args.cut_icgray_p[1000 - t_int],
                        )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(
                            image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0)
                        )
                        dists = dists.view(
                            [
                                args.cut_overview[1000 - t_int] + args.cut_innercut[1000 - t_int],
                                n,
                                -1,
                            ]
                        )
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(
                            losses.sum().item()
                        )  # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += (
                            torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0]
                            / cutn_batches
                        )
                tv_losses = tv_loss(x_in)
                if use_secondary_model is True:
                    range_losses = range_loss(out)
                else:
                    range_losses = range_loss(out["pred_xstart"])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                loss = (
                    tv_losses.sum() * tv_scale
                    + range_losses.sum() * range_scale
                    + sat_losses.sum() * sat_scale
                )
                if init is not None and args.init_scale:
                    init_losses = lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * args.init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if torch.isnan(x_in_grad).any() == False:
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    # print("NaN'd")
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if args.clamp_grad and x_is_NaN == False:
                magnitude = grad.square().mean().sqrt()
                return (
                    grad * magnitude.clamp(max=args.clamp_max) / magnitude
                )  # min=-0.02, min=-clamp_max,
            return grad

        if diffusion_model == "stable_diffusion":

            def cond_fn(x, t, init=None):

                # global cur_t
                # cur_step = cur_t
                t = 1000 - t
                t = t[0]
                x = x.detach()
                with torch.enable_grad():
                    x_is_NaN = False
                    global clamp_start_, clamp_max
                    x = x.requires_grad_()
                    x_in = sd_model.decode_first_stage(x)
                    # display_handler(x_in,t,1,False)
                    n = x_in.shape[0]
                    # clip_guidance_scale = clip_guidance_index[t]
                    # make_cutouts = {}
                    x_in_grad = torch.zeros_like(x_in)
                    # for i in clip_list:
                    #     make_cutouts[i] = MakeCutouts(clip_size[i][0] if type(clip_size[i]) is tuple else clip_size[i],
                    #     Overview= cut_overview[t],
                    #     InnerCrop = cut_innercut[t],
                    #     IC_Size_Pow=cut_ic_pow, IC_Grey_P = cut_icgray_p[t],
                    #     cut_blur_n = cut_blur_n[t]
                    #     )
                    #     cutn = cut_overview[t]+cut_innercut[t]
                    # for j in range(cutn_batches):
                    # losses=0
                    # for i in clip_list:
                    #     clip_in = clip_normalize[i](make_cutouts[i](x_in.add(1).div(2)).to("cuda"))
                    #     image_embeds = clip_model[i].encode_image(clip_in).float().unsqueeze(0).expand([target_embeds[i].shape[0],-1,-1])
                    #     target_embeds_temp = target_embeds[i]
                    #     if i == 'ViT-B-32--openai' and experimental_aesthetic_embeddings:
                    #       aesthetic_embedding = torch.from_numpy(np.load(f'aesthetic-predictor/vit_b_32_embeddings/rating{experimental_aesthetic_embeddings_score}.npy')).to(device)
                    #       aesthetic_query = target_embeds_temp + aesthetic_embedding * experimental_aesthetic_embeddings_weight
                    #       target_embeds_temp = (aesthetic_query) / torch.linalg.norm(aesthetic_query)
                    #     if i == 'ViT-L-14--openai' and experimental_aesthetic_embeddings:
                    #       aesthetic_embedding = torch.from_numpy(np.load(f'aesthetic-predictor/vit_l_14_embeddings/rating{experimental_aesthetic_embeddings_score}.npy')).to(device)
                    #       aesthetic_query = target_embeds_temp + aesthetic_embedding * experimental_aesthetic_embeddings_weight
                    #       target_embeds_temp = (aesthetic_query) / torch.linalg.norm(aesthetic_query)
                    #     target_embeds_temp = target_embeds_temp.unsqueeze(1).expand([-1,cutn*n,-1])
                    #     dists = spherical_dist_loss(image_embeds, target_embeds_temp)
                    #     dists = dists.mean(1).mul(weights[i].squeeze()).mean()
                    #     losses+=dists*clip_guidance_scale #* (2 if i in ["ViT-L-14-336--openai", "RN50x64--openai", "ViT-B-32--laion2b_e16"] else (.4 if "cloob" in i else 1))
                    #     if i == "ViT-L-14-336--openai" and aes_scale !=0:
                    #         aes_loss = (aesthetic_model_336(F.normalize(image_embeds, dim=-1))).mean()
                    #         losses -= aes_loss * aes_scale
                    #     if i == "ViT-L-14--openai" and aes_scale !=0:
                    #         aes_loss = (aesthetic_model_224(F.normalize(image_embeds, dim=-1))).mean()
                    #         losses -= aes_loss * aes_scale
                    #     if i == "ViT-B-16--openai" and aes_scale !=0:
                    #         aes_loss = (aesthetic_model_16(F.normalize(image_embeds, dim=-1))).mean()
                    #         losses -= aes_loss * aes_scale
                    #     if i == "ViT-B-32--openai" and aes_scale !=0:
                    #         aes_loss = (aesthetic_model_32(F.normalize(image_embeds, dim=-1))).mean()
                    #         losses -= aes_loss * aes_scale
                    # x_in_grad += torch.autograd.grad(losses, x_in)[0] / cutn_batches / len(clip_list)
                    # losses += dists
                    # losses = losses / len(clip_list)
                    # gc.collect()

                    # loss =  losses
                    loss = torch.zeros(1).cuda().requires_grad_()
                    # del losses
                    # if symmetric_loss_scale != 0: loss +=  symmetric_loss(x_in) * symmetric_loss_scale

                    if init is not None and init_scale:

                        init_losses = lpips_model(x_in, init)
                        loss = loss + init_losses.sum() * init_scale
                    # raise Exception (x.shape, init.shape, t.shape, loss, x_in.shape, x_in_grad.shape)
                    print(loss, args.init_scale, init_scale, init.mean(), init.std())
                    x_in_grad += torch.autograd.grad(loss, x_in)[0]
                    if torch.isnan(x_in_grad).any() == False:
                        grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                    else:
                        # print("NaN'd")
                        x_is_NaN = True
                        grad = torch.zeros_like(x)
                if args.clamp_grad and x_is_NaN == False:
                    magnitude = grad.square().mean().sqrt()
                    return (
                        grad * magnitude.clamp(max=args.clamp_max) / magnitude
                    )  # min=-0.02, min=-clamp_max,
                return grad

            # cond_fn = None

        if diffusion_model != "stable_diffusion":
            if args.diffusion_sampling_mode == "ddim":
                sample_fn = diffusion.ddim_sample_loop_progressive
            else:
                sample_fn = diffusion.plms_sample_loop_progressive

        image_display = Output()
        for i in range(args.n_batches):
            if args.animation_mode == "None":
                display.clear_output(wait=True)
                batchBar = tqdm(range(args.n_batches), desc="Batches")
                batchBar.n = i
                batchBar.refresh()
            print("")
            display.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            steps = int(get_scheduled_arg(frame_num, steps_schedule))
            style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
            skip_steps = int(steps - steps * style_strength)
            cur_t = diffusion.num_timesteps - skip_steps - 1
            total_steps = cur_t

            if perlin_init:
                init = regen_perlin()

            consistency_mask = None
            if check_consistency and frame_num > 0:
                frame1_path = f"{videoFramesFolder}/{frame_num:06}.jpg"
                if reverse_cc_order:
                    weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                else:
                    weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"
                consistency_mask = load_cc(weights_path, blur=consistency_blur)
            if diffusion_model != "stable_diffusion":
                if args.diffusion_sampling_mode == "ddim":
                    samples = sample_fn(
                        model,
                        (batch_size, 3, args.side_y, args.side_x),
                        clip_denoised=clip_denoised,
                        model_kwargs={},
                        cond_fn=cond_fn,
                        progress=True,
                        skip_timesteps=skip_steps,
                        init_image=init,
                        randomize_class=randomize_class,
                        eta=eta,
                        mask=consistency_mask,
                        inpainting_stop=inpainting_stop,
                        early_stop=early_stop,
                    )
                else:
                    samples = sample_fn(
                        model,
                        (batch_size, 3, args.side_y, args.side_x),
                        clip_denoised=clip_denoised,
                        model_kwargs={},
                        cond_fn=cond_fn,
                        progress=True,
                        skip_timesteps=skip_steps,
                        init_image=init,
                        randomize_class=randomize_class,
                        order=2,
                    )
            if diffusion_model == "stable_diffusion":
                if VERBOSE:
                    print(args.side_x, args.side_y, init_image)
                # init = Image.open(fetch(init_image)).convert('RGB')

                # init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
                # init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
                text_prompt = args.prompts_series[frame_num]
                neg_prompt = args.neg_prompts_series[frame_num]
                if VERBOSE:
                    print("init_scale pre sd run", init_scale)
                # init_latent_scale = args.init_latent_scale
                # if frame_num>0:
                #   init_latent_scale = args.frames_latent_scale
                steps = int(get_scheduled_arg(frame_num, steps_schedule))
                init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
                init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)
                style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
                skip_steps = int(steps - steps * style_strength)
                cfg_scale = get_scheduled_arg(frame_num, cfg_scale_schedule)
                if VERBOSE:
                    printf("skip_steps b4 run_sd: ", skip_steps)

                init_grad_img = None
                if init_grad:
                    init_grad_img = f"{videoFramesFolder}/{frame_num+1:06}.jpg"
                sample, latent = run_sd(
                    args,
                    init_image=init_image,
                    skip_timesteps=skip_steps,
                    H=args.side_y,
                    W=args.side_x,
                    text_prompt=text_prompt,
                    neg_prompt=neg_prompt,
                    steps=steps,
                    seed=seed,
                    init_scale=init_scale,
                    init_latent_scale=init_latent_scale,
                    cfg_scale=cfg_scale,
                    cond_fn=None,
                    init_grad_img=init_grad_img,
                )

                filename = f"{args.batch_name}({args.batchNum})_{frame_num:06}.png"
                # if warp_mode == 'use_raw':torch.save(sample,f'{batchFolder}/{filename[:-4]}_raw.pt')
                if warp_mode == "use_latent":
                    torch.save(latent, f"{batchFolder}/{filename[:-4]}_lat.pt")
                samples = sample * (steps - skip_steps)
                samples = [{"pred_xstart": sample} for sample in samples]
                # for j, sample in enumerate(samples):
                # print(j, sample["pred_xstart"].size)
                # raise Exception
                if VERBOSE:
                    print(sample[0][0].shape)
                image = TF.to_pil_image(sample[0][0].add(1).div(2).clamp(0, 1))
                if warp_towards_init != "off" and frame_num != 0:
                    if warp_towards_init == "init":
                        warp_init_filename = f"{videoFramesFolder}/{frame_num+1:06}.jpg"
                    else:
                        warp_init_filename = init_image
                    print("warping towards init")
                    init_pil = PIL.Image.open(warp_init_filename)
                    image = warp_towards_init_fn(image, init_pil)

                display.clear_output(wait=True)
                fit(image, display_size).save("progress.png")
                display.display(display.Image("progress.png"))
                if frame_num > 0:
                    global first_latent
                    global first_latent_source

                    def get_frame_from_color_mode(mode, offset):
                        if mode == "stylized_frame_offset":
                            if VERBOSE:
                                print(f"the stylized frame with offset {offset}.")
                            filename = f"{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-offset:06}.png"
                        if mode == "stylized_frame":
                            if VERBOSE:
                                print(f"the stylized frame number {offset}.")
                            filename = (
                                f"{batchFolder}/{args.batch_name}({args.batchNum})_{offset:06}.png"
                            )
                        if mode == "init_frame_offset":
                            if VERBOSE:
                                print(f"the raw init frame with offset {offset}.")
                            filename = f"{videoFramesFolder}/{frame_num-offset:06}.jpg"
                        if mode == "init_frame":
                            if VERBOSE:
                                print(f"the raw init frame number {offset}.")
                            filename = f"{videoFramesFolder}/{offset:06}.jpg"
                        return filename

                    if "frame" in normalize_latent:

                        def img2latent(img_path):
                            frame2 = PIL.Image.open(img_path)
                            frame2pil = frame2.convert("RGB").resize(image.size, warp_interp)
                            frame2pil = np.array(frame2pil)
                            frame2pil = (frame2pil / 255.0)[None, ...].transpose(0, 3, 1, 2)
                            frame2pil = 2 * torch.from_numpy(frame2pil).cuda().half() - 1.0
                            frame2pil = sd_model.get_first_stage_encoding(
                                sd_model.encode_first_stage(frame2pil)
                            ).half()
                            return frame2pil

                        try:
                            if VERBOSE:
                                print("Matching latent to:")
                            filename = get_frame_from_color_mode(
                                normalize_latent, normalize_latent_offset
                            )
                            match_latent = img2latent(filename)
                            first_latent = match_latent
                            first_latent_source = filename
                            # print(first_latent_source, first_latent)
                        except:
                            if VERBOSE:
                                print(traceback.format_exc())
                            print(f"Frame with offset/position {normalize_latent_offset} not found")
                            if "init" in normalize_latent:
                                try:
                                    filename = f"{videoFramesFolder}/{0:06}.jpg"
                                    match_latent = img2latent(filename)
                                    first_latent = match_latent
                                    first_latent_source = filename
                                except:
                                    pass
                            print(f"Color matching the 1st frame.")

                    if colormatch_frame != "off":
                        try:
                            print("Matching color to:")
                            filename = get_frame_from_color_mode(
                                colormatch_frame, colormatch_offset
                            )
                            match_frame = PIL.Image.open(filename)
                            first_frame = match_frame
                            first_frame_source = filename

                        except:
                            print(f"Frame with offset/position {colormatch_offset} not found")
                            if "init" in colormatch_frame:
                                try:
                                    filename = f"{videoFramesFolder}/{0:06}.jpg"
                                    match_frame = PIL.Image.open(filename)
                                    first_frame = match_frame
                                    first_frame_source = filename
                                except:
                                    pass
                            print(f"Color matching the 1st frame.")
                        print("Colormatch source - ", first_frame_source)
                        image = PIL.Image.fromarray(
                            match_color_var(
                                first_frame,
                                image,
                                opacity=color_match_frame_str,
                                f=colormatch_method_fn,
                                regrain=colormatch_regrain,
                            )
                        )

                if mask_result and check_consistency and frame_num > 0:
                    diffuse_inpaint_mask_blur = 15
                    diffuse_inpaint_mask_thresh = 220
                    if VERBOSE:
                        print("imitating inpaint")
                    frame1_path = f"{videoFramesFolder}/{frame_num:06}.jpg"
                    weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                    consistency_mask = load_cc(weights_path, blur=consistency_blur)
                    consistency_mask = cv2.GaussianBlur(
                        consistency_mask,
                        (diffuse_inpaint_mask_blur, diffuse_inpaint_mask_blur),
                        cv2.BORDER_DEFAULT,
                    )
                    consistency_mask = np.where(
                        consistency_mask < diffuse_inpaint_mask_thresh / 255.0, 0, 1.0
                    )

                    # consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
                    if warp_mode == "use_image":
                        consistency_mask = cv2.GaussianBlur(
                            consistency_mask, (3, 3), cv2.BORDER_DEFAULT
                        )
                        init_img_prev = PIL.Image.open(init_image)
                        if VERBOSE:
                            print(init_img_prev.size, consistency_mask.shape, image.size)
                        cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                        image_masked = np.array(image) * (1 - consistency_mask) + np.array(
                            init_img_prev
                        ) * (consistency_mask)

                        # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                        image_masked = PIL.Image.fromarray(image_masked.round().astype("uint8"))
                        # image = image_masked.resize(image.size, warp_interp)
                        image = image_masked
                    if warp_mode == "use_latent":
                        if invert_mask:
                            consistency_mask = 1 - consistency_mask
                        init_lat_prev = torch.load("prevFrameScaled_lat.pt")
                        sample_masked = sd_model.decode_first_stage(latent.cuda().half())[0]
                        image_prev = TF.to_pil_image(sample_masked.add(1).div(2).clamp(0, 1))

                        cc_small = consistency_mask[::8, ::8, 0]
                        latent = latent.cpu() * (1 - cc_small) + init_lat_prev * cc_small
                        torch.save(latent, "prevFrameScaled_lat.pt")

                        # image_prev = PIL.Image.open(f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png')
                        torch.save(latent, "prevFrame_lat.pt")
                        # cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                        # image_prev = PIL.Image.open('prevFrameScaled.png')
                        image_masked = np.array(image) * (1 - consistency_mask) + np.array(
                            image_prev
                        ) * (consistency_mask)

                        # # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                        image_masked = PIL.Image.fromarray(image_masked.round().astype("uint8"))
                        # image = image_masked.resize(image.size, warp_interp)
                        image = image_masked

                if frame_num == 0:
                    save_settings(args)
                if args.animation_mode != "None":
                    # sys.exit(os.getcwd(), 'cwd')
                    if warp_mode == "use_image":
                        image.save("prevFrame.png")
                    else:
                        torch.save(latent, "prevFrame_lat.pt")
                filename = f"{args.batch_name}({args.batchNum})_{frame_num:06}.png"
                image.save(f"{batchFolder}/{filename}")
                # np.save(latent, f'{batchFolder}/{filename[:-4]}.npy')
                if args.animation_mode == "Video Input":
                    # If turbo, save a blended image
                    if turbo_mode and frame_num > 0:
                        # Mix new image with prevFrameScaled
                        blend_factor = (1) / int(turbo_steps)
                        if warp_mode == "use_image":
                            newFrame = cv2.imread("prevFrame.png")  # This is already updated..
                            prev_frame_warped = cv2.imread("prevFrameScaled.png")
                            blendedImage = cv2.addWeighted(
                                newFrame, blend_factor, prev_frame_warped, (1 - blend_factor), 0.0
                            )
                            cv2.imwrite(f"{batchFolder}/{filename}", blendedImage)
                        if warp_mode == "use_latent":
                            newFrame = torch.load("prevFrame_lat.pt").cuda()
                            prev_frame_warped = torch.load("prevFrameScaled_lat.pt").cuda()
                            blendedImage = newFrame * (blend_factor) + prev_frame_warped * (
                                1 - blend_factor
                            )
                            blendedImage = get_image_from_lat(blendedImage)
                            blendedImage.save(f"{batchFolder}/{filename}")

                else:
                    image.save(f"{batchFolder}/{filename}")

            # with run_display:
            # display.clear_output(wait=True)
            # o = 0
            # for j, sample in enumerate(samples):
            #   cur_t -= 1
            #   # if (cur_t <= stop_early-2):
            #   #   print(cur_t)
            #   #   break
            #   intermediateStep = False
            #   if args.steps_per_checkpoint is not None:
            #       if j % steps_per_checkpoint == 0 and j > 0:
            #         intermediateStep = True
            #   elif j in args.intermediate_saves:
            #     intermediateStep = True
            #   with image_display:
            #     if j % args.display_rate == 0 or cur_t == -1 or cur_t == stop_early-1 or intermediateStep == True:

            #         for k, image in enumerate(sample['pred_xstart']):
            #             # tqdm.write(f'Batch {i}, step {j}, output {k}:')
            #             current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
            #             percent = math.ceil(j/total_steps*100)
            #             if args.n_batches > 0:
            #               #if intermediates are saved to the subfolder, don't append a step or percentage to the name
            #               if (cur_t == -1 or cur_t == stop_early-1) and args.intermediates_in_subfolder is True:
            #                 save_num = f'{frame_num:06}' if animation_mode != "None" else i
            #     filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
            #   else:
            #     #If we're working with percentages, append it
            #     if args.steps_per_checkpoint is not None:
            #       filename = f'{args.batch_name}({args.batchNum})_{i:06}-{percent:02}%.png'
            #     # Or else, iIf we're working with specific steps, append those
            #     else:
            #       filename = f'{args.batch_name}({args.batchNum})_{i:06}-{j:03}.png'
            # image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
            # if frame_num > 0:
            #   print('times per image', o); o+=1
            #   image = PIL.Image.fromarray(match_color_var(first_frame, image, f=PT.lab_transfer))
            #   # image.save(f'/content/{frame_num}_{cur_t}_{o}.jpg')
            #   # image = PIL.Image.fromarray(match_color_var(first_frame, image))

            # #reapply init image on top of
            # if mask_result and check_consistency and frame_num>0:
            #   diffuse_inpaint_mask_blur = 15
            #   diffuse_inpaint_mask_thresh = 220
            #   print('imitating inpaint')
            #   frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
            #   weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
            #   consistency_mask = load_cc(weights_path, blur=consistency_blur)
            #   consistency_mask = cv2.GaussianBlur(consistency_mask,
            #                           (diffuse_inpaint_mask_blur,diffuse_inpaint_mask_blur),cv2.BORDER_DEFAULT)
            #   consistency_mask = np.where(consistency_mask<diffuse_inpaint_mask_thresh/255., 0, 1.)
            #   consistency_mask = cv2.GaussianBlur(consistency_mask,
            #                           (3,3),cv2.BORDER_DEFAULT)

            #   # consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
            #   init_img_prev = PIL.Image.open(init_image)
            #   print(init_img_prev.size, consistency_mask.shape, image.size)
            #   cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
            #   image_masked = np.array(image)*(1-consistency_mask) + np.array(init_img_prev)*(consistency_mask)

            #   # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
            #   image_masked = PIL.Image.fromarray(image_masked.round().astype('uint8'))
            #   # image = image_masked.resize(image.size, warp_interp)
            #   image = image_masked

            # if j % args.display_rate == 0 or cur_t == -1 or cur_t == stop_early-1:
            #   image.save('progress.png')
            #   display.clear_output(wait=True)
            #   display.display(display.Image('progress.png'))
            # if args.steps_per_checkpoint is not None:
            #   if j % args.steps_per_checkpoint == 0 and j > 0:
            #     if args.intermediates_in_subfolder is True:
            #       image.save(f'{partialFolder}/{filename}')
            #     else:
            #       image.save(f'{batchFolder}/{filename}')
            # else:
            #   if j in args.intermediate_saves:
            #     if args.intermediates_in_subfolder is True:
            #       image.save(f'{partialFolder}/{filename}')
            #     else:
            #       image.save(f'{batchFolder}/{filename}')
            # if (cur_t == -1) | (cur_t == stop_early-1):
            #   if cur_t == stop_early-1: print('early stopping')
            # if frame_num == 0:
            #   save_settings()
            # if args.animation_mode != "None":
            #   # sys.exit(os.getcwd(), 'cwd')
            #   image.save('prevFrame.png')
            # image.save(f'{batchFolder}/{filename}')
            # if args.animation_mode == 'Video Input':
            #   # If turbo, save a blended image
            #   if turbo_mode and frame_num > 0:
            #     # Mix new image with prevFrameScaled
            #     blend_factor = (1)/int(turbo_steps)
            #     newFrame = cv2.imread('prevFrame.png') # This is already updated..
            #     prev_frame_warped = cv2.imread('prevFrameScaled.png')
            #     blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
            #     cv2.imwrite(f'{batchFolder}/{filename}',blendedImage)
            #   else:
            #     image.save(f'{batchFolder}/{filename}')

            # if frame_num != args.max_frames-1:
            #   display.clear_output()

            plt.plot(np.array(loss_values), "r")
    batchBar.close()


def save_settings(args_in):
    setting_list = {
        "text_prompts": text_prompts,
        "user_comment": user_comment,
        "image_prompts": image_prompts,
        # 'clip_guidance_scale': clip_guidance_scale,
        # 'tv_scale': tv_scale,
        "range_scale": range_scale,
        "sat_scale": sat_scale,
        # 'cutn': cutn,
        # 'cutn_batches': cutn_batches,
        "max_frames": max_frames,
        "interp_spline": interp_spline,
        # 'rotation_per_frame': rotation_per_frame,
        "init_image": init_image,
        # 'init_scale': init_scale,
        # 'skip_steps': skip_steps,
        # 'zoom_per_frame': zoom_per_frame,
        # 'frames_scale': frames_scale,
        # 'frames_skip_steps': frames_skip_steps,
        # 'perlin_init': perlin_init,
        # 'perlin_mode': perlin_mode,
        # 'skip_augs': skip_augs,
        # 'randomize_class': randomize_class,
        # 'clip_denoised': clip_denoised,
        "clamp_grad": clamp_grad,
        "clamp_max": clamp_max,
        "seed": seed,
        "fuzzy_prompt": fuzzy_prompt,
        "rand_mag": rand_mag,
        "eta": eta,
        "width": width_height[0],
        "height": width_height[1],
        "diffusion_model": diffusion_model,
        # 'use_secondary_model': use_secondary_model,
        # 'steps': steps,
        "diffusion_steps": diffusion_steps,
        # 'diffusion_sampling_mode': diffusion_sampling_mode,
        # 'ViTB32': ViTB32,
        # 'ViTB16': ViTB16,
        # 'ViTL14': ViTL14,
        # 'ViTL14_336': ViTL14_336,
        # 'RN101': RN101,
        # 'RN50': RN50,
        # 'RN50x4': RN50x4,
        # 'RN50x16': RN50x16,
        # 'RN50x64': RN50x64,
        # 'cut_overview': str(cut_overview),
        # 'cut_innercut': str(cut_innercut),
        # 'cut_ic_pow': cut_ic_pow,
        # 'cut_icgray_p': str(cut_icgray_p),
        # 'key_frames': key_frames,
        "max_frames": max_frames,
        "video_init_path": video_init_path,
        "extract_nth_frame": extract_nth_frame,
        "flow_video_init_path": flow_video_init_path,
        "flow_extract_nth_frame": flow_extract_nth_frame,
        "video_init_seed_continuity": video_init_seed_continuity,
        "turbo_mode": turbo_mode,
        "turbo_steps": turbo_steps,
        "turbo_preroll": turbo_preroll,
        # warp settings
        "flow_warp": flow_warp,
        # 'flow_blend':flow_blend,
        "check_consistency": check_consistency,
        "turbo_frame_skips_steps": turbo_frame_skips_steps,
        "forward_weights_clip": forward_weights_clip,
        "forward_weights_clip_turbo_step": forward_weights_clip_turbo_step,
        # 'disable_cc_for_turbo_frames' : disable_cc_for_turbo_frames,
        "padding_ratio": padding_ratio,
        "padding_mode": padding_mode,
        "consistency_blur": consistency_blur,
        "inpaint_blend": inpaint_blend,
        "match_color_strength": match_color_strength,
        "high_brightness_threshold": high_brightness_threshold,
        "high_brightness_adjust_ratio": high_brightness_adjust_ratio,
        "low_brightness_threshold": low_brightness_threshold,
        "low_brightness_adjust_ratio": low_brightness_adjust_ratio,
        "stop_early": stop_early,
        "high_brightness_adjust_fix_amount": high_brightness_adjust_fix_amount,
        "low_brightness_adjust_fix_amount": low_brightness_adjust_fix_amount,
        "max_brightness_threshold": max_brightness_threshold,
        "min_brightness_threshold": min_brightness_threshold,
        "enable_adjust_brightness": enable_adjust_brightness,
        # 'init_latent_scale':init_latent_scale,
        # 'frames_latent_scale':frames_latent_scale,
        # 'cfg_scale':cfg_scale,
        "dynamic_thresh": dynamic_thresh,
        "warp_interp": warp_interp,
        "fixed_code": fixed_code,
        "blend_code": blend_code,
        "normalize_code": normalize_code,
        "mask_result": mask_result,
        "reverse_cc_order": reverse_cc_order,
        "flow_lq": flow_lq,
        "use_predicted_noise": use_predicted_noise,
        "clip_guidance_scale": clip_guidance_scale,
        "clip_type": clip_type,
        "clip_pretrain": clip_pretrain,
        "missed_consistency_weight": missed_consistency_weight,
        "overshoot_consistency_weight": overshoot_consistency_weight,
        "edges_consistency_weight": edges_consistency_weight,
        "style_strength_schedule": style_strength_schedule,
        "flow_blend_schedule": flow_blend_schedule,
        "steps_schedule": steps_schedule,
        "init_scale_schedule": init_scale_schedule,
        "latent_scale_schedule": latent_scale_schedule,
        "latent_scale_template": latent_scale_template,
        "init_scale_template": init_scale_template,
        "steps_template": steps_template,
        "style_strength_template": style_strength_template,
        "flow_blend_template": flow_blend_template,
        "make_schedules": make_schedules,
        "normalize_latent": normalize_latent,
        "normalize_latent_offset": normalize_latent_offset,
        "colormatch_frame": colormatch_frame,
        "use_karras_noise": use_karras_noise,
        "end_karras_ramp_early": end_karras_ramp_early,
        "use_background_mask": use_background_mask,
        "apply_mask_after_warp": apply_mask_after_warp,
        "background": background,
        "background_source": background_source,
        "mask_source": mask_source,
        "extract_background_mask": extract_background_mask,
        "mask_video_path": mask_video_path,
        "negative_prompts": negative_prompts,
        "invert_mask": invert_mask,
        "warp_strength": warp_strength,
        "flow_override_map": flow_override_map,
        "cfg_scale_schedule": cfg_scale_schedule,
        "respect_sched": respect_sched,
        "color_match_frame_str": color_match_frame_str,
        "colormatch_offset": colormatch_offset,
        # 'latent_fixed_norm':latent_fixed_norm,
        "latent_fixed_mean": latent_fixed_mean,
        "latent_fixed_std": latent_fixed_std,
        "colormatch_method": colormatch_method,
        "colormatch_regrain": colormatch_regrain,
        "warp_mode": warp_mode,
        "use_patchmatch_inpaiting": use_patchmatch_inpaiting,
        "blend_latent_to_init": blend_latent_to_init,
        "warp_towards_init": warp_towards_init,
        "init_grad": init_grad,
        "grad_denoised": grad_denoised,
    }
    setting_list.update(args_in)
    # print('Settings:', setting_list)
    with open(f"{batchFolder}/{batch_name}({batchNum})_settings.txt", "w+") as f:  # save settings
        json.dump(setting_list, f, ensure_ascii=False, indent=4)


# %%
# @title 1.6 Define the secondary diffusion model


def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, c),
            ConvBlock(c, c),
            SkipBlock(
                [
                    nn.AvgPool2d(2),
                    ConvBlock(c, c * 2),
                    ConvBlock(c * 2, c * 2),
                    SkipBlock(
                        [
                            nn.AvgPool2d(2),
                            ConvBlock(c * 2, c * 4),
                            ConvBlock(c * 4, c * 4),
                            SkipBlock(
                                [
                                    nn.AvgPool2d(2),
                                    ConvBlock(c * 4, c * 8),
                                    ConvBlock(c * 8, c * 4),
                                    nn.Upsample(
                                        scale_factor=2, mode="bilinear", align_corners=False
                                    ),
                                ]
                            ),
                            ConvBlock(c * 8, c * 4),
                            ConvBlock(c * 4, c * 2),
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        ]
                    ),
                    ConvBlock(c * 4, c * 2),
                    ConvBlock(c * 2, c),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ]
            ),
            ConvBlock(c * 2, c),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock(
                [
                    self.down,
                    ConvBlock(cs[0], cs[1]),
                    ConvBlock(cs[1], cs[1]),
                    SkipBlock(
                        [
                            self.down,
                            ConvBlock(cs[1], cs[2]),
                            ConvBlock(cs[2], cs[2]),
                            SkipBlock(
                                [
                                    self.down,
                                    ConvBlock(cs[2], cs[3]),
                                    ConvBlock(cs[3], cs[3]),
                                    SkipBlock(
                                        [
                                            self.down,
                                            ConvBlock(cs[3], cs[4]),
                                            ConvBlock(cs[4], cs[4]),
                                            SkipBlock(
                                                [
                                                    self.down,
                                                    ConvBlock(cs[4], cs[5]),
                                                    ConvBlock(cs[5], cs[5]),
                                                    ConvBlock(cs[5], cs[5]),
                                                    ConvBlock(cs[5], cs[4]),
                                                    self.up,
                                                ]
                                            ),
                                            ConvBlock(cs[4] * 2, cs[4]),
                                            ConvBlock(cs[4], cs[3]),
                                            self.up,
                                        ]
                                    ),
                                    ConvBlock(cs[3] * 2, cs[3]),
                                    ConvBlock(cs[3], cs[2]),
                                    self.up,
                                ]
                            ),
                            ConvBlock(cs[2] * 2, cs[2]),
                            ConvBlock(cs[2], cs[1]),
                            self.up,
                        ]
                    ),
                    ConvBlock(cs[1] * 2, cs[1]),
                    ConvBlock(cs[1], cs[0]),
                    self.up,
                ]
            ),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


# %% [markdown]
# # 2. Diffusion and CLIP model settings

# %%
# @title init main sd run function, cond_fn, color matching for SD
init_latent = None
target_embed = None


mask_result = False
early_stop = 0
inpainting_stop = 0
warp_interp = PIL.Image.BILINEAR

# init SD
from glob import glob
import argparse, os, sys
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

os.chdir(f"{root_dir}")


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


from kornia import augmentation as KA

aug = KA.RandomAffine(0, (1 / 14, 1 / 14), p=1, padding_mode="border")
from torch.nn import functional as F


def sd_cond_fn(
    x,
    t,
    denoised,
    init_image_sd,
    init_latent,
    init_scale,
    init_latent_scale,
    target_embed,
    **kwargs,
):
    with torch.cuda.amp.autocast():
        # init_latent_scale,  init_scale, clip_guidance_scale, target_embed, init_latent, clamp_grad, clamp_max,
        # **kwargs):
        # global init_latent_scale
        # global init_scale
        global clip_guidance_scale
        # global target_embed
        # print(target_embed.shape)
        global clamp_grad
        global clamp_max
        loss = 0.0
        if grad_denoised:
            x = denoised
        grad = torch.zeros_like(x)
        if sat_scale > 0 or init_scale > 0 or clip_guidance_scale > 0:
            with torch.autocast("cuda"):
                denoised_small = denoised[:, :, ::2, ::2]
                denoised_img = (
                    model_wrap_cfg.inner_model.inner_model.differentiable_decode_first_stage(
                        denoised_small
                    )
                )

        if clip_guidance_scale > 0:
            # compare text clip embeds with denoised image embeds
            # denoised_img = model_wrap_cfg.inner_model.inner_model.differentiable_decode_first_stage(denoised);# print(denoised.requires_grad)
            # print('d b',denoised.std(), denoised.mean())
            denoised_img = denoised_img[0].add(1).div(2)
            denoised_img = normalize(denoised_img)
            denoised_t = denoised_img.half().cuda()[None, ...]
            # print('d a',denoised_t.std(), denoised_t.mean())
            image_embed = get_image_embed(denoised_t).half()

            # image_embed = get_image_embed(denoised.add(1).div(2)).half()
            loss = spherical_dist_loss(image_embed, target_embed).sum() * clip_guidance_scale

        if init_latent_scale > 0:
            # compare init image latent with denoised latent
            loss += spherical_dist_loss(denoised, init_latent).sum() * init_latent_scale

        if sat_scale > 0:
            loss += torch.abs(denoised_img - denoised_img.clamp(min=-1, max=1)).mean()

        if init_scale > 0:
            # compare init image with denoised latent image via lpips
            # print('init_image_sd', init_image_sd)

            loss += lpips_model(denoised_img, init_image_sd[:, :, ::2, ::2]).sum() * init_scale

        if loss != 0.0:
            grad = -torch.autograd.grad(loss, x)[0]
            if torch.isnan(grad).any():
                return torch.zeros_like(x)
            if VERBOSE:
                print(loss, grad.max())
            if clamp_grad:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=clamp_max) / magnitude

        return grad


import cv2

try:
    from python_color_transfer.color_transfer import ColorTransfer, Regrain
except:
    os.chdir(root_dir)
    gitclone("https://github.com/pengbo-learn/python-color-transfer")
    sys.path.append("./python-color-transfer")

from python_color_transfer.color_transfer import ColorTransfer, Regrain

PT = ColorTransfer()


def match_color_var(stylized_img, raw_img, opacity=1.0, f=PT.pdf_transfer, regrain=False):
    img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype("uint8"), cv2.COLOR_RGB2BGR)
    img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype("uint8"), cv2.COLOR_RGB2BGR)
    img_arr_ref = cv2.resize(
        img_arr_ref, (img_arr_in.shape[1], img_arr_in.shape[0]), interpolation=cv2.INTER_CUBIC
    )

    # img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
    img_arr_col = f(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
    if regrain:
        img_arr_col = RG.regrain(img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
    img_arr_col = img_arr_col * opacity + img_arr_in * (1 - opacity)
    img_arr_reg = cv2.cvtColor(img_arr_col.round().astype("uint8"), cv2.COLOR_BGR2RGB)

    return img_arr_reg


# https://gist.githubusercontent.com/trygvebw/c71334dd127d537a15e9d59790f7f5e1/raw/ed0bed6abaf75c0f1b270cf6996de3e07cbafc81/find_noise.py

import torch
import numpy as np

# import k_diffusion as K

from PIL import Image
from torch import autocast
from einops import rearrange, repeat


def pil_img_to_torch(pil_img, half=False):
    image = np.array(pil_img).astype(np.float32) / 255.0
    image = rearrange(torch.from_numpy(image), "h w c -> c h w")
    if half:
        image = image.half()
    return (2.0 * image - 1.0).unsqueeze(0)


def pil_img_to_latent(model, img, batch_size=1, device="cuda", half=True):
    init_image = pil_img_to_torch(img, half=half).to(device)
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    if half:
        return model.get_first_stage_encoding(model.encode_first_stage(init_image.half()))
    return model.get_first_stage_encoding(model.encode_first_stage(init_image))


def find_noise_for_image(model, x, prompt, steps, cond_scale=0.0, verbose=False, normalize=True):

    with torch.no_grad():
        with autocast("cuda"):
            uncond = model.get_learned_conditioning([""])
            cond = model.get_learned_conditioning([prompt])

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    if verbose:
        print(sigmas)

    with torch.no_grad():
        with autocast("cuda"):
            for i in trange(1, len(sigmas)):
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
                cond_in = torch.cat([uncond, cond])

                c_out, c_in = [
                    K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)
                ]

                if i == 1:
                    t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
                else:
                    t = dnw.sigma_to_t(sigma_in)

                eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

                denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

                if i == 1:
                    d = (x - denoised) / (2 * sigmas[i])
                else:
                    d = (x - denoised) / sigmas[i - 1]

                dt = sigmas[i] - sigmas[i - 1]
                x = x + d * dt
            print(x.shape)
            if normalize:
                return (x / x.std()) * sigmas[-1]
            else:
                return x


# karras noise
# https://github.com/Birch-san/stable-diffusion/blob/693c8a336aa3453d30ce403f48eb545689a679e5/scripts/txt2img_fork.py#L62-L81
sys.path.append("./k-diffusion")


def get_premature_sigma_min(
    steps: int, sigma_max: float, sigma_min_nominal: float, rho: float
) -> float:
    min_inv_rho = sigma_min_nominal ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    ramp = (steps - 2) * 1 / (steps - 1)
    sigma_min = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigma_min


pred_noise = None


def run_sd(
    opt,
    init_image,
    skip_timesteps,
    H,
    W,
    text_prompt,
    neg_prompt,
    steps,
    seed,
    init_scale,
    init_latent_scale,
    cfg_scale,
    cond_fn=None,
    init_grad_img=None,
):
    seed_everything(seed)
    # global cfg_scale
    if VERBOSE:
        print(
            "seed",
            "clip_guidance_scale",
            "init_scale",
            "init_latent_scale",
            "clamp_grad",
            "clamp_max",
            "init_image",
            "skip_timesteps",
            "cfg_scale",
        )
        print(
            seed,
            clip_guidance_scale,
            init_scale,
            init_latent_scale,
            clamp_grad,
            clamp_max,
            init_image,
            skip_timesteps,
            cfg_scale,
        )
    global start_code
    global pred_noise
    global frame_num
    global normalize_latent
    global first_latent
    global first_latent_source
    global use_karras_noise
    global end_karras_ramp_early
    global latent_fixed_norm
    global latent_norm_4d
    global latent_fixed_mean
    global latent_fixed_std
    global n_mean_avg
    global n_std_avg

    batch_size = 1
    scale = cfg_scale

    C = 4  # 4
    f = 8  # 8
    H = H
    W = W
    if VERBOSE:
        print(W, H, "WH")
    prompt = text_prompt[0]
    neg_prompt = neg_prompt[0]
    ddim_steps = steps

    # init_latent_scale = 0. #20
    prompt_clip = prompt

    assert prompt is not None
    prompts = [prompt]
    if VERBOSE:
        print("prompts", prompts, text_prompt)

    precision_scope = autocast

    t_enc = ddim_steps - skip_timesteps

    if init_image is not None:
        if isinstance(init_image, str):
            if not init_image.endswith("_lat.pt"):
                with torch.no_grad():
                    init_image_sd = load_img_sd(init_image, size=(W, H)).cuda().half()
                    init_latent = sd_model.get_first_stage_encoding(
                        sd_model.encode_first_stage(init_image_sd)
                    ).half()
                    x0 = init_latent
            if init_image.endswith("_lat.pt"):
                init_latent = torch.load(init_image).cuda().half()
                init_image_sd = None
                x0 = init_latent

    if init_grad_img is not None:
        print("Replacing init image for cond fn")
        init_image_sd = load_img_sd(init_grad_img, size=(W, H)).cuda().half()

    if blend_latent_to_init > 0.0 and first_latent is not None:
        print("Blending to latent ", first_latent_source)
        x0 = x0 * (1 - blend_latent_to_init) + blend_latent_to_init * first_latent
    if normalize_latent != "off" and first_latent is not None:
        if VERBOSE:
            print("norm to 1st latent")
            print("latent source - ", first_latent_source)
        # noise2 - target
        # noise - modified

        if latent_norm_4d:
            n_mean = first_latent.mean(dim=(2, 3), keepdim=True)
            n_std = first_latent.std(dim=(2, 3), keepdim=True)
        else:
            n_mean = first_latent.mean()
            n_std = first_latent.std()

        if n_mean_avg is None and n_std_avg is None:
            n_mean_avg = n_mean.clone().detach().cpu().numpy()[0, :, 0, 0]
            n_std_avg = n_std.clone().detach().cpu().numpy()[0, :, 0, 0]
        else:
            n_mean_avg = (
                n_mean_avg * n_smooth
                + (1 - n_smooth) * n_mean.clone().detach().cpu().numpy()[0, :, 0, 0]
            )
            n_std_avg = (
                n_std_avg * n_smooth
                + (1 - n_smooth) * n_std.clone().detach().cpu().numpy()[0, :, 0, 0]
            )

        if VERBOSE:
            print("n_stats_avg (mean, std): ", n_mean_avg, n_std_avg)
        if normalize_latent == "user_defined":
            n_mean = latent_fixed_mean
            if isinstance(n_mean, list) and len(n_mean) == 4:
                n_mean = np.array(n_mean)[None, :, None, None]
            n_std = latent_fixed_std
            if isinstance(n_std, list) and len(n_std) == 4:
                n_std = np.array(n_std)[None, :, None, None]
        if latent_norm_4d:
            n2_mean = x0.mean(dim=(2, 3), keepdim=True)
        else:
            n2_mean = x0.mean()
        x0 = x0 - (n2_mean - n_mean)
        if latent_norm_4d:
            n2_std = x0.std(dim=(2, 3), keepdim=True)
        else:
            n2_std = x0.std()
        x0 = x0 / (n2_std / n_std)

    if clip_guidance_scale > 0:
        # text_features = clip_model.encode_text(text)
        target_embed = F.normalize(
            clip_model.encode_text(open_clip.tokenize(prompt_clip).cuda()).float()
        )
    else:
        target_embed = None
    cond_fn_partial = partial(
        sd_cond_fn,
        init_image_sd=init_image_sd,
        init_latent=init_latent,
        init_scale=init_scale,
        init_latent_scale=init_latent_scale,
        target_embed=target_embed,
    )
    model_fn = make_cond_model_fn(model_wrap_cfg, cond_fn_partial)
    model_fn = make_static_thresh_model_fn(model_fn, dynamic_thresh)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with precision_scope("cuda"):
                with sd_model.ema_scope():
                    tic = time.time()
                    all_samples = []
                    uc = None
                    if True:
                        if scale != 1.0:
                            uc = sd_model.get_learned_conditioning([neg_prompt])

                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = sd_model.get_learned_conditioning(prompts)

                        shape = [C, H // f, W // f]
                        if use_karras_noise:

                            rho = 7.0
                            # 14.6146
                            sigma_max = model_wrap.sigmas[-1].item()
                            sigma_min_nominal = model_wrap.sigmas[0].item()
                            # get the "sigma before sigma_min" from a slightly longer ramp
                            # https://github.com/crowsonkb/k-diffusion/pull/23#issuecomment-1234872495
                            premature_sigma_min = get_premature_sigma_min(
                                steps=steps + 1,
                                sigma_max=sigma_max,
                                sigma_min_nominal=sigma_min_nominal,
                                rho=rho,
                            )
                            sigmas = K.sampling.get_sigmas_karras(
                                n=steps,
                                sigma_min=premature_sigma_min
                                if end_karras_ramp_early
                                else sigma_min_nominal,
                                sigma_max=sigma_max,
                                rho=rho,
                                device="cuda",
                            )
                        else:
                            sigmas = model_wrap.get_sigmas(ddim_steps)
                        extra_args = {"cond": c, "uncond": uc, "cond_scale": scale}
                        if skip_timesteps > 0:
                            # using non-random start code
                            if fixed_code:
                                if start_code is None:
                                    if VERBOSE:
                                        print("init start code")
                                    start_code = torch.randn_like(
                                        x0
                                    )  # * sigmas[ddim_steps - t_enc - 1]
                                if normalize_code:
                                    noise2 = torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1]
                                    if latent_norm_4d:
                                        n_mean = noise2.mean(dim=(2, 3), keepdim=True)
                                    else:
                                        n_mean = noise2.mean()
                                    if latent_norm_4d:
                                        n_std = noise2.std(dim=(2, 3), keepdim=True)
                                    else:
                                        n_std = noise2.std()
                                    # n_mean = noise2.mean()
                                    # n_std = noise2.std()
                                noise = torch.randn_like(x0)
                                noise = (
                                    start_code * (blend_code) + (1 - blend_code) * noise
                                ) * sigmas[ddim_steps - t_enc - 1]
                                if normalize_code:
                                    # n2_mean = noise.mean()
                                    # noise = noise - (n2_mean-n_mean)
                                    # n2_std = noise.std()
                                    # noise = noise/(n2_std/n_std)

                                    if latent_norm_4d:
                                        n2_mean = noise.mean(dim=(2, 3), keepdim=True)
                                    else:
                                        n2_mean = noise.mean()
                                    noise = noise - (n2_mean - n_mean)
                                    if latent_norm_4d:
                                        n2_std = noise.std(dim=(2, 3), keepdim=True)
                                    else:
                                        n2_std = noise.std()
                                    noise = noise / (n2_std / n_std)

                                    # noise = torch.roll(noise,shifts = (3,3), dims=(2,3)) #not helping
                                    # print('noise randn at this time',noise2.mean(), noise2.std(), noise2.min(), noise2.max())
                                    # print('start code noise randn', start_code.mean(), start_code.std(), start_code.min(), start_code.max())
                                    # print('noise randn balanced',noise.mean(), noise.std(), noise.min(), noise.max())
                                # xi = x0 + noise
                            # elif use_predicted_noise:
                            # #  if frame_num>0:
                            #   print('using predicted noise')
                            #   init_image_pil = PIL.Image.open(init_image)
                            #   pred_noise = find_noise_for_image(sd_model, x0, prompts[0], t_enc, cond_scale=scale, verbose=False, normalize=True)
                            #   xi = pred_noise#*sigmas[ddim_steps - t_enc -1]

                            else:
                                noise = torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1]
                            xi = x0 + noise
                            # print(xi.mean(), xi.std(), xi.min(), xi.max())
                            sigma_sched = sigmas[ddim_steps - t_enc - 1 :]
                            samples_ddim = K.sampling.sample_euler(
                                model_fn, xi, sigma_sched, extra_args=extra_args
                            )
                        else:
                            x = torch.randn([batch_size, *shape], device=device) * sigmas[0]
                            samples_ddim = K.sampling.sample_lms(
                                model_fn, x, sigmas, extra_args=extra_args
                            )
                        if first_latent is None:
                            if VERBOSE:
                                print("setting 1st latent")
                            first_latent_source = "samples ddim (1st frame output)"
                            first_latent = samples_ddim
                        x_samples_ddim = sd_model.decode_first_stage(samples_ddim)
                        printf(
                            "x_samples_ddim",
                            x_samples_ddim.min(),
                            x_samples_ddim.max(),
                            x_samples_ddim.std(),
                            x_samples_ddim.mean(),
                        )
                        scale_raw_sample = False
                        if scale_raw_sample:
                            m = x_samples_ddim.mean()
                            x_samples_ddim -= m
                            r = (x_samples_ddim.max() - x_samples_ddim.min()) / 2

                            x_samples_ddim /= r
                            x_samples_ddim += m
                            if VERBOSE:
                                printf(
                                    "x_samples_ddim scaled",
                                    x_samples_ddim.min(),
                                    x_samples_ddim.max(),
                                    x_samples_ddim.std(),
                                    x_samples_ddim.mean(),
                                )

                        all_samples.append(x_samples_ddim)
    return all_samples, samples_ddim


# %%
# @markdown ####**Models Settings:**
# @markdown #####temporarily off
diffusion_model = "stable_diffusion"
use_secondary_model = False
diffusion_sampling_mode = "ddim"
##@markdown #####**Custom model:**
custom_path = ""

##@markdown #####**CLIP settings:**
ViTB32 = False
ViTB16 = False
ViTL14 = False
ViTL14_336 = False
RN101 = False
RN50 = False
RN50x4 = False
RN50x16 = False
RN50x64 = False

## @markdown If you're having issues with model downloads, check this to compare SHA's:
check_model_SHA = False
use_checkpoint = True
model_256_SHA = "983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a"
model_512_SHA = "9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648"
model_256_comics_SHA = "f587fd6d2edb093701931e5083a13ab6b76b3f457b60efd1aa873d60ee3d6388"
model_secondary_SHA = "983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a"

model_256_link = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
)
model_512_link = "https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt"
model_256_comics_link = "https://github.com/Sxela/DiscoDiffusion-Warp/releases/download/v0.1.0/256x256_openai_comics_faces_by_alex_spirin_084000.pt"
model_secondary_link = (
    "https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth"
)

model_256_path = f"{model_path}/256x256_diffusion_uncond.pt"
model_512_path = f"{model_path}/512x512_diffusion_uncond_finetune_008100.pt"
model_256_comics_path = f"{model_path}/256x256_openai_comics_faces_by_alex_spirin_084000.pt"
model_secondary_path = f"{model_path}/secondary_model_imagenet_2.pth"

model_256_downloaded = False
model_512_downloaded = False
model_secondary_downloaded = False
model_256_comics_downloaded = False

# Download the diffusion model
if diffusion_model == "256x256_diffusion_uncond":
    if os.path.exists(model_256_path) and check_model_SHA:
        print("Checking 256 Diffusion File")
        with open(model_256_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
        if hash == model_256_SHA:
            print("256 Model SHA matches")
            model_256_downloaded = True
        else:
            print("256 Model SHA doesn't match, redownloading...")
            wget(model_256_link, model_path)
            model_256_downloaded = True
    elif os.path.exists(model_256_path) and not check_model_SHA or model_256_downloaded == True:
        print("256 Model already downloaded, check check_model_SHA if the file is corrupt")
    else:
        wget(model_256_link, model_path)
        model_256_downloaded = True
elif diffusion_model == "512x512_diffusion_uncond_finetune_008100":
    if os.path.exists(model_512_path) and check_model_SHA:
        print("Checking 512 Diffusion File")
        with open(model_512_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
        if hash == model_512_SHA:
            print("512 Model SHA matches")
            model_512_downloaded = True
        else:
            print("512 Model SHA doesn't match, redownloading...")
            wget(model_512_link, model_path)
            model_512_downloaded = True
    elif os.path.exists(model_512_path) and not check_model_SHA or model_512_downloaded == True:
        print("512 Model already downloaded, check check_model_SHA if the file is corrupt")
    else:
        wget(model_512_link, model_path)
        model_512_downloaded = True
elif diffusion_model == "256x256_openai_comics_faces_by_alex_spirin_084000":
    if os.path.exists(model_256_comics_path) and check_model_SHA:
        print("Checking 256 Comics Diffusion File")
        with open(model_256_comics_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
        if hash == model_256_comics_SHA:
            print("256 Comics Model SHA matches")
            model_256_comics_downloaded = True
        else:
            print("256 Comics SHA doesn't match, redownloading...")
            wget(model_256_comics_link, model_path)
            model_256_comics_downloaded = True
    elif (
        os.path.exists(model_256_comics_path)
        and not check_model_SHA
        or model_256_comics_downloaded == True
    ):
        print("256 Comics Model already downloaded, check check_model_SHA if the file is corrupt")
    else:
        wget(model_256_comics_link, model_path)
        model_256_comics_downloaded = True


# Download the secondary diffusion model v2
if use_secondary_model == True:
    if os.path.exists(model_secondary_path) and check_model_SHA:
        print("Checking Secondary Diffusion File")
        with open(model_secondary_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
        if hash == model_secondary_SHA:
            print("Secondary Model SHA matches")
            model_secondary_downloaded = True
        else:
            print("Secondary Model SHA doesn't match, redownloading...")
            wget(model_secondary_link, model_path)
            model_secondary_downloaded = True
    elif (
        os.path.exists(model_secondary_path)
        and not check_model_SHA
        or model_secondary_downloaded == True
    ):
        print("Secondary Model already downloaded, check check_model_SHA if the file is corrupt")
    else:
        wget(model_secondary_link, model_path)
        model_secondary_downloaded = True

model_config = model_and_diffusion_defaults()
if diffusion_model == "512x512_diffusion_uncond_finetune_008100":
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": 1000,  # No need to edit this, it is taken care of later.
            "rescale_timesteps": True,
            "timestep_respacing": 250,  # No need to edit this, it is taken care of later.
            "image_size": 512,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": use_checkpoint,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )
elif diffusion_model == "256x256_diffusion_uncond":
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": 1000,  # No need to edit this, it is taken care of later.
            "rescale_timesteps": True,
            "timestep_respacing": 250,  # No need to edit this, it is taken care of later.
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": use_checkpoint,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )
elif diffusion_model == "256x256_openai_comics_faces_by_alex_spirin_084000":
    model_config.update(
        {
            "attention_resolutions": "16",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": "ddim100",
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 128,
            "num_heads": 1,
            "num_res_blocks": 2,
            "use_checkpoint": use_checkpoint,
            "use_fp16": True,
            "use_scale_shift_norm": False,
        }
    )

model_default = model_config["image_size"]

if use_secondary_model:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(
        torch.load(f"{model_path}/secondary_model_imagenet_2.pth", map_location="cpu")
    )
    secondary_model.eval().requires_grad_(False).to(device)

clip_models = []
if ViTB32 is True:
    clip_models.append(clip.load("ViT-B/32", jit=False)[0].eval().requires_grad_(False).to(device))
if ViTB16 is True:
    clip_models.append(clip.load("ViT-B/16", jit=False)[0].eval().requires_grad_(False).to(device))
if ViTL14 is True:
    clip_models.append(clip.load("ViT-L/14", jit=False)[0].eval().requires_grad_(False).to(device))
if ViTL14_336 is True:
    clip_models.append(
        clip.load("ViT-L/14@336px", jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN50 is True:
    clip_models.append(clip.load("RN50", jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x4 is True:
    clip_models.append(clip.load("RN50x4", jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x16 is True:
    clip_models.append(clip.load("RN50x16", jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x64 is True:
    clip_models.append(clip.load("RN50x64", jit=False)[0].eval().requires_grad_(False).to(device))
if RN101 is True:
    clip_models.append(clip.load("RN101", jit=False)[0].eval().requires_grad_(False).to(device))

normalize = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)
lpips_model = lpips.LPIPS(net="vgg").to(device)

if diffusion_model == "custom":
    model_config.update(
        {
            "attention_resolutions": "16",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": "ddim100",
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 128,
            "num_heads": 1,
            "num_res_blocks": 2,
            "use_checkpoint": use_checkpoint,
            "use_fp16": True,
            "use_scale_shift_norm": False,
        }
    )

# %% [markdown]
# # 3. Settings

# %%
# @markdown ####**Basic Settings:**
# !settings
# TODO video_init_path and other vid_input downstrems
batch_name = vid_input.split(".")[0]  # @param{type: 'string'}
steps = 250
##@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
# stop_early = 0  #@param{type: 'number'}
stop_early = 0
stop_early = min(steps - 1, stop_early)
# @markdown Specify desired output size here.\
# @markdown Don't forget to rerun all steps after changing the width height (including forcing optical flow generation)
width_height = [1024, 576]  # @param{type: 'raw'}
clip_guidance_scale = 13000  # @param{type: 'number'}
tv_scale = 15000  # @param{type: 'number'}
range_scale = 1  # @param{type: 'number'}
sat_scale = 2000  # @param{type: 'number'}
cutn_batches = 4
skip_augs = False

# @markdown ---

# @markdown ####**Init Settings:**
init_image = ""  # @param{type: 'string'}
init_scale = 0
##@param{type: 'integer'}
skip_steps = 25
##@param{type: 'integer'}
##@markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.\
##@markdown A good init_scale for Stable Diffusion is 0*


# Get corrected sizes
side_x = (width_height[0] // 64) * 64
side_y = (width_height[1] // 64) * 64
if side_x != width_height[0] or side_y != width_height[1]:
    print(f"Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.")
width_height = (side_x, side_y)
# Update Model Settings
timestep_respacing = f"ddim{steps}"
diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
model_config.update(
    {
        "timestep_respacing": timestep_respacing,
        "diffusion_steps": diffusion_steps,
    }
)

# Make folder for batch
batchFolder = f"{outDirPath}/{batch_name}"
createPath(batchFolder)

vidFolder = f"{outDirPath}/{batch_name}"
batchFolder = vidFolder
settingsFile = "settings.txt"
if michael_mode:
    # TODO mode to continue batch
    # Prefix settings file so it's easier to find at top of dir list
    settingsFile = "_settings.txt"
    num_dirs = 0
    if os.path.exists(vidFolder):
        lst = os.listdir(vidFolder)
        num_dirs = len(lst)
    batchFolder = os.path.join(vidFolder, time.strftime(f"_{9999 - num_dirs}_%m_%d__%H_%M"))

# %% [markdown]
# ### Animation Settings

# %%
# @markdown Create a looping video from single init image\
# @markdown Use this if you just want to test settings. This will create a small video (1 sec = 24 frames)\
# @markdown This way you will be able to iterate faster without the need to process flow maps for a long final video before even getting to testing prompts.
# @markdown You'll need to manually input the resulting video path into the next cell.

use_looped_init_image = False  # @param {'type':'boolean'}
video_duration_sec = 2  # @param {'type':'number'}
if use_looped_init_image:
    ffmpeg_cmd = (
        f"ffmpeg -loop 1 -i '{init_image}' -c:v libx264 -t '{video_duration_sec}' -pix_fmt yuv420p"
        f" -vf scale={side_x}:{side_y} '{root_dir}/out.mp4' -y"
    )
    subprocess.run(
        ffmpeg_cmd.split(" "),
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    print("Video saved to ", f"{root_dir}/out.mp4")

# %%
##@markdown ####**Animation Mode:**
animation_mode = "Video Input"


# @markdown ---

# @markdown ####**Video Input Settings:**

video_init_path = os.path.join(initDirPath, vid_input)

extract_nth_frame = 1  # @param {type: 'number'}
# @markdown *Specify frame range. end_frame=0 means fill the end of video*
start_frame = 0  # @param {type: 'number'}
end_frame = 0  # @param {type: 'number'}
if end_frame <= 0 or end_frame == None:
    end_frame = 99999999999999999999999999999
# @markdown ####Separate guiding video (optical flow source):
# @markdown Leave blank to use the first video.
flow_video_init_path = ""  # @param {type: 'string'}
flow_extract_nth_frame = 2  # @param {type: 'number'}
if flow_video_init_path == "":
    flow_video_init_path = None

store_frames_on_google_drive = False  # @param {type: 'boolean'}
video_init_seed_continuity = True
# @markdown #####**Video Optical Flow Settings:**
flow_warp = True  # @param {type: 'boolean'}
# cal optical flow from video frames and warp prev frame with flow
flow_blend = 0.999
##@param {type: 'number'} #0 - take next frame, 1 - take prev warped frame
check_consistency = True  # @param {type: 'boolean'}
# cal optical flow from video frames and warp prev frame with flow


def extractFrames(video_path, output_path, nth_frame, start_frame, end_frame):
    if not os.path.exists(output_path):
        createPath(output_path)

    lst = os.listdir(output_path)
    folder_size = len(lst)

    if folder_size > 1 and not force_vid_extract:
        print(f"Vid already extracted to: {output_path}")
        return
    else:
        print(f"Exporting Video Frames (1 every {nth_frame})...")

    try:
        for f in [o.replace("\\", "/") for o in glob(output_path + "/*.jpg")]:
            # for f in pathlib.Path(f'{output_path}').glob('*.jpg'):
            pathlib.Path(f).unlink()
    except:
        print("error deleting frame ", f)
    # vf = f'select=not(mod(n\\,{nth_frame}))'
    vf = f"select=between(n\\,{start_frame}\\,{end_frame}) , select=not(mod(n\\,{nth_frame}))"
    if os.path.exists(video_path):
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    f"{video_path}",
                    "-vf",
                    f"{vf}",
                    "-vsync",
                    "vfr",
                    "-q:v",
                    "2",
                    "-loglevel",
                    "error",
                    "-stats",
                    f"{output_path}/%06d.jpg",
                ],
                stdout=subprocess.PIPE,
            ).stdout.decode("utf-8")
        except:
            subprocess.run(
                [
                    "ffmpeg.exe",
                    "-i",
                    f"{video_path}",
                    "-vf",
                    f"{vf}",
                    "-vsync",
                    "vfr",
                    "-q:v",
                    "2",
                    "-loglevel",
                    "error",
                    "-stats",
                    f"{output_path}/%06d.jpg",
                ],
                stdout=subprocess.PIPE,
            ).stdout.decode("utf-8")

    else:
        sys.exit(f"\nERROR!\n\nVideo not found: {video_path}.\nPlease check your video path.\n")

videoFramesFolder = "//\\"
if animation_mode == "Video Input":
    if store_frames_on_google_drive:  # suggested by Chris the Wizard#8082 at discord
        videoFramesFolder = f"{batchFolder}/videoFrames"
        flowVideoFramesFolder = (
            f"{batchFolder}/flowVideoFrames" if flow_video_init_path else videoFramesFolder
        )
    else:
        videoFramesFolder = f"{root_dir}/videoFrames"
        flowVideoFramesFolder = (
            f"{root_dir}/flowVideoFrames" if flow_video_init_path else videoFramesFolder
        )
    if not is_colab:
        videoFramesFolder = f"{batchFolder}/videoFrames"
        flowVideoFramesFolder = (
            f"{batchFolder}/flowVideoFrames" if flow_video_init_path else videoFramesFolder
        )
    if michael_mode:
        videoFramesFolder = f"{vidFolder}/videoFrames"
        flowVideoFramesFolder = (
            f"{vidFolder}/flowVideoFrames" if flow_video_init_path else videoFramesFolder
        )

    extractFrames(video_init_path, videoFramesFolder, extract_nth_frame, start_frame, end_frame)
    if flow_video_init_path:
        print(flow_video_init_path, flowVideoFramesFolder, flow_extract_nth_frame)
        extractFrames(
            flow_video_init_path,
            flowVideoFramesFolder,
            flow_extract_nth_frame,
            start_frame,
            end_frame,
        )

##@markdown ---

##@markdown ####**2D Animation Settings:**
##@markdown `zoom` is a multiplier of dimensions, 1 is no zoom.
##@markdown All rotations are provided in degrees.

key_frames = True
# #@param {type:"boolean"}
max_frames = 10000
##@param {type:"number"}

if animation_mode == "Video Input":
    max_frames = len(glob(f"{videoFramesFolder}/*.jpg"))

interp_spline = (  # Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
    "Linear"
)
##@markdown ####**Coherency Settings:**
##@markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default for Stable Diffusion is 0.
frames_scale = 0
##@markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
frames_skip_steps = "0: (0.75)" # This is only used in Video Input Legacy
# I don't think any of these are used
angle = "0:(0)"
zoom = "0: (1), 10: (1.05)"
translation_x = "0: (0)"
translation_y = "0: (0)"
translation_z = "0: (10.0)"
rotation_3d_x = "0: (0)"
rotation_3d_y = "0: (0)"
rotation_3d_z = "0: (0)"
midas_depth_model = "dpt_large"
midas_weight = 0.3
near_plane = 200
far_plane = 10000
fov = 40
# padding_mode = 'border'
# sampling_mode = 'bicubic'

# ======= TURBO MODE
# @markdown ---
# @markdown ####**Turbo Mode:**
# @markdown (Starts after frame 1,) skips diffusion steps and just uses flow map to warp images for skipped frames.
# @markdown Speeds up rendering by 2x-4x, and may improve image coherence between frames. frame_blend_mode smooths abrupt texture changes across 2 frames.
# @markdown For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo

turbo_mode = False  # @param {type:"boolean"}
turbo_steps = "3"  # @param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 1  # frames

# insist turbo be used only w 3d anim.
if turbo_mode and animation_mode != "Video Input":
    print("=====")
    print("Turbo mode only available with 3D animations. Disabling Turbo.")
    print("=====")
    turbo_mode = False

# @markdown ---


vr_mode = False

vr_eye_angle = 0.5

vr_ipd = 5.0

# insist turbo be used only w 3d anim.
if vr_mode and animation_mode != "Video Input":
    print("=====")
    print("VR mode only available with 3D animations. Disabling VR.")
    print("=====")
    turbo_mode = False


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


def get_inbetweens(key_frames, integer=False):
    """Given a dict with frame numbers as keys and a parameter value as values,
    return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
    Any values not provided in the input dict are calculated by linear interpolation between
    the values of the previous and next provided frames. If there is no previous provided frame, then
    the value is equal to the value of the next provided frame, or if there is no next provided frame,
    then the value is equal to the value of the previous provided frame. If no frames are provided,
    all frame values are NaN.

    Parameters
    ----------
    key_frames: dict
        A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
    integer: Bool, optional
        If True, the values of the output series are converted to integers.
        Otherwise, the values are floats.

    Returns
    -------
    pd.Series
        A Series with length max_frames representing the parameter values for each frame.

    Examples
    --------
    >>> max_frames = 5
    >>> get_inbetweens({1: 5, 3: 6})
    0    5.0
    1    5.0
    2    5.5
    3    6.0
    4    6.0
    dtype: float64

    >>> get_inbetweens({1: 5, 3: 6}, integer=True)
    0    5
    1    5
    2    5
    3    6
    4    6
    dtype: int64
    """
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    interp_method = interp_spline

    if interp_method == "Cubic" and len(key_frames.items()) <= 3:
        interp_method = "Quadratic"

    if interp_method == "Quadratic" and len(key_frames.items()) <= 2:
        interp_method = "Linear"

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    # key_frame_series = key_frame_series.interpolate(method=intrp_method,order=1, limit_direction='both')
    key_frame_series = key_frame_series.interpolate(
        method=interp_method.lower(), limit_direction="both"
    )
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


def split_prompts(prompts):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series


if key_frames:
    try:
        frames_skip_steps_series= get_inbetweens(parse_key_frames(frames_skip_steps))
    except RuntimeError as e:
        print(
            f'{frames_skip_steps} not formatted as keyframe'
        )
        frames_skip_steps = f"0: ({frames_skip_steps})"
        frames_skip_steps_series = get_inbetweens(parse_key_frames(frames_skip_steps))

    try:
        angle_series = get_inbetweens(parse_key_frames(angle))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `angle` correctly for key frames.\n"
            "Attempting to interpret `angle` as "
            f'"0: ({angle})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        angle = f"0: ({angle})"
        angle_series = get_inbetweens(parse_key_frames(angle))

    try:
        zoom_series = get_inbetweens(parse_key_frames(zoom))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `zoom` correctly for key frames.\n"
            "Attempting to interpret `zoom` as "
            f'"0: ({zoom})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        zoom = f"0: ({zoom})"
        zoom_series = get_inbetweens(parse_key_frames(zoom))

    try:
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

else:
    angle = float(angle)
    zoom = float(zoom)
    translation_x = float(translation_x)
    translation_y = float(translation_y)
    translation_z = float(translation_z)
    rotation_3d_x = float(rotation_3d_x)
    rotation_3d_y = float(rotation_3d_y)
    rotation_3d_z = float(rotation_3d_z)

# %%
# @title Video Masking

# @markdown Generate background mask from your init video or use a video as a mask
mask_source = "init_video"  # @param ['init_video','mask_video']
# @markdown Check to rotoscope the video and create a mask from it. If unchecked, the raw monochrome video will be used as a mask.
extract_background_mask = True  # @param {'type':'boolean'}
# @markdown Specify path to a mask video for mask_video mode.
mask_video_path = ""  # @param {'type':'string'}
if extract_background_mask:
    os.chdir(root_dir)
    pip_res = subprocess.run(
        ["pip", "install", "av", "pims"], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    print(pip_res)
    if not os.path.exists("RobustVideoMattingCLI"):
        gitclone("https://github.com/Sxela/RobustVideoMattingCLI")
    if mask_source == "init_video":
        videoFramesAlpha = videoFramesFolder + "Alpha"
        createPath(videoFramesAlpha)
        pip_res = subprocess.run(
            [
                "python",
                f"{root_dir}/RobustVideoMattingCLI/rvm_cli.py",
                "--input_path",
                f"{videoFramesFolder}",
                "--output_alpha",
                f"{root_dir}/alpha.mp4",
            ],
            stdout=subprocess.PIPE,
        ).stdout.decode("utf-8")
        print(pip_res)
        extractFrames(f"{root_dir}/alpha.mp4", f"{videoFramesAlpha}", 1, 0, 999999999)
    if mask_source == "mask_video":
        videoFramesAlpha = videoFramesFolder + "Alpha"
        createPath(videoFramesAlpha)
        maskVideoFrames = videoFramesFolder + "Mask"
        createPath(maskVideoFrames)
        extractFrames(
            mask_video_path, f"{maskVideoFrames}", extract_nth_frame, start_frame, end_frame
        )
        pip_res = subprocess.run(
            [
                "python",
                f"{root_dir}/RobustVideoMattingCLI/rvm_cli.py",
                "--input_path",
                f"{maskVideoFrames}",
                "--output_alpha",
                f"{root_dir}/alpha.mp4",
            ],
            stdout=subprocess.PIPE,
        ).stdout.decode("utf-8")
        print(pip_res)
        extractFrames(f"{root_dir}/alpha.mp4", f"{videoFramesAlpha}", 1, 0, 999999999)
else:
    if mask_source == "init_video":
        videoFramesAlpha = videoFramesFolder
    if mask_source == "mask_video":
        videoFramesAlpha = videoFramesFolder + "Alpha"
        createPath(videoFramesAlpha)
        extractFrames(
            mask_video_path, f"{videoFramesAlpha}", extract_nth_frame, start_frame, end_frame
        )
        # extract video


# %%
# @title Install Color Transfer and RAFT
##@markdown Run once per session. Doesn't download again if model path exists.
##@markdown Use force download to reload raft models if needed
force_download = False  # @param {type:'boolean'}
# import wget
import zipfile, shutil

if (os.path.exists(f"{root_dir}/raft")) and force_download:
    try:
        shutil.rmtree(f"{root_dir}/raft")
    except:
        print("error deleting existing RAFT model")
if (not (os.path.exists(f"{root_dir}/raft"))) or force_download:
    os.chdir(root_dir)
    print("cloning WarpFusion")
    gitclone("https://github.com/Sxela/WarpFusion")

try:
    from python_color_transfer.color_transfer import ColorTransfer, Regrain
except:
    os.chdir(root_dir)
    gitclone("https://github.com/pengbo-learn/python-color-transfer")

os.chdir(root_dir)
sys.path.append("./python-color-transfer")

if animation_mode == "Video Input":
    os.chdir(root_dir)
    if not os.path.exists("flow_tools"):
        gitclone("https://github.com/Sxela/flow_tools")

    # %cd "{root_dir}/"
    # !git clone https://github.com/princeton-vl/RAFT
    # %cd "{root_dir}/RAFT"
    # if os.path.exists(f'{root_path}/RAFT/models') and force_download:
    #   try:
    #     print('forcing model redownload')
    #     shutil.rmtree(f'{root_path}/RAFT/models')
    #   except:
    #     print('error deleting existing RAFT model')

    # if (not (os.path.exists(f'{root_path}/RAFT/models/raft-things.pth'))) or force_download:

    #   !curl -L https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip -o "{root_dir}/RAFT/models.zip"

    #   with zipfile.ZipFile(f'{root_dir}/RAFT/models.zip', 'r') as zip_ref:
    #       zip_ref.extractall(f'{root_path}/RAFT/')


# %%
# @title Define color matching and brightness adjustment
os.chdir(f"{root_dir}/python-color-transfer")
from python_color_transfer.color_transfer import ColorTransfer, Regrain

os.chdir(root_path)

PT = ColorTransfer()
RG = Regrain()


def match_color(stylized_img, raw_img, opacity=1.0):
    img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype("uint8"), cv2.COLOR_RGB2BGR)
    img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype("uint8"), cv2.COLOR_RGB2BGR)
    # img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
    img_arr_col = PT.pdf_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
    img_arr_reg = RG.regrain(img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
    img_arr_reg = img_arr_reg * opacity + img_arr_in * (1 - opacity)
    img_arr_reg = cv2.cvtColor(img_arr_reg.round().astype("uint8"), cv2.COLOR_BGR2RGB)
    return img_arr_reg


from PIL import Image, ImageOps, ImageStat, ImageEnhance


def get_stats(image):
    stat = ImageStat.Stat(image)
    brightness = sum(stat.mean) / len(stat.mean)
    contrast = sum(stat.stddev) / len(stat.stddev)
    return brightness, contrast


# implemetation taken from https://github.com/lowfuel/progrockdiffusion


def adjust_brightness(image):

    brightness, contrast = get_stats(image)
    if brightness > high_brightness_threshold:
        print(" Brightness over threshold. Compensating!")
        filter = ImageEnhance.Brightness(image)
        image = filter.enhance(high_brightness_adjust_ratio)
        image = np.array(image)
        image = (
            np.where(
                image > high_brightness_threshold, image - high_brightness_adjust_fix_amount, image
            )
            .clip(0, 255)
            .round()
            .astype("uint8")
        )
        image = PIL.Image.fromarray(image)
    if brightness < low_brightness_threshold:
        print(" Brightness below threshold. Compensating!")
        filter = ImageEnhance.Brightness(image)
        image = filter.enhance(low_brightness_adjust_ratio)
        image = np.array(image)
        image = (
            np.where(
                image < low_brightness_threshold, image + low_brightness_adjust_fix_amount, image
            )
            .clip(0, 255)
            .round()
            .astype("uint8")
        )
        image = PIL.Image.fromarray(image)

    image = np.array(image)
    image = (
        np.where(image > max_brightness_threshold, image - high_brightness_adjust_fix_amount, image)
        .clip(0, 255)
        .round()
        .astype("uint8")
    )
    image = (
        np.where(image < min_brightness_threshold, image + low_brightness_adjust_fix_amount, image)
        .clip(0, 255)
        .round()
        .astype("uint8")
    )
    image = PIL.Image.fromarray(image)
    return image


# %%
# @title Define optical flow functions for Video input animation mode only
# if animation_mode == 'Video Input Legacy':
DEBUG = False

# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


from torch import Tensor

# if True:
if animation_mode == "Video Input":
    in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
    flo_folder = in_path + "_out_flo_fwd"
    # the main idea comes from neural-style-tf frame warping with optical flow maps
    # https://github.com/cysmith/neural-style-tf
    # path = f'{root_dir}/RAFT/core'
    # import sys
    # sys.path.append(f'{root_dir}/RAFT/core')
    # %cd {path}

    # from utils.utils import InputPadder

    class InputPadder:
        """Pads images such that dimensions are divisible by 8"""

        def __init__(self, dims, mode="sintel"):
            self.ht, self.wd = dims[-2:]
            pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
            pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
            if mode == "sintel":
                self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
            else:
                self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

        def pad(self, *inputs):
            return [F.pad(x, self._pad, mode="replicate") for x in inputs]

        def unpad(self, x):
            ht, wd = x.shape[-2:]
            c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
            return x[..., c[0] : c[1], c[2] : c[3]]

    # from raft import RAFT
    import numpy as np
    import argparse, PIL, cv2
    from PIL import Image
    from tqdm.notebook import tqdm
    from glob import glob
    import torch
    import scipy.ndimage

    args2 = argparse.Namespace()
    args2.small = False
    args2.mixed_precision = True

    TAG_CHAR = np.array([202021.25], np.float32)

    def writeFlow(filename, uv, v=None):
        """
        https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py
        Copyright 2017 NVIDIA CORPORATION

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.

        Write optical flow to file.

        If v is None, uv is assumed to contain both u and v channels,
        stacked in depth.
        Original code by Deqing Sun, adapted from Daniel Scharstein.
        """
        nBands = 2

        if v is None:
            assert uv.ndim == 3
            assert uv.shape[2] == 2
            u = uv[:, :, 0]
            v = uv[:, :, 1]
        else:
            u = uv

        assert u.shape == v.shape
        height, width = u.shape
        f = open(filename, "wb")
        # write the header
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)
        f.close()

    # def load_cc(path, blur=2):
    #   weights = np.load(path)
    #   if blur>0: weights = scipy.ndimage.gaussian_filter(weights, [blur, blur])
    #   weights = np.repeat(weights[...,None],3, axis=2)

    #   if DEBUG: print('weight min max mean std', weights.shape, weights.min(), weights.max(), weights.mean(), weights.std())
    #   return weights

    def load_cc(path, blur=2):
        multilayer_weights = np.array(PIL.Image.open(path)) / 255
        weights = np.ones_like(multilayer_weights[..., 0])
        weights *= multilayer_weights[..., 0].clip(1 - missed_consistency_weight, 1)
        weights *= multilayer_weights[..., 1].clip(1 - overshoot_consistency_weight, 1)
        weights *= multilayer_weights[..., 2].clip(1 - edges_consistency_weight, 1)

        if blur > 0:
            weights = scipy.ndimage.gaussian_filter(weights, [blur, blur])
        weights = np.repeat(weights[..., None], 3, axis=2)

        if DEBUG:
            print(
                "weight min max mean std",
                weights.shape,
                weights.min(),
                weights.max(),
                weights.mean(),
                weights.std(),
            )
        return weights

    def load_img(img, size):
        img = PIL.Image.open(img).convert("RGB").resize(size, warp_interp)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...].cuda()

    def get_flow(frame1, frame2, model, iters=20, half=True):
        # print(frame1.shape, frame2.shape)
        padder = InputPadder(frame1.shape)
        frame1, frame2 = padder.pad(frame1, frame2)
        if half:
            frame1, frame2 = frame1.half(), frame2.half()
        # print(frame1.shape, frame2.shape)
        _, flow12 = model(frame1, frame2)
        flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

        return flow12

    def warp_flow(img, flow, mul=1.0):
        h, w = flow.shape[:2]
        flow = flow.copy()
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        # print('flow stats', flow.max(), flow.min(), flow.mean())
        # print(flow)
        flow *= mul
        # print('flow stats mul', flow.max(), flow.min(), flow.mean())
        # res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        res = cv2.remap(img, flow, None, cv2.INTER_LANCZOS4)

        return res

    def makeEven(_x):
        return _x if (_x % 2 == 0) else _x + 1

    def fit(img, maxsize=512):
        maxdim = max(*img.size)
        if maxdim > maxsize:
            # if True:
            ratio = maxsize / maxdim
            x, y = img.size
            size = (makeEven(int(x * ratio)), makeEven(int(y * ratio)))
            img = img.resize(size, warp_interp)
        return img

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
    ):

        if isinstance(flo_path, str):
            flow21 = np.load(flo_path)
        else:
            flow21 = flo_path
        # print('loaded flow from ', flo_path, ' witch shape ', flow21.shape)
        pad = int(max(flow21.shape) * pad_pct)
        flow21 = np.pad(flow21, pad_width=((pad, pad), (pad, pad), (0, 0)), mode="constant")
        # print('frame1.size, frame2.size, padded flow21.shape')
        # print(frame1.size, frame2.size, flow21.shape)

        frame1pil = np.array(
            frame1.convert("RGB")
        )  # .resize((flow21.shape[1]-pad*2,flow21.shape[0]-pad*2),warp_interp))
        frame1pil = np.pad(frame1pil, pad_width=((pad, pad), (pad, pad), (0, 0)), mode=padding_mode)
        if video_mode:
            warp_mul = 1.0
        frame1_warped21 = warp_flow(frame1pil, flow21, warp_mul)
        frame1_warped21 = frame1_warped21[
            pad : frame1_warped21.shape[0] - pad, pad : frame1_warped21.shape[1] - pad, :
        ]

        frame2pil = np.array(
            frame2.convert("RGB").resize(
                (flow21.shape[1] - pad * 2, flow21.shape[0] - pad * 2), warp_interp
            )
        )
        # if not video_mode: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
        if weights_path:
            forward_weights = load_cc(weights_path, blur=consistency_blur)
            # print('forward_weights')
            # print(forward_weights.shape)
            if not video_mode:
                frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)

            forward_weights = forward_weights.clip(forward_clip, 1.0)
            if use_patchmatch_inpaiting > 0 and warp_mode == "use_image":
                if not is_colab:
                    print("Patchmatch only working on colab/linux")
                if not video_mode and is_colab:
                    print("patchmatching")
                    # print(np.array(blended_w).shape, forward_weights[...,0][...,None].shape )
                    patchmatch_mask = (forward_weights[..., 0][..., None] * -255.0 + 255).astype(
                        "uint8"
                    )
                    frame2pil = np.array(frame2pil) * (
                        1 - use_patchmatch_inpaiting
                    ) + use_patchmatch_inpaiting * np.array(
                        patch_match.inpaint(frame1_warped21, patchmatch_mask, patch_size=5)
                    )
                    # blended_w = PIL.Image.fromarray(blended_w)
            blended_w = frame2pil * (1 - blend) + blend * (
                frame1_warped21 * forward_weights + frame2pil * (1 - forward_weights)
            )
        else:
            if not video_mode:
                frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
            blended_w = frame2pil * (1 - blend) + frame1_warped21 * (blend)

        blended_w = PIL.Image.fromarray(blended_w.round().astype("uint8"))
        # if use_patchmatch_inpaiting and warp_mode == 'use_image':
        #           print('patchmatching')
        #           print(np.array(blended_w).shape, forward_weights[...,0][...,None].shape )
        #           patchmatch_mask = (forward_weights[...,0][...,None]*-255.+255).astype('uint8')
        #           blended_w = patch_match.inpaint(blended_w, patchmatch_mask, patch_size=5)
        #           blended_w = PIL.Image.fromarray(blended_w)
        if not video_mode:
            if enable_adjust_brightness:
                blended_w = adjust_brightness(blended_w)
        return blended_w

    def warp_lat(
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
    ):
        warp_downscaled = True
        flow21 = np.load(flo_path)
        pad = int(max(flow21.shape) * pad_pct)
        if warp_downscaled:
            flow21 = flow21.transpose(2, 0, 1)[None, ...]
            flow21 = torch.nn.functional.interpolate(
                torch.from_numpy(flow21), scale_factor=1 / 8, mode="bilinear"
            )
            flow21 = flow21.numpy()[0].transpose(1, 2, 0) / 8
            # flow21 = flow21[::8,::8,:]/8

        flow21 = np.pad(flow21, pad_width=((pad, pad), (pad, pad), (0, 0)), mode="constant")

        if not warp_downscaled:
            frame1 = torch.nn.functional.interpolate(frame1, scale_factor=8)
        frame1pil = frame1.cpu().numpy()[0].transpose(1, 2, 0)

        frame1pil = np.pad(frame1pil, pad_width=((pad, pad), (pad, pad), (0, 0)), mode=padding_mode)
        if video_mode:
            warp_mul = 1.0
        frame1_warped21 = warp_flow(frame1pil, flow21, warp_mul)
        frame1_warped21 = frame1_warped21[
            pad : frame1_warped21.shape[0] - pad, pad : frame1_warped21.shape[1] - pad, :
        ]
        if not warp_downscaled:
            frame2pil = frame2.convert("RGB").resize(
                (flow21.shape[1] - pad * 2, flow21.shape[0] - pad * 2), warp_interp
            )
        else:
            frame2pil = frame2.convert("RGB").resize(
                ((flow21.shape[1] - pad * 2) * 8, (flow21.shape[0] - pad * 2) * 8), warp_interp
            )
        frame2pil = np.array(frame2pil)
        frame2pil = (frame2pil / 255.0)[None, ...].transpose(0, 3, 1, 2)
        frame2pil = 2 * torch.from_numpy(frame2pil).cuda().half() - 1.0
        frame2pil = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame2pil)).half()
        if not warp_downscaled:
            frame2pil = torch.nn.functional.interpolate(frame2pil, scale_factor=8)
        frame2pil = frame2pil.cpu().numpy()[0].transpose(1, 2, 0)
        # if not video_mode: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
        if weights_path:
            forward_weights = load_cc(weights_path, blur=consistency_blur)
            print(forward_weights[..., :1].shape, "forward_weights.shape")
            forward_weights = np.repeat(forward_weights[..., :1], 4, axis=-1)
            # print('forward_weights')
            # print(forward_weights.shape)
            print(
                "frame2pil.shape, frame1_warped21.shape, flow21.shape",
                frame2pil.shape,
                frame1_warped21.shape,
                flow21.shape,
            )
            forward_weights = forward_weights.clip(forward_clip, 1.0)
            if warp_downscaled:
                forward_weights = forward_weights[::8, ::8, :]
                print(forward_weights.shape, "forward_weights.shape")
            blended_w = frame2pil * (1 - blend) + blend * (
                frame1_warped21 * forward_weights + frame2pil * (1 - forward_weights)
            )
        else:
            if not video_mode and not warp_mode == "use_latent":
                frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
            blended_w = frame2pil * (1 - blend) + frame1_warped21 * (blend)
        blended_w = blended_w.transpose(2, 0, 1)[None, ...]
        blended_w = torch.from_numpy(blended_w)
        if not warp_downscaled:
            # blended_w = blended_w[::8,::8,:]
            blended_w = torch.nn.functional.interpolate(
                blended_w, scale_factor=1 / 8, mode="bilinear"
            )

        return blended_w  # torch.nn.functional.interpolate(torch.from_numpy(blended_w), scale_factor = 1/8)

    in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
    flo_folder = in_path + "_out_flo_fwd"

    temp_flo = in_path + "_temp_flo"
    flo_fwd_folder = in_path + "_out_flo_fwd"
    flo_bck_folder = in_path + "_out_flo_bck"

    os.chdir(root_path)


# %%
# @title Generate optical flow and consistency maps
# @markdown Run once per init video\
# @markdown If you are getting **"AttributeError: module 'PIL.TiffTags' has no attribute 'IFD'"** error,\
# @markdown just click **"Runtime" - "Restart and Run All"** once per session.
# hack to get pillow to work w\o restarting
# if you're running locally, just restart this runtime, no need to edit PIL files.
if is_colab:
    filedata = None
    with open("/usr/local/lib/python3.7/dist-packages/PIL/TiffImagePlugin.py", "r") as file:
        filedata = file.read()
    filedata = filedata.replace('(TiffTags.IFD, "L", "long"),', '#(TiffTags.IFD, "L", "long"),')
    with open("/usr/local/lib/python3.7/dist-packages/PIL/TiffImagePlugin.py", "w") as file:
        file.write(filedata)
import gc

force_flow_generation = True  # @param {type:'boolean'}
# @markdown Use lower quality model (half-precision).\
# @markdown Uses half the vram, allows fitting 1500x1500+ frames into 16gigs, which the original full-precision RAFT can't do.
flow_lq = True  # @param {type:'boolean'}
# @markdown Save human-readable flow images along with motion vectors. Check /{your output dir}/videoFrames/out_flo_fwd folder.
flow_save_img_preview = True  # @param {type:'boolean'}
in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
flo_folder = in_path + "_out_flo_fwd"
# @markdown reverse_cc_order - on - default value (like in older notebooks). off - reverses consistency computation
reverse_cc_order = True  # @param {type:'boolean'}
if not flow_warp:
    print("flow_wapr not set, skipping")

if (animation_mode == "Video Input") and (flow_warp):
    flows = glob(flo_folder + "/*.*")
    if (len(flows) > 0) and not force_flow_generation:
        print(
            f"Skipping flow generation:\nFound {len(flows)} existing flow files in current working"
            f" folder: {flo_folder}.\nIf you wish to generate new flow files, check"
            " force_flow_generation and run this cell again."
        )

    if (len(flows) == 0) or force_flow_generation:

        frames = sorted(glob(in_path + "/*.*"))
        if len(frames) < 2:
            print(
                f"WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your"
                " video input.\nPlease check your video path."
            )
        if len(frames) >= 2:
            if flow_lq:
                raft_model = torch.jit.load(f"{root_dir}/WarpFusion/raft/raft_half.jit").eval()
            # raft_model = torch.nn.DataParallel(RAFT(args2))
            else:
                raft_model = torch.jit.load(f"{root_dir}/WarpFusion/raft/raft_fp32.jit").eval()
            # raft_model.load_state_dict(torch.load(f'{root_path}/RAFT/models/raft-things.pth'))
            # raft_model = raft_model.module.cuda().eval()

            for f in pathlib.Path(f"{flo_fwd_folder}").glob("*.*"):
                f.unlink()

            temp_flo = in_path + "_temp_flo"
            flo_fwd_folder = in_path + "_out_flo_fwd"

            pathlib.Path(flo_fwd_folder).mkdir(parents=True, exist_ok=True)
            pathlib.Path(temp_flo).mkdir(parents=True, exist_ok=True)
            cc_path = f"{root_dir}/flow_tools/check_consistency.py"
            with torch.no_grad():
                for frame1, frame2 in tqdm(zip(frames[:-1], frames[1:]), total=len(frames) - 1):
                    frame1 = frame1.replace("\\", "/")
                    out_flow21_fn = f"{flo_fwd_folder}/{frame1.split('/')[-1]}"

                    frame1 = load_img(frame1, width_height)
                    frame2 = load_img(frame2, width_height)

                    flow21 = get_flow(frame2, frame1, raft_model, half=flow_lq)
                    if flow_save_img_preview:
                        PIL.Image.fromarray(flow_to_image(flow21)).save(
                            out_flow21_fn + ".jpg", quality=90
                        )
                    np.save(out_flow21_fn, flow21)

                    if check_consistency:
                        flow12 = get_flow(frame1, frame2, raft_model, half=flow_lq)
                        if flow_save_img_preview:
                            PIL.Image.fromarray(flow_to_image(flow12)).save(
                                out_flow21_fn + "_12" + ".jpg", quality=90
                            )
                        np.save(out_flow21_fn + "_12", flow12)
                        gc.collect()

            del raft_model
            gc.collect()
            if check_consistency:
                fwd = f"{flo_fwd_folder}/*jpg.npy"
                bwd = f"{flo_fwd_folder}/*jpg_12.npy"

                if reverse_cc_order:
                    # old version, may be incorrect
                    print("Doing bwd->fwd cc check")

                    pip_res = subprocess.run(
                        [
                            "python",
                            cc_path,
                            "--flow_fwd",
                            fwd,
                            "--flow_bwd",
                            bwd,
                            "--output",
                            flo_fwd_folder,
                            "--image-output",
                            "--output_postfix='-21_cc'",
                            "--blur=0.",
                            "--save_separate_channels",
                            "--skip_numpy_output",
                        ],
                        stdout=subprocess.PIPE,
                    ).stdout.decode("utf-8")
                    print(pip_res)
                else:
                    print("Doing fwd->bwd cc check")
                    pip_res = subprocess.run(
                        [
                            "python",
                            cc_path,
                            "--flow_fwd",
                            bwd,
                            "--flow_bwd",
                            fwd,
                            "--output",
                            flo_fwd_folder,
                            "--image-output",
                            "--output_postfix='-21_cc'",
                            "--blur=0.",
                            "--save_separate_channels",
                            "--skip_numpy_output",
                        ],
                        stdout=subprocess.PIPE,
                    ).stdout.decode("utf-8")
                    print(pip_res)
                for f in pathlib.Path(flo_fwd_folder).glob("*jpg_12.npy"):
                    f.unlink()


# %%
# @title Consistency map mixing
# @markdown You can mix consistency map layers separately\
# @markdown missed_consistency_weight - masks pixels that have missed their expected position in the next frame \
# @markdown overshoot_consistency_weight - masks pixels warped from outside the frame\
# @markdown edges_consistency_weight - masks moving objects' edges\
# @markdown The default values to simulate previous versions' behavior are 1,1,1

missed_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
overshoot_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
edges_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}

# %% [markdown]
# # Load up a stable.
#
# Don't forget to place your checkpoint at /content/ and change the path accordingly.
#
#
# You need to log on to https://huggingface.co and
#
# get checkpoints here -
# https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
#
# https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
# or
# https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt
#
# You can pick 1.2 or 1.3 as well, just be sure to grab the "original" flavor.
#

# %%
# @markdown specify path to your Stable Diffusion checkpoint (the "original" flavor)
# @title define SD + K functions, load model

import argparse
import math, os, time

os.chdir(f"{root_dir}/src/taming-transformers")
import taming

os.chdir(f"{root_dir}")

import accelerate
import torch
import torch.nn as nn
from tqdm.notebook import trange, tqdm

sys.path.append("./k-diffusion")
import k_diffusion as K
from pytorch_lightning import seed_everything

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from torch import autocast
import numpy as np

from einops import rearrange
from torchvision.utils import make_grid
import PIL.Image
from torchvision import transforms


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


import gc


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    del pl_sd
    gc.collect()
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda().half()
    model.eval()
    return model


import clip
from kornia import augmentation as KA
from torch.nn import functional as F
from resize_right import resize


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


from einops import rearrange, repeat


def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            # with torch.no_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            # print(denoised.requires_grad)
            # with torch.enable_grad():
            # denoised = denoised.detach().requires_grad_()
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            # print(cond_grad.requires_grad)
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised

    return model_fn


def make_static_thresh_model_fn(model, value=1.0):
    def model_fn(x, sigma, **kwargs):
        return model(x, sigma, **kwargs).clamp(-value, value)

    return model_fn


def get_image_embed(x):
    if x.shape[2:4] != clip_size:
        x = resize(x, out_shape=clip_size, pad_mode="reflect")
    # print('clip', x.shape)
    # x = clip_normalize(x).cuda()
    x = clip_model.encode_image(x).float()
    return F.normalize(x)


def load_img_sd(path, size):
    # print(type(path))
    # print('load_sd',path)

    image = Image.open(path).convert("RGB")
    # print(f'loaded img with size {image.size}')
    image = image.resize(size, resample=PIL.Image.LANCZOS)
    # w, h = image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

    # image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    if VERBOSE:
        print(f"resized to {image.size}")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


# import lpips
# lpips_model = lpips.LPIPS(net='vgg').to(device)

dynamic_thresh = 2.0
device = "cuda"
config_path = f"{root_dir}/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
model_path = "/content/drive/MyDrive/models/sd-v1-4.ckpt"  # @param {'type':'string'}
import pickle

if model_path.endswith(".pkl"):
    with open(model_path, "rb") as f:
        sd_model = pickle.load(f).cuda().eval()
else:
    config = OmegaConf.load(config_path)
    sd_model = load_model_from_config(config, model_path)

model_wrap = K.external.CompVisDenoiser(sd_model)
sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
model_wrap_cfg = CFGDenoiser(model_wrap)

# import pickle
# with open('sd_v1_4.pkl', 'wb') as f:
#   pickle.dump(sd_model, f)

# @markdown If you're having crashes (CPU out of memory errors) while running this cell on standard colab env, consider saving the model as pickle.\
# @markdown You can save the pickled model on your google drive and use it instead of the usual stable diffusion model.\
# @markdown To do that, run the notebook with a high-ram env, run all cells before and including this cell as well, and save pickle in the next cell. Then you can switch to a low-ram env and load the pickled model.

# %%


# %%
# @title Save loaded model
# @markdown For this cell to work you need to load model in the previous cell.\
# @markdown Saves an already loaded model as an object file, that weights less, loads faster, and requires less CPU RAM.\
# @markdown After saving model as pickle, you can then load it as your usual stable diffusion model in thecell above.\
# @markdown The model will be saved under the same name with .pkl extenstion.
save_model_pickle = False  # @param {'type':'boolean'}
save_folder = "/content/drive/MyDrive/models/"  # @param {'type':'string'}
if save_folder != "" and save_model_pickle:
    os.makedirs(save_folder, exist_ok=True)
    out_path = save_folder + model_path.replace("\\", "/").split("/")[-1].split(".")[0] + ".pkl"
    with open(out_path, "wb") as f:
        pickle.dump(sd_model, f)
    print("Model successfully saved as: ", out_path)

# %% [markdown]
# # CLIP guidance

# %%
# @title CLIP guidance settings
# @markdown You can use clip guidance to further push style towards your text input.\
# @markdown Please note that enabling it (by using clip_guidance_scale>0) will greatly increase render times and VRAM usage.\
# @markdown For now it does 1 sample of the whole image per step (similar to 1 outer_cut in discodiffusion).

# clip_type, clip_pretrain = 'ViT-B-32-quickgelu', 'laion400m_e32'
# clip_type, clip_pretrain ='ViT-L-14', 'laion2b_s32b_b82k'
clip_type = "ViT-H-14"  # @param ['ViT-L-14','ViT-B-32-quickgelu', 'ViT-H-14']
if clip_type == "ViT-H-14":
    clip_pretrain = "laion2b_s32b_b79k"
if clip_type == "ViT-L-14":
    clip_pretrain = "laion2b_s32b_b82k"
if clip_type == "ViT-B-32-quickgelu":
    clip_pretrain = "laion400m_e32"

clip_guidance_scale = 0  # @param {'type':"number"}
if clip_guidance_scale > 0:
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        clip_type, pretrained=clip_pretrain
    )
    _ = clip_model.half().cuda().eval()
    clip_size = clip_model.visual.image_size
    for param in clip_model.parameters():
        param.requires_grad = False
else:
    try:
        del clip_model
        gc.collect()
    except:
        pass

# %% [markdown]
# # Extra Settings
#  Seed, clamp grad

# %%
# markdown ####**Saving:**

intermediate_saves = None
intermediates_in_subfolder = True
# markdown Intermediate steps will save a copy at your specified intervals. You can either format it as a single integer or a list of specific steps

# markdown A value of `2` will save a copy at 33% and 66%. 0 will save none.

# markdown A value of `[5, 9, 34, 45]` will save at steps 5, 9, 34, and 45. (Make sure to include the brackets)


if type(intermediate_saves) is not list:
    if intermediate_saves:
        steps_per_checkpoint = math.floor((steps - skip_steps - 1) // (intermediate_saves + 1))
        steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
        print(f"Will save every {steps_per_checkpoint} steps")
    else:
        steps_per_checkpoint = steps + 10
else:
    steps_per_checkpoint = None

if intermediate_saves and intermediates_in_subfolder is True:
    partialFolder = f"{batchFolder}/partials"
    createPath(partialFolder)

    # @markdown ---

# @markdown ####**Advanced Settings:**
# @markdown *There are a few extra advanced settings available if you double click this cell.*


perlin_init = False
perlin_mode = "mixed"
set_seed = "random_seed"  # @param{type: 'string'}


# @markdown *Clamp grad is used with any of the init_scales or sat_scale above 0*\
# @markdown Clamp grad limits the amount various criterions, controlled by *_scale parameters, are pushing the image towards the desired result.\
# @markdown For example, high scale values may cause artifacts, and clamp_grad removes this effect.
# @markdown 0.7 is a good clamp_max value.
eta = 0.55
clamp_grad = True  # @param{type: 'boolean'}
clamp_max = 0.7  # @param{type: 'number'}


# EXTRA ADVANCED SETTINGS:
randomize_class = True
clip_denoised = False
fuzzy_prompt = False
rand_mag = 0.05


# markdown ---

# markdown ####**Cutn Scheduling:**
# markdown Format: `[40]*400+[20]*600` = 40 cuts for the first 400 /1000 steps, then 20 for the last 600/1000

# markdown cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn.

cut_overview = "[16]*400+[12]*200+[8]*200+[4]*200"
cut_innercut = "[0]*400+[4]*200+[8]*200+[12]*200"
cut_ic_pow = 1
cut_icgray_p = "[0.2]*100+[0]*900"


# %% [markdown]
# # Prompts
# `animation_mode: None` will only use the first set. `animation_mode: 2D / Video` will run through them per the set frames and hold on the last one.

# %%
from external_prompts import text_prompts, negative_prompts, image_prompts


# %% [markdown]
# # Warp Turbo Smooth Settings

# %% [markdown]
# turbo_frame_skips_steps - allows to set different frames_skip_steps for turbo frames. None means turbo frames are warped only without diffusion
#
# soften_consistency_mask - clip the lower values of consistency mask to this value. Raw video frames will leak stronger with lower values.
#
# soften_consistency_mask_for_turbo_frames - same, but for turbo frames
#
#
#
#

# %%
# @title ##Warp Turbo Smooth Settings
# @markdown Skip steps for turbo frames. Select 100% to skip diffusion rendering for turbo frames completely.
turbo_frame_skips_steps = (  # @param ['70%','75%','80%','85%', '90%', '95%', '100% (don`t diffuse turbo frames, fastest)']
    "100% (don`t diffuse turbo frames, fastest)"
)

if turbo_frame_skips_steps == "100% (don`t diffuse turbo frames, fastest)":
    turbo_frame_skips_steps = None
else:
    turbo_frame_skips_steps = int(turbo_frame_skips_steps.split("%")[0]) / 100
# None - disable and use default skip steps

# @markdown ###Consistency mask postprocessing
# @markdown ####Soften consistency mask
# @markdown Lower values mean less stylized frames and more raw video input in areas with fast movement, but fewer trails add ghosting.\
# @markdown Gives glitchy datamoshing look.\
# @markdown Higher values keep stylized frames, but add trails and ghosting.

soften_consistency_mask = 0  # @param {type:"slider", min:0, max:1, step:0.1}
forward_weights_clip = soften_consistency_mask
# 0 behaves like consistency on, 1 - off, in between - blends
soften_consistency_mask_for_turbo_frames = 0  # @param {type:"slider", min:0, max:1, step:0.1}
forward_weights_clip_turbo_step = soften_consistency_mask_for_turbo_frames
# None - disable and use forward_weights_clip for turbo frames, 0 behaves like consistency on, 1 - off, in between - blends
# @markdown ####Blur consistency mask.
# @markdown Softens transition between raw video init and stylized frames in occluded areas.
consistency_blur = 1  # @param


# disable_cc_for_turbo_frames = False #@param {"type":"boolean"}
# disable consistency for turbo frames, the same as forward_weights_clip_turbo_step = 1, but a bit faster

# @markdown ###Frame padding
# @markdown Increase padding if you have a shaky\moving camera footage and are getting black borders.

padding_ratio = 0.2  # @param {type:"slider", min:0, max:1, step:0.1}
# relative to image size, in range 0-1
padding_mode = "reflect"  # @param ['reflect','edge','wrap']


# safeguard the params
if turbo_frame_skips_steps is not None:
    turbo_frame_skips_steps = min(max(0, turbo_frame_skips_steps), 1)
forward_weights_clip = min(max(0, forward_weights_clip), 1)
if forward_weights_clip_turbo_step is not None:
    forward_weights_clip_turbo_step = min(max(0, forward_weights_clip_turbo_step), 1)
padding_ratio = min(max(0, padding_ratio), 1)
##@markdown ###Inpainting
##@markdown Inpaint occluded areas on top of raw frames. 0 - 0% inpainting opacity (no inpainting), 1 - 100% inpainting opacity. Other values blend between raw and inpainted frames.

inpaint_blend = 0
##@param {type:"slider", min:0,max:1,value:1,step:0.1}

# @markdown ###Color matching
# @markdown Match color of inconsistent areas to unoccluded ones, after inconsistent areas were replaced with raw init video or inpainted\
# @markdown 0 - off, other values control effect opacity

match_color_strength = 0  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.1'}

disable_cc_for_turbo_frames = False

# %% [markdown]
# # Automatic Brightness Adjustment

# %%
# @markdown ###Automatic Brightness Adjustment
# @markdown Automatically adjust image brightness when its mean value reaches a certain threshold\
# @markdown Ratio means the vaue by which pixel values are multiplied when the thresjold is reached\
# @markdown Fix amount is being directly added to\subtracted from pixel values to prevent oversaturation due to multiplications\
# @markdown Fix amount is also being applied to border values defined by min\max threshold, like 1 and 254 to keep the image from having burnt out\pitch black areas while still being within set high\low thresholds


# @markdown The idea comes from https://github.com/lowfuel/progrockdiffusion

enable_adjust_brightness = False  # @param {'type':'boolean'}
high_brightness_threshold = 180  # @param {'type':'number'}
high_brightness_adjust_ratio = 0.97  # @param {'type':'number'}
high_brightness_adjust_fix_amount = 2  # @param {'type':'number'}
max_brightness_threshold = 254  # @param {'type':'number'}
low_brightness_threshold = 40  # @param {'type':'number'}
low_brightness_adjust_ratio = 1.03  # @param {'type':'number'}
low_brightness_adjust_fix_amount = 2  # @param {'type':'number'}
min_brightness_threshold = 1  # @param {'type':'number'}


# %% [markdown]
# # Video masking (render-time)

# %%
# @title Video mask settings
# @markdown Check to enable background masking during render. Not recommended, better use masking when creating the output video for more control and faster testing.
use_background_mask = False  # @param {'type':'boolean'}
# @markdown Check to invert the mask.
invert_mask = False  # @param {'type':'boolean'}
# @markdown Apply mask right before feeding init image to the model. Unchecking will only mask current raw init frame.
apply_mask_after_warp = True  # @param {'type':'boolean'}
# @markdown Choose background source to paste masked stylized image onto: image, color, init video.
background = "init_video"  # @param ['image', 'color', 'init_video']
# @markdown Specify the init image path or color depending on your background source choice.
background_source = "red"  # @param {'type':'string'}


# %% [markdown]
# # stable-settings

# %%
warp_mode = "use_image"  # @param ['use_latent', 'use_image']
warp_towards_init = "off"  # @param ['stylized', 'off']

if warp_towards_init != "off":
    if flow_lq:
        raft_model = torch.jit.load(f"{root_dir}/WarpFusion/raft/raft_half.jit").eval()
    # raft_model = torch.nn.DataParallel(RAFT(args2))
    else:
        raft_model = torch.jit.load(f"{root_dir}/WarpFusion/raft/raft_fp32.jit").eval()

# %%
# DD-style losses, renders 2 times slower (!) and more memory intensive :D

latent_scale_schedule = [
    0,
    0,
]  # controls coherency with previous frame in latent space. 0 is a good starting value. 1+ render slower, but may improve image coherency. 100 is a good value if you decide to turn it on.
init_scale_schedule = [
    0,
    0,
]  # controls coherency with previous frame in pixel space. 0 - off, 1000 - a good starting value if you decide to turn it on.
init_grad = True  # True - compare result to real frame, False - to stylized frame

grad_denoised = True  # fastest, on by default, calc grad towards denoised x instead of input x

# %%
# !sched
steps_schedule = {
    0: steps,
}  # schedules total steps. useful with low strength, when you end up with only 10 steps at 0.2 strength x50 steps. Increasing max steps for low strength gives model more time to get to your text prompt
style_strength_schedule = [
    0.6,
]  # [0.5]+[0.2]*149+[0.3]*3+[0.2] #use this instead of skip steps. It means how many steps we should do. 0.8 = we diffuse for 80% steps, so we skip 20%. So for skip steps 70% use 0.3
flow_blend_schedule = [
    0.8
]  # for example [0.1]*3+[0.999]*18+[0.3] will fade-in for 3 frames, keep style for 18 frames, and fade-out for the rest
cfg_scale_schedule = [7.5]  # text2image strength, 7.5 is a good default
blend_json_schedules = (
    True  # True - interpolate values between keyframes. False - use latest keyframe
)

dynamic_thresh = 50

fixed_code = False  # you can use this with fast moving videos, but be careful with still images
blend_code = 0.2  # high values make the output collapse
normalize_code = True

warp_strength = 1  # leave 1 for no change. 1.01 is already a strong value.
flow_override_map = (
    []
)  # [*range(1,15)]+[16]*10+[*range(17+10,17+10+20)]+[18+10+20]*15+[*range(19+10+20+15,9999)] #map flow to frames. set to [] to disable.  [1]*10+[*range(10,9999)] repeats 1st frame flow 10 times, then continues as usual

# %%
# danger (and barely used) zone
use_predicted_noise = False
user_comment = "testing cc layers"

mask_result = False  # imitates inpainting by leaving only inconsistent areas to be diffused

use_karras_noise = False  # Should work better with current sample, needs more testing.
end_karras_ramp_early = False

warp_interp = PIL.Image.LANCZOS
VERBOSE = True

use_patchmatch_inpaiting = 0

blend_latent_to_init = 0


# %% [markdown]
# # Frame correction (latent & color matching)

# %%
# @title Frame correction
# @markdown Match frame pixels or latent to other frames to preven oversaturation and feedback loop artifacts
# @markdown ###Latent matching
# @markdown Match the range of latent vector towards the 1st frame or a user defined range. Doesn't restrict colors, but may limit contrast.
normalize_latent = (  # @param ['off', 'first_latent', 'user_defined', 'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset']
    "init_frame_offset"
)
# @markdown in offset mode, specifies the offset back from current frame, and 0 means current frame. In non-offset mode specifies the fixed frame number. 0 means the 1st frame.

normalize_latent_offset = 0  # @param {'type':'number'}
# @markdown User defined stats to normalize the latent towards
latent_fixed_mean = 0.0  # @param {'type':'raw'}
latent_fixed_std = 0.9  # @param {'type':'raw'}
# @markdown Match latent on per-channel basis
latent_norm_4d = True  # @param {'type':'boolean'}
# @markdown ###Color matching
# @markdown Color match frame towards stylized or raw init frame. Helps prevent images going deep purple. As a drawback, may lock colors to the selected fixed frame. Select stylized_frame with colormatch_offset = 0 to reproduce previous notebooks.
colormatch_frame = (  # @param ['off', 'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset']
    "off"
)
# @markdown Color match strength. 1 mimics legacy behavior
color_match_frame_str = 1  # @param {'type':'number'}
# @markdown in offset mode, specifies the offset back from current frame, and 0 means current frame. In non-offset mode specifies the fixed frame number. 0 means the 1st frame.
colormatch_offset = 0  # @param {'type':'number'}
colormatch_method = "LAB"  # @param ['LAB', 'PDF', 'mean']
colormatch_method_fn = PT.lab_transfer
if colormatch_method == "LAB":
    colormatch_method_fn = PT.pdf_transfer
if colormatch_method == "mean":
    colormatch_method_fn = PT.mean_std_transfer
# @markdown Match source frame's texture
colormatch_regrain = False  # @param {'type':'boolean'}


# %% [markdown]
# # Content-aware scheduling

# %%
# @title Content-aware scheduing
# @markdown Allows automated settings scheduling based on video frames difference. If a scene changes, it will be detected and reflected in the schedule.\
# @markdown rmse function is faster than lpips, but less precise.\
# @markdown After the analysis is done, check the graph and pick a threshold that works best for your video. 0.5 is a good one for lpips, 1.2 is a good one for rmse. Don't forget to adjust the templates with new threshold in the cell below.


def load_img_lpips(path, size=(512, 512)):
    image = Image.open(path).convert("RGB")
    image = image.resize(size, resample=Image.LANCZOS)
    # print(f'resized to {image.size}')
    image = np.array(image).astype(np.float32) / 127
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = normalize(image)
    return image.cuda()


diff = None
analyze_video = True  # @param {'type':'boolean'}

diff_function = "lpips"  # @param ['rmse','lpips','rmse+lpips']


def rmse(x, y):
    return torch.sqrt(torch.mean((x - y) ** 2))


def joint_loss(x, y):
    return rmse(x, y) * lpips_model(x, y)


diff_func = rmse
if diff_function == "lpips":
    diff_func = lpips_model
if diff_function == "rmse+lpips":
    diff_func = joint_loss

if analyze_video:
    diff = [0]
    frames = sorted(glob(f"{videoFramesFolder}/*.jpg"))
    from tqdm.notebook import trange

    for i in trange(1, len(frames)):
        with torch.no_grad():
            diff.append(
                diff_func(load_img_lpips(frames[i - 1]), load_img_lpips(frames[i]))
                .sum()
                .mean()
                .detach()
                .cpu()
                .numpy()
            )

    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = [12.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    y = diff
    plt.title(f"{diff_function} frame difference")
    plt.plot(y, color="red")
    calc_thresh = np.percentile(np.array(diff), 97)
    plt.axhline(y=calc_thresh, color="b", linestyle="dashed")

    plt.show()
    print(f"suggested threshold: {calc_thresh.round(2)}")

# %%
# @title Plot threshold vs frame difference
# @markdown The suggested threshold may be incorrect, so you can plot your value and see if it covers the peaks.
if diff is not None:
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = [12.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    y = diff
    plt.title(f"{diff_function} frame difference")
    plt.plot(y, color="red")
    calc_thresh = np.percentile(np.array(diff), 97)
    plt.axhline(y=calc_thresh, color="b", linestyle="dashed")
    user_threshold = 0.55  # @param {'type':'raw'}
    plt.axhline(y=user_threshold, color="r")

    plt.show()
    peaks = []
    for i, d in enumerate(diff):
        if d > user_threshold:
            peaks.append(i)
    print(f"Peaks at frames: {peaks} for user_threshold of {user_threshold}")
else:
    print("Please analyze frames in the previous cell  to plot graph")

# %%
# , threshold
# @title Create schedules from frame difference
def adjust_schedule(diff, normal_val, new_scene_val, thresh, falloff_frames, sched=None):
    diff_array = np.array(diff)

    diff_new = np.zeros_like(diff_array)
    diff_new = diff_new + normal_val

    for i in range(len(diff_new)):
        el = diff_array[i]
        if sched is not None:
            diff_new[i] = get_scheduled_arg(i, sched)
        if el > thresh or i == 0:
            diff_new[i] = new_scene_val
            if falloff_frames > 0:
                for j in range(falloff_frames):
                    if i + j > len(diff_new) - 1:
                        break
                    # print(j,(falloff_frames-j)/falloff_frames, j/falloff_frames )
                    falloff_val = normal_val
                    if sched is not None:
                        falloff_val = get_scheduled_arg(i + falloff_frames, sched)
                    diff_new[i + j] = (
                        new_scene_val * (falloff_frames - j) / falloff_frames
                        + falloff_val * j / falloff_frames
                    )
    return diff_new


def check_and_adjust_sched(sched, template, diff, respect_sched=True):
    if template is None or template == "" or template == []:
        return sched
    normal_val, new_scene_val, thresh, falloff_frames = template
    sched_source = None
    if respect_sched:
        sched_source = sched
    return list(
        adjust_schedule(diff, normal_val, new_scene_val, thresh, falloff_frames, sched_source)
        .astype("float")
        .round(3)
    )


# @markdown fill in templates for schedules you'd like to create from frames' difference\
# @markdown leave blank to use schedules from previous cells\
# @markdown format: **[normal value, high difference value, difference threshold, falloff from high to normal (number of frames)]**\
# @markdown For example, setting flow blend template to [0.999, 0.3, 0.5, 5] will use 0.999 everywhere unless a scene has changed (frame difference >0.5) and then set flow_blend for this frame to 0.3 and gradually fade to 0.999 in 5 frames

latent_scale_template = ""  # @param {'type':'raw'}
init_scale_template = ""  # @param {'type':'raw'}
steps_template = ""  # @param {'type':'raw'}
style_strength_template = [0.2, 0.5, 0.55, 3]  # @param {'type':'raw'}
flow_blend_template = [0.75, 0.9, 0.55, 3]  # @param {'type':'raw'}
cfg_scale_template = [10, 15, 0.55, 5]  # @param {'type':'raw'}

# @markdown Turning this off will disable templates and will use schedules set in previous cell
make_schedules = False  # @param {'type':'boolean'}
# @markdown Turning this on will respect previously set schedules and only alter the frames with peak difference
respect_sched = True  # @param {'type':'boolean'}
if make_schedules:
    if diff is None:
        sys.exit(
            f"\nERROR!\n\nframes were not anayzed. Please enable analyze_video in the previous"
            f" cell, run it, and then run this cell again\n"
        )

    latent_scale_schedule = check_and_adjust_sched(
        latent_scale_schedule, latent_scale_template, diff, respect_sched
    )
    init_scale_schedule = check_and_adjust_sched(
        init_scale_schedule, init_scale_template, diff, respect_sched
    )
    steps_schedule = int(
        check_and_adjust_sched(steps_schedule, steps_template, diff, respect_sched)
    )
    style_strength_schedule = check_and_adjust_sched(
        style_strength_schedule, style_strength_template, diff, respect_sched
    )
    flow_blend_schedule = check_and_adjust_sched(
        flow_blend_schedule, flow_blend_template, diff, respect_sched
    )
    cfg_scale_schedule = check_and_adjust_sched(
        cfg_scale_schedule, cfg_scale_template, diff, respect_sched
    )

# shift+1 required


# %% [markdown]
# # 4. Diffuse!
# if you are having OOM or PIL error here click "restart and run all" once.

# %%
# @title Do the Run!
# @markdown Preview max size
display_size = 512  # @param


def printf(*msg, file=f"{root_dir}/log.txt"):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(file, "a") as f:
        msg = f'{dt_string}> {" ".join([str(o) for o in (msg)])}'
        print(msg, file=f)


printf("--------Beginning new run------")
##@markdown `n_batches` ignored with animation modes.
display_rate = 9999999
##@param{type: 'number'}
n_batches = 1
##@param{type: 'number'}
start_code = None
first_latent = None
first_latent_source = "not set"
os.chdir(root_dir)
n_mean_avg = None
n_std_avg = None
n_smooth = 0.5
# Update Model Settings
timestep_respacing = f"ddim{steps}"
diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
model_config.update(
    {
        "timestep_respacing": timestep_respacing,
        "diffusion_steps": diffusion_steps,
    }
)

batch_size = 1


def move_files(start_num, end_num, old_folder, new_folder):
    for i in range(start_num, end_num):
        old_file = old_folder + f"/{batch_name}({batchNum})_{i:06}.png"
        new_file = new_folder + f"/{batch_name}({batchNum})_{i:06}.png"
        os.rename(old_file, new_file)


# @markdown ---


resume_run = False  # @param{type: 'boolean'}
run_to_resume = "latest"  # @param{type: 'string'}
resume_from_frame = "latest"  # @param{type: 'string'}
retain_overwritten_frames = False  # @param{type: 'boolean'}
if retain_overwritten_frames is True:
    retainFolder = f"{batchFolder}/retained"
    createPath(retainFolder)



if animation_mode == "Video Input":
    frames = sorted(glob(in_path + "/*.*"))
    if len(frames) == 0:
        sys.exit(
            "ERROR: 0 frames found.\nPlease check your video input path and rerun the video"
            " settings cell."
        )
    flows = glob(flo_folder + "/*.*")
    if (len(flows) == 0) and flow_warp:
        sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")


if resume_run:
    if run_to_resume == "latest":
        try:
            batchNum
        except:
            batchNum = len(glob(f"{batchFolder}/{batch_name}(*)_{settingsFile}")) - 1
    else:
        batchNum = int(run_to_resume)
    if resume_from_frame == "latest":
        start_frame = len(glob(batchFolder + f"/{batch_name}({batchNum})_*.png"))
        if (
            animation_mode != "Video Input"
            and turbo_mode == True
            and start_frame > turbo_preroll
            and start_frame % int(turbo_steps) != 0
        ):
            start_frame = start_frame - (start_frame % int(turbo_steps))
    else:
        start_frame = int(resume_from_frame) + 1
        if (
            animation_mode != "Video Input"
            and turbo_mode == True
            and start_frame > turbo_preroll
            and start_frame % int(turbo_steps) != 0
        ):
            start_frame = start_frame - (start_frame % int(turbo_steps))
        if retain_overwritten_frames is True:
            existing_frames = len(glob(batchFolder + f"/{batch_name}({batchNum})_*.png"))
            frames_to_save = existing_frames - start_frame
            print(f"Moving {frames_to_save} frames to the Retained folder")
            move_files(start_frame, existing_frames, batchFolder, retainFolder)
else:
    start_frame = 0
    batchNum = len(glob(batchFolder + "/*.txt"))
    while (
        os.path.isfile(f"{batchFolder}/{batch_name}({batchNum})_{settingsFile}") is True
        or os.path.isfile(f"{batchFolder}/{batch_name}-{batchNum}_{settingsFile}") is True
    ):
        batchNum += 1

print(f"Starting Run: {batch_name}({batchNum}) at frame {start_frame}")

if set_seed == "random_seed":
    random.seed()
    seed = random.randint(0, 2**32)
    # print(f'Using seed: {seed}')
else:
    seed = int(set_seed)

args = {
    "batchNum": batchNum,
    "prompts_series": split_prompts(text_prompts) if text_prompts else None,
    "neg_prompts_series": split_prompts(negative_prompts) if negative_prompts else None,
    "image_prompts_series": split_prompts(image_prompts) if image_prompts else None,
    "seed": seed,
    "display_rate": display_rate,
    "n_batches": n_batches if animation_mode == "None" else 1,
    "batch_size": batch_size,
    "batch_name": batch_name,
    "steps": steps,
    "diffusion_sampling_mode": diffusion_sampling_mode,
    "width_height": width_height,
    "clip_guidance_scale": clip_guidance_scale,
    "tv_scale": tv_scale,
    "range_scale": range_scale,
    "sat_scale": sat_scale,
    "cutn_batches": cutn_batches,
    "init_image": init_image,
    "init_scale": init_scale,
    "skip_steps": skip_steps,
    "side_x": side_x,
    "side_y": side_y,
    "timestep_respacing": timestep_respacing,
    "diffusion_steps": diffusion_steps,
    "animation_mode": animation_mode,
    "video_init_path": video_init_path,
    "extract_nth_frame": extract_nth_frame,
    "video_init_seed_continuity": video_init_seed_continuity,
    "key_frames": key_frames,
    "max_frames": max_frames if animation_mode != "None" else 1,
    "interp_spline": interp_spline,
    "start_frame": start_frame,
    "angle": angle,
    "zoom": zoom,
    "translation_x": translation_x,
    "translation_y": translation_y,
    "translation_z": translation_z,
    "rotation_3d_x": rotation_3d_x,
    "rotation_3d_y": rotation_3d_y,
    "rotation_3d_z": rotation_3d_z,
    "midas_depth_model": midas_depth_model,
    "midas_weight": midas_weight,
    "near_plane": near_plane,
    "far_plane": far_plane,
    "fov": fov,
    "padding_mode": padding_mode,
    # 'sampling_mode': sampling_mode,
    "angle_series": angle_series,
    "zoom_series": zoom_series,
    "translation_x_series": translation_x_series,
    "translation_y_series": translation_y_series,
    "translation_z_series": translation_z_series,
    "rotation_3d_x_series": rotation_3d_x_series,
    "rotation_3d_y_series": rotation_3d_y_series,
    "rotation_3d_z_series": rotation_3d_z_series,
    "frames_scale": frames_scale,
    "frames_skip_steps_series": frames_skip_steps_series,
    "frames_skip_steps": frames_skip_steps,
    "text_prompts": text_prompts,
    "image_prompts": image_prompts,
    "cut_overview": eval(cut_overview),
    "cut_innercut": eval(cut_innercut),
    "cut_ic_pow": cut_ic_pow,
    "cut_icgray_p": eval(cut_icgray_p),
    "intermediate_saves": intermediate_saves,
    "intermediates_in_subfolder": intermediates_in_subfolder,
    "steps_per_checkpoint": steps_per_checkpoint,
    "perlin_init": perlin_init,
    "perlin_mode": perlin_mode,
    "set_seed": set_seed,
    "eta": eta,
    "clamp_grad": clamp_grad,
    "clamp_max": clamp_max,
    "skip_augs": skip_augs,
    "randomize_class": randomize_class,
    "clip_denoised": clip_denoised,
    "fuzzy_prompt": fuzzy_prompt,
    "rand_mag": rand_mag,
    # 'init_latent_scale': init_latent_scale,
    # 'frames_latent_scale': frames_latent_scale
}

args = SimpleNamespace(**args)

print("Prepping model...")
model, diffusion = create_model_and_diffusion(**model_config)
if diffusion_model == "stable_diffusion":
    pass
else:
    if diffusion_model == "custom":
        model.load_state_dict(torch.load(custom_path, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(f"{model_path}/{diffusion_model}.pt", map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()

import traceback

gc.collect()
torch.cuda.empty_cache()
do_run()
print("n_stats_avg (mean, std): ", n_mean_avg, n_std_avg)
# try:
#   do_run()
# except KeyboardInterrupt:
#     pass
# except Exception as e:
#     print(e)
#     print(traceback.format_exc())
# finally:
#     print('Seed used:', seed)
#     gc.collect()
#     torch.cuda.empty_cache()

# %% [markdown]
# # 5. Create the video

# %%
import PIL

# @title ### **Create video**
# @markdown Video file will save in the same folder as your images.
from tqdm.notebook import trange

skip_video_for_run_all = False  # @param {type: 'boolean'}
# @markdown ### **Video masking (post-processing)**
# @markdown Use previously generated background mask during video creation
use_background_mask_video = True  # @param {type: 'boolean'}
invert_mask_video = False  # @param {type: 'boolean'}
# @markdown Choose background source: image, color, init video.
background_video = "init_video"  # @param ['image', 'color', 'init_video']
# @markdown Specify the init image path or color depending on your background video source choice.
background_source_video = "red"  # @param {type: 'string'}
blend_mode = "optical flow"  # @param ['None', 'linear', 'optical flow']
# if (blend_mode == "optical flow") & (animation_mode != 'Video Input Legacy'):
# @markdown ### **Video blending (post-processing)**
#   print('Please enable Video Input mode and generate optical flow maps to use optical flow blend mode')
blend = 0.5  # @param {type: 'number'}
check_consistency = True  # @param {type: 'boolean'}
postfix = ""
if use_background_mask_video:
    postfix += "_mask"

# @markdown ### **Video settings**
if skip_video_for_run_all == True:
    print("Skipping video creation, uncheck skip_video_for_run_all if you want to run it")

else:
    # import subprocess in case this cell is run without the above cells
    import subprocess
    from base64 import b64encode

    latest_run = batchNum

    folder = batch_name  # @param
    run = latest_run  # @param
    final_frame = "final_frame"

    init_frame = 1  # @param {type:"number"} This is the frame where the video will start
    last_frame = final_frame  # @param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
    fps = 24  # @param {type:"number"}
    # view_video_in_cell = True #@param {type: 'boolean'}

    frames = []
    # tqdm.write('Generating video...')

    if last_frame == "final_frame":
        last_frame = len(glob(batchFolder + f"/{folder}({run})_*.png"))
        print(f"Total frames: {last_frame}")

    image_path = f"{outDirPath}/{folder}/{folder}({run})_%06d.png"
    filepath = f"{outDirPath}/{folder}/{folder}({run}).mp4"

    if (blend_mode == "optical flow") & (True):
        image_path = f"{outDirPath}/{folder}/flow/{folder}({run})_%06d.png"
        postfix += "_flow"
        video_out = batchFolder + f"/video"
        os.makedirs(video_out, exist_ok=True)
        filepath = f"{video_out}/{folder}({run})_{postfix}.mp4"
        if last_frame == "final_frame":
            last_frame = len(glob(batchFolder + f"/flow/{folder}({run})_*.png"))
        flo_out = batchFolder + f"/flow"
        # !rm -rf {flo_out}/*

        # !mkdir "{flo_out}"
        os.makedirs(flo_out, exist_ok=True)

        frames_in = sorted(glob(batchFolder + f"/{folder}({run})_*.png"))

        frame0 = PIL.Image.open(frames_in[0])
        if use_background_mask_video:
            frame0 = apply_mask(
                frame0, 0, background_video, background_source_video, invert_mask_video
            )
        frame0.save(flo_out + "/" + frames_in[0].replace("\\", "/").split("/")[-1])

        for i in trange(init_frame, min(len(frames_in), last_frame)):

            frame1_path = frames_in[i - 1]
            frame2_path = frames_in[i]

            frame1 = PIL.Image.open(frame1_path)
            frame2 = PIL.Image.open(frame2_path)
            # if use_background_mask_video:
            #   frame1 = apply_mask(frame1, i-1, background_video, background_source_video)
            #   frame2 = apply_mask(frame2, i, background_video, background_source_video)
            frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4])+1):06}.jpg"
            flo_path = f"{flo_folder}/{frame1_stem}.npy"
            weights_path = None
            if check_consistency:
                if reverse_cc_order:
                    weights_path = f"{flo_folder}/{frame1_stem}-21_cc.jpg"
                else:
                    weights_path = f"{flo_folder}/{frame1_stem}_12-21_cc.jpg"
            # print(i, frame1_path, frame2_path, flo_path)
            frame = warp(
                frame1,
                frame2,
                flo_path,
                blend=blend,
                weights_path=weights_path,
                pad_pct=padding_ratio,
                padding_mode=padding_mode,
                inpaint_blend=0,
                video_mode=True,
            )
            if use_background_mask_video:
                frame = apply_mask(
                    frame, i, background_video, background_source_video, invert_mask_video
                )
            frame.save(batchFolder + f"/flow/{folder}({run})_{i:06}.png")
    if blend_mode == "linear":
        image_path = f"{outDirPath}/{folder}/blend/{folder}({run})_%06d.png"
        postfix += "_blend"
        video_out = batchFolder + f"/video"
        os.makedirs(video_out, exist_ok=True)
        filepath = f"{video_out}/{folder}({run})_{postfix}.mp4"
        if last_frame == "final_frame":
            last_frame = len(glob(batchFolder + f"/blend/{folder}({run})_*.png"))
        blend_out = batchFolder + f"/blend"
        pathlib.Path(blend_out).mkdir(exist_ok=True)
        frames_in = glob(batchFolder + f"/{folder}({run})_*.png")

        frame0 = PIL.Image.open(frames_in[0])
        if use_background_mask_video:
            frame0 = apply_mask(
                frame0, 0, background_video, background_source_video, invert_mask_video
            )
        frame0.save(flo_out + "/" + frames_in[0].replace("\\", "/").split("/")[-1])

        for i in trange(1, len(frames_in)):
            frame1_path = frames_in[i - 1]
            frame2_path = frames_in[i]

            frame1 = PIL.Image.open(frame1_path)
            frame2 = PIL.Image.open(frame2_path)
            # if use_background_mask_video:
            #   frame1 = apply_mask(frame1, i-1, background_video, background_source_video)
            #   frame2 = apply_mask(frame2, i, background_video, background_source_video)
            frame = PIL.Image.fromarray(
                (np.array(frame1) * (1 - blend) + np.array(frame2) * (blend))
                .round()
                .astype("uint8")
            )
            if use_background_mask_video:
                frame = apply_mask(
                    frame, i, background_video, background_source_video, invert_mask_video
                )
            frame.save(batchFolder + f"/blend/{folder}({run})_{i:06}.png")

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

    process = subprocess.Popen(
        cmd, cwd=f"{batchFolder}", stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
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


# %%
# @title Shutdown runtime
# @markdown Useful with the new Colab policy.\
# @markdown If on, shuts down the runtime after every cell has been run successfully.
from google.colab import runtime

shut_down_after_run_all = False  # @param {'type':'boolean'}
if shut_down_after_run_all:
    runtime.unassign()

# %% [markdown]
# # Compare settings

# %%
# @title Insert paths to two settings.txt files to compare
file1 = ""  # @param {'type':'string'}
file2 = ""  # @param {'type':'string'}
if file1 != "" and files2 != "":
    import json

    with open(file1, "rb") as f:
        f1 = json.load(f)
    with open(file2, "rb") as f:
        f2 = json.load(f)
    joint_keys = set(list(f1.keys()) + list(f2.keys()))
    print(f'Comparing\n{file1.split("/")[-1]}\n{file2.split("/")[-1]}\n')
    for key in joint_keys:
        if key in f1.keys() and key in f2.keys() and f1[key] != f2[key]:
            print(f"{key}: {f1[key]} -> {f2[key]}")
        if key in f1.keys() and key not in f2.keys():
            print(f"{key}: {f1[key]} -> <variable missing>")
        if key not in f1.keys() and key in f2.keys():
            print(f"{key}: <variable missing> -> {f2[key]}")

main_style = "oil painting, octane render"  # , by kilian eng"
# other = (
#     "abstract psychedelic vines, intricate, surreal, in dawn, sunny, happy, epic, photograph ,"
#     " Artwork by Wes Wilson, Tokio Aoyama, vladimir kush, concept art, trending on art station  ,"
#     " octane render , Hyperrealistic"
# )
other = (
    "a close up digital painting of microscopic organisms in the style of 70s psychedelic poster"
    " art, bright, sunny, happy, octane render, trending on artstation, cinematic,"
    " hyper realism, high detail, octane render, 8k"
)
# MACRO
# Peter Max
# "a digital painting of microscopic organisms in the style of 70s psychedelic poster art, bright, sunny, happy, octane render, trending on"
# " artstation, cinematic, hyper realism, high detail, octane render, 8k"
# josef thoma
# woodcut?
text_prompts = {
    "0": (
        "A photograph, amazing   cinematic film still , concept art    laughing    cauliflower"
        " Cthulhu  in a   farm ,  in  dawn,  art by  wes anderson  ,   RHADS ,  Beatrix potter ,"
        " colorful,   Internal glow , large expressive eyes , large expressive eyes , large"
        " expressive eyes ,  4k ,  subsurface scattering  8k, 4k, hd, intricate and highly"
        " realistic, trending on art station, wide range of colors, orange, blue, photorealistic,"
        " dramatic shadows, photorealistic, dramatic shadows, highly detailed, hyperreal, long"
        " view, vista"
    ),
    "100": (
        "A photograph, remarkable   oil painting ,  greek marble statue    hysterical    Tardigrade"
        "  in a   world made of skittles ,  in  dawn,  art by  Moebius Jean Giraud ,   Salvador"
        " Dali ,  Alejandro Jodorowsky ,   redshift render ,  Volumetric ,  raytracing ,  realistic"
        "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
        " dramatic shadows, highly detailed, hyperreal, long view, vista"
    ),
    "200": (
        "A photograph, amazing   daguerrotype ,  surrealist painting    raging    tsar bomba"
        " explosion  in a   world made of skittles ,  in  evening ,  art by  Mark Ryden ,  Margaret"
        " Keane ,  Jon Marro ,    intricate and highly detailed ,   Photorealistic ,  "
        " Photorealistic,  Surreal  8k, 4k, hd, intricate and highly realistic, trending on art"
        " station, photorealistic, dramatic shadows, highly detailed, hyperreal, long view, vista"
    ),
    "300": (
        "A photograph, remarkable   cinematic film still ,  oil painting    planet scale   "
        " Cerberus  in a   space suit  ,  in  dawn,  art by  Walt Disney ,  ross draws ,  MAX ERNST"
        " ,   HDR ,  Studio light ,  Maximum texture ,   Photorealistic 8k, 4k, hd, intricate and"
        " highly realistic, trending on art station, photorealistic, dramatic shadows, highly"
        " detailed, hyperreal, long view, vista"
    ),
}

negative_prompts = {0: ["red, face, humans, people, text, blurry, unfocused"]}

image_prompts = {
    # 0:['ImagePromptsWorkButArentVeryGood.png:2',],
}
# main_prompt = f"green forest and trees, {main_style}"
# animation_prompts = {
#     0: main_prompt,
#     anim_args_dict["switch_frame"] + 24 * 15: f"a massive root system, {main_style}",
#     anim_args_dict["switch_frame"]
#     + 24 * 30: f"the gears of nature turning in a forest, {main_style}",
#     # 20: "a beautiful banana, trending on Artstation",
#     # 30: "a beautiful coconut, trending on Artstation",
#     # 40: "a beautiful durian, trending on Artstation",
# }

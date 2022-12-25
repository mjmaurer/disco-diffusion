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
cadence = 54
text_prompts = {
    0 * cadence: (
        "A photograph, amazing cinematic film still , cartoon bear made of yarn in a lush forest "
        " , in  dawn,  art by peter max , RHADS , "
        " colorful, psychedelic , Internal glow , large expressive eyes , large expressive eyes , large"
        " expressive eyes ,  4k ,  subsurface scattering  8k, 4k, hd, intricate and highly"
        " realistic, trending on art station, wide range of colors, green , blue, photorealistic,"
        " dramatic shadows, photorealistic, dramatic shadows, highly detailed, hyperreal, long"
        " view, vista"
    ),
    cadence: (
        "A photograph, remarkable  oil painting , greek marble statue of cartoon bears "
        " in space ,  in  dawn,  art by  Moebius Jean Giraud , "
        " Alejandro Jodorowsky ,  redshift render ,  Volumetric ,  raytracing ,  realistic"
        "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
        " dramatic shadows, highly detailed, hyperreal, long view, vista"
    ),
    2 * cadence: (
        "A photograph, remarkable   oil painting , muscular cartoon bear astronauts  "
        " in space ,  in  dawn,  art by Peter Max , Jon Marro "
        " , redshift render ,  Volumetric ,  raytracing ,  realistic"
        "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
        " dramatic shadows, highly detailed, hyperreal, long view, vista"
    ),
    3 * cadence: (
        "A photograph, remarkable   oil painting , cartoon bears wearing suits "
        " in a futuristic city ,  in  dawn,  art by Vladimir Kush , Mark Ryden "
        " , redshift render ,  Volumetric ,  raytracing ,  realistic"
        "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
        " dramatic shadows, highly detailed, hyperreal, long view, vista"
    ),
    # "200": (
    #     "A photograph, amazing   daguerrotype ,  surrealist painting    raging    tsar bomba"
    #     " explosion  in a   world made of skittles ,  in  evening ,  art by  Mark Ryden ,  Margaret"
    #     " Keane ,  Jon Marro ,    intricate and highly detailed ,   Photorealistic ,  "
    #     " Photorealistic,  Surreal  8k, 4k, hd, intricate and highly realistic, trending on art"
    #     " station, photorealistic, dramatic shadows, highly detailed, hyperreal, long view, vista"
    # ),
    # "300": (
    #     "A photograph, remarkable   cinematic film still ,  oil painting    planet scale   "
    #     " Cerberus  in a   space suit  ,  in  dawn,  art by  Walt Disney ,  ross draws ,  MAX ERNST"
    #     " ,   HDR ,  Studio light ,  Maximum texture ,   Photorealistic 8k, 4k, hd, intricate and"
    #     " highly realistic, trending on art station, photorealistic, dramatic shadows, highly"
    #     " detailed, hyperreal, long view, vista"
    # ),
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

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
# organisms

# p = [
#     (
#         "A photograph, amazing cinematic film still , cartoon bear made of yarn in a forest "
#         " , art by RHADS , "
#         " colorful, Internal glow , large expressive eyes , large expressive eyes , large"
#         " expressive eyes ,  4k ,  subsurface scattering  8k, 4k, hd, intricate and highly"
#         " realistic, trending on art station, wide range of colors, green , blue, photorealistic,"
#         " dramatic shadows, photorealistic, dramatic shadows, highly detailed, hyperreal, long"
#         " view, vista"
#     ),
#     (
#         "A photograph, remarkable , digital painting , microscopic organisms "
#         " , in  dawn, art by  Moebius Jean Giraud , "
#         " Alejandro Jodorowsky ,  redshift render ,  Volumetric ,  raytracing ,  realistic"
#         "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
#         " dramatic shadows, highly detailed, hyperreal, long view, vista"
#     ),
#     (
#         "A photograph, remarkable  oil painting , cartoon bear astronauts  "
#         " in space ,  in  dawn,  art by Peter Max , Jon Marro "
#         " , redshift render ,  Volumetric ,  raytracing ,  realistic"
#         "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
#         " dramatic shadows, highly detailed, hyperreal, long view, vista"
#     ),
#     (
#         "A photograph, remarkable planet scale , cartoon bears wearing suits "
#         " in a futuristic city ,  in  dawn,  art by Vladimir Kush "
#         " , redshift render ,  Volumetric ,  raytracing ,  realistic"
#         "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
#         " dramatic shadows, highly detailed, hyperreal, long view, vista"
#     ),
#     (
#         "A photograph, remarkable , woodcut art , abtract cartoon bears on a farm "
#         " ,  in  dawn,  art by Mark Ryden "
#         " , redshift render ,  Volumetric ,  raytracing ,  realistic"
#         "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
#         " dramatic shadows, highly detailed, hyperreal, long view, vista"
#     ),
#     (
#         "A epic  cinematic film still and digital illustration of an adorable fluffy and plushy "
#         " and silly  cartoon bear in a   world made of vines , in morning , art by  Wes"
#         " Anderson  And   Margaret Keane  And    Zdzisław Beksiński , made from  soft felt "
#         " intricate and highly detailed, realistic, extreme detail, fine detail, dramatic lighting,"
#         " volumetric lighting, extreme fluffy, cute, adorable, large expressive eyes, long view,"
#         " exotic colors, mutant, cyberpunk colors, DSLR,  sharp focus, extremely detailed realistic"
#         " and highly detailed, 4k, 8k, hd, photorealistic, dramatic shadows"
#     ),
#     (
#         "A  amazing  cinematic film still and pencil illustration of a realistic  cartoon bear "
#         " swimmin in the ocean , in  dawn, art by ernst"
#         " haeckl  And  Lucian Freud , intricate and highly detailed, realistic, extreme detail,"
#         " fine detail, dramatic lighting, volumetric lighting, extreme fluffy, cute, adorable,"
#         " large expressive eyes, long view, exotic colors, mutant, cyberpunk colors, DSLR,  sharp"
#         " focus, extremely detailed realistic and highly detailed, 4k, 8k, hd, photorealistic,"
#         " dramatic shadows"
#     ),
# ]
p = [
    (
        "A photograph, amazing cinematic film still , santa flying through space "
        " , art by RHADS , "
        " colorful, Internal glow , large expressive eyes , large expressive eyes , large"
        " expressive eyes ,  4k ,  subsurface scattering  8k, 4k, hd, intricate and highly"
        " realistic, trending on art station, wide range of colors, green , blue, photorealistic,"
        " dramatic shadows, photorealistic, dramatic shadows, highly detailed, hyperreal, long"
        " view, vista"
    ),
    (
        "A photograph, remarkable , digital painting , cartoon of santas elves dancing "
        " , in  dawn, art by Peter Max , "
        " Alejandro Jodorowsky ,  redshift render ,  Volumetric ,  raytracing ,  realistic"
        "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
        " dramatic shadows, highly detailed, hyperreal, long view, vista"
    ),
    (
        "A photograph, remarkable , woodcut art , christmas presents under the christmas tree "
        " , in  dawn, by Jon Marro, "
        " , redshift render ,  Volumetric ,  raytracing ,  realistic"
        "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
        " dramatic shadows, highly detailed, hyperreal, long view, vista"
    ),
    (
        "A photograph, remarkable planet scale , 70s psychedelic poster art , cats wearing santa hats "
        " in  dawn,  art by  Wes Wilson "
        " , redshift render ,  Volumetric ,  raytracing ,  realistic"
        "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
        " dramatic shadows, highly detailed, hyperreal, long view, vista"
    ),
    (
        "A photograph, remarkable , architecture diagram , christmas trees "
        " ,  art by Mark Ryden "
        " , redshift render ,  Volumetric ,  raytracing ,  realistic"
        "  8k, 4k, hd, intricate and highly realistic, trending on art station, photorealistic,"
        " dramatic shadows, highly detailed, hyperreal, long view, vista"
    ),
    (
        "A  amazing  cinematic film still , a portrait of santa claus made of yarn , "
        " in  dawn, art by ernst"
        " haeckl  And  Lucian Freud , intricate and highly detailed, realistic, extreme detail,"
        " fine detail, dramatic lighting, volumetric lighting, extreme fluffy, cute, adorable,"
        " large expressive eyes, long view, exotic colors, mutant, cyberpunk colors, DSLR,  sharp"
        " focus, extremely detailed realistic and highly detailed, 4k, 8k, hd, photorealistic,"
        " dramatic shadows"
    ),
]
cadence = 54

text_prompts = {
    i * cadence * 2: p[i % len(p)]
    for i in range(20)
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

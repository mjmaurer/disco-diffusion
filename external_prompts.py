main_style = "oil painting, octane render"  # , by kilian eng"
# other = (
#     "abstract psychedelic vines, intricate, surreal, in dawn, sunny, happy, epic, photograph ,"
#     " Artwork by Wes Wilson, Tokio Aoyama, vladimir kush, concept art, trending on art station  ,"
#     " octane render , Hyperrealistic"
# )
other = (
    "70s psychedelic poster art of a forest and clouds, octane render, trending on artstation, cinematic, hyper realism,"
    " high detail, octane render, 8k"
)
# josef thoma
# woodcut?
text_prompts = {
    0: [
        (
            # "A forest with massive trees engraves with cartoon bears, "
            # "Massive fallen trees in a psychadelic forest, " + main_style
            "Massive fallen trees in a colorful psychadelic forest, " + main_style
        ),
        # main_style,
        # "blurry, shallow depth of field, bokeh:-4",
    ],
    60: [
        (
            # "A forest with massive trees engraves with cartoon bears, "
            # "Massive fallen trees in a psychadelic forest, " + main_style
            "A psychedelic city skyline, " + main_style
        ),
        # main_style,
        # "blurry, shallow depth of field, bokeh:-4",
    ],
}

negative_prompts = {0: ["person, human, man, woman, body, text, blurry, unfocused"]}

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

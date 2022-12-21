main_style = "oil painting, octane render"  # , by kilian eng"
other = (
    "fantasy, lush, bright psychedelic forest, in dawn, woodcut, sunny, happy, epic, photograph ,"
    " Artwork by Mark Ryden , Slawek Fedorczuk , trending on art station , filigree , octane render"
    " , Hyperrealistic"
)
# woodcut?
text_prompts = {
    0: [
        other
        # (
        #     # "A forest with massive trees engraves with cartoon bears, "
        #     # "Massive fallen trees in a psychadelic forest, " + main_style
        #     "Trees and tree branches in a colorful psychadelic forest"
        # ),
        # main_style,
        # "blurry, shallow depth of field, bokeh:-4",
    ],
}

negative_prompts = {0: ["purple, text, blurry, unfocused"]}

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

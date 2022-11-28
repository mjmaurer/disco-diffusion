main_style = "vibrant, octane render, by kilian eng"

# text_prompts = {
#     0: [
#         (
#             # "A forest with massive trees engraves with cartoon bears, "
#             "Massive tree branches and cartoon bears, "
#             + main_style
#         ),
#     ],
# }
text_prompts = {
    0: [
        "a beautiful breathtaking highly-detailed intricate portrait painting of a young woman with"
        " long hair, against a backdrop of stars in the style of alphonse mucha and ilya kuvshinov"
        " and peter mohrbacher, ross tran rossdraws, watercolor, featured on artstation, 70mm,"
        " rendered in octane"
    ],
    550: [
        "a highly detailed matte painting of a robot, No Man's Sky Screenshot, tall grass field,"
        " broken machinery, Simon Stalenhag , featured on Artstation."
    ],
    1000: [
        "a beautiful breathtaking highly-detailed intricate portrait painting of a blissful"
        " ignorant enlightened young sxelaqwertyop dancing against a backdrop of stars in the style"
        " of alphonse mucha and ilya kuvshinov and peter mohrbacker, ross tran rossdraws,"
        " watercolor, featured on artstation, 70mm, rendered in octane"
    ],
    1500: [
        "a highly detailed matte painting of a robot sxelaqwertyop, No Man's Sky Screenshot, tall"
        " grass field, broken machinery, Simon Stalenhag , featured on Artstation."
    ],
}

negative_prompts = {0: ["text, logo, cropped, two heads, four arms, lazy eye, blurry, unfocused"]}

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

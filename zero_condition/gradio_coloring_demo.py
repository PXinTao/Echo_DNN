#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import boxx
from boxx import *
import time
import numpy as np
from condition256_gen import DDNInference, crop_and_resize
import threading
from PIL import Image
import io
import os

# pip install gradio huggingface_hub
import gradio as gr
from huggingface_hub import hf_hub_download

H, W = 256, 256
ddn = None


ddn_asset_dir = "../../ddn_asset"
ddn_asset_filenames = [
    "v32-00003-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch64_k64-shot-200000.pkl",
    "ffhq_example/FFHQ-test4.png",
    "ffhq_example/FFHQ-00526.png",
    "rgba_zscg_example/FFHQ-00526_guided_rgba.png",
    "ffhq_example/FFHQ-test2.png",
    "ffhq_example/FFHQ-test5.png",
    "rgba_zscg_example/FFHQ-test5_guided_rgba.png",
]
ddn_asset_paths = {
    filename: f"{ddn_asset_dir}/{filename}" for filename in ddn_asset_filenames
}
for filename, v in ddn_asset_paths.items():
    if not os.path.exists(v):
        try:
            ddn_asset_paths[filename] = hf_hub_download(
                repo_id="diyer22/ddn_asset",
                filename=filename,
                local_files_only=True,
                etag_timeout=1,
            )
        except Exception as e:
            print(f"Downloading {filename}")
            ddn_asset_paths[filename] = hf_hub_download(
                repo_id="diyer22/ddn_asset", filename=filename
            )


def get_model():
    with threading.Lock():
        global ddn
        if ddn is None:
            ddn = DDNInference(
                ddn_asset_paths[
                    "v32-00003-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch64_k64-shot-200000.pkl"
                ]
            )
        return ddn


def generate(example_description, input_condition_img, editor_value, prompt_value=None):
    print(f"{boxx.localTimeStr(True)} {boxx.increase('requests')}th requests")
    boxx.tree([example_description, input_condition_img, editor_value, prompt_value])
    # print("layer sum:", tree(editor_value["layers"]) or editor_value["layers"][0].sum())
    ddn = get_model()
    n = n_samples
    if prompt_value:
        n = n_samples_with_clip
    guided_rgba = editor_value["layers"][0] if len(editor_value["layers"]) else None
    d = ddn.coloring_demo_inference(
        input_condition_img,
        n_samples=n,
        guided_rgba=guided_rgba,
        clip_prompt=prompt_value,
    )
    stage_last_predicts = d["stage_last_predicts_np"]
    # tree(stage_last_predicts)
    if 0:  # dump to /tmp with timestamp for debugging
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if guided_rgba is not None:
            boxx.imsave(
                f"/tmp/condition256_gen_{timestamp}_guided_rgba.png",
                guided_rgba,
            )
            boxx.imsave(
                f"/tmp/condition256_gen_{timestamp}_background.png",
                rgba_edit["background"],
            )
        if input_condition_img is not None:
            boxx.imsave(
                f"/tmp/condition256_gen_{timestamp}_input_condition_img.png",
                input_condition_img,
            )
        if prompt_value:
            with open(f"/tmp/condition256_gen_{timestamp}_prompt.txt", "w") as f:
                f.write(prompt_value)
        # stage_last_predicts is a dict with resolution keys, get the largest resolution
        largest_resolution_key = max(
            stage_last_predicts.keys(), key=lambda x: int(x.split("x")[0])
        )
        stage_last_predict = stage_last_predicts[largest_resolution_key]
        for i, img in enumerate(stage_last_predict):
            if img is not None:
                boxx.imsave(
                    f"/tmp/condition256_gen_{timestamp}_stage_last_predict_{i}.png",
                    img,
                )
    return flatten_results(stage_last_predicts)


def input_condition_img_callback(input_condition_img, editor_value):
    if isinstance(input_condition_img, np.ndarray):
        if input_condition_img.shape[0] != H or input_condition_img.shape[1] != W:
            input_condition_img = crop_and_resize(input_condition_img, (H, W))
    if input_condition_img is None:
        editor_value["background"] = None
        editor_value["composite"] = None
    else:
        layers = editor_value.get("layers", [])
        if (
            len(layers) > 0
            and layers[0].sum() > 0
            and layers[0].shape[:2] != input_condition_img.shape[:2]
        ):  # if the first layer is not empty and has different shape, empty it
            editor_value["layers"] = []
        editor_value["background"] = np.uint8(input_condition_img.mean(-1).round()) // 2
        editor_value["composite"] = None
    return editor_value


def flatten_results(results):
    return sum(
        [
            (results[key] + [None] * n_samples)[:n_samples]
            for key in sorted(result_blocks)
        ],
        [],
    )


def read_as_input_condition_img(png_path):
    img = boxx.imread(png_path)
    return crop_and_resize(img, (H, W))


with gr.Blocks() as demo:
    gr.Markdown(
        """# DDN coloring demo
- [Discrete Distribution Networks (DDN)](https://discrete-distribution-networks.github.io/) is a **novel generative model** with unique properties.
- This demo showcases DDN's features through a coloring task, particularly highlighting its **Zero-Shot Conditional Generation (ZSCG)** capability.
- (Optional) Users can guide generation process using **color strokes** and **CLIP prompts**.
- This DDN model is trained on face dataset (FFHQ) with $K=64$, $L=55$ 
"""
    )
    gr.HTML("<hr style='padding:0!important'>", container=False)
    with gr.Row():
        with gr.Row():
            default_rgb = read_as_input_condition_img(
                ddn_asset_paths["ffhq_example/FFHQ-test4.png"]
            )
            default_edit = dict(
                background=np.uint8(default_rgb.mean(-1)) // 2,
                layers=[],
                composite=None,
            )
            upload_block = gr.Image(
                np.uint8(default_rgb.mean(-1)),
                label="input_condition_img",
                format="png",
                width=W,
                height=H,
            )
            brush = gr.Brush()
            brush.colors = [
                "rgb(255,0,0)",
                "rgb(0,255,0)",
                "rgb(0,0,255)",
                "rgb(0,255,255)",
                "rgb(255,0,255)",
                "rgb(255,255,0)",
                "rgb(0,0,0)",
                "rgb(64,64,64)",
                "rgb(128,128,128)",
                "rgb(192,192,192)",
                "rgb(255,255,255)",
                "rgb(39,106,167)",
                "rgb(251,217,196)",
                "rgb(255,194,116)",
                "rgb(229,220,138)",
                "rgb(203,172,225)",
                "rgb(98, 59, 31)",
            ] + brush.colors
            brush.default_size = 8
            brush.default_color = "rgb(0,255,0)"
            editor_block = gr.ImageEditor(
                default_edit,
                elem_id="color-stroke",
                label="color stroke",
                type="numpy",
                crop_size="1:1",
                width=256,
                height=256,
                brush=brush,
                layers=False,
                format="png",
                # canvas_size=(256, 256),   # 画布固定为 256×256
                # fixed_canvas=True,        # 上传 / 更换 background 时自动缩放到画布大小
            )
            with gr.Column():
                prompt_block = gr.Textbox(
                    "",
                    interactive=True,
                    label="CLIP prompt:",
                )
                description_block = gr.Textbox(
                    "",
                    label="description",
                    visible=False,
                )
                gr.HTML(
                    """<span style="">
                            <br><br><br><br>
                            JUST click this button to do coloring:
                        </span>""",
                    container=False,
                    padding=False,
                )
                button = gr.Button("Generate ➡️")
    gr.HTML("<hr style='padding:0!important'>", container=False)
    upload_block.input(
        input_condition_img_callback,
        inputs=[upload_block, editor_block],
        outputs=editor_block,
    )

    clean_img_args = dict(
        interactive=False,
        show_label=False,
        show_download_button=False,
        min_width=1,
        show_fullscreen_button=False,
        format="png",
        # container=False
    )
    n_samples = 12
    n_samples_with_clip = 4
    min_size = W
    result_blocks = {}
    # gr.Markdown("## Results")
    # gr.Markdown("Each column is a sample, each row is last predict of a stage")
    with gr.Row():
        for sample_idx in range(n_samples):
            with gr.Column(min_width=W):
                for stage_idx in range(9)[::-1]:
                    size = 2**stage_idx
                    key = f"{size}x{size}"
                    if size < min_size:
                        continue
                    result_image = gr.Image(
                        width=max(80, size + 16), height=size + 2, **clean_img_args
                    )
                    result_blocks[key] = result_blocks.get(key, []) + [result_image]
    button.click(
        generate,
        inputs=[description_block, upload_block, editor_block, prompt_block],
        outputs=flatten_results(result_blocks),
    )

    example_rgb = boxx.imread(ddn_asset_paths["ffhq_example/FFHQ-test4.png"])
    example_img = np.uint8(
        example_rgb.mean(
            -1,
        )
    )
    if 0:
        gr.HTML("<hr>")
        with gr.Row() as example1:
            with gr.Column():
                example_img_gr1 = gr.Image(
                    example_img, width=64 + 16, height=64 + 2, **clean_img_args
                )
            with gr.Column():
                gr.Markdown("example1: Musk\n\nprompt: None")

    gr.HTML("<hr>")

    guided_rgba = example_rgb[:]
    # guided_rgba[:] = [[255,194,116]]
    # guided_rgba[:] = [[229,220,138]]
    guided_rgba[:] = [[203, 172, 225]]
    mask = np.ones_like(guided_rgba)[..., :1] * 0
    mask[: len(mask) // 10 :, : len(mask) // 10] = 255
    guided_rgba = np.concatenate([guided_rgba, mask], axis=-1)

    editor_value_example = dict(
        background=example_img // 2, layers=[guided_rgba], composite=guided_rgba
    )

    def read_as_example_input(png_path, guided_path=None):
        condition_img = read_as_input_condition_img(png_path)
        background = np.uint8(condition_img.mean(-1)) // 2
        if guided_path is None:
            layers = []
        else:
            guided_rgba_526 = boxx.imread(guided_path)
            import cv2

            # resize to 1024x1024 to avoid bug in gradio(same size will not load)
            guided_rgba_526 = boxx.resize(
                guided_rgba_526, (1024, 1024), interpolation=cv2.INTER_NEAREST
            )
            # compress real resolution for fast transport png
            background = boxx.resize(
                boxx.resize(background, (128, 128)),
                (1024, 1024),
                interpolation=cv2.INTER_NEAREST,
            )
            layers = [guided_rgba_526]
        return [
            condition_img,
            dict(
                background=background,
                layers=layers,
                composite=None,
            ),
            "",
        ]

    example_inputs = []
    example_inputs.append(
        [
            "random coloring",
            np.uint8(default_rgb.mean(-1)),
            default_edit,
            "",
        ]
    )
    _rgb, rgba_edit = read_as_example_input(
        ddn_asset_paths["ffhq_example/FFHQ-00526.png"],
        ddn_asset_paths["rgba_zscg_example/FFHQ-00526_guided_rgba.png"],
    )[:2]

    # print("layer sum:", tree(rgba_edit["layers"]) or rgba_edit["layers"][0].sum())
    example_inputs.append(
        [
            "ZSCG by color stroke",
            _rgb,
            rgba_edit,
            "",
        ]
    )
    _rgb, _edit = read_as_example_input(ddn_asset_paths["ffhq_example/FFHQ-test2.png"])[
        :2
    ]
    example_inputs.append(
        [
            "ZSCG by CLIP prompt",
            np.uint8(_rgb.mean(-1)),
            _edit,
            "colorful portrait",
        ]
    )
    example_inputs.append(
        [
            "CLIP + color stroke",
            *read_as_example_input(
                ddn_asset_paths["ffhq_example/FFHQ-test5.png"],
                ddn_asset_paths["rgba_zscg_example/FFHQ-test5_guided_rgba.png"],
            )[:2],
            "portrait with purple hair",
        ]
    )
    # example_inputs += [
    #     read_as_example_input(png_path)
    #     for png_path in sorted(glob.glob(f"{ddn_asset_dir}/ffhq_example/*.png"))
    # ]
    gr.Examples(
        example_inputs,
        inputs=[description_block, upload_block, editor_block, prompt_block],
        outputs=flatten_results(result_blocks),
        fn=generate,
        # cache_mode="lazy",
        # cache_examples=True, # json.decoder.JSONDecodeError: with CLIP
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """*Tip: example with color strokes may cause gradio an UI bug on IOS*"""
            )
        with gr.Column():
            # back to top
            gr.HTML(
                """
            <div style="text-align: right; width: 100%;">
                <a href="#color-stroke" style="text-decoration: none; color: #000;">
                    <button style="border: 1px solid #888; text-align: center; 
                        padding: 10px 20px; border-radius: 5px; cursor: pointer; 
                        margin-left: auto; display: inline-block;">
                        Back to Top
                    </button>
                </a>
            </div>
            """,
                container=False,
            )
    gr.HTML("<hr>")

    # GIF area
    def convert_to_gif(*flattened_results):
        """Convert valid result blocks to GIF animation"""
        if not flattened_results or all(img is None for img in flattened_results):
            return None

        # Rebuild result_blocks structure from flattened results
        result_dict = {}
        idx = 0
        for sample_idx in range(n_samples):
            for stage_idx in range(9)[::-1]:
                size = 2**stage_idx
                key = f"{size}x{size}"
                if size < min_size:
                    continue

                if key not in result_dict:
                    result_dict[key] = []

                if idx < len(flattened_results):
                    result_dict[key].append(flattened_results[idx])
                else:
                    result_dict[key].append(None)
                idx += 1

        # Collect valid images, grouped by stages
        stage_images = {}

        for sample_idx in range(n_samples):
            for stage_idx in range(9)[::-1]:
                size = 2**stage_idx
                key = f"{size}x{size}"
                if size < min_size:
                    continue

                if key in result_dict and sample_idx < len(result_dict[key]):
                    result_img = result_dict[key][sample_idx]
                    if result_img is not None:
                        if key not in stage_images:
                            stage_images[key] = []
                        # Convert to PIL Image
                        if isinstance(result_img, np.ndarray):
                            img = Image.fromarray(result_img)
                        else:
                            img = result_img
                        stage_images[key].append(img)

        if not stage_images:
            return None

        # Use largest size images to create GIF
        largest_size = max(stage_images.keys(), key=lambda x: int(x.split("x")[0]))
        frames = stage_images[largest_size]

        if not frames:
            return None

        # Ensure all frames are the same size and high quality
        # Resize all frames to consistent size if needed
        target_size = (W, H)
        processed_frames = []
        for frame in frames:
            if frame.size != target_size:
                # Use high-quality resampling
                frame = frame.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to RGB if needed (GIF supports RGB)
            if frame.mode != "RGB":
                frame = frame.convert("RGB")

            processed_frames.append(frame)

        # Create GIF in memory with improved quality settings
        gif_buffer = io.BytesIO()
        processed_frames[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=processed_frames[1:],
            duration=500,  # Slower animation for better viewing (500ms = 2 fps)
            loop=0,
            optimize=False,  # Disable optimization to preserve quality
            disposal=2,  # Clear frame before next one for better quality
        )
        gif_buffer.seek(0)

        # Load the GIF back as PIL Image to preserve animation
        gif_img = Image.open(gif_buffer)
        return gif_img

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Convert Results to GIF Animation")
            convert_gif_button = gr.Button("Convert to GIF", variant="secondary")
        with gr.Column():
            gif_output = gr.Image(
                label="GIF",
                type="pil",
                width=W,
                height=H,
                interactive=False,
                show_download_button=True,
                format="gif",
            )

    # Connect the conversion function to button
    convert_gif_button.click(
        convert_to_gif,
        inputs=flatten_results(result_blocks),  # Use all result_blocks as input
        outputs=gif_output,
    )

if __name__ == "__main__":
    demo.launch(
        debug=True,
        share=True,
        server_name="0.0.0.0",
        server_port=17860,
        show_error=True,
    )

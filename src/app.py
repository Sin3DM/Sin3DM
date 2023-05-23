import gradio as gr
import os
from functools import partial
from sample import sample_diffusion, decode
from utils.parser_util import encoding_log_dir, diffusion_log_dir, load_and_overwrite_args
from utils.common_util import seed_all
from utils import dist_util

TITLE = "Sin3DM: Learning a Diffusion Model from a Single 3D Textured Shape"
DESCRIPTION = '''
Sin3DM learns from a single 3D textured shape and generates high-quality variations.
Default setting takes 30-50s and ~3GB VRAM to generate 4 samples (on an NVIDIA A6000).
Reduce marching cube resolution, number of faces and texture resolution to speed up.
'''
DEVICE_IDX = 0

CKPT_DIR = "checkpoints"
EXAMPLE_NAMES = sorted(os.listdir(CKPT_DIR))


class EmptyArgs():
    pass


def main(
    example,
    n_samples=4,
    rand_seed=0,
    mc_reso=256,
    n_faces=10000,
    tex_reso=2048,
    resize_x=1,
    resize_y=1,
    resize_z=1,
    use_ddim=False,
    ddim_steps=100,
    device_idx=0,
    ):

    seed_all(rand_seed)

    # load args
    exp_tag = os.path.join(CKPT_DIR, example)
    args = EmptyArgs()
    args.tag = exp_tag
    args.gpu_id = device_idx

    args.n_samples = n_samples
    args.output = "tmp"
    args.resize = (resize_x, resize_y, resize_z)
    args.use_ddim = use_ddim
    args.timestep_respacing = str(ddim_steps) if use_ddim else ""
    args.reso = mc_reso
    args.n_faces = n_faces
    args.texreso = tex_reso
    args.vox = False
    args.copy_mtl = False
    args.file_format = "glb"

    # print all args
    print("----- Sampling args -----")
    for k, v in args.__dict__.items():
        print("{0:20}".format(k), v)

    # check existence
    if not os.path.exists(args.tag):
        raise ValueError(f"Experiment log does not exist: {args.tag}")
    
    # load saved model args
    enc_log_dir = encoding_log_dir(args.tag)
    diff_log_dir = diffusion_log_dir(args.tag)
    load_and_overwrite_args(args, os.path.join(enc_log_dir, "args.json"))
    load_and_overwrite_args(args, os.path.join(diff_log_dir, "args.json"), ignore_keys=["timestep_respacing"])

    dist_util.setup_dist(args.gpu_id)

    feat_paths = sample_diffusion(args)
    print("diffusion done.")

    decode(args, feat_paths)
    print("decoding done.")

    gltf_paths = [os.path.join(os.path.dirname(feat_path), "object.glb") for feat_path in feat_paths]
    
    if len(gltf_paths) < 4:
        gltf_paths += [None] * (4 - len(gltf_paths))
    
    return gltf_paths


def find_input_path(example):
    path = os.path.join(CKPT_DIR, example, "input.glb")
    if not os.path.exists(path):
        path = os.path.join(CKPT_DIR, example, "input.gltf")
        if not os.path.exists(path):
            print(f"Input file does not exist: {path}")
            path = None
    return path


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown("# " + TITLE)
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(variant="panel", scale=0.8):
            ref_block = gr.Model3D(value=find_input_path(EXAMPLE_NAMES[0]), label="Training shape")
            
            with gr.Row():
                input_block = gr.Dropdown(
                    EXAMPLE_NAMES, value=EXAMPLE_NAMES[0], label="Example",
                )
                n_sample_sld = gr.Slider(1, 4, value=4, step=1, label="Number of Samples")
            
            with gr.Row():
                mcreso_num = gr.Number(value=256, label="MC Resolution", precision=0)
                nfaces_num = gr.Number(value=10000, label="Number of Faces", precision=0)
                texreso_num = gr.Number(value=2048, label="Texture Resolution", precision=0)
            
            with gr.Row():
                resizex_sld = gr.Slider(0.5, 2, value=1, step=0.1, label="Resize X")
                resizey_sld = gr.Slider(0.5, 2, value=1, step=0.1, label="Resize Y")
                resizez_sld = gr.Slider(0.5, 2, value=1, step=0.1, label="Resize Z")

            with gr.Row():
                seed_num = gr.Number(value=0, label="Random Seed", precision=0)
                ddim_chk = gr.Checkbox(value=False, label="Use DDIM")
                ddim_steps = gr.Slider(50, 200, value=200, step=10, label="DDIM Steps")

            with gr.Row():
                run_btn = gr.Button('Run Generation', variant='primary')

        with gr.Column(variant="panel", scale=1.2):
            with gr.Row():
                output_block1 = gr.Model3D(label="Generated 1")
                output_block2 = gr.Model3D(label="Generated 2")
            with gr.Row():
                output_block3 = gr.Model3D(label="Generated 3")
                output_block4 = gr.Model3D(label="Generated 4")

    input_block.change(fn=find_input_path, inputs=[input_block], outputs=[ref_block])
    
    run_btn.click(fn=partial(main, device_idx=DEVICE_IDX),
                    inputs=[input_block, n_sample_sld, seed_num, mcreso_num, nfaces_num, texreso_num, 
                            resizex_sld, resizey_sld, resizez_sld, ddim_chk, ddim_steps],
                    outputs=[output_block1, output_block2, output_block3, output_block4])

demo.launch(enable_queue=True, share=True)

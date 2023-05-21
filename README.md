# Sin3DM

Please visit our [project page]() for more details.


## Installation
Install required packages via pip
```bash
pip install -r requirements.txt
```


## Data & Pretrained models
TBA.


## Get started
We provide a script on one example for data generation, training, sampling and evaluation. 
```bash
bash scripts/run_single.sh
```


#### Data generation
We create training data by sampling points from a textured mesh:
```bash
cd data
python mesh_sampler.py -s {OBJ_PATH} -d {SAVE_NPZ_PATH} --n_surf 5000000
```
Add `--watertight` flag if the mesh is already watertight. Use `mesh_sampler_pbr.py` for pbr models.


#### Training
Train the autoencoder and then the diffusion model:
```bash
cd src
python train.py --tag {EXP_DIR} --data_path {NPZ_PATH} --gpu_id 0
```


#### Sampling
Generate new textured meshes:
```bash
cd src
python sample.py --tag {EXP_DIR} --n_samples 10 --n_faces 50000 --output results10 --gpu_id 0
```
- `--n_faces` specifies the desired number of mesh triangles. 
- Use `--resize` for retargeting, e.g., `--resize 1 1 1.5`.
- By default, it uses DDPM sampling with 1000 steps. Add `--use_ddim --timestep_respacing '100'` for DDIM sampling.


#### Evaluation
Geometry quality (SSFID) relies on a pretrained 3D shape classifier. Please download it from [here](https://drive.google.com/file/d/1HjnDudrXsNY4CYhIGhH4Q0r3-NBnBaiC/view?usp=sharing) and put `Clsshapenet_128.pth` under `evaluation` folder.
First, render each model from 8 views:
```bash
cd rendering
python mvrender_script.py -s {RESULT_DIR} -g 0 -bl {BLENDER_PATH}
```
`BLENDER_PATH` is the path to the Blender program.

Then, run evaluation for all the metrics:
```bash
cd evaluation
python eval_full.py -s {RESULT_DIR} -r {NPZ_DATA_DIR} -g 0
```
`NPZ_DATA_DIR` is the parent folder of the data path `NPZ_PATH`.


#### Rendering
See `rendering` for the rendering scripts that are used to produce paper figures.


## Acknowledgments
The code is partly based on [guided-diffusion](https://github.com/openai/guided-diffusion), [BlenderToolBox](https://github.com/HTDerekLiu/BlenderToolbox) and [SinGAN](https://github.com/tamarott/SinGAN). We thank the authors of these works for sharing their code.

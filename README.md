## NeuMan: Neural Human Radiance Field from a Single Video

This repository is a reference implementation for NeuMan. NeuMan reconstructs both the background scene and an animatable human from a single video using neural radiance fields.

[[Paper]](https://arxiv.org/abs/2203.12575) 

 <p align="center">
  <img src="./resources/teaser.gif" height="260">
</p>

### Video demos

Novel view and novel pose synthesis

[[Bike]](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/bike.mp4)
[[Citron]](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/citron.mp4)
[[Parking lot]](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/demo3.mp4)
[[Jogging]](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/jogging.mp4)
[[Lab]](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/lab.mp4)
[[Seattle]](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/seattle.mp4)

Compositional Synthesis

[[Handshake]](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/handshake.mp4)
[[Dance]](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/dance.mp4)

### Environment

To create the environment using Conda:

```sh
conda env create -f environment.yml
```

Alternately, you can create the environment by executing:

```sh
conda create -n neuman_env python=3.7 -y;
conda activate neuman_env;
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch;
# For RTX 30 series GPU with CUDA version 11.x, please use:
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c fvcore -c iopath -c conda-forge fvcore iopath;
conda install -c bottler nvidiacub;
conda install pytorch3d -c pytorch3d;
conda install -c conda-forge igl;
pip install opencv-python joblib open3d imageio tensorboardX chumpy lpips scikit-image ipython matplotlib;
```

Notice that `pytorch3d` requires a specific version of pytorch, in our case `pytorch=1.8.0`.

Activate the environment:

```sh 
conda activate neuman_env
```

### Demo

- Download SMPL weights:
  - Registration is required to download the UV map(Download UV map in OBJ format) from [SMPL](https://smpl.is.tue.mpg.de/download.php).
  - Download neutral SMPL weights(SMPLIFY_CODE_V2.ZIP) from [SMPLify](https://smplify.is.tue.mpg.de/download.php), extract `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` and rename it to `SMPL_NEUTRAL.pkl`.
  - Put the all the downloaded files into `./data/smplx` folder with following structure:
   ```bash
    .
    └── data
        └── smplx
            ├── smpl
            │   └── SMPL_NEUTRAL.pkl
            └── smpl_uv.obj
   ```

- Download NeuMan dataset and pretrained models:
  - Data ([download](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/dataset.zip))
  - Pretrained models ([download](https://docs-assets.developer.apple.com/ml-research/datasets/neuman/pretrained.zip))

  Alternately, run the following script to set up data and pretrained models.

  ```sh
  bash setup_data_and_models.sh
  ```
- (*Optional*) Download AMASS dataset for reposing:
  - AMASS dataset is used for rendering novel poses, specifically `render_reposing.py` and `render_gathering.py`.
  - We used SFU mocap(SMPL+H G) subset, please download from [AMASS](https://amass.is.tue.mpg.de/download.php).
  - Put the downloaded mocap data in to `./data/SFU` folder.
   ```bash
    .
    └── data
        └── SFU
            ├── 0005
            ├── 0007
            │   ...
            └── 0018
   ```

- Render using pretrained model

  Render 360 views of a canonical human:
      
  ```sh
  python render_360.py --scene_dir ./data/bike --weights_path ./out/bike_human/checkpoint.pth.tar --mode canonical_360
  ```
     
  Render 360 views of a posed human:
      
  ```sh
  python render_360.py --scene_dir ./data/bike --weights_path ./out/bike_human/checkpoint.pth.tar --mode posed_360
  ```

  Render test views of a sequence, and evaluate the metrics:
     
  ```sh
  python render_test_views.py --scene_dir ./data/bike --weights_path ./out/bike_human/checkpoint.pth.tar
  ```
      
  Render novel poses with the background:
      
  ```sh
  python render_reposing.py --scene_dir ./data/bike --weights_path ./out/bike_human/checkpoint.pth.tar --motion_name=jumpandroll
  ```
      
  Render telegathering:
      
  ```sh
  python render_gathering.py --actors parkinglot seattle citron --scene_dir ./data/seattle --weights_path ./out/seattle_human/checkpoint.pth.tar
  ```


### Training

- Download NeuMan dataset

- Train scene NeRF
```sh
python train.py --scene_dir ./data/bike/ --name=bike_background --train_mode=bkg
```

- Train human NeRF
```sh
python train.py --scene_dir ./data/bike  --name=bike_human --load_background=bike_background --train_mode=smpl_and_offset
```

### Use your own video

- Preprocess: Check [preprocess](./preprocess/README.md)

### Citation

```
@inproceedings{jiang2022neuman,
  title={NeuMan: Neural Human Radiance Field from a Single Video},
  author={Jiang, Wei and Yi, Kwang Moo and Samei, Golnoosh and Tuzel, Oncel and Ranjan, Anurag},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  year={2022}
}
```

### License

The code is released under the [LICENSE](./LICENSE) terms.

# Revisit Event Generation Model: Self-Supervised Learning of Event-to-Video Reconstruction with Implicit Neural Representations (ECCV2024)
### [Project Page](https://vlislab22.github.io/EvINR/) | [Paper](xxxx) | [Data](https://pan.baidu.com/s/1grYAM5GTq2mURvvUMBmaWg?pwd=2u2q)
[![Minimal EvINR in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pYlZ9UU1nsB1XIUc25jZD7z_97pGssQg?usp=sharing)<br>

[Zipeng Wang](https://scholar.google.com/citations?user=3w7X6NYAAAAJ),
[Yunfan Lu](https://scholar.google.com/citations?user=nPUR_0sAAAAJ),
[Addison Lin Wang](https://vlislab22.github.io/vlislab/linwang.html)<br>
The Hong Kong University of Science and Technology (Guangzhou).

This is the official implementation of the paper "Revisit Event Generation Model: Self-Supervised Learning of Event-to-Video Reconstruction with Implicit Neural Representations" .

## Abstract
![Framework](assets/framework.png)
*Reconstructing intensity frames from event data while maintaining high temporal resolution and dynamic range is crucial for bridging the gap between event-based and frame-based computer vision. 
Previous approaches have depended on supervised learning on synthetic data, which lacks interpretability and risk over-fitting to the setting of the event simulator. 
Recently, self-supervised learning (SSL) based methods, which primarily utilize per-frame optical flow to estimate intensity via photometric constancy,  has been actively investigated. However, they are vulnerable to errors in the case of inaccurate optical flow.
This paper proposes a novel SSL event-to-video reconstruction approach, dubbed EvINR, which eliminates the need for labeled data or optical flow estimation.
Our core idea is to reconstruct intensity frames by directly addressing the event generation model, essentially a partial differential equation (PDE) that describes how events are generated based on the time-varying brightness signals.
Specifically, we utilize an implicit neural representation (INR), which takes in spatiotemporal coordinate (x, y, t) and predicts intensity values, to represent the solution of the event generation equation. 
The INR, parameterized as a fully-connected Multi-layer Perceptron (MLP), can be optimized with its temporal derivatives supervised by events.
To make EvINR feasible for online requisites, we propose several acceleration techniques that substantially expedite the training process. 
Comprehensive experiments demonstrate that our EvINR surpasses previous SSL methods by 38% w.r.t. Mean Squared Error (MSE) and is comparable or superior to SoTA supervised methods.*

## Google Colab
<!-- If you want to do a quick test with EvINR, we have written a [Colab](https://colab.research.google.com/drive/1pYlZ9UU1nsB1XIUc25jZD7z_97pGssQg?usp=sharing) with minimal implementation, which can be viewed online (which means you don't have to install anything!) -->
If you're eager to give EvINR a spin without the hassle of installations, check out our [Colab notebook](https://colab.research.google.com/drive/1pYlZ9UU1nsB1XIUc25jZD7z_97pGssQg?usp=sharing) for a quick and easy test run. It's all set up for you to play around with online!

## Overview
This repository is organized as follows:

* event_data.py: Loads event data and stacks them into event frames.
* model.py: Contains our neural network solver for event-based video reconstruction.
* utils.py: Contains utility functions for event data manipulation and visualization.
* train.py: Contains the training routine.
* scripts/: Converts common event datasets into formats used in our work.
## AED Dataset
We release our ALPIX Event Dataset (AED) [here](https://pan.baidu.com/s/1grYAM5GTq2mURvvUMBmaWg?pwd=2u2q) (password: 2u2q).

## Data Preparation
We currently provide conversion scripts for the following datasets: [IJRR](https://rpg.ifi.uzh.ch/davis_data.html), [HQF](https://drive.google.com/drive/folders/18Xdr6pxJX0ZXTrXW9tK0hC3ZpmKDIt6_), and [CED](https://rpg.ifi.uzh.ch/CED.html). Our AED dataset does not require further conversion.

To process your own dataset, please convert the event data into a numpy array with the shape $[N \times 4]$, where N is the total number of events. The properties of each event should be $(t, x, y, p)$, where $p \in \{-1, 1\}$.

## Training
We provide the example commands to train EvINR on different dataset.

### IJRR and HQF
```
python train.py -n EXP_NAME -d DATA_PATH --H 240 --W 180
```

### CED
```
python train.py -n EXP_NAME -d DATA_PATH --H 260 --W 346 --color_event
```

### AED
```
python train.py -n EXP_NAME -d DATA_PATH --H 408 --W 306 --event_thresh 0.25
```

<!-- [![siren_video](https://img.youtube.com/vi/Q2fLWGBeaiI/0.jpg)](https://www.youtube.com/watch?v=Q2fLWGBeaiI)


## Google Colab
If you want to experiment with Siren, we have written a [Colab](https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb).
It's quite comprehensive and comes with a no-frills, drop-in implementation of SIREN. It doesn't require 
installing anything, and goes through the following experiments / SIREN properties:
* Fitting an image
* Fitting an audio signal
* Solving Poisson's equation
* Initialization scheme & distribution of activations
* Distribution of activations is shift-invariant
* Periodicity & behavior outside of the training range. 


## Get started
If you want to reproduce all the results (including the baselines) shown in the paper, the videos, point clouds, and 
audio files can be found [here](https://drive.google.com/drive/folders/1_iq__37--Hw7FJOEUK1tX7mdp8SKB368K?usp=sharing).

You can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate siren
```

## High-Level structure
The code is organized as follows:
* dataio.py loads training and testing data.
* training.py contains a generic training routine.
* modules.py contains layers and full neural network modules.
* meta_modules.py contains hypernetwork code.
* utils.py contains utility functions, most promintently related to the writing of Tensorboard summaries.
* diff_operators.py contains implementations of differential operators.
* loss_functions.py contains loss functions for the different experiments.
* make_figures.py contains helper functions to create the convergence videos shown in the video.
* ./experiment_scripts/ contains scripts to reproduce experiments in the paper.

## Reproducing experiments
The directory `experiment_scripts` contains one script per experiment in the paper.

To monitor progress, the training code writes tensorboard summaries into a "summaries"" subdirectory in the logging_root.

### Image experiments
The image experiment can be reproduced with
```
python experiment_scripts/train_img.py --model_type=sine
```
The figures in the paper were made by extracting images from the tensorboard summaries. Example code how to do this can
be found in the make_figures.py script.

### Audio experiments
This github repository comes with both the "counting" and "bach" audio clips under ./data.

They can be trained with
```
python experiment_scipts/train_audio.py --model_type=sine ---Wav_path=<path_to_audio_file>
```

### Video experiments
The "bikes" video sequence comes with scikit-video and need not be downloaded. The cat video can be downloaded with the
link above.

To fit a model to a video, run
```
python experiment_scipts/train_video.py --model_type=sine --experiment_name bikes_video
```

### Poisson experiments
For the poisson experiments, there are three separate scripts: One for reconstructing an image from its gradients 
(train_poisson_grad_img.py), from its laplacian (train_poisson_lapl_image.py), and to combine two images 
(train_poisson_gradcomp_img.py).

Some of the experiments were run using the BSD500 datast, which you can download [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).

### SDF Experiments
To fit a Signed Distance Function (SDF) with SIREN, you first need a pointcloud in .xyz format that includes surface normals.
If you only have a mesh / ply file, this can be accomplished with the open-source tool Meshlab.

To reproduce our results, we provide both models of the Thai Statue from the 3D Stanford model repository and the living room used in our paper
for download here.

To start training a SIREN, run:
```
python experiments_scripts/train_single_sdf.py --model_type=sine --point_cloud_path=<path_to_the_model_in_xyz_format> --batch_size=250000 --experiment_name=experiment_1
```
This will regularly save checkpoints in the directory specified by the rootpath in the script, in a subdirectory "experiment_1". 
The batch_size is typically adjusted to fit in the entire memory of your GPU. 
Our experiments show that with a 256, 3 hidden layer SIREN one can set the batch size between 230-250'000 for a NVidia GPU with 12GB memory.

To inspect a SDF fitted to a 3D point cloud, we now need to create a mesh from the zero-level set of the SDF. 
This is performed with another script that uses a marching cubes algorithm (adapted from the DeepSDF github repo) 
and creates the mesh saved in a .ply file format. It can be called with:
```
python experiments_scripts/test_single_sdf.py --checkpoint_path=<path_to_the_checkpoint_of_the_trained_model> --experiment_name=experiment_1_rec 
```
This will save the .ply file as "reconstruction.ply" in "experiment_1_rec" (be patient, the marching cube meshing step takes some time ;) )
In the event the machine you use for the reconstruction does not have enough RAM, running test_sdf script will likely freeze. If this is the case, 
please use the option --resolution=512 in the command line above (set to 1600 by default) that will reconstruct the mesh at a lower spatial resolution.

The .ply file can be visualized using a software such as [Meshlab](https://www.meshlab.net/#download) (a cross-platform visualizer and editor for 3D models).

### Helmholtz and wave equation experiments
The helmholtz and wave equation experiments can be reproduced with the train_wave_equation.py and train_helmholtz.py scripts.

## Torchmeta
We're using the excellent [torchmeta](https://github.com/tristandeleu/pytorch-meta) to implement hypernetworks. We 
realized that there is a technical report, which we forgot to cite - it'll make it into the camera-ready version! -->

## Citation
If you find our work useful in your research, please cite:
```
```

## Contact
If you have any questions, please feel free to email the authors or raise an issue.

## Acknowledgments
Our code follows the awesome [Siren](https://github.com/vsitzmann/siren/) repository. We thanks them for the inspiring work.
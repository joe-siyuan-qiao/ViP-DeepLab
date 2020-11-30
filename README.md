# ViP-DeepLab

## Introduction

In this repository, we present the datasets and the toolkits of [ViP-DeepLab](arxiv).
ViP-DeepLab is a unified model attempting to tackle the long-standing and challenging inverse projection problem in vision, which we model as restoring the point clouds from perspective image sequences while providing each point with instance-level semantic interpretations.
Solving this problem requires the vision models to predict the spatial location, semantic class,
and temporally consistent instance label for each 3D point.
ViP-DeepLab approaches it by jointly performing monocular depth estimation and video panoptic segmentation.
We name this joint task as Depth-aware Video Panoptic Segmentation, and propose a new evaluation metric along with two derived datasets for it.
This repository includes the datasets and the toolkits.

[![Demo](readme_srcs/ViP-DeepLab.gif)](www.cs.jhu.edu/~syqiao/ViP-DeepLab/ViP-DeepLab_v3.mp4)

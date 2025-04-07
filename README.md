# UpmixAI: Automatic Blind Stereo-to-Surround Upmixing Using Music Source Separation Deep Neural Networks
## Abstract
Blind stereo-to-surround upmixing aims to convert stereo audio into a multichannel surround mix without access to original multitrack recordings. Traditional upmixing approaches rely on Primary-Ambient Extraction (PAE) using digital signal processing (DSP) techniques, which often suffer from limited generalization and spatial artifacts. This work proposes a deep learning-based alternative, integrating high-complexity state-of-the-art (SOTA) music source separation (MSS) models to enhance ambient signal extraction and stereo decorrelation.

We evaluate the efficacy of SOTA MSS architectures for separating primary and ambient components for improved spatial and spectral coherence. The selected deep neural network (DNN) architecture is trained on a custom dataset constructed from open-source music data and assessed using both objective SDR and SI-SDR metrics and a controlled participant listening study.

Experimental results demonstrate that MSS models can be successfully adapted for PAE, outperforming prior deep learning approaches in the literature and matching the performance of traditional DSP-based upmixing methods. These findings validate the proposed data-driven framework as a viable and perceptually effective solution for automatic blind stereo-to-surround upmixing that utilizes deep learning throughout the entire processing chain.

---

## Contents
This repository contains all the code, data, and assets used in the experiments:

- **Model Architectures**  
  PyTorch implementations of:
  - [Mel-RoFormer](https://ieeexplore.ieee.org/abstract/document/10446843) (Lu et al., 2024)
  - [MLP](https://ieeexplore.ieee.org/abstract/document/8461459) (Ibrahim et al., 2018)

- **Data Analysis Scripts**  
  Scripts for:
  - Objective metric evaluation (e.g., SDR, SI-SDR, LSD)
  - Subjective listening test analysis

- **Raw Evaluation Data**  
  Includes:
  - Metric scores for all systems
  - Participant responses from listening tests
  - Summary statistics and visualizations

---

## Audio Examples & Model Checkpoints

All pretrained weights, training datasets, and audio examples are available on the [Hugging Face](https://huggingface.co/nick7ong/).

Contents:
- Model weights for all trained models
- Processed training dataset (PAEDB)
- Test audio examples for all evaluation




# 
<p align="center">
  <h1 align="center"> <ins>CSBSR</ins>:<br> Joint Learning of Blind Super-Resolution and
Crack Segmentation for Realistic Degraded Images</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=lijAs5AAAAAJ">Yuki Kondo</a>
    ·
    <a href="https://scholar.google.com/citations?user=Tgbsbs8AAAAJ">Norimichi Ukita</a>
  </p>
  <h2 align="center"><p>
    <a href="https://arxiv.org/abs/2302.12491" align="center">Paper</a> | 
    <a href="https://yuki-11.github.io/CSBSR" align="center">Project Page (Coming soon...)</a>
  </p></h2>
  <div align="center"></div>
</p>
<br/>
<p align="center">
    <img src="https://raw.githubusercontent.com/Yuki-11/CSBSR/main/fig/result_csbsr_github.png" alt="example" width=80%>
    <br>
    <em>Crack segmentation challenges for synthetically-degraded images given by low resolution and anisotropic Gaussian blur. Our method (f) CSBSR succeeds in detecting cracks in the most detail</em>
</p>

## Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## About the Project
This paper proposes crack segmentation augmented by super resolution (SR) with deep neural networks. In the proposed method, a SR network is jointly trained with a binary segmentation network in an end-to-end manner. This joint learning allows the SR network to be optimized for improving segmentation results. For realistic scenarios, the SR network is extended from non-blind to blind for processing a low-resolution image degraded by unknown blurs. The joint network is improved by our proposed two extra paths that further encourage the mutual optimization between SR and segmentation. Comparative experiments with State-of-the-Art segmentation methods demonstrate the superiority of our joint learning, and various ablation studies prove the effects of our contributions.

## Getting Starte
In your Python environment, run:
```bash
pip install -r requirement.txt
```
🚧 **Code will be released soon! Stay tuned.** 🚧

## Usage
🚧 **Code will be released soon! Stay tuned.** 🚧

## License

Distributed under the Apache-2.0 license License. See `LICENSE` for more information.


## BibTeX
If you find our models useful, please consider citing our paper!
```
@article{kondo2023csbsr,
  title={Joint Learning of Blind Super-Resolution and Crack Segmentation for Realistic Degraded Images},
  author={Kondo, Yuki and Ukita, Norimichi},
  journal={arXiv preprint arXiv:2302.12491},
  year={2023}
}
```

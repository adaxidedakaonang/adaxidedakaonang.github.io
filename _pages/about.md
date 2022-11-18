---
permalink: /
title: "Ciao! I'm 刘(Liu) 畅(Chang), from XDU, China!"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Ciao! I'm a 2nd-year PhD student in school of AI at Xidian University, Xian, China. My supervisor is [Prof. Niu Yi](https://web.xidian.edu.cn/niuyi/index.html). My current research interest is continual learning in semantic segmentation. Now I'm a visiting student of [LTTM](https://web.xidian.edu.cn/niuyi/index.html) at [UNIPD](https://www.unipd.it/) supported by [China scholarship Council (CSC)](https://www.chinesescholarshipcouncil.com/), working with [Prof. Pietro Zanuttigh](https://lttm.dei.unipd.it/nuovo/staff/zanuttigh.html). 

I was a Master student (2018.9-2021.3) educated at the Xidian University. I graduated with a bachelor's degree in School of Electronic Engineering from Xidian University, China. I'm a reviewer of [Displays](https://www.sciencedirect.com/journal/displays).


Publications
======
### NL-CALIC Soft Decoding Using Strict Constrained Wide-Activated Recurrent Residual Network
In this work, we propose a normalized Tanh activate strategy and a lightweight wide-activate recurrent structure to solve three key challenges of the soft-decoding of near-lossless codes: 1. How to add an effective strict constrained peak absolute error (PAE) boundary to the network; 2. An end-to-end solution that is suitable for different quantization steps (compression ratios). 3. Simple structure that favors the GPU and FPGA implementation. To this end, we propose a Wide-activated Recurrent structure with a normalized Tanh activate strategy for Soft-Decoding (WRSD). Experiments demonstrate the effectiveness of the proposed WRSD technique that WRSD outperforms better than the state-of-the-art soft decoders with less than 5% number of parameters, and every computation node of WRSD requires less than 64KB storage for the parameters which can be easily cached by most of the current consumer-level GPUs.

[paper here](https://ieeexplore.ieee.org/abstract/document/9662665)

[code here](https://github.com/dota-109/WRSD)

![images](/images/papers/WRSD/wrsd1.png)

### Retinex-guided Channel-grouping based Patch Swap for Arbitrary Style Transfer (under review)
The basic principle of the patch-matching based style transfer is to substitute the patches of the content image feature maps by the closest patches from the style image feature maps. Since the finite features harvested from one single aesthetic style image are inadequate to represent the rich textures of the content natural image, existing techniques treat the full-channel style feature patches as simple signal tensors and create new
style feature patches via signal-level fusion. In this paper, we propose a Retinex theory guided, channel-grouping based patch swap technique to group the style feature maps into surface and texture channels, and the new features are created by the combination of these two groups, which can be regarded as a semantic-level fusion of the raw style features. In addition, we
provide complementary fusion and multi-scale generation strategy to prevent unexpected black area and over-stylised results respectively. Experimental results demonstrate that the proposed method outperforms the existing techniques in providing more style-consistent textures while keeping the content fidelity.

[conference verison here](https://ieeexplore.ieee.org/abstract/document/9190962)

![images](/images/papers/ST/style_transfer.png)
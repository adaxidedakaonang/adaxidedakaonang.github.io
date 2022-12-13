---
title: "NL-CALIC Soft Decoding Using Strict Constrained Wide-Activated Recurrent Residual Network"
collection: publications
permalink: /publication/2009-10-01-paper-title-number-1
excerpt: 'This paper is about the number 1. The number 2 is left for future work.'
date: 2021-12-01
venue: 'Journal 1'
paperurl: 'http://adaxidedakaonang.github.io/files/WRSD.pdf'
citation: '@article{niu2021nl,
  title={NL-CALIC Soft Decoding Using Strict Constrained Wide-Activated Recurrent Residual Network},
  author={Niu, Yi and Liu, Chang and Ma, Mingming and Li, Fu and Chen, Zhiwen and Shi, Guangming},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={1243--1257},
  year={2021},
  publisher={IEEE}
}'
---
In this work, we propose a normalized Tanh activate strategy and a lightweight wide-activate recurrent structure to solve three key challenges of the soft-decoding of near-lossless codes: 1. How to add an effective strict constrained peak absolute error (PAE) boundary to the network; 2. An end-to-end solution that is suitable for different quantization steps (compression ratios). 3. Simple structure that favors the GPU and FPGA implementation. To this end, we propose a Wide-activated Recurrent structure with a normalized Tanh activate strategy for Soft-Decoding (WRSD). Experiments demonstrate the effectiveness of the proposed WRSD technique that WRSD outperforms better than the state-of-the-art soft decoders with less than 5% number of parameters, and every computation node of WRSD requires less than 64KB storage for the parameters which can be easily cached by most of the current consumer-level GPUs.

[Download paper here](https://ieeexplore.ieee.org/abstract/document/9662665)

[code here](https://github.com/dota-109/WRSD)


![images](/images/papers/WRSD/wrsd1.png)
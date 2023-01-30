---
title: "NL-CALIC Soft Decoding Using Strict Constrained Wide-Activated Recurrent Residual Network"
collection: publications
permalink: /publication/2009-10-01-paper-title-number-1
date: 2021-12-01
venue: 'IEEE Transactions on Image Processing'
paperurl: 'http://adaxidedakaonang.github.io/files/WRSD.pdf'
citation: 'Niu Y, Liu C, Ma M, et al. NL-CALIC Soft Decoding Using Strict Constrained Wide-Activated Recurrent Residual Network[J]. IEEE Transactions on Image Processing, 2021, 31: 1243-1257.'
---
In this work, we propose a normalized Tanh activate strategy and a lightweight wide-activate recurrent structure to solve three key challenges of the soft-decoding of near-lossless codes: 1. How to add an effective strict constrained peak absolute error (PAE) boundary to the network; 2. An end-to-end solution that is suitable for different quantization steps (compression ratios). 3. Simple structure that favors the GPU and FPGA implementation. To this end, we propose a Wide-activated Recurrent structure with a normalized Tanh activate strategy for Soft-Decoding (WRSD). Experiments demonstrate the effectiveness of the proposed WRSD technique that WRSD outperforms better than the state-of-the-art soft decoders with less than 5% number of parameters, and every computation node of WRSD requires less than 64KB storage for the parameters which can be easily cached by most of the current consumer-level GPUs.

[Download paper here](https://ieeexplore.ieee.org/abstract/document/9662665)

[code here](https://github.com/dota-109/WRSD)

## Performance
![images](/images/papers/WRSD/wrsd1.png)

## Structure
![images](/images/papers/WRSD/nta.png)
![images](/images/papers/WRSD/wrsd_structure.png)
## quality
![images](/images/papers/WRSD/performance.png)
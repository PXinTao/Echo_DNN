# Weights README
You can download the weights through two channels:
- Huggingface Models: https://huggingface.co/diyer22/ddn_asset/tree/main  
- HTTP server: http://113.44.140.251:9000/ddn/weights/

The weight provided:  
1. `Recommended File Name`: Description
    - Download CMD
2. `cifar-ddn.pkl`: unconditional training of CIFAR-10, with FID score 52
    - `wget -O weights/cifar-ddn.pkl http://113.44.140.251:9000/ddn/weights/v15-00035-cifar10-32x32-cifar_blockn32_outputk64_chain.dropout0.05_fp32_goon.v15.22-shot-087808.pkl`
3. `ffhq-64x64-ddn.pkl`: unconditional training of FFHQ-64x64
    - `wget -O weights/ffhq-64x64-ddn.pkl http://113.44.140.251:9000/ddn/weights/v15-00018-ffhq-64x64-blockn64_outputk512_chain.dropout0.05-shot-117913.pkl`
4. `ffhq-256x256-coloring-ddn.pkl`: conditional training of FFHQ-256x256, the conditon is grayscale image with random weights of RGB channel
    - `wget -O weights/ffhq-256x256-coloring-ddn.pkl http://113.44.140.251:9000/ddn/weights/v32-00003-ffhq-256x256-ffhq256_cond.color_chain.dropout0.05_batch64_k64-shot-200000.pkl`

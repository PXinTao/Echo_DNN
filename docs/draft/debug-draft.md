```bash
# docker debug on 3060
docker run -it --gpus all --net=host -v `pwd`:/workspace --user $(id -u):$(id -g) -e http_proxy -e https_proxy diyer22/ddn bash

python train.py --data datasets/cifar10-32x32.zip --fp16=1 --outdir=training-runs --batch-gpu=8 --batch=16 --desc=cifar_fp16_blockn32_outputk64_chain.dropout0.05 --chain-dropout 0.05 --max-blockn=32 --max-outputk 64 --duration 0.00001


python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```


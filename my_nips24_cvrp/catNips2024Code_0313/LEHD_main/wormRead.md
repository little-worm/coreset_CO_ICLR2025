# worm对LEHD的复现

对应的论文是：

```
Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization
（Nips2023）
```

---



## 原始的Dependencies

```bash
Python=3.8.6
torch==1.12.1
numpy==1.23.3
matplotlib==3.5.2
tqdm==4.64.1
pytz==2022.1
```



## Worm Installation

- 手动安装

```
conda create --name worm python=3.9
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
pip install pytz
pip install tqdm
```

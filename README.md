---
title: License Plate Detection
emoji: 😻
colorFrom: purple
colorTo: purple
sdk: docker
pinned: false
license: cc-by-nc-2.0
app_port: 7860
---

### Installation
Developed and tested on `python==3.8.10`

It is recommended to create a virtual enviroment by
```
python3 -m venv .env
source .env/bin/activate
```
1. Install `pytorch` and `torchvision` by
```bash
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```
2. Install requirements by
```bash
pip3 install -r requirements.txt
pip3 install cython-bbox
```
3. Download weights from `s3`.

### Run from terminal
```bash
 python main.py data/test_images
```
- `sample_imgs` is folder containing images

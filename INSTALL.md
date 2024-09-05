## Installation
The codebases are built on top of [Detectron2](https://detectron2.readthedocs.io/tutorials/install.html).


### Dependencies and Installation
```bash
conda create --name frozenseg python=3.10 -y
conda activate frozenseg
conda install pytorch==2.3.1 torchvision==0.18.1 -c pytorch -c nvidia

# under your working directory
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

git clone https://github.com/chenxi52/FrozenSeg.git
cd FrozenSeg
pip install -r requirements.txt

# compile CUDA kernel for MSDeformAttn
cd frozenseg/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```
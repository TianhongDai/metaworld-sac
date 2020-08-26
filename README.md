# Meta World Single Tasks

## Requirement
- torch == 1.2.0
- metaworld

## Install Metaworld (use old version!)
```
https://github.com/rlworkgroup/metaworld.git
cd metaworld/
git checkout 2361d353d0895d5908156aec71341d4ad09dd3c2
pip install -e .
```

## Demo
Download pretrained models from [here](https://drive.google.com/file/d/19zdmws5rFrH_2KjAl4GnwrtpeBxgwPIG/view?usp=sharing).
```
python demo.py --env-name='pick_place' --render
```
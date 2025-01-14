# Smart-Model-then-Control

_This work innovates traditional model predictive control for the scheduling of thermostatically controlled loads. Inspired by "Smart Predict-then-Optimize", a "Smart Model-then-Control" strategy is proposed to learn a cost-effective model for the downstream control task. The actual control costs are reduced in multiple building types._

Codes for submitted Paper "A Smart Model-then-Control Strategy for the Scheduling of Thermostatically Controlled Loads".

Authors: Xueyuan Cui, Boyuan Liu, Yehui Li, and Yi Wang.

## Requirements
Python version: 3.8.17

The must-have packages can be installed by running
```
pip install requirements.txt
```

## Experiments
### Data
All the data for experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1U4RE0EGJgCrL_LJvFmMf_LiXID7o4P38?usp=sharing).

### Reproduction
To reproduce the experiments of the proposed methods and comparisons ('Lat_MB', 'Lat_MF', 'Ori_MB', and 'Ori_MF'), please run
```
cd Codes/
python Lat_MB.py
python Lat_MF.py
python Ori_MB.py
python Ori_MF.py
```
To reproduce the experiments of generating latent and original models, please run
```
cd Codes/
python Lat_model.py
python Ori_model.py
```
To reproduce the experiments of ground-truth results, please run
```
cd Codes/
python Ground_truth.py
```
Note: There is NO multi-GPU/parallelling training in our codes. 


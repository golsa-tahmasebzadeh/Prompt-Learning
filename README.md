## Prompt Learning for Image Classification
This repository contains scripts for prompt learning based on the Context Optimization (CoOp) technique.
## Installation
``` bash
git clone https://github.com/golsa-tahmasebzadeh/prompt_learning.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Download Data
Download the data from [here](https://drive.google.com/file/d/1m3RU_epZXiEi2n1ByGMBnlvjI1vT5j8m/view?usp=sharing) and put in the root directory.

## Training 
``` bash
cd sbatch
sbatch train_coop.sh
```

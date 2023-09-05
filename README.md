## Prompt Learning for Image Classification
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
Change the virtual environment path in the sbatch file.
``` bash
cd sbatch
sbatch train_coop.sh
```

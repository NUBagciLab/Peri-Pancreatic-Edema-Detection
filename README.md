# Detection of Peri-Pancreatic Edema using Deep Learning and Radiomics Techniques

## Abstract
![Workflow](https://github.com/NUBagciLab/Peri-Pancreatic-Edema-Detection/blob/main/Fig1.jpg)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.
```bash
pip install -r requirements.txt
```

## Dataset
The dataset used in this project has the following structure.
```bash
-- data
   |-- raw
   |   |-- raw
   |   |-- preprocessed
   |-- ROI
   |   |-- PeripancreaticEdema
   |   |-- Pancreas
   |-- ROI_slices
   |   |-- PeripancreaticEdema
   |   |-- Pancreas
```

## Source code changes
Four models was redefined in the pytorch source code. Modifyied source code is also included.
These code should be put as the following path:
```bash
path/to/your/torchvision/model/redefined_model.py
```
check these models ![here](https://github.com/NUBagciLab/Peri-Pancreatic-Edema-Detection/tree/main/models_2d/changed%20source%20code)
## Citations

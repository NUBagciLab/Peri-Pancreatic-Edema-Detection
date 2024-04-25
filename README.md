# Detection of Peri-Pancreatic Edema using Deep Learning and Radiomics Techniques

## Abstract
![Workflow](https://github.com/NUBagciLab/Peri-Pancreatic-Edema-Detection/blob/main/Fig1.jpg)

This study introduces a novel CT dataset sourced from 255 patients with pancreatic diseases, featuring annotated pancreas segmentation masks and corresponding diagnostic labels for peri-pancreatic edema condition.  With the novel dataset, we first evaluate the efficacy of the LinTransUNet model, a linear Transformer based segmentation algorithm, to segment the pancreas accurately from CT imaging data. Then, we use segmented pancreas regions with two distinctive machine learning classifiers to identify existence of peri-pancreatic edema: deep learning-based models and a radiomics-based eXtreme Gradient Boosting (XGBoost). The LinTransUNet achieved promising results, with a Dice coefficient of 80.85%, and mIoU of 68.73%. Among the nine benchmarked classification models for peri-pancreatic edema detection, Swin-Tiny transformer model demonstrated the highest recall of 98.85% and precision of 98.38%. Comparatively, the radiomics-based XGBoost model achieved an accuracy of 79.61% and recall of 91.05%, showcasing its potential as a supplementary diagnostic tool given its rapid processing speed and reduced training time. 
To our knowledge, this is the first study aiming to detect peri-pancreatic edema automatically. We propose to use modern deep learning architectures and radiomics together and created a benchmarking for the first time for this particular problem, impacting clinical evaluation of pancreatitis, specifically detecting peri-pancreatic edema.


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
check these models [here](https://github.com/NUBagciLab/Peri-Pancreatic-Edema-Detection/tree/main/models_2d/changed%20source%20code)
## Citations

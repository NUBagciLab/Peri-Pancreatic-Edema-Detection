# Detection of Peri-Pancreatic Edema using Deep Learning and Radiomics Techniques



**Portals:**

[![Dataset]](https://osf.io/wyth7/.)
[![Paper]](https://arxiv.org/abs/2404.17064#:~:text=Detection%20of%20Peri%2DPancreatic%20Edema%20using%20Deep%20Learning%20and%20Radiomics%20Techniques,-Ziliang%20Hong%2C%20Debesh&text=Identifying%20peri%2Dpancreatic%20edema%20is,in%20pancreatitis%20diagnosis%20and%20management.)
[![EMBC]](https://embc.embs.org/2024/)
</div>

**Authors:**

-Ziliang Hong<sup>3</sup>, Debesh Jha1<sup>1</sup>, Koushik Biswas<sup>1</sup>, Zheyuan Zhang<sup>1</sup>, Yury Velichko<sup>1</sup>, Cemal Yazici<sup>3</sup>, Temel Tirkes<sup>4</sup>,
Amir Borhani<sup>1</sup>, Baris Turkbey<sup>2</sup>, Alpay Medetalibeyoglu<sup>1</sup>, Gorkem Durak<sup>1</sup>, Ulas Bagci<sup>1</sup>


**Affiliations:** 

<sup>1</sup> Machine & Hybrid Intelligence Lab, Northwestern University

<sup>2</sup> National Cancer Institute

<sup>3</sup> University of Illionis at Chicago  

<sup>4</sup> Indiana University  

## Abstract
![Workflow](https://github.com/NUBagciLab/Peri-Pancreatic-Edema-Detection/blob/main/Fig1.jpg)

This study introduces a novel CT dataset sourced from 255 patients with pancreatic diseases, featuring annotated pancreas segmentation masks and corresponding diagnostic labels for peri-pancreatic edema condition.  With the novel dataset, we first evaluate the efficacy of the LinTransUNet model, a linear Transformer based segmentation algorithm, to segment the pancreas accurately from CT imaging data. Then, we use segmented pancreas regions with two distinctive machine learning classifiers to identify existence of peri-pancreatic edema: deep learning-based models and a radiomics-based eXtreme Gradient Boosting (XGBoost). 

The LinTransUNet achieved promising results, with a Dice coefficient of 80.85%, and mIoU of 68.73%. Among the nine benchmarked classification models for peri-pancreatic edema detection, Swin-Tiny transformer model demonstrated the highest recall of 98.85% and precision of 98.38%. Comparatively, the radiomics-based XGBoost model achieved an accuracy of 79.61% and recall of 91.05%, showcasing its potential as a supplementary diagnostic tool given its rapid processing speed and reduced training time. 

To our knowledge, this is the first study aiming to detect peri-pancreatic edema automatically. We propose to use modern deep learning architectures and radiomics together and created a benchmarking for the first time for this particular problem, impacting clinical evaluation of pancreatitis, specifically detecting peri-pancreatic edema.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.
```bash
pip install -r requirements.txt
```

## Dataset
The trainable data used in this project has the following structure. ROI are sourced form peri-pancreatic image and segmentation mask.
```bash
-- data
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
```bash
@article{hong2024detection,
  title={Detection of Peri-Pancreatic Edema using Deep Learning and Radiomics Techniques},
  author={Hong, Ziliang and Jha, Debesh and Biswas, Koushik and Zhang, Zheyuan and Velichko, Yury and Yazici, Cemal and Tirkes, Temel and Borhani, Amir and Turkbey, Baris and Medetalibeyoglu, Alpay and others},
  journal={arXiv preprint arXiv:2404.17064},
  year={2024}
}
```


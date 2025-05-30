# PTMFusionNet

This is the code repository of the paper **PTMFusionNet: A Deep Learning Approach for Predicting Disease Related Post-Translational Modification and Classifying Disease Subtypes**.

## Introduction
![image](https://github.com/Jie-Ni/PTMFusionNet/raw/main/Flowchart%20of%20PTMFusionNet.gif)

With the advancement of technologies such as mass spectrometry, it has become possible to simultaneously perform large-scale detection of protein intensity and corresponding post-translational modification (PTM) information, thereby facilitating clinical diagnosis and treatment. However, existing PTM information is insufficient to fully integrate with protein expression data. We propose a deep learning method called PTMFusionNet, which predicts potential disease-related PTMs and integrates them with protein expression data to classify disease subtypes. PTMFusionNet includes two Graph Convolutional Network (GCN) models: the Layer Attention Graph Convolutional Network (LAGCN) and the Feature Weighting Graph Convolutional Network (FWGCN). LAGCN is used to predict PTM potentiality scores, while FWGCN integrates these scores with protein expression data for disease subtype classification. Experimental results across three datasets (KIPAN, COADREAD, and THCA) demonstrate that PTMFusionNet outperforms benchmark algorithms in accuracy, F1 score, and AUC, highlighting its robustness in identifying critical PTM biomarkers and advancing disease subtyping.

## Getting Started
To get start with PTMFusionNet, please follow the instructions below.

### Clone the repository
```bash
git clone git@github.com:Jie-Ni/PTMFusionNet.git
```

### Prerequisites
```bash
pip install -r requirements.txt
```

### To run the main script of disease subtypes classification
```bash
python Main_Important\ PTM\ biomarker\ identification.py
```

### To run the main script of important PTM biomarker identification
```bash
python Main_Important\ PTM\ biomarker\ identification.py
```

## Contact
If you have any questions, please feel free to get touch with me, my email is njie AT seu DOT edu DOT com

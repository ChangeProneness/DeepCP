# DeepCP

This repository contains the source code and datasets of the paper with the title **Predicting Change-Proneness with a Deep Learning Model: Incorporating Structural Dependencies**.

## Abstract
Change-proneness prediction aims to identify source files that deserve developers' attention for early quality improvement, ultimately reducing future maintenance costs. Existing studies have utilized diverse change features and machine learning techniques to construct effective prediction models. However, these approaches do not leverage the grouping of change features for the learning of feature representations; and the roles of dependency files in influencing the change-proneness of the target file are not adequately exploited. To this end, in this paper, we propose a novel approach called *DeepCP*, which utilizes a deep learning model to predict the change-proneness of source files. The key rationale of DeepCP is that the grouping of change features should be considered when constructing deep learning models, given that feeding the sequence of features directly into models has a detrimental impact on the learning of feature representations. To achieve this, DeepCP utilizes different subnetworks to learn different categories of features and merges the resulting representations with a multi-head attention network to generate the source file representation. Another key rationale of DeepCP is that the features of dependency files should be exploited because the change-proneness of dependency files significantly influences that of the target file (ripple effects). To accomplish this, DeepCP first learns the representations of the target file and its structurally dependent files and then integrates the resulting representations with an attention network. Evaluation results demonstrate that DeepCP outperforms the state-of-the-art approach in predicting change-proneness, and structural dependencies can further enhance the performance of DeepCP.

## File structure
This repository is structured as follows:

- ConstructDataset
  > source code for constructing training datasets, including the computation of metrics and dependencies
- Model
  - TrainTest.py
    > source code for training and evaluating the DeepCP model 
  - baseline
    > source code for the baseline approach
  - models
    > source code for the DeepCP model
- Evaluation
  - Dataset
    - CommitList
      > commit sequences of the projects in our evaluation
    - datasets.zip
      > datasets constructed form the projects in our evaluation
  - GenerateTableFigure
      > source code for generating the tables and figures in the section of evaluation
  - Results
      > evaluation results of DeepCP and CNN

## How to use DeepCP to predict the change-proneness of your source files?
1. Construct a training dataset using the code in the directory of `ConstructDataset`
2. Train a model using the code in the source file of `Model/TrainTest.py`
3. Compute features of the target source file using the code in `ConstructDataset`
4. Predict the change-proneness of the target file using the trained model

# DeepCP

## File structure
- ConstructDataset
  > source code for computing metrics and dependencies
- Evaluation
  - Dataset
    - CommitList
      > commit sequences of the projects in our evaluation
    - datasets.zip
      > datasets constructed form the projects in our evaluation
  - GenerateTableFigure
      > source code for generating the tables and figures in the section of evaluation
- Model
  - TrainTest.py
    > source code for train and evaluate the DeepCP model 
  - baseline
    > source code for the baseline approach
  - models
    > source code for the DeepCP model

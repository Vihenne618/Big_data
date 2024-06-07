## Group_Project_PB01:  **Predicting Contact Maps in Bioinformatics**

Contact map prediction is a bioinformatics problem, and more specifically a protein structure prediction classification task. The ECBLD’14 Big Data competition provided a highly imbalanced dataset to solve this problem. 

This project implements a local method based on gradient boosting tree based on Apache Spark, which improves a new scheme for big data processing. Since the original data for this project was an extremely unbalanced binary categorical data, an undersampling method was used and chunks of loaded data were used to train the model. Meanwhile, in the model training module, the local method is implemented in Spark's Pipeline based on the gradient boosting tree that. Finally, the results of the projections are evaluated using the evaluation parameters.

### Getting started

1. Place the training data and test data files into the data folder(./data).
2. Execute the mian.py file

### Project structure

```
|-data					:Data Storage Directory
|	|--models				:Training models storage directory
|	|--result				:Prediction results storage directory
|	|--testdata.csv			:Partial test data
|	|--traindata.csv		:Partial training data
|-data_processing		:Data loading and processing module
|	|--data_process.py		:executable programme
|	|--data_process.ipynb	:Development documentation
|	|--README.md
|-evaluating			:Prediction and Evaluation Module
|	|--evaluation.py		:executable programme
|	|--evaluation.ipynb		:Development documentation
|	|--README.md
|-model_training		:Model Training Module
|	|--model_train.py		:executable programme
|	|--model_train.ipynb	:Development documentation
|	|--README.md:
|-mian.py				:Main Program Entry
|-mian_chunk.py			:Program entry(chunked reads)
|-README.md
|-readme_img			:Illustrate the picture of the document
```

1. There are two programme entrances to this project:
   - main.py is the main entry point of the program and uses the data in the data directory for training and prediction.
   - main.py strings together all the py files. Implement the total functionality.
   - mian_chunk.py is the chunk execution portal, which mainly implements chunk loading data from the original data source (arff file) to train the model and make predictions.
2. There are two files (.py & .ipynb) in the catalogue for data processing, model training, and evaluation.
   - .py files are executable programs that are primarily responsible for abstracting functional code into interfaces.
   - .ipynb file is an instruction file for the programme development process, presenting the functionality in a step-by-step manner.

### Rreferences

- http://cruncher.ico2s.org/bdcomp/
- https://github.com/triguero/ROSEFW-RF
- https://www.sciencedirect.com/science/article/abs/pii/S0950705115002130
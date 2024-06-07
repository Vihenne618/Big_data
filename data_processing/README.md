# Group_Project_PB01



## Data Extraction 

- In this step, I extracted 20,000 pieces of data from the training_set.arff file, and the ratio of positive and negative cases was 1 to 1. And 2000 pieces of data were extracted from the test_set.arff file with a 1:1 ratio of positive and negative. The subdata sets used are downsampled balanced data sets and stored in csv files. The code, written in arfftocsv.ipynb, reextracts the desired size of the subdataset by changing the value of max_num

## Data Processing

- In this step, I use pyspark to read the data and store it in the dataframe. For class attributes, I use sequential encoding to re-encode the class attributes into integers. For numerical attributes, I do min-max normalization of their values

## Data Chunk Load

In the mian_chunk.py file, the source data file is read, because the data file is too large, so this project samples the data loaded in chunks: 

- The training data is loaded in chunks and one chunk at a time is loaded to train to generate several gradient boosting trees.
- Save the trained tree to the directory of "./data/models".
- Predictive data is loaded in chunks, and each chunk is predicted with all the models, and voting is used to arrive at a prediction.
- Store the prediction results to a file(./data/result) and perform the next chunked prediction iteration.
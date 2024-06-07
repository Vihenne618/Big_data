
import data_processing.data_process as data_process
import model_training.model_train as model_train
from model_training.model_train import LocalGBTClassifierModel
from pyspark.sql import SparkSession
import evaluating.evaluation as evaluation
import time
import gc
import os

# 1. initialize the SparkSession
spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("PB01_Spark_App") \
    .config("spark.default.parallelism", "5") \
    .config("spark.executor.memory","16g") \
    .getOrCreate()
# data path
train_data_path = "./data/training_set.arff"
test_data_path = "./data/test_set.arff"
# model storage path
model_paths = "./data/models/local_models.pkl"
# predict data storage path
prediction_data_paths = "./data/result/prediction_data.csv"

# clear the local moedls
model_train.clear_models(model_paths, prediction_data_paths)
schema, data_type, string_col = data_process.get_data_schema()
# classify model
model = None

# 2. Load train data in blocks and train the model
train_chunk_size = 20000
chunk_loader = data_process.read_data_in_chunks(spark, train_data_path, train_chunk_size, schema, data_type)
index = 1
try:
    while True:
        print("\nBlock [" + str(index) + "] starts running.")
        # 2.1 load the chunk data
        chunk_data = next(chunk_loader)
        print(" - 1.Load the block data finish.")

        # 2.2 train the model
        model = model_train.operate(spark, chunk_data, model_paths)
        print(" - 2.Block model training finish.")

        # 2.3 Release cache, memory
        chunk_data.unpersist()
        chunk_data = None
        gc.collect()
        print("Block [" + str(index) + "] running finish.\n")
        index += 1
except RuntimeError as e:
    print("Load Train data Exception: ",e)
    pass


# 3. Load test data in blocks and predict
test_chunk_size = 2000
chunk_loader = data_process.read_data_in_chunks(spark, test_data_path, test_chunk_size, schema, data_type)
index_p = 1
# init the model
if model is None:
    print("The model is untrained and cannot make predictions.")
try:
    while True:
        print("\nBlock [" + str(index_p) + "] starts running.")
        # 3.1 init the model
        chunk_data = next(chunk_loader)
        print(" - 1.Load the block data finish.")

        # 3.2 predict and save predict result in file
        evaluation.predict(spark=spark, model=model, test_data=chunk_data, file_paths=prediction_data_paths)
        print(" - 2.Block model prediction finish.")

        # 3.3 Release cache, memory
        chunk_data.unpersist()
        chunk_data = None
        gc.collect()
        print("Block [" + str(index_p) + "] running finish.\n")
        index_p += 1
except RuntimeError as e:
    print("Load Test data Exception: ",e)
    pass

# start_time = time.time()
# end_time = time.time()
# print("Running time：", end_time - start_time, "s")
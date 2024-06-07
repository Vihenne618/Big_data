import data_processing.data_process as data_process
import model_training.model_train as model_train
import evaluating.evaluation as evaluation
from pyspark.sql import SparkSession
import time

# initialize the SparkSession
spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("PB01_Spark_App") \
    .config("spark.default.parallelism", "10") \
    .getOrCreate()

# data path
train_data_path = "./data/traindata.csv"
test_data_path = "./data/testdata.csv"
# models path
model_paths = "./data/models/local_models.pkl"

# clear the local moedls
model_train.clear_models(model_paths, None)

# step1: data process
train_data_processed, test_data_processed = data_process.operate(spark, train_data_path, test_data_path)
print("Step 1: Data process finish")

# step2: model train
model = model_train.operate(spark, train_data_processed, model_paths)
print("Step 2:Model train finish")

# step3: evalution
accuracy, auc, metrics = evaluation.operate(spark, model, test_data_processed)
print("Step 2:Model predict finish")
print("Test Accuracy = {:.2f}%".format(accuracy * 100))
print("Area Under ROC = ", auc)
print("Confusion Matrix:")
print(metrics)



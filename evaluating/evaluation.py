from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import os
import gc
import csv

# Predict and Evaluation
# @ Parameters:
#     spark: (SparkSession)
#     test_data: (DataFrame)
#     model: (PipelineModel)
# @ accuracy
#     
def operate(spark, model, test_data):
  label_col = "class {0,1}"
  predictions = model.transform(test_data)
  evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
  accuracy = evaluator.evaluate(predictions)


  evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
  auc = evaluator.evaluate(predictions)
  # print("Area Under ROC = ", auc)

  pred_and_labels = predictions.select("prediction", label_col).rdd
  # Convert data type
  pred_and_labels = pred_and_labels.map(lambda row: (float(row.prediction), float(row[label_col])))
  metrics = MulticlassMetrics(pred_and_labels)
  confusion_matrix = metrics.confusionMatrix().toArray()
  return  accuracy, auc, confusion_matrix


# Prediction
# @ Parameters:
#     spark: (SparkSession)
#     test_data: (DataFrame)
#     model: (PipelineModel) model Object
#     file_paths: (String) prediction result file path
# @ accuracy
#     
def predict(spark, model, test_data, file_paths):
  # make prediction for test data
  predictions = model.transform(test_data)
  evaluator = MulticlassClassificationEvaluator(labelCol="class {0,1}", metricName="accuracy")
  accuracy = evaluator.evaluate(predictions)
  print("Test Accuracy = {:.2f}%".format(accuracy * 100))

  # save the prediction result in file
  data = predictions.collect()
  if not os.path.exists(file_paths):
      with open(file_paths, 'w', newline='') as f:
          writer = csv.writer(f)
          writer.writerow(predictions.columns)
  with open(file_paths, 'a', newline='') as f:
      writer = csv.writer(f)
      for row in data:
          writer.writerow(row)

  # Release cache, memory
  predictions, data, evaluator  = None, None, None
  gc.collect()
  return None
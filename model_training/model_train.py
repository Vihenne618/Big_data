from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, ChiSqSelector
from pyspark.ml import Pipeline,Transformer,Estimator
from pyspark.sql.functions import col
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
import pandas as pd
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.ml.util import MLWritable, MLReadable
import joblib
import pickle
import os
import gc
from pyspark.ml import PipelineModel


# Local gradient boosted tree Estimator
class LocalGBTClassifier(Estimator):
  def __init__(self, featuresCol, labelCol, model_storage_path):
    super(LocalGBTClassifier, self).__init__()
    self.featuresCol = featuresCol
    self.labelCol = labelCol
    self.model_storage_path = model_storage_path

  def setFeaturesCol(self, value):
    return self._set(featuresCol=value)

  def getFeaturesCol(self):
    return self.featuresCol

  def setLabelCol(self, value):
    return self._set(labelCol=value)

  def getLabelCol(self):
      return self.labelCol
  
  # Distribute data to partitions to train partition models
  def _fit(self, dataset):
    features_col = self.getFeaturesCol()
    label_col = self.getLabelCol()
    num_partitions = dataset.rdd.getNumPartitions()
    # spilt the data and distribute to partitons
    model_list = dataset.select(col(features_col), col(label_col)) \
                        .repartition(num_partitions) \
                        .rdd \
                        .mapPartitions(self._train_partition)\
                        .collect()
    classifier_model = LocalGBTClassifierModel(self.featuresCol, self.labelCol, model_list, self.model_storage_path)
    # Release cache, memory
    features_col, label_col, num_partitions, model_list = None, None, None, None
    gc.collect()
    return classifier_model

  # Training the model within the partition
  def _train_partition(self, interator):
    # create partition_data
    partition_data = pd.DataFrame.from_records(interator)
    x = partition_data.iloc[:, 0].tolist()
    y = partition_data.iloc[:, 1].tolist()
    sample_weights = [1 if label == 0 else 9 for label in y]
    # Training Partitioned Gradient Boosting Trees
    partition_gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    partition_gb_classifier.fit(x, y, sample_weights)
    # Release cache, memory
    x, y, partition_data, sample_weights = None, None, None, None
    gc.collect()
    return [partition_gb_classifier]





# Local gradient boosted tree
class LocalGBTClassifierModel(Transformer, MLWritable, MLReadable):
  def __init__(self, featuresCol, labelCol, local_models, model_storage_path):
    self.uid = "local_gbt_classifier_model"
    self.featuresCol = featuresCol
    self.labelCol = labelCol
    self.model_storage_path = model_storage_path
    # Save the data model incrementally
    self.local_models = saveModels(local_models, model_storage_path)

  # Make partition predictions and aggregate the results
  def _transform(self, dataset):
    num_partitions = dataset.rdd.getNumPartitions()
    features_col = self.featuresCol
    label_col = self.labelCol
    # load spark
    spark = SparkSession.builder.getOrCreate()
    # broadcast the local models to all partiton
    local_models_broadcast = spark.sparkContext.broadcast(self.local_models)

    # predict within the partition
    def transform_partition(interator):
      # load all local models
      local_models = local_models_broadcast.value
      # create partition_data
      partition_data = pd.DataFrame.from_records(interator)
      x = partition_data.iloc[:, 0].tolist()
      y = partition_data.iloc[:, 1].tolist()
      # predict in all partition model
      predictions = [model.predict(x) for model in local_models]
      # Vote on each sample and select the category with the most votes as the final prediction
      final_predictions = []
      for sample_predictions in zip(*predictions):
          counter = Counter(sample_predictions)
          final_predictions.append(float(counter.most_common(1)[0][0]))
      # generate the result
      result = list(zip(y, final_predictions, final_predictions))
      # Release memory
      x, y, predictions, partition_data, final_predictions, local_models = None, None, None, None, None, None
      gc.collect()
      return result
    
    schema = StructType([
      StructField(label_col, IntegerType(), True),
      StructField("prediction", DoubleType(), True),
      StructField("rawPrediction", DoubleType(), True)
    ])
    result = dataset.repartition(num_partitions) \
                  .select(col(features_col), col(label_col)) \
                  .rdd \
                  .mapPartitions(transform_partition) \
                  .toDF(schema=schema)
    # Release memory
    num_partitions, features_col, label_col, schema = None, None, None, None
    gc.collect()
    return result

  def save(self, file_path):
    with open(file_path, "wb") as f:
      pickle.dump(self.model, f)

  @classmethod
  def load(cls, file_path):
    with open(file_path, "rb") as f:
      model = pickle.load(f)
    return cls(model)






# save the models
def saveModels(models, model_paths):
  models_all = loadModels(model_paths)
  if models is not None and len(models) > 0:
    models_all.extend(models)
  if(len(models_all) > 0) :
    joblib.dump(models_all, model_paths)
  return models_all

# load the models
def loadModels(model_paths):
  models = None
  try:
    models = joblib.load(model_paths)
  except (FileNotFoundError, IOError) as e:
    models = []
  return models

# clear the models:
def clear_models(model_paths, prediction_data_paths):
  if model_paths is not None and os.path.exists(model_paths):
    os.remove(model_paths)

  if prediction_data_paths is not None and os.path.exists(prediction_data_paths):
    os.remove(prediction_data_paths)





# Training Models
# @ Parameters:
#     spark: (SparkSession)
#     train_data: (DataFrame)
#     test_data: (DataFrame)
# @ return: model(PipelineModel)
#     
def operate(spark, train_data, model_paths):
  label_col = "class {0,1}"
  feature_cols = train_data.columns
  feature_cols.remove(label_col)

  # Build vectorAssembler
  vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

  # Feature Selection with ChiSqSelector
  selector = ChiSqSelector(percentile=0.1, featuresCol="features", outputCol="select_features", labelCol=label_col)

  # Build a GBT model
  # gbt = GBTClassifier(maxIter=10, stepSize=0.01, featuresCol="select_features", labelCol=label_col)
  gbt = LocalGBTClassifier(featuresCol="features", labelCol=label_col, model_storage_path=model_paths)

  # Creat Pipeline
  pipeline = Pipeline(stages=[vector_assembler, selector, gbt])

  # train the model
  cv_model = pipeline.fit(train_data)

  # Release memory
  vector_assembler, selector, gbt, pipeline = None, None, None, None
  gc.collect()
  return cv_model



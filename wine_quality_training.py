from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # Initializing Spark session
    spark = SparkSession.builder.appName("Wine Quality Training").getOrCreate()
    
    # Paths to datasets 
    training_data_path = "hdfs:///user/hadoop/trainingDataset.csv"
    validation_data_path = "hdfs:///user/hadoop/validationDataset.csv"

    # stored in HDFS  
    # training_data_path = "/home/hadoop/Projects/WineQualityPrediction/TrainingDataset.csv"
    # validation_data_path = "/home/hadoop/Projects/WineQualityPrediction/ValidationDataset.csv"


    # Reading training & validation data
    train_df = spark.read.option("delimiter", ";").csv(training_data_path, header=True, inferSchema=True)
    validate_df = spark.read.option("delimiter", ";").csv(validation_data_path, header=True, inferSchema=True)

    # Removing quotes from CSV headers
    def clean_header(df):
        old_column_names = df.schema.names
        clean_column_names = [name.replace('"', '') for name in old_column_names]
        for old_name, new_name in zip(old_column_names, clean_column_names):
            df = df.withColumnRenamed(old_name, new_name)
        return df

    # Header cleaning
    train_df = clean_header(train_df)
    validate_df = clean_header(validate_df)

    # Feature Columns List
    feature_columns = [col for col in train_df.columns if col != 'quality']

    # Assembling features into a single vector column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Logistic regression model
    lr = LogisticRegression(featuresCol="features", labelCol="quality")

    # pipeline with stages: vector assembler and logistic regression
    pipeline = Pipeline(stages=[assembler, lr])

    # Training model with the train dataset using Pipeline stages
    pipe_model = pipeline.fit(train_df)

    # Saving Pipeline model results
    # model_path = "hdfs:///user/hadoop/Models/wine_quality_training"
    model_path = "/home/hadoop/Projects/WineQualityPrediction/Models/wine_quality_training"
    pipe_model.write().overwrite().save(model_path)

    # Generating predictions for train and validation datasets
    try:
        train_prediction = pipe_model.transform(train_df)
        test_prediction = pipe_model.transform(validate_df)
    except Exception as e:
        print("ERROR: Check CSV input data format")

    # Setting up evaluator callssification object to generate prediction metrics
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")

    # Calculating F1 and accuracy of model with Train and Validation datasets
    train_accuracy = evaluator.evaluate(train_prediction, {evaluator.metricName: "accuracy"})
    test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})
    train_F1score = evaluator.evaluate(train_prediction, {evaluator.metricName: "f1"})
    test_F1score = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})


    # Printing the metrics
    print("EVALUATION METRICS:")
    print("Train Accuracy = ", train_accuracy)
    print("Train F1 score = ", train_F1score)
    print("Test Accuracy = ", test_accuracy)
    print("Test F1 score = ", test_F1score)


    # Saving results onto results.txt
    fp = open("/job/results.txt", "w")
    fp.write("\n")
    fp.write("Train F1 score =  %s\n" % train_F1score)
    fp.write("Train Accuracy = %s\n" % train_accuracy)
    fp.write("\n")
    fp.write("Test F1 score =  %s\n" % test_F1score)
    fp.write("Test Accuracy = %s\n" % test_accuracy)
    fp.close()


if __name__ == "__main__":
    main()

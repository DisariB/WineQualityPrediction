from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # Initializing Spark session
    spark = SparkSession.builder.appName("Wine Quality Testing").getOrCreate()
    
    # Path to saved model
    model_path = "/home/hadoop/Projects/WineQualityPrediction/Models/wine_quality_training"
    
    # Path to testing dataset
    testing_data_path = "/home/hadoop/Projects/WineQualityPrediction/TestingDataset.csv"
    
    # Loading trained model
    model = PipelineModel.load(model_path)
    
    # testing data
    test_df = spark.read.option("delimiter", ";").csv(testing_data_path, header=True, inferSchema=True)
    
    # Removing quotes from CSV headers
    def clean_header(df):
        old_column_names = df.schema.names
        clean_column_names = [name.replace('"', '') for name in old_column_names]
        for old_name, new_name in zip(old_column_names, clean_column_names):
            df = df.withColumnRenamed(old_name, new_name)
        return df

    # Clean headers
    test_df = clean_header(test_df)

    # Generating preidctions for testing dataset
    test_prediction = model.transform(test_df)

    # Evaluator for classification object to generate prediction metrics
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")

    # F1 and accuracy of model with testing dataset
    test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})
    test_F1score = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})

    # Printing metrics
    print("TESTING METRICS:")
    print("Test Accuracy = ", test_accuracy)
    print("Test F1 score = ", test_F1score)

    # Saving results
    with open("/home/hadoop/Projects/WineQualityPrediction/testing_results.txt", "w") as fp:
        fp.write("Test F1 score = %s\n" % test_F1score)
        fp.write("Test Accuracy = %s\n" % test_accuracy)

if __name__ == "__main__":
    main()

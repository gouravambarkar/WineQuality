package org.apache.maven.mavenproject;

import java.io.IOException;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityTrainingApp {
    public static void main(String[] args) throws IOException {
        // Initialize Spark
        SparkConf sparkConf = new SparkConf().setAppName("WineQualityTrainingApp");
        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);
        SparkSession sparkSession = SparkSession.builder().appName("WineQualityTrainingApp").getOrCreate();

        // Load training data from S3
        Dataset<Row> trainingDataset = sparkSession.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("s3://<doker-1.bucket>/TrainingDataset.csv");

        // Define feature columns
        String[] inputColumns = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"};
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(inputColumns)
                .setOutputCol("features");

        // Split dataset into training and validation sets
        Dataset<Row>[] splitData = trainingDataset.randomSplit(new double[]{0.8, 0.2}, 42L);
        Dataset<Row> trainDataset = splitData[0];
        Dataset<Row> validationDataset = splitData[1];

        // Create LogisticRegression model
        LogisticRegression logisticRegression = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFamily("multinomial")
                .setLabelCol("quality");

        // Set up pipeline with feature transformation and model
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{vectorAssembler, logisticRegression});

        // Train the model
        PipelineModel pipelineModel = pipeline.fit(trainDataset);

        // Evaluate model on validation dataset
        Dataset<Row> predictions = pipelineModel.transform(validationDataset);
        MulticlassClassificationEvaluator multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = multiclassClassificationEvaluator.evaluate(predictions);
        System.out.println("F1 score on validation data: " + f1Score);

        // Save the trained model
        pipelineModel.write().overwrite().save("s3://<doker-1.bucket>/wineQualityModel");

        javaSparkContext.close();
    }
}

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPredictor {
    public static void main(String[] args) {
        // Initialize Spark
        SparkConf sparkConf = new SparkConf().setAppName("WineQualityPredictor");
        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);
        SparkSession sparkSession = SparkSession.builder().appName("WineQualityPredictor").getOrCreate();

        // Load the trained model from S3
        PipelineModel pipelineModel = PipelineModel.load("s3://<doker-1.bucket>/wineQualityModel");

        // Load the test dataset from S3
        Dataset<Row> testDataset = sparkSession.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("s3://<doker-1.bucket>/TestDataset.csv");

        // Make predictions on the test dataset
        Dataset<Row> predictions = pipelineModel.transform(testDataset);

        // Evaluate the model performance on the test dataset
        MulticlassClassificationEvaluator multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = multiclassClassificationEvaluator.evaluate(predictions);
        System.out.println("F1 score on test data: " + f1Score);

        javaSparkContext.close();
    }
}

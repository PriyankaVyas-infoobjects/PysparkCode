from pyspark import SparkConf, SparkContext
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
#sc=SparkContext("local", "Spark Demo")
import os
os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk1.8.0_291"
os.environ["SPARK_HOME"] = "C:/spark/spark-3.1.1-bin-hadoop2.7"
os.environ['HADOOP_HOME'] = "C://Hadoop/bin/winutils"
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('abc').getOrCreate()
df = spark.read.csv("C://Users/HP/Downloads/hiring.csv", inferSchema=True, header=True)
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext.getOrCreate();
sqlContext = SQLContext(sc)
from pyspark.sql.functions import mean
mean_val=df.select(mean(df['test_score'])).collect()
mean_score=mean_val[0][0]
df=df.na.fill(mean_score,['test_score'])
columns = ['experience', 'test_score', 'interview_score', 'salary']
import flask
from flask import Flask, jsonify
app = Flask(__name__)
@app.route('/')
def hello_world():
    return "Hello,World!"
@app.route('/appi/')
def appi():
    assembler = VectorAssembler(inputCols=['experience', 'test_score', 'interview_score'], outputCol='features')
    output = assembler.transform(df)
    output.select('features', 'salary').show(5)
    final_data = output.select('features', 'salary')
    from pyspark import SparkContext
    from pyspark.sql import SQLContext
    sc = SparkContext.getOrCreate();
    sqlContext = SQLContext(sc)
    train_data, test_data = final_data.randomSplit([0.9, 0.1])
    model = LinearRegression(featuresCol='features', labelCol='salary')
    # pass train_data to train model
    trained_model = model.fit(train_data)
    # evaluating model trained for Rsquared error
    results = trained_model.evaluate(train_data)
    print(results)
    unlabeled_data = test_data.select('features')
    unlabeled_data.show(5)
    predictions = trained_model.transform(unlabeled_data)


    pre = predictions.select(predictions['prediction']).collect()
    return jsonify(pre)
if __name__=="__main__":
    app.run(debug= True)

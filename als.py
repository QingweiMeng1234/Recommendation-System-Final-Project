from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import explode, dense_rank, count, desc, sum, row_number, countDistinct, col, expr, collect_list, broadcast
from pyspark.sql.window import Window
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.feature import StringIndexer
import sys 

def main(spark, file_path1, file_path2, file_path3):
    interaction_t=spark.read.parquet(file_path1, header=True, schema='user_id INT, recording_msid STRING, timestamp INT')
    interaction_t.createOrReplaceTempView('interaction_t')
    interaction_t=interaction_t.repartition(1000, "recording_msid")
    #interaction_t.show()
    track=spark.read.parquet(file_path2, header=True, schema='recording_msid STRING, artist_name STRING, track_name STRING, recording_mbid STRING')
    track.createOrReplaceTempView('track')
    #track.show()
     
    track=track.withColumn("recording_mbid", expr("CASE WHEN recording_mbid IS NULL THEN recording_msid ELSE recording_mbid END"))
    track.createOrReplaceTempView('track')
    track = spark.sql('select *, dense_rank() over(partition by 1 order by recording_mbid) as rn from track')   
    track.show() 
    track=track.repartition(1000, "recording_msid")
    track.createOrReplaceTempView('track')

    data_t=track.join(interaction_t, ['recording_msid'])
    data_t=data_t.withColumn("recording_mbid", expr("CASE WHEN recording_mbid IS NULL THEN recording_msid ELSE recording_mbid END"))
    data_t=data_t.drop("artist_name", "track_name","__index_level_0__","timestamp","recording_msid", "recording_mbid").withColumnRenamed('rn', 'recording_mbid')
    data_t.show()
    beta=10000
    popularity=data_t.groupBy('recording_mbid').agg((countDistinct('user_id')/(count('*')+beta)).alias("popularity")).orderBy("popularity", ascending = False) 
    popularity=popularity.repartition(100, "recording_mbid")
    data_t=data_t.repartition(1000,"recording_mbid")
    popularity=data_t.join(popularity,["recording_mbid"]) 
    popularity.show()
   
    interaction_v=spark.read.parquet(file_path3, header=True, schema='user_id INT, recording_msid STRING, timestamp INT')
    interaction_v.createOrReplaceTempView('interaction_v')
   
    data_v=track.join(interaction_v, 'recording_msid')
    data_v=data_v.withColumn("recording_mbid", expr("CASE WHEN recording_mbid IS NULL THEN recording_msid ELSE recording_mbid END"))
    data_v=data_v.drop("artist_name", "track_name","__index_level_0__","timestamp","recording_msid", "recording_mbid").withColumnRenamed('rn', 'recording_mbid')
    data_v.show()
    
    
    data_t.createOrReplaceTempView('data_t')
    data_v.createOrReplaceTempView('data_v')
    popularity.createOrReplaceTempView('popularity')

    als = ALS(rank=100, regParam=0.01, alpha=1, maxIter=3, implicitPrefs=True, userCol="user_id", itemCol="recording_mbid", ratingCol="popularity", nonnegative=True ,coldStartStrategy="drop")
    model = als.fit(popularity)
    users_valid = data_v.select('user_id').distinct()
    
    predictions = model.recommendForUserSubset(users_valid, 100)
    pred_label = predictions.select('user_id','recommendations.recording_mbid')
    true_label = data_v.groupBy("user_id").agg(collect_list("recording_mbid").alias("groundtruth"))
    pred_label=pred_label.repartition(1000, "user_id")
    true_label=true_label.repartition(1000, "user_id")
    predictions_rdd = pred_label.join(true_label, ['user_id'], 'inner').rdd.map(lambda row: (row[1], row[2]))   
    
    
    metrics = RankingMetrics(predictions_rdd)
    mean_ap=metrics.meanAveragePrecisionAt(100)
    NDCG=metrics.ndcgAt(100)
    print("Precision at 100:", metrics.precisionAt(100))
    print("ALS Mean average precision at ", beta, ": {:.8f}".format(mean_ap))
    print("ALS Average NDCG at ", beta, ": {:.8f}".format(NDCG))

    
    
    
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('als').getOrCreate()

    # Get file_path for dataset to analyze
    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]
    file_path3 = sys.argv[3] 

    main(spark, file_path1, file_path2, file_path3)

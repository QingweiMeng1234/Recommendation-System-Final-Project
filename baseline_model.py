import sys
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, desc, sum, row_number, countDistinct, col, expr, collect_list
from pyspark.mllib.evaluation import RankingMetrics

def main(spark, file_path1, file_path2,file_path3):
    interaction_t=spark.read.parquet(file_path1, header=True, schema='user_id INT, recording_msid STRING, timestamp INT')
    interaction_t.createOrReplaceTempView('interaction_t')
    #interaction_t.show()
    track=spark.read.parquet(file_path2, header=True, schema='recording_msid STRING, artist_name STRING, track_name STRING, recording_mbid STRING')
    track.createOrReplaceTempView('track')
    #track.show()

    data_t=track.join(interaction_t, 'recording_msid')
    data_t=data_t.withColumn("recording_mbid", expr("CASE WHEN recording_mbid IS NULL THEN recording_msid ELSE recording_mbid END"))
    data_t=data_t.drop("artist_name", "track_name","__index_level_0__","timestamp","recording_msid")
    data_t.show()
    beta=100000
    popularity=data_t.groupBy('recording_mbid').agg((countDistinct('user_id')/(count('*')+beta)).alias("popularity")).orderBy("popularity", ascending = False).limit(100)  
    popularity.show()
   
    interaction_v=spark.read.parquet(file_path3, header=True, schema='user_id INT, recording_msid STRING, timestamp INT')
    interaction_v.createOrReplaceTempView('interaction_v')
   
    data_v=track.join(interaction_v, 'recording_msid')
    data_v=data_v.withColumn("recording_mbid", expr("CASE WHEN recording_mbid IS NULL THEN recording_msid ELSE recording_mbid END"))
    data_v=data_v.drop("artist_name", "track_name","__index_level_0__","timestamp","recording_msid")
    #ground_truth=data_v.groupBy('recording_mbid').agg((countDistinct('user_id')/(count('*')+beta)).alias("actual_popularity"))
    #ground_truth.show()
    #data_v=data_v.join(ground_truth,"recording_mbid")
    #ranked_val=data_v.join(popularity, "recording_mbid")
    #ranked_val.show()
    #ranked_val=ranked_val.groupBy("user_id").agg(collect_list("recording_mbid").alias("popularity"), collect_list("recording_mbid").alias("actual_popularity"))
    #predictions_rdd=ranked_val.rdd.map(lambda x: (x[1], x[2]))
    
    
    val_ratings = data_v.groupBy("user_id").agg(collect_list("recording_mbid").alias("groundtruth"))
    
    popularity= popularity.agg(collect_list(col("recording_mbid")).alias("recording_mbid"))
    predictions_rdd = val_ratings.select("groundtruth").crossJoin(popularity.select("recording_mbid")).rdd#.map(lambda row: (row[0], row[1]))
    metrics=RankingMetrics(predictions_rdd)
    mean_ap=metrics.meanAveragePrecisionAt(100)
    NDCG=metrics.ndcgAt(100)
    print("Mean average precision at ", beta, ": {:.8f}".format(mean_ap))
    print("Average NDCG at ", beta, ": {:.8f}".format(NDCG))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline_model').getOrCreate()

    # Get file_path for dataset to analyze
    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]
    file_path3 = sys.argv[3] 

    main(spark, file_path1, file_path2, file_path3)

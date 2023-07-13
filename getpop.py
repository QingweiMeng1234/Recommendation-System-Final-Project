# Fetch data

def main(spark, userID):
    df = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
    df.createOrReplaceTempView('df')
    df= spark.sql('select *, rand() as rn from df where user_id in (select user_id from (select user_id, count(*) from df GROUP BY user_id HAVING count(*)>=10) a)')
    df.createOrReplaceTempView('df')
    train_df=spark.sql('select user_id, recording_msid, timestamp from df where rn<=0.8')
    val_df=spark.sql('select user_id, recording_msid, timestamp from df where rn>0.8') 
    train_df.write.option("compression", "snappy").option("spark.sql.files.maxRecordsPerFile",3).parquet("interactions_train_small.parquet")
    val_df.write.option("compression", "snappy").option("spark.sql.files.maxRecordsPerFile",3).parquet("interactions_val_small.parquet")

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
    #track.show()
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
    popularity.write.option("compression", "snappy").option("spark.sql.files.maxRecordsPerFile",3).parquet("popularity_small.parquet")

    #df = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')
    #df.createOrReplaceTempView('df')
    #df= spark.sql('select *, rand() as rn from df where user_id in (select user_id from (select user_id, count(*) from df GROUP BY user_id HAVING count(*)>=10) a)')
    #df.createOrReplaceTempView('df')
    #train_df=spark.sql('select user_id, recording_msid, timestamp from df where rn<=0.8')
    #val_df=spark.sql('select user_id, recording_msid, timestamp from df where rn>0.8') 
    #train_df.write.option("compression", "snappy").option("spark.sql.files.maxRecordsPerFile",3).parquet("interactions_train.parquet")
    #val_df.write.option("compression", "snappy").option("spark.sql.files.maxRecordsPerFile",3).parquet("interactions_val.parquet")
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('partition').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)




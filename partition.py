import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc


def main(spark, userID):
    df = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
    df.createOrReplaceTempView('df')
    df= spark.sql('select *, rand() as rn from df where user_id in (select user_id from (select user_id, count(*) from df GROUP BY user_id HAVING count(*)>=10) a)')
    df.createOrReplaceTempView('df')
    train_df=spark.sql('select user_id, recording_msid, timestamp from df where rn<=0.8')
    val_df=spark.sql('select user_id, recording_msid, timestamp from df where rn>0.8') 
    train_df.write.option("compression", "snappy").option("spark.sql.files.maxRecordsPerFile",3).parquet("interactions_train_small.parquet")
    val_df.write.option("compression", "snappy").option("spark.sql.files.maxRecordsPerFile",3).parquet("interactions_val_small.parquet")


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

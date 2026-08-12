"""
Dùng chung cho toàn bộ pipeline.
Gọi get_spark() ở đầu mỗi file thay vì tạo SparkSession mới.
"""

from pyspark.sql import SparkSession


def get_spark(app_name: str = "Rossmann_BigData") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"[SparkSession] '{app_name}' — Spark {spark.version}")
    return spark

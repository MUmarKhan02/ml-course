"""Week 2, Day 4 (part 2): w2d4_pyspark.py
PySpark hands-on with the same store data.

Needs Java 17+ (adoptium.net) and: uv pip install pyspark
No Java? Run it on Google Colab (!pip install pyspark) after uploading the two CSV files.

Run: python w2d4_pyspark.py
"""

import shutil
import sys

if shutil.which("java") is None and "JAVA_HOME" not in __import__("os").environ:
    sys.exit("Java not found. Install Java 17+ from adoptium.net, or run this file on Google Colab.")

from pyspark.sql import SparkSession, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

# 1. Start Spark (local[*] = use every CPU core on this laptop)
spark = SparkSession.builder.appName("week2-day4").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Read CSVs into Spark DataFrames
orders = spark.read.csv("orders_raw.csv", header=True, inferSchema=True)
customers = spark.read.csv("customers_raw.csv", header=True, inferSchema=True)
orders.printSchema()
print("partitions:", orders.rdd.getNumPartitions())

# 3. Clean with transformations (lazy: nothing runs yet)
clean = (
    orders.dropDuplicates()
    .withColumn("amount", F.regexp_replace("amount", r"[$,]", "").cast("double"))
    .withColumn("returned", F.col("returned") == "yes")
    .dropna(subset=["amount"])
)
customers = customers.withColumn("state", F.upper(F.trim(F.col("state"))))
print("plan built, nothing computed yet")

# 4. Actions trigger the work
print("clean orders:", clean.count())
clean.show(3)

# 5. groupBy + aggregations
(clean.groupBy("category")
    .agg(F.count("*").alias("orders"),
         F.round(F.sum("amount"), 0).alias("revenue"),
         F.round(F.avg(F.col("returned").cast("int")), 3).alias("return_rate"))
    .orderBy(F.desc("revenue"))
    .show())

# 6. Join (broadcast the small table) + revenue per customer by state
joined = clean.join(F.broadcast(customers.select("customer_id", "state", "is_member")), on="customer_id", how="inner")
(joined.filter(~F.col("returned"))
    .groupBy("state")
    .agg(F.round(F.sum("amount"), 2).alias("revenue"), F.countDistinct("customer_id").alias("customers"))
    .withColumn("rev_per_customer", F.round(F.col("revenue") / F.col("customers"), 2))
    .orderBy(F.desc("rev_per_customer"))
    .show(3))

# 7. Window function: each customer's second order
w = Window.partitionBy("customer_id").orderBy("order_date")
second = clean.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 2)
print("customers with a 2nd order:", second.count())

# 8. Spark SQL on a temporary view
clean.createOrReplaceTempView("orders")
spark.sql("""
    SELECT category, ROUND(AVG(amount), 2) AS avg_order
    FROM orders
    GROUP BY category
    ORDER BY avg_order DESC
""").show()

# 9. Write partitioned Parquet (how data lakes on S3 / ADLS / GCS are organized)
try:
    clean.write.mode("overwrite").partitionBy("category").parquet("spark_out/orders_by_category")
    print("wrote spark_out/orders_by_category/category=.../part-*.parquet")
except Exception as exc:  # Windows without winutils.exe / HADOOP_HOME
    if "HADOOP_HOME" in str(exc) or "winutils" in str(exc):
        print("Skipped writing Parquet: on Windows Spark needs winutils.exe + HADOOP_HOME. Run this step on Colab or Linux/Mac.")
    else:
        raise

spark.stop()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

spark = (
    SparkSession.builder
    .appName("Rossmann_01_EDA_Spark").getOrCreate()
    .coonfig("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("Spark version:", spark.version)
HDFS_ROOT = "hdfs://localhost:9000/user/user/rossmann/"
#Ghi 1 file csv ra hdfs
def save_single_csv(sdf, hdfs_path):
    tmp = hdfs_path + "_tmp"
    (sdf.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(tmp))
    
    hadoop = spark._jvm.org.apache.hadoop
    conf = spark._jsc.hadoopConfiguration()
    
    uri = spark._jvm.java.net.URI(tmp)
    fs = hadoop.fs.FileSystem.get(uri, conf)

    Path = hadoop.fs.Path
    part = None
    for st in fs.listStatus(Path(tmp)):
        nm = str(st.getPath().getName())
        if nm.startswith("part-") and nm.endswith(".csv"):
            part = st.getPath()
            break
    dst = Path(hdfs_path)
    if fs.exists(dst):
        fs.delete(dst, True)
    fs.rename(part, dst)
    fs.delete(Path(tmp), True)
    print("Da ghi:", hdfs_path)


#Đếm missing values
def show_null_counts(df, title):
    print(title)
    df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).show()


#Tiền xử lý bộ dữ liệu sale
df_sales = spark.read.csv(f"{HDFS_ROOT}/sale.csv", header=True, inferSchema=True)
df_sales = df_sales.withColumn("Date", F.to_date("Date", "M/d/yyyy"))
df_sales = df_sales.orderBy("Date")

print("sale.csv shape :", (df_sales.count(), len(df_sales.columns)))   # ~ (1017209, 8)
df_sales.show(5)
df_sales.describe().show()

show_null_counts(df_sales, "Missing values - sale.csv:")


#Tiền xử lý bộ dữ liệu store
df_store = spark.read.csv(f"{HDFS_ROOT}/store.csv", header=True, inferSchema=True)

print("store.csv shape:", (df_store.count(), len(df_store.columns)))   # ~ (1115, 10)
df_store.show(5)
df_store.describe().show()

show_null_counts(df_store, "Missing values - store.csv (truoc xu ly):")

median_cd = df_store.approxQuantile("CompetitionDistance", [0.5], 0.001)[0]
print("Median CompetitionDistance:", median_cd)
df_store = df_store.na.fill({"CompetitionDistance": median_cd})

df_store = df_store.na.fill(0).na.fill("0")

show_null_counts(df_store, "Missing values - store.csv (sau xu ly):")
df_store.show(5)

#Kết hợp 2 bộ dữ liệu
df = df_sales.join(df_store, on="Store", how="left")
print("Merged shape:", (df.count(), len(df.columns))) 
df.show(5)


df_open = df.filter(F.col("Sales") > 0).cache()

#Kiểm tra phân phối của biến Sales và log(Sales)
buckets, counts = df_open.select("Sales").rdd.map(lambda r: float(r[0])).histogram(50)
centers = [(buckets[i] + buckets[i + 1]) / 2 for i in range(len(counts))]
bw = buckets[1] - buckets[0]

lb, lc = (df_open.select(F.log1p("Sales").alias("ls")).rdd
          .map(lambda r: float(r[0])).histogram(50))
lcenters = [(lb[i] + lb[i + 1]) / 2 for i in range(len(lc))]
lbw = lb[1] - lb[0]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(centers, counts, width=bw, color="steelblue")
ax[0].set_title("Phan phoi cua Sales"); ax[0].set_xlabel("Sales"); ax[0].set_ylabel("Frequency")
ax[1].bar(lcenters, lc, width=lbw, color="seagreen")
ax[1].set_title("Phan phoi cua Log(Sales)"); ax[1].set_xlabel("log(1+Sales)")
plt.tight_layout(); plt.show()

#Giá trị trung bình của Sales theo ngày
daily_pd =(df_open.groupBy("Date").agg(F.sum("Sales").alias("avg")).
                orderBy("Date").toPandas())
promo_pd = (df_open.filter(F.col("Promo") == 1)
            .groupBy("Date").agg(F.avg("Sales").alias("avg"))
            .orderBy("Date").toPandas())
plt.figure(figsize=(14, 4))
plt.plot(daily_pd["Date"], daily_pd["avg"], color="steelblue", linewidth=1)
plt.scatter(promo_pd["Date"], promo_pd["avg"], color="red", s=5, label="Co Promo")
plt.title("Gia tri Sales trung binh theo ngay", weight="bold"); plt.legend(); plt.show()
# Gia tri trung binh cua Sales theo ngay trong tuan & theo thang
dow_pd = (df_open.groupBy("DayOfWeek").agg(F.avg("Sales").alias("avg"))
          .orderBy("DayOfWeek").toPandas())
month_pd = (df_open.groupBy(F.month("Date").alias("Month"))
            .agg(F.avg("Sales").alias("avg")).orderBy("Month").toPandas())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow_pd["avg"], color="steelblue")
axes[0].set_title("Sales theo ngay trong tuan", weight="bold"); axes[0].set_ylabel("Avg Sales")
axes[1].bar(month_pd["Month"], month_pd["avg"], color="green")
axes[1].set_title("Sales theo thang", weight="bold"); axes[1].set_xlabel("Thang")
plt.tight_layout(); plt.show()

#Kiem tra moi tuong quan giua Sales va cac bien
num_cols = [c for c, t in df.dtypes
            if t in ("int", "bigint", "double", "float", "smallint", "tinyint")]
vec_df = (VectorAssembler(inputCols=num_cols, outputCol="cv", handleInvalid="skip")
          .transform(df.na.drop(subset=num_cols)).select("cv"))
corr_mat = Correlation.corr(vec_df, "cv").head()[0].toArray()
corr_pd = pd.DataFrame(corr_mat, index=num_cols, columns=num_cols)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_pd, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Ma tran tuong quan giua Sales va cac bien", weight="bold"); plt.show()
print(corr_pd["Sales"].drop("Sales").sort_values(ascending=False))

df_open.unpersist()


#xuat data ra hdfs
save_single_csv(df, f"{HDFS_ROOT}/cleaned_rossmann.csv")
print("Da xuat cleaned_rossmann.csv, shape:", (df.count(), len(df.columns)))

spark.stop()
print("[01] Done.")
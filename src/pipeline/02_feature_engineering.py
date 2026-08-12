import json
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, UnivariateFeatureSelector
from pyspark.ml.stat import Correlation
from pyspark.ml.stat import FValueTest



spark = (
    SparkSession.builder
    .appName("Rossmann_02_FeatureEngineering_Spark")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
HDFS_ROOT = "hdfs://localhost:9000/user/project/rossmann"

#Hàm bổ trợ lưu DataFrame -> CSV rồi push lên HDFS
def save_single_csv(sdf, hdfs_path):
    tmp = hdfs_path + "__tmp"
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

#Hàm bổ trợ lưu object -> JSON rồi push lên HDFS
def save_json_hdfs(obj, hdfs_path):
    content = json.dumps(obj, ensure_ascii=False, indent=2)
    hadoop = spark._jvm.org.apache.hadoop
    fs = hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    out = fs.create(hadoop.fs.Path(hdfs_path), True)
    out.write(bytearray(content, "utf-8"))
    out.close()
    print("Da ghi:", hdfs_path)

#Bổ trợ One-hot Encoding
def add_dummies(df, col, drop_first=True):
    cats = [r[0] for r in df.select(col).distinct().collect() if r[0] is not None]
    cats = sorted(cats, key=lambda x: str(x))
    if drop_first:
        cats = cats[1:]
    new_cols = []
    for c in cats:
        safe = f"{col}_{str(c).replace(' ', '_')}"
        df = df.withColumn(safe, F.when(F.col(col) == c, 1).otherwise(0))
        new_cols.append(safe)
    return df, new_cols

df = spark.read.csv(f"{HDFS_ROOT}/cleaned_rossmann.csv", header=True, inferSchema=True)
df = df.withColumn("Date", F.to_date("Date"))
print("cleaned_rossmann:", (df.count(), len(df.columns)))

df = (df
      .withColumn("Year",       F.year("Date"))
      .withColumn("Month",      F.month("Date"))
      .withColumn("WeekOfYear", F.weekofyear("Date"))
      .withColumn("DayOfWeek", ((F.dayofweek(F.col("Date")) + 5) % 7 + 1).cast("int")))
df = df.withColumn("IsWeekend", F.when(F.col("DayOfWeek").isin(6, 7), 1).otherwise(0))

w_store = Window.partitionBy("Store").orderBy("Date")
for lag in [1, 3, 7, 14]:
    df = df.withColumn(f"Sales_lag_{lag}",     F.lag("Sales", lag).over(w_store))
    df = df.withColumn(f"Customers_lag_{lag}", F.lag("Customers", lag).over(w_store))

df = df.filter(F.col("Open") == 1)

for w in [7, 14]:
    win = Window.partitionBy("Store").orderBy("Date").rowsBetween(-w, -1)
    df = df.withColumn(f"Sales_roll_mean_{w}", F.avg("Sales").over(win))
    df = df.withColumn(f"Sales_roll_std_{w}",  F.stddev("Sales").over(win))
    df = df.withColumn(f"Sales_roll_max_{w}",  F.max("Sales").over(win))

df = (df
      .withColumn("Promo_Weekend", F.col("Promo") * F.col("IsWeekend"))
      .withColumn("Promo_Month",   F.col("Promo") * F.col("Month")))

df, _ = add_dummies(df, "DayOfWeek", drop_first=False)
for col in ["StateHoliday", "StoreType", "Assortment"]:
    df = df.withColumn(col, F.col(col).cast("string"))
    df, _ = add_dummies(df, col, drop_first=True)

from pyspark import StorageLevel

n_before = df.count()
df = df.na.drop()
df = df.orderBy("Date")
df = df.persist(StorageLevel.MEMORY_AND_DISK)

print("Số dòng trước:", n_before, " sau khi dropna:", df.count())
print("Đã persist df với StorageLevel.MEMORY_AND_DISK")


exclude_cols = ["Date", "Sales", "Sale", "Customers", "PromoInterval", "Store",
                "StateHoliday", "StoreType", "Assortment", "DayOfWeek",
                "Sales_per_Customer"]  
numeric_types = ("int", "bigint", "double", "float", "smallint", "tinyint", "long")
candidate_features = [c for c, t in df.dtypes
                      if c not in exclude_cols and t in numeric_types]
print("So feature ung vien:", len(candidate_features))


assembler = VectorAssembler(inputCols=candidate_features,
                            outputCol="features_vec", handleInvalid="skip")
assembled = assembler.transform(df)

selector = UnivariateFeatureSelector(
    featuresCol="features_vec",
    outputCol="selectedFeatures",
    labelCol="Sales",
    selectionMode="fpr"          
)
selector.setFeatureType("continuous")
selector.setLabelType("continuous")
selector.setSelectionThreshold(0.05)  

sel_model = selector.fit(assembled)
selected_idx = list(sel_model.selectedFeatures)
features = [candidate_features[i] for i in selected_idx]
dropped  = [c for c in candidate_features if c not in features]

print(f"So feature DA CHON: {len(features)} / {len(candidate_features)}")
print("Feature da chon :", features)
print("Feature bi loai :", dropped)

assembler_corr = VectorAssembler(inputCols=candidate_features + ["Sales"],
                                 outputCol="corr_vec", handleInvalid="skip")
corr_mat = Correlation.corr(
    assembler_corr.transform(df).select("corr_vec"), "corr_vec"
).head()[0].toArray()
r_with_sales = np.nan_to_num(corr_mat[:-1, -1], nan=0.0)   

n = df.count()
F_stat  = (n - 2) * (r_with_sales ** 2) / (1 - r_with_sales ** 2 + 1e-12)  
p_value = f_dist.sf(F_stat, 1, n - 2)                                        

result = pd.DataFrame({
    "feature":  candidate_features,
    "abs_corr": np.round(np.abs(r_with_sales), 4),
    "p_value":  np.round(p_value, 6),
})
result["selected"] = result["feature"].isin(features)
result = result.sort_values("abs_corr", ascending=False).reset_index(drop=True)

print(f"Giu {result['selected'].sum()}/{len(result)} feature")
print(result)

#Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

plot_df = result.sort_values("abs_corr", ascending=True).reset_index(drop=True)
xmax    = max(plot_df["abs_corr"].max(), 1e-6)

norm   = Normalize(vmin=0, vmax=xmax)
greens = cm.get_cmap("Greens")
colors = [greens(0.35 + 0.6 * norm(v)) if sel else "#c0392b"
          for v, sel in zip(plot_df["abs_corr"], plot_df["selected"])]

fig, ax = plt.subplots(figsize=(11, 12))
ax.barh(plot_df["feature"], plot_df["abs_corr"], color=colors,
        edgecolor="white", linewidth=0.6)

for y, v in enumerate(plot_df["abs_corr"]):
    ax.text(v + xmax * 0.012, y, f"{v:.3f}", va="center", ha="left",
            fontsize=8, color="#333333")

ax.set_xlim(0, xmax * 1.14)
ax.set_xlabel("|Tương quan Pearson với Sales|  (càng cao = liên hệ càng mạnh)",
              fontsize=11, fontweight="bold")
ax.set_title(f"Kết Quả Lựa Chọn Đặc Trưng — Giữ {int(result['selected'].sum())}/{len(result)} Đặc Trưng",
             fontsize=14, fontweight="bold", pad=14)

ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=8)

leg = ax.legend(handles=[Patch(color=greens(0.75), label="Giữ lại"),
                         Patch(color="#c0392b",     label="Loại bỏ")],
                loc="lower right", frameon=True, fontsize=10, title="Trạng thái")
leg.get_title().set_fontweight("bold")

plt.tight_layout()
plt.savefig("feature_selection.png", dpi=150, bbox_inches="tight")
plt.show()

save_single_csv(df, f"{HDFS_ROOT}/df_features.csv")
save_json_hdfs(features, f"{HDFS_ROOT}/features.json")

spark.stop()
print("[02] Done.")


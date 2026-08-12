# Import các thư viện cần thiết

import time
import threading
import math
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, FloatType, StringType, DateType
)

# Cấu hình cho hệ thống

HDFS_BASE        = "hdfs://localhost:9000/user/project/rossmann"
SOURCE_CSV       = f"{HDFS_BASE}/cleaned_rossmann.csv"
STREAM_INPUT_DIR = f"{HDFS_BASE}/stream_input"       # Producer ghi vào đây
STREAM_OUTPUT_DIR= f"{HDFS_BASE}/stream_output"      
CHECKPOINT_DIR   = f"{HDFS_BASE}/stream_checkpoint"  # Spark checkpoint

ANOMALY_DIR      = f"{HDFS_BASE}/stream_anomalies"   # Nơi ghi nhận các cảnh báo bất thường

ROWS_PER_BATCH   = 500      # Số dòng mỗi file batch
PRODUCER_INTERVAL= 4        # Giây giữa mỗi lần producer ghi file mới
TRIGGER_INTERVAL = "5 seconds"  # Spark sẽ xử lý mỗi 5 giây
MAX_BATCHES      = 15       # Dừng sau bao nhiêu batch

# Ngưỡng phát hiện bất thường theo quy tắc 3-sigma (z-score):
# z = (Sales - mean_lich_su) / std_lich_su của chính cửa hàng đó.
# |z| > 3  ->  chỉ xấp xỉ 0.3% dữ liệu bình thường bị rơi ra ngoài (quy tac 68-95-99.7)
# -> giá trị nằm ngoài khoảng này được xem là bất thường.
Z_THRESHOLD      = 3.0

# Mau ANSI cho terminal (do = sut giam, xanh la = tang vot)
import os as _os
_os.system("")          # bật ANSI color trên Windows PowerShell / CMD
RED    = "\033[91m"     # Sụt giảm bất thường
GREEN  = "\033[92m"     # Tăng đột biến
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Khai báo Schema

ROSSMANN_SCHEMA = StructType([
    StructField("Store",         IntegerType(), True),
    StructField("DayOfWeek",     IntegerType(), True),
    StructField("Date",          StringType(),  True),  # đọc string, parse sau
    StructField("Sales",         FloatType(),   True),
    StructField("Customers",     IntegerType(), True),
    StructField("Open",          IntegerType(), True),
    StructField("Promo",         IntegerType(), True),
    StructField("StateHoliday",  StringType(),  True),
    StructField("SchoolHoliday", IntegerType(), True),
    StructField("StoreType",     StringType(),  True),
    StructField("Assortment",    StringType(),  True),
    StructField("CompetitionDistance", FloatType(), True),
    StructField("Year",          IntegerType(), True),
    StructField("Month",         IntegerType(), True),
    StructField("WeekOfYear",    IntegerType(), True),
    StructField("IsWeekend",     IntegerType(), True),
])

# Khởi tạo SparkSession

def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("Rossmann_Structured_Streaming")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "false")
        # Tăng stack size JVM (-Xss) để tránh StackOverflowError do regex
        # Đệ quy sau khi Spark don dep checkpoint lúc dùng streaming query.
        .config("spark.driver.extraJavaOptions", "-Xss16m")
        .config("spark.executor.extraJavaOptions", "-Xss16m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print("SparkSession khởi động thành công")
    print(f"Spark version : {spark.version}")
    print(f"Master        : {spark.sparkContext.master}")
    return spark

# Khởi tạo Producer - Chạy trong Thread riêng

def write_single_csv_hdfs(spark, sdf, hdfs_file_path):
    """
    Ghi sdf thành 1 file CSV PHẲNG trên HDFS
    Cách làm: Ta sẽ ghi ra folder tạm  tìm part-file  rename thành file phẳng
    Đây cũng là pattern chuẩn cho file streaming source:
    File chỉ xuất hiện trong thư mục theo dõi sau khi đã ghi xong hoàn toàn,
    tránh việc Spark đọc phải file đang ghi dở.
    """
    # Bước 1: Ghi ra folder tạm (coalesce(1) gop thanh 1 part-file duy nhat)
    tmp = hdfs_file_path + "__tmp"
    (sdf.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(tmp))
 
    # Bước 2: Dùng Hadoop FileSystem API (qua JVM) để thao tác với file trên HDFS
    hadoop = spark._jvm.org.apache.hadoop
    conf   = spark._jsc.hadoopConfiguration()
    uri    = spark._jvm.java.net.URI(tmp)
    fs     = hadoop.fs.FileSystem.get(uri, conf)
    Path   = hadoop.fs.Path
 
    # Tìm file part-xxxxx.csv trong folder tạm
    part = None
    for st in fs.listStatus(Path(tmp)):
        nm = str(st.getPath().getName())
        if nm.startswith("part-") and nm.endswith(".csv"):
            part = st.getPath()
            break
 
    dst = Path(hdfs_file_path)
    if fs.exists(dst):
        fs.delete(dst, False)
    fs.rename(part, dst)          # rename part-file thành file phẳng
    fs.delete(Path(tmp), True)    # xoá folder tạm
 
 
def run_producer(spark: SparkSession, stop_event: threading.Event):
    """
    Producer giúp ta giả lập các cửa hàng Rossmann gửi dữ liệu lên HDFS theo thời gian thực.
    Mỗi PRODUCER_INTERVAL giây, một file CSV nhỏ với khoảng 500 dòng sẽ được ghi vào
    stream_input/, kích hoạt Spark Structured Streaming xử lý micro-batch mới.
    """
    # Đọc toàn bộ dataset gốc từ HDFS 1 lần duy nhất
    print("\nProducer: Đang đọc toàn bộ dataset từ HDFS...")
 
    try:
        df_full = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(SOURCE_CSV)
        )
        df_full = df_full.filter(F.year(F.to_date(F.col("Date"))) == 2015)
        total_rows = df_full.count()
        print(f"Tổng số dòng dataset: {total_rows:,}")
    except Exception as e:
        print(f"Producer: Không đọc được CSV từ HDFS — {e}")
        stop_event.set()
        return
 
    # Tính số batch cần thiết
    n_batches = min(MAX_BATCHES, math.ceil(total_rows / ROWS_PER_BATCH))
    print(f"Sẽ gửi {n_batches} batch × {ROWS_PER_BATCH} dòng/batch")
    print(f"Ghi vào: {STREAM_INPUT_DIR}/\n")
 
    # Lấy về Pandas để dễ slice theo batch
    pdf = df_full.toPandas()
 
    # Lặp lại qua từng batch: lấy ra 500 dòng rồi ghi thành 1 file -> gửi file dần dần
    for batch_idx in range(n_batches):
        if stop_event.is_set():
            break
 
        start_row = batch_idx * ROWS_PER_BATCH
        end_row   = min(start_row + ROWS_PER_BATCH, total_rows)
        batch_pdf = pdf.iloc[start_row:end_row]
 
        # Ghi batch lên HDFS dưới dạng FILE PHẲNG, để Spark readStream có thể xử lý được
        output_path = f"{STREAM_INPUT_DIR}/batch_{batch_idx:04d}.csv"
        batch_sdf = spark.createDataFrame(batch_pdf)
        write_single_csv_hdfs(spark, batch_sdf, output_path)
 
        ts = time.strftime("%H:%M:%S")
        print(
            f"  [{ts}]  Producer gửi batch {batch_idx+1:02d}/{n_batches} "
            f"— dòng {start_row:,}{end_row:,} "
            f"({end_row - start_row} records)"
        )
 
        time.sleep(PRODUCER_INTERVAL)
 
    print("\nProducer: Đã gửi xong tất cả các batch.")
    stop_event.set()

# Streaming query giúp giải quyết bài toán anomaly detection

def run_streaming(spark: SparkSession, stop_event: threading.Event):
    """
    Spark Structured Streaming PHÁT HIỆN BẤT THƯỜNG real-time.

    Ý tưởng triển khai:
      1. Chúng ta sẽ tính BASELINE từ dữ liệu lịch sử: mỗi cửa hàng có 
      doanh thu trung bình (mean) và độ lệch chuẩn (std) riêng. Cửa hàng lớn mean cao, cửa hàng 
      nhỏ mean thấp — baseline cá nhân hoá cho TỪNG cửa hàng.
      2. Mỗi micro-batch data đến, tính z-score cho từng bản ghi:
      z = (Sales_hôm_nay - mean_cửa_hàng) / std_cửa_hàng
      3. |z| > 3    BẤT THƯỜNG (quy tắc 3-sigma). Phân biệt:
      z < -3    SỤT GIẢM bất thường (vd: mất điện, hết hàng, máy POS lỗi)
      z > +3    TĂNG đột biến  (vd: sự kiện, khuyến mãi ngoài kế hoạch)
    """
    # BƯỚC 1: Tính baseline từ lịch sử
    print("\nĐang tính BASELINE doanh thu mỗi cửa hàng từ lịch sử...")
    hist = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(SOURCE_CSV)
            .filter(
                (F.col("Sales") > 0) &
                (F.col("Open") == 1) &
                (F.year(F.to_date(F.col("Date"))) <= 2014)          # Dùng dữ liệu 2013-2014 để tính baseline
            ))

    baseline_df = (
        hist.groupBy("Store")
        .agg(
            F.round(F.avg("Sales"), 2).alias("BaseMean"),
            F.round(F.stddev("Sales"), 2).alias("BaseStd"),
            F.count("*").alias("BaseCount"),
        )
        # Bỏ cửa hàng không đủ dữ liệu để tính std đáng tin (std null hoặc = 0)
        .filter((F.col("BaseStd").isNotNull()) & (F.col("BaseStd") > 0))
        .cache()
    )
    n_base = baseline_df.count()
    print(f"Đã tính baseline cho {n_base:,} cửa hàng từ dữ liệu trong giai đoạn 2013–2014")
    print(f"Ngưỡng phát hiện: |z-score| > {Z_THRESHOLD} theo quy tắc 3-sigma\n")

    # BƯỚC 2: readStream từ HDFS
    print("Streaming: Khởi động readStream...")
    print(f"Đọc từ      : {STREAM_INPUT_DIR}/")
    print(f"Ghi cảnh báo: {ANOMALY_DIR}/")
    print(f"Trigger     : mỗi {TRIGGER_INTERVAL}\n")

    stream_df = (
        spark.readStream
        .schema(ROSSMANN_SCHEMA)
        .option("header", "true")
        .option("maxFilesPerTrigger", 1)
        .csv(STREAM_INPUT_DIR)
        .filter((F.col("Sales") > 0) & (F.col("Open") == 1))
    )

    # Bộ đếm tổng (mutable để closure foreachBatch cập nhật được)
    stats = {"total": 0, "anomaly": 0, "batches": 0}

    # BƯỚC 3: Thực hiện xử lý từng micro-batch bằng foreachBatch
    def process_batch(batch_df, batch_id):
        if batch_df.rdd.isEmpty():
            return
        batch_df = batch_df.cache()
        n_records = batch_df.count()

        # Join với baseline (broadcast vì baseline nhỏ ~1115 dòng)
        joined = batch_df.join(F.broadcast(baseline_df), on="Store", how="inner")

        scored = (
            joined
            .withColumn("ZScore",
                        F.round((F.col("Sales") - F.col("BaseMean"))
                                / F.col("BaseStd"), 2))
            .withColumn("DeviationPct",
                        F.round((F.col("Sales") - F.col("BaseMean"))
                                / F.col("BaseMean") * 100, 1))
        )

        (
        scored.select("Store", "Date", "Sales", "BaseMean", "ZScore", "DeviationPct")
        .write.mode("append")
        .option("header", "true")
        .csv(STREAM_OUTPUT_DIR))

        anomalies = scored.filter(F.abs(F.col("ZScore")) > Z_THRESHOLD).cache()
        n_anom = anomalies.count()

        stats["batches"] += 1
        stats["total"]   += n_records
        stats["anomaly"] += n_anom

        ts = time.strftime("%H:%M:%S")
        # Phân cách giữa các micro-batch cho dễ đọc
        print()
        print(f"{BOLD}{CYAN}[{ts}] Micro-batch #{stats['batches']:02d}{RESET}  "
              f"|  {n_records} ban ghi  "
              f"|  {n_anom} bat thuong")

        # In chi tiết các cảnh báo
        if n_anom > 0:
            rows = (anomalies
                    .orderBy(F.abs(F.col("ZScore")).desc())
                    .limit(15)
                    .collect())
            for r in rows:
                if r["ZScore"] < 0:
                    color, tag = RED, "SUT GIAM"     # do = sut giam
                else:
                    color, tag = GREEN, "TANG VOT"   # xanh la = tang vot
                print(f"{color}    [{tag}]  Store {r['Store']:>4}  |  {r['Date']}  |  "
                      f"Sales = {r['Sales']:>7.0f}   "
                      f"(TB lich su {r['BaseMean']:>7.0f}, "
                      f"lech {r['DeviationPct']:>+6.1f}%, z = {r['ZScore']:>+5.2f}){RESET}")

            # Ghi cảnh báo ra HDFS (append từng batch vào cùng folder)
            try:
                (anomalies
                 .select("Store", "Date", "Sales", "BaseMean", "BaseStd",
                         "DeviationPct", "ZScore")
                 .write.mode("append")
                 .option("header", "true")
                 .csv(ANOMALY_DIR))
            except Exception as e:
                print(f"Không ghi được cảnh báo ra HDFS: {e}")
            anomalies.unpersist()

        batch_df.unpersist()

    query = (
        stream_df.writeStream
        .foreachBatch(process_batch)
        .trigger(processingTime=TRIGGER_INTERVAL)
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/anomaly")
        .queryName("rossmann_anomaly_detection")
        .start()
    )

    print("Streaming anomaly detection đang chạy. Chờ producer gửi data...\n")
    print("─" * 70)

    # Chờ đến khi producer xong
    while not stop_event.is_set():
        time.sleep(2)

    # Dừng gracefully: đợi batch hiện tại xử lý xong rồi mới stop,
    # Tránh interrupt giữa chừng.
    print("\nĐang dừng streaming query...")
    try:
        query.processAllAvailable()   # Chờ xử lý hết data đang chờ
    except Exception:
        pass
    try:
        query.stop()
        query.awaitTermination(timeout=20)
    except Exception:
        pass

    # Lưu tổng kết vào session để main đọc lại
    stop_event.stats = stats
    print("Streaming dừng thành công.")
    print(f"Tổng kết: {stats['batches']} batch | "
          f"{stats['total']:,} bản ghi | "
          f"{stats['anomaly']} bất thường được phát hiện.")
    
# Khởi tạo hàm Main
    
def main():
    print()
    print("ROSSMANN REAL-TIME SALES STREAMING PIPELINE")
    print("Big Data | NHÓM 7 | UEH")
    print()
    print(f"HDFS Base      : {HDFS_BASE}")
    print(f"Source CSV     : {SOURCE_CSV}")
    print(f"Stream Input   : {STREAM_INPUT_DIR}")
    print(f"Anomaly Out    : {ANOMALY_DIR}")
    print(f"Z-threshold    : {Z_THRESHOLD} (3-sigma)")
    print(f"Rows/batch     : {ROWS_PER_BATCH}")
    print(f"Producer delay : {PRODUCER_INTERVAL}s")
    print(f"Trigger        : {TRIGGER_INTERVAL}")
    print(f"Max batches    : {MAX_BATCHES}")
    print()

    spark = create_spark_session()

    # Khởi tạo HDFS directories và xoá checkpoint cũ
    try:
        hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI.create("hdfs://localhost:9000"),
            spark._jsc.hadoopConfiguration()
        )
        HPath = spark._jvm.org.apache.hadoop.fs.Path

        # Xoá và tạo lại stream_input
        p = HPath(STREAM_INPUT_DIR)
        if hadoop_fs.exists(p): hadoop_fs.delete(p, True)
        hadoop_fs.mkdirs(p)
        print(f"Đã tạo thư mục input: {STREAM_INPUT_DIR}")

        # Xoá và tạo lại stream_input
        p = HPath(STREAM_OUTPUT_DIR)
        if hadoop_fs.exists(p): hadoop_fs.delete(p, True)
        hadoop_fs.mkdirs(p)
        print(f"Đã tạo thư mục output: {STREAM_OUTPUT_DIR}")

        # Xoá checkpoint cũ để tránh conflict khi chạy lại
        p = HPath(CHECKPOINT_DIR)
        if hadoop_fs.exists(p):
            hadoop_fs.delete(p, True)
            print(f"Đã xoá checkpoint cũ: {CHECKPOINT_DIR}")

        # Xoá và tạo lại stream_anomalies, nơi ghi cảnh báo bất thường
        p = HPath(ANOMALY_DIR)
        if hadoop_fs.exists(p): hadoop_fs.delete(p, True)
        hadoop_fs.mkdirs(p)
        print(f"Đã tạo thư mục anomalies: {ANOMALY_DIR}\n")

    except Exception as e:
        print(f"Không thể khởi tạo HDFS: {e}")
        print("Chạy thủ công trong PowerShell:")
        print("hadoop fs -rm -r /user/project/rossmann/stream_input")
        print("hadoop fs -rm -r /user/project/rossmann/stream_checkpoint")
        print("hadoop fs -mkdir /user/project/rossmann/stream_input\n")


    stop_event = threading.Event()

    # Khởi động Streaming trước
    streaming_thread = threading.Thread(
        target=run_streaming,
        args=(spark, stop_event),
        daemon=True,
        name="StreamingThread"
    )
    streaming_thread.start()

    # Đợi streaming khởi động xong
    time.sleep(8)

    # Khởi động Producer trong thread riêng
    producer_thread = threading.Thread(
        target=run_producer,
        args=(spark, stop_event),
        name="ProducerThread"
    )
    producer_thread.start()

    # Chờ producer xong
    producer_thread.join()

    # Đợi thêm 2 trigger để Spark xử lý hết batch cuối
    print(f"\nCho them {int(TRIGGER_INTERVAL.split()[0]) * 2}s de Spark xu ly batch cuoi...")
    time.sleep(int(TRIGGER_INTERVAL.split()[0]) * 2)

    stop_event.set()
    streaming_thread.join(timeout=15)

    # Tổng kết các cảnh báo bất thường từ HDFS
    print()
    print("KẾT QUẢ CUỐI — CÁC CỬA HÀNG CÓ DOANH THU BẤT THƯỜNG")
    print()
    try:
        anomaly_df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(ANOMALY_DIR)
        )
        n_anom = anomaly_df.count()

        if n_anom == 0:
            print("\n  Không phát hiện cửa hàng nào bất thường trong kỳ stream này.")
        else:
            n_drop = anomaly_df.filter(F.col("ZScore") < 0).count()
            n_rise = anomaly_df.filter(F.col("ZScore") > 0).count()
            print(f"\n{BOLD}Tong so canh bao bat thuong : {n_anom}{RESET}")
            print(f"  {RED}Sut giam bat thuong : {n_drop}{RESET}")
            print(f"  {GREEN}Tang dot bien       : {n_rise}{RESET}")
            print()

            # 15 bất thường nghiêm trọng nhất (|z-score| lớn nhất)
            print("TOP 15 BAT THUONG NGHIEM TRONG NHAT (|z-score| cao nhat):")
            print("─" * 78)
            print(f"{BOLD}{'Loai':<10}{'Store':>6}{'Date':>14}{'Sales':>10}"
                  f"{'TB_LichSu':>12}{'Lech%':>9}{'ZScore':>9}{RESET}")
            print("─" * 78)
            top = (anomaly_df
                   .withColumn("absZ", F.abs(F.col("ZScore")))
                   .orderBy(F.col("absZ").desc())
                   .drop("absZ")
                   .limit(15)
                   .collect())
            for r in top:
                if r["ZScore"] < 0:
                    color, tag = RED, "SUT GIAM"
                else:
                    color, tag = GREEN, "TANG VOT"
                print(f"{color}{tag:<10}{r['Store']:>6}{str(r['Date']):>14}"
                      f"{r['Sales']:>10.0f}{r['BaseMean']:>12.0f}"
                      f"{r['DeviationPct']:>+8.1f}%{r['ZScore']:>+9.2f}{RESET}")
            print("─" * 78)
    except Exception as e:
        print(f"Không đọc được cảnh báo: {e}")
        print("(Có thể không có bất thường nào trong kỳ stream — folder rỗng.)")
        print("Kiểm tra HDFS tại:", ANOMALY_DIR)

    print()
    print("PIPELINE HOÀN THÀNH")
    print(f"Được lưu tại HDFS: {ANOMALY_DIR}")
    print()

    spark.stop()


if __name__ == "__main__":
    main()

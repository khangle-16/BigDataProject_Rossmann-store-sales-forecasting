from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Rossmann_SparkSQL").config("spark.sql.shuffle.partitions",4).getOrCreate()

spark.sparkContext.setLogLevel("WARN")


HDFS_ROOT = "hdfs://localhost:9000/user/project/rossmann"

df_sales = spark.read.csv(f"{HDFS_ROOT}/sale.csv", header = True, inferSchema=True)
df_stores = spark.read.csv(f"{HDFS_ROOT}/store.csv", header=True, inferSchema=True)

df_sales.createOrReplaceTempView("sales")
df_stores.createOrReplaceTempView("stores")

df_sales.cache()
df_stores.cache()

print(f"Sales : {df_sales.count():,} rows x {len(df_sales.columns)} cols")
print(f"Stores : {df_stores.count():,} rows x {len(df_stores.columns)} cols")

#Query 1: Ta sử dụng GroupBY, Aggeration và Join để tìm Doanh thu trunh bình theo từng loại cửa hàng

spark.sql("""
    SELECT
          s.StoreType,
          COUNT(DISTINCT sa.Store) AS so_cua_hang,
          ROUND(AVG(sa.Sales),2) AS doanh_thu_tb,
          ROUND(SUM(sa.Sales),2) AS tong_doanh_thu,
          ROUND(AVG(sa.Customers),2 AS khach_hang_tb
    FROM sales sa
    JOIN stores s ON sa.Store=s.Store
    WHERE sa.Open = 1 AND sa.Sales > 0
    GROUP BY s.StoreType
    ORDER BY doanh_thu_tb DESC
""").show()

#Query 2: Ta sử dụng GroupBY và Aggeration để so sánh doanh thu có Promo và không có Promo

spark.sql("""
    SELECT
          Promo,
          COUNT(*) AS so_ngay_ghi_nhan,
          ROUND(AVG(Sales),2) AS doanh_thu_tb,
          ROUND(AVG(Customers),2) as khach_tb,
          ROUND(AVG(Sales/Customers),2) as doanh_thu_per_khach
    FROM sales
    WHERE Open = 1 AND Sales > 0 AND Customers > 0
    GROUP BY Promo
    ORDER BY Promo
""").show()

#Query 3: Ta sử dụng Time series và GroupBy để doanh thu trung bình theo từng tháng trong năm

spark.sql("""
    SELECT
          MONTH(Date) as thang,
          ROUND(AVG(Sales),2) as doanh_thu_tb,
          ROUND(SUM(Sales),2) as tong_doanh_thu,
          ROUND(AVG(Customers),2) AS khach_tb
    FROM sales
    WHERE Open = 1 AND Sales > 0
    GROUP BY MONTH(Date)
    ORDER BY thang
""").show(12)

#Query 4: Ta dùng Window Function: RANK để xếp hạng top 10 cửa hàng có doanh thu trung bình cao nhất

spark.sql("""
    SELECT *
    FROM (
        SELECT
            sa.Store,
            s.StoreType,
            ROUND(AVG(sa.Sales), 2)         AS doanh_thu_tb,
            ROUND(SUM(sa.Sales), 2)         AS tong_doanh_thu,
            RANK() OVER (
                PARTITION BY s.StoreType
                ORDER BY AVG(sa.Sales) DESC
            )                               AS xep_hang
        FROM sales sa
        JOIN stores s ON sa.Store = s.Store
        WHERE sa.Open = 1 AND sa.Sales > 0
        GROUP BY sa.Store, s.StoreType
    )
    WHERE xep_hang <= 3
    ORDER BY StoreType, xep_hang
""").show()

# Query 5: Ta dùng Time Series + Window Function để kiểm tra tốc độ tăng trưởng doanh thu tháng so với tháng trước

spark.sql("""
    SELECT
        nam, thang,
        tong_doanh_thu,
        LAG(tong_doanh_thu) OVER (ORDER BY nam, thang)  AS thang_truoc,
        ROUND(
            (tong_doanh_thu
             - LAG(tong_doanh_thu) OVER (ORDER BY nam, thang))
            / LAG(tong_doanh_thu) OVER (ORDER BY nam, thang) * 100
        , 2)                                            AS tang_truong_pct
    FROM (
        SELECT
            YEAR(Date)          AS nam,
            MONTH(Date)         AS thang,
            ROUND(SUM(Sales), 2) AS tong_doanh_thu
        FROM sales
        WHERE Open = 1 AND Sales > 0
        GROUP BY YEAR(Date), MONTH(Date)
    )
    ORDER BY nam, thang
""").show(36)

# Query 6: Ta dùng Subquery + Join + Group By để kiểm tra cửa hàng có doanh thu dưới mức trung bình toàn hệ thống

spark.sql("""
    SELECT
        sa.Store,
        s.StoreType,
        ROUND(AVG(sa.Sales), 2)             AS doanh_thu_tb,
        ROUND(tb.avg_he_thong, 2)           AS tb_he_thong,
        ROUND(AVG(sa.Sales) - tb.avg_he_thong, 2) AS chenh_lech
    FROM sales sa
    JOIN stores s  ON sa.Store = s.Store
    JOIN (
        SELECT AVG(Sales) AS avg_he_thong
        FROM sales
        WHERE Open = 1 AND Sales > 0
    ) tb ON 1 = 1
    WHERE sa.Open = 1 AND sa.Sales > 0
    GROUP BY sa.Store, s.StoreType, tb.avg_he_thong
    HAVING AVG(sa.Sales) < tb.avg_he_thong
    ORDER BY doanh_thu_tb ASC
    LIMIT 20
""").show()

#Query 7: Sử dụng Group By + Aggregation để tìm doanh thu theo ngày trong tuần — ngày nào bán được nhiều nhất
spark.sql("""
    SELECT
        DayOfWeek,
        ROUND(AVG(Sales), 2)                AS doanh_thu_tb,
        ROUND(AVG(Customers), 2)            AS khach_tb,
        COUNT(*)                            AS so_ngay_ghi_nhan
    FROM sales
    WHERE Open = 1 AND Sales > 0
    GROUP BY DayOfWeek
    ORDER BY DayOfWeek
""").show()

#Query 8: Sử dụng Join + Group By + CASE WHEN xem ảnh hưởng của khoảng cách đối thủ cạnh tranh đến doanh thu

spark.sql("""
    SELECT
        CASE
            WHEN s.CompetitionDistance < 500  THEN 'Rat gan  (<500m)'
            WHEN s.CompetitionDistance < 1000 THEN 'Gan      (500m-1km)'
            WHEN s.CompetitionDistance < 5000 THEN 'Vua      (1km-5km)'
            ELSE                                   'Xa       (>5km)'
        END                                     AS nhom_khoang_cach,
        COUNT(DISTINCT sa.Store)                AS so_cua_hang,
        ROUND(AVG(sa.Sales), 2)                 AS doanh_thu_tb,
        ROUND(AVG(sa.Customers), 2)             AS khach_tb
    FROM sales sa
    JOIN stores s ON sa.Store = s.Store
    WHERE sa.Open = 1 AND sa.Sales > 0
    GROUP BY
        CASE
            WHEN s.CompetitionDistance < 500  THEN 'Rat gan  (<500m)'
            WHEN s.CompetitionDistance < 1000 THEN 'Gan      (500m-1km)'
            WHEN s.CompetitionDistance < 5000 THEN 'Vua      (1km-5km)'
            ELSE                                   'Xa       (>5km)'
        END
    ORDER BY doanh_thu_tb DESC
""").show()

#Query 9: Sử dụng Window Function xem doanh thu tích lũy theo thời gian của top 5 cửa hàng
spark.sql("""
    SELECT
        Store,
        Date,
        Sales,
        SUM(Sales) OVER (
            PARTITION BY Store
            ORDER BY Date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                   AS doanh_thu_tich_luy
    FROM sales
    WHERE Store IN (
        SELECT Store FROM sales
        WHERE Open = 1 AND Sales > 0
        GROUP BY Store
        ORDER BY SUM(Sales) DESC
        LIMIT 5
    )
    AND Open = 1
    ORDER BY Store, Date
""").show(30)

#Query 10: so sánh doanh thu quý 2014 và 2015
print("\n"+ "="*60)
print("Query 10: So sánh doanh thu quý 2014 và 2015")
print("="*60)

spark.sql("""
    SELECT
        YEAR(sa.Date)                       AS nam,
        QUARTER(sa.Date)                    AS quy,
        s.StoreType,
        ROUND(SUM(sa.Sales), 2)             AS tong_doanh_thu,
        ROUND(AVG(sa.Sales), 2)             AS doanh_thu_tb,
        COUNT(DISTINCT sa.Store)            AS so_cua_hang
    FROM sales sa
    JOIN stores s ON sa.Store = s.Store
    WHERE sa.Open = 1
      AND sa.Sales > 0
      AND YEAR(sa.Date) IN (2014, 2015)
    GROUP BY YEAR(sa.Date), QUARTER(sa.Date), s.StoreType
    ORDER BY StoreType, nam, quy
""").show(40)

#Query 11: Phân nhóm cửa hàng theo doanh thu trung bình
print("\n" + "="*60)
print("QUERY 11: Phân nhóm cửa hàng theo doanh thu (NTILE Quartile)")
print("="*60)

spark.sql("""
    SELECT
        nhom,
        COUNT(*)                            AS so_cua_hang,
        ROUND(MIN(doanh_thu_tb), 2)         AS doanh_thu_thap_nhat,
        ROUND(MAX(doanh_thu_tb), 2)         AS doanh_thu_cao_nhat,
        ROUND(AVG(doanh_thu_tb), 2)         AS doanh_thu_tb_nhom
    FROM (
        SELECT
            Store,
            ROUND(AVG(Sales), 2)            AS doanh_thu_tb,
            NTILE(4) OVER (
                ORDER BY AVG(Sales) DESC
            )                               AS nhom
        FROM sales
        WHERE Open = 1 AND Sales > 0
        GROUP BY Store
    )
    GROUP BY nhom
    ORDER BY nhom
""").show()

#Query 12: Tỷ lệ ngày đóng cửa theo cửa hàng
print("\n" + "="*60)
print("QUERY 12: Tỷ lệ ngày đóng cửa theo cửa hàng")
print("="*60)

spark.sql("""
    SELECT
        s.StoreType,
        COUNT(*)                                AS tong_ngay,
        SUM(CASE WHEN sa.Open = 0 THEN 1 END)  AS ngay_dong_cua,
        ROUND(
            SUM(CASE WHEN sa.Open = 0 THEN 1 END)
            / COUNT(*) * 100
        , 2)                                    AS ty_le_dong_cua_pct,
        ROUND(AVG(
            CASE WHEN sa.Open = 1 THEN sa.Sales END
        ), 2)                                   AS doanh_thu_tb_ngay_mo
    FROM sales sa
    JOIN stores s ON sa.Store = s.Store
    GROUP BY s.StoreType
    ORDER BY ty_le_dong_cua_pct DESC
""").show()
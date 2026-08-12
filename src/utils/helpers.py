"""
Đây là các hàm mà nhóm sẽ dùng chung giữa các file:
  - fill_lag_rolling()  : Điền lag/rolling features cho dự báo đệ quy
  - plot_forecast()     : Vẽ biểu đồ forecast dark-background
  - predict_spark()     : Wrap MLlib PipelineModel để predict 1 row pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. fill_lag_rolling
def fill_lag_rolling(df_ref: pd.DataFrame, i: int,
                     hist_df_full: pd.DataFrame,
                     cust_dow_avg: dict) -> None:
    """
    Hàm này giúp chúng ta điền giá trị lag và rolling feature tại dòng i của df_ref. Dùng trong vòng lặp dự báo đệ quy.
    """
    for lag in [1, 3, 7, 14]:
        idx = i - lag
        df_ref.loc[i, f"Sales_lag_{lag}"] = (
            df_ref.loc[idx, "Sales"] if idx >= 0 else np.nan
        )
        val = df_ref.loc[idx, "Customers"] if idx >= 0 else np.nan
        if pd.isna(val):
            val = cust_dow_avg.get(df_ref.loc[i, "Date"].dayofweek + 1, 0)
        df_ref.loc[i, f"Customers_lag_{lag}"] = val

    for w in [7, 14]:
        history = df_ref.loc[max(0, i - w): i - 1, "Sales"].dropna()
        df_ref.loc[i, f"Sales_roll_mean_{w}"] = (
            history.mean() if len(history) > 0 else np.nan
        )
        df_ref.loc[i, f"Sales_roll_std_{w}"] = (
            history.std() if len(history) > 1 else 0.0
        )
        df_ref.loc[i, f"Sales_roll_max_{w}"] = (
            history.max() if len(history) > 0 else np.nan
        )

    s_lag1 = df_ref.loc[i, "Sales_lag_1"]
    c_lag1 = df_ref.loc[i, "Customers_lag_1"]
    df_ref.loc[i, "Sales_per_Customer"] = (
        s_lag1 / c_lag1
        if (not pd.isna(s_lag1) and not pd.isna(c_lag1) and c_lag1 != 0)
        else hist_df_full["Sales_per_Customer"].median()
    )

# 2. plot_forecast
def plot_forecast(store_id, plot_hist_df, df_future, pred_no_promo, avg_sales,
                  df_promo_predicted=None, pred_with_promo=None,
                  days_on=None, days_off=None) -> None:
    """
    Đây là hàm đung để vẽ biểu đồ forecast (dark background)
    - Luôn vẽ: Dữ liệu lịch sử của 60 ngày và đường baseline no-promo
    - Tuỳ chọn: Kịch bản khi doanh nghiệp có apply promo vào chương trình của họ
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 8))

    COLOR_ACTUAL   = "#00e5ff"
    COLOR_BASELINE = "#ffaa00"
    COLOR_PROMO    = "#39ff14"
    COLOR_AVG      = "#ff0055"
    COLOR_GRID     = "#444444"

    # Lịch sử
    ax.plot(plot_hist_df["Date"], plot_hist_df["Sales"],
            linestyle="-", marker="o", markersize=5, color=COLOR_ACTUAL,
            linewidth=2, label="Actual Demand (Last 60 days)")
    ax.fill_between(plot_hist_df["Date"], plot_hist_df["Sales"],
                    color=COLOR_ACTUAL, alpha=0.15)

    # Baseline
    ax.plot(df_future["Date"], pred_no_promo,
            linestyle="--", marker=".", markersize=4, color=COLOR_BASELINE,
            linewidth=1.5, alpha=0.8, label="Forecast (No Promo)")

    # Promo scenario (tuỳ chọn)
    if df_promo_predicted is not None and pred_with_promo is not None:
        ax.plot(df_promo_predicted["Date"], pred_with_promo,
                linestyle="-", marker="o", markersize=5, color=COLOR_PROMO,
                linewidth=2.5,
                label=f"Forecasted Demand (Promo {days_on} On / {days_off} Off)")
        ax.fill_between(df_promo_predicted["Date"], pred_with_promo,
                        color=COLOR_PROMO, alpha=0.15)

    # Avg line
    ax.axhline(avg_sales, linestyle=":", color=COLOR_AVG, linewidth=2,
               label=f"Avg Historical Sales ({avg_sales:,.0f} €)")

    # Styling
    ax.axvspan(df_future["Date"].min(), df_future["Date"].max(),
               color="#ffffff", alpha=0.03)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(color=COLOR_GRID, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(colors="white", labelsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b \\'%y"))
    plt.xticks(rotation=0)

    title_suffix = (
        "with Promo Scenario" if df_promo_predicted is not None
        else "Baseline (No Promo)"
    )
    plt.title(
        f"Store {store_id} – Forecast vs Actual | {title_suffix}",
        color="white", fontsize=18, fontweight="bold", pad=20
    )
    plt.xlabel("Date", color="#aaaaaa", fontsize=12, labelpad=10)
    plt.ylabel("Sales", color="#aaaaaa", fontsize=12, labelpad=10)
    plt.legend(loc="upper left", frameon=True, facecolor="#111111",
               edgecolor="#444444", labelcolor="white",
               fontsize=11, framealpha=0.85)
    plt.tight_layout()
    plt.show()
    plt.style.use("default")


# 3. predict_spark
def make_predict_spark(spark, model, features: list):
    """
    Đây là hàm sẽ trả về hàm predict_spark(row_df) đã được bind với spark session,
    model và danh sách features.

    Chúng ta dùng cách này để tránh import vòng và giữ spark/model là dependency
    được truyền vào từ ngoài.

    Có thể lấy ví dụ:
        predict_spark = make_predict_spark(spark, best_model, features)
        pred = predict_spark(X_current)   # Với X_current: 1-row pandas DataFrame
    """
    def predict_spark(row_df: pd.DataFrame) -> float:
        row_sdf = spark.createDataFrame(row_df[features].astype("float64"))
        result  = model.transform(row_sdf)
        return float(result.select("prediction").collect()[0][0])

    return predict_spark
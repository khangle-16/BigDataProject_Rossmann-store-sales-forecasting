# Import các thư viện cần thiết
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import base64
import os
import json

#Khởi tạo hàm đọc file hình ảnh

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Không tìm thấy tệp hình ảnh '{bin_file}'.")
        return ""
    
#Cấu hình trang: Tiêu đề, icon và layout toàn màn hình

st.set_page_config(
    page_title="Rossmann Forecaster",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed"
)

img_path = 'image_cacfcb.jpg'
img_base64 = get_base64_of_bin_file(img_path)

st.markdown(f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{padding-top: 1rem; max-width: 85rem;}}

    .glass-card {{
        background-color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #f3f4f6;
        height: 100%;
    }}

    .hero-banner {{
        background: linear-gradient(90deg, rgba(227,6,19,0.9) 0%, rgba(227,6,19,0.3) 100%),
                    url('data:image/jpeg;base64,{img_base64}');
        background-size: cover;
        background-position: center 20%;
        border-radius: 1.5rem;
        padding: 4.5rem 2rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(227,6,19,0.3);
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 2rem; }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        color: #E30613 !important;
        border-bottom: 3px solid #E30613 !important;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

#Header: Tên thương hiệu + logo và thông tin người dùng

c1, c2 = st.columns([3, 1])
logo_path = 'logo.jpg'
logo_base64 = get_base64_of_bin_file(logo_path)

with c1:
    st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 16px; padding-bottom: 15px;'>
            <div style='background-color: white; padding: 10px; border-radius: 12px;
                        border: 1px solid #E5E7EB; display: flex; justify-content: center;
                        align-items: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                <img src="data:image/png;base64,{logo_base64}"
                     style="width: 48px; height: 48px; object-fit: contain;" alt="Rossmann Logo">
            </div>
            <div>
                <h1 style='color: #E30613; font-size: 28px; margin: 0;
                           font-weight: 800; letter-spacing: -0.5px;'>ROSSMANN</h1>
                <p style='color: #6B7280; font-size: 12px; margin: 0;
                          text-transform: uppercase; letter-spacing: 2px; font-weight: bold;'>
                    Sales Forecaster
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
        <div style='display: flex; align-items: center; justify-content: flex-end;
                    gap: 12px; padding-top: 10px;'>
            <div style='text-align: right;'>
                <p style='margin: 0; font-size: 16px; font-weight: 800; color: #000000 !important;
                          font-family: "Segoe UI", Roboto, Arial, sans-serif; letter-spacing: 0.5px;'>
                    Người Dùng
                </p>
                <p style='margin: 0; font-size: 12px; font-weight: 700; color: #047857 !important;
                          display: flex; align-items: center; justify-content: flex-end; gap: 5px;
                          letter-spacing: 0.5px; font-family: "Segoe UI", Roboto, Arial, sans-serif;'>
                    <span style='height: 8px; width: 8px; background-color: #10B981;
                                 border-radius: 50%; display: inline-block;
                                 box-shadow: 0 0 4px #10B981;'></span>
                    Đang hoạt động
                </p>
            </div>
            <div style='width: 42px; height: 42px; border-radius: 50%; background-color: #F3F4F6;
                        border: 1px solid #D1D5DB; display: flex; justify-content: center;
                        align-items: center; font-size: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                ?
            </div>
        </div>
    """, unsafe_allow_html=True)

# Cấu hình đường dẫn đọc file trên HDFS

_HDFS_ROOT          = "hdfs://localhost:9000/user/project/rossmann"
MLLIB_MODEL_PATH    = f"{_HDFS_ROOT}/models/GBTRegressor"
CLEANED_CSV_HDFS    = f"{_HDFS_ROOT}/cleaned_rossmann.csv"
FEATURES_JSON_HDFS  = f"{_HDFS_ROOT}/features.json"


def _find_hdfs_cmd():
    """
    Tìm đường dẫn đầy đủ tới lệnh hdfs.
    Trên Windows lệnh thật là hdfs.cmd (subprocess không tự thêm .cmd như
    PowerShell), nên thử lần lượt: hdfs.cmd / hdfs trong PATH, rồi
    %HADOOP_HOME%\\bin. Tránh lỗi 'Không tìm thấy lệnh hdfs'.
    """
    import shutil
    for name in ("hdfs.cmd", "hdfs"):
        found = shutil.which(name)
        if found:
            return found
    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home:
        for name in ("hdfs.cmd", "hdfs"):
            cand = os.path.join(hadoop_home, "bin", name)
            if os.path.exists(cand):
                return cand
    return None

# Khởi động SparkSession và load model GBT đã train vào từ HDFS.
# @st.cache_resource: chỉ chạy 1 lần, giữ lại để rerun cho các lần sau.

@st.cache_resource(show_spinner="Đang khởi động Spark & load MLlib model...")
def load_model():
    try:
        from pyspark.ml import PipelineModel
        from pyspark.sql import SparkSession

        spark = (SparkSession.builder
                 .appName("RossmannWebapp")
                 .config("spark.sql.shuffle.partitions", "4")
                 .config("spark.driver.memory", "2g")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        model = PipelineModel.load(MLLIB_MODEL_PATH)
        return model, spark
    except Exception as e:
        st.error(f"Không load được MLlib model: {e}")
        return None, None

# Dự báo hàng loạt: pandas -> spark DataFrame -> model.transform -> Lấy cột prediction

def mllib_predict_batch(pipeline_model, spark, X_pandas, features):
    X_input = X_pandas[features].copy().astype("float64")
    sdf = spark.createDataFrame(X_input)
    preds_sdf = pipeline_model.transform(sdf)
    return preds_sdf.select("prediction").toPandas()["prediction"].values

# Nạp data từ hdfs để tạo các features (lag, rolling, Promo, one-hot)

@st.cache_data(show_spinner="Đang load dữ liệu từ HDFS...")
def load_and_preprocess_data():
    try:
        from pyspark.sql import SparkSession
        _spark = (SparkSession.builder
                  .appName("RossmannWebapp")
                  .config("spark.sql.shuffle.partitions", "4")
                  .config("spark.driver.memory", "2g")
                  .getOrCreate())
        _spark.sparkContext.setLogLevel("ERROR")
        # Đọc trực tiếp từ HDFS rồi chuyển sang pandas để xử lý phía sau
        df = (_spark.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv(CLEANED_CSV_HDFS)
              .toPandas())
    except Exception:
        # Fallback dữ liệu giả lập nếu không kết nối được HDFS
        dates = pd.date_range("2015-05-01", "2015-07-31")
        data = []
        for s in [1, 2, 3]:
            for d in dates:
                data.append([s, d, np.random.randint(3000, 8000),
                             np.random.randint(300, 800),
                             np.random.choice([0, 1]), 0, 0, 'a', 'a'])
        df = pd.DataFrame(data, columns=[
            'Store', 'Date', 'Sales', 'Customers', 'Promo',
            'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment'
        ])

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Store', 'Date']).reset_index(drop=True)

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1
    df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)

    lags = [1, 3, 7, 14]
    for lag in lags:
        df[f'Sales_lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
        df[f'Customers_lag_{lag}'] = df.groupby('Store')['Customers'].shift(lag)

    windows = [7, 14]
    for w in windows:
        df[f'Sales_roll_mean_{w}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=w).mean())
        df[f'Sales_roll_std_{w}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=w).std())
        df[f'Sales_roll_max_{w}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=w).max())

    df['Promo_Weekend'] = df['Promo'] * df['IsWeekend']
    df['Promo_Month'] = df['Promo'] * df['Month']
    df['Sales_per_Customer'] = np.where(
        df['Customers'] == 0, 0, df['Sales'] / df['Customers'])

    df = pd.get_dummies(df, columns=['DayOfWeek'], prefix='DOW')
    df = df.dropna(
        subset=[f'Sales_lag_{lag}' for lag in lags] +
               [f'Sales_roll_mean_{w}' for w in windows]
    ).reset_index(drop=True)

    categorical_cols = [c for c in ['StateHoliday', 'StoreType', 'Assortment']
                        if c in df.columns]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.sort_values(by='Date')

    exclude_cols = ['Date', 'Sales', 'Customers', 'PromoInterval', 'Store']
    features = [col for col in df.columns if col not in exclude_cols]
    return df, features

# Điền lag/rolling cho 1 ngày trong tương lai dựa trên các ngày đã dự báo

def fill_lag_rolling(df_ref, i, cust_dow_avg, hist_df_full):
    for lag in [1, 3, 7, 14]:
        idx = i - lag
        df_ref.loc[i, f"Sales_lag_{lag}"] = (
            df_ref.loc[idx, "Sales"] if idx >= 0 else np.nan)
        val = df_ref.loc[idx, "Customers"] if idx >= 0 else np.nan
        if pd.isna(val):
            val = cust_dow_avg.get(df_ref.loc[i, "Date"].dayofweek + 1, 0)
        df_ref.loc[i, f"Customers_lag_{lag}"] = val
 
    for w in [7, 14]:
        history = df_ref.loc[max(0, i - w): i - 1, "Sales"].dropna()
        df_ref.loc[i, f"Sales_roll_mean_{w}"] = (
            history.mean() if len(history) > 0 else np.nan)
        df_ref.loc[i, f"Sales_roll_std_{w}"] = (
            history.std() if len(history) > 1 else 0.0)
        df_ref.loc[i, f"Sales_roll_max_{w}"] = (
            history.max() if len(history) > 0 else np.nan)
 
    s_lag1 = df_ref.loc[i, "Sales_lag_1"]
    c_lag1 = df_ref.loc[i, "Customers_lag_1"]
    df_ref.loc[i, "Sales_per_Customer"] = (
        s_lag1 / c_lag1
        if (not pd.isna(s_lag1) and not pd.isna(c_lag1) and c_lag1 != 0)
        else hist_df_full["Sales_per_Customer"].median()
    )

# Nạp data và model khi mở app

model, spark_session = load_model()
df, features = load_and_preprocess_data()

try:
    # Đọc features.json trực tiếp từ HDFS (qua SparkContext.textFile)
    _features_str = "\n".join(
        spark_session.sparkContext.textFile(FEATURES_JSON_HDFS).collect())
    mllib_features = json.loads(_features_str)
except Exception:
    mllib_features = features

for key in ['bi_report', 'baseline_report', 'df_chart_display']:
    if key not in st.session_state:
        st.session_state[key] = None

# Thông tin tên các tab trong web app
tab_home, tab_predict, tab_analytics = st.tabs([
    "TRANG CHỦ", "DỰ BÁO", "THỐNG KÊ"
])



#TAB 1 : Trang chủ
with tab_home:
    st.markdown("""
        <div class="hero-banner">
            <h2 style="font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;">
                Chào mừng đến với Rossmann Forecasting App
            </h2>
            <p style="font-size: 1rem; color: rgba(255,255,255,0.8);
                      max-width: 600px; margin-bottom: 1.5rem;">
                Hệ thống dự báo doanh thu thông minh dành cho chuỗi cửa hàng
                bán lẻ dược phẩm hàng đầu Châu Âu.
            </p>
        </div>
    """, unsafe_allow_html=True)

    mask = (df['Date'] >= '2013-01-01') & (df['Date'] <= '2015-07-31')
    df_history = df[mask]
    avg_sales_hist = df_history['Sales'].mean() if not df_history.empty else 0
    avg_cust_hist = df_history['Customers'].mean() if not df_history.empty else 0
    total_stores = df['Store'].nunique()

    def render_info_card(icon, title, value, desc):
        return f"""
        <div class="glass-card">
            <div style="width: 40px; height: 40px; border-radius: 10px;
                        background-color: #FEF2F2; display: flex;
                        justify-content: center; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 20px;">{icon}</span>
            </div>
            <p style="font-size: 10px; font-weight: bold; color: #6B7280;
                      text-transform: uppercase; letter-spacing: 1px; margin:0;">{title}</p>
            <p style="font-size: 28px; font-weight: 900; color: #111827;
                      margin: 5px 0;">{value}</p>
            <p style="font-size: 12px; color: #6B7280; margin:0;">{desc}</p>
        </div>
        """

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(render_info_card(
            "📍", "Mạng lưới chi nhánh",
            f"{total_stores:,.0f}", "Cửa hàng đang hoạt động."
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(render_info_card(
            "💰", "Doanh thu trung bình",
            f"€{avg_sales_hist:,.0f}", "Mỗi ngày/cửa hàng (01/2013 - 07/2015)"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(render_info_card(
            "👥", "Lượt khách trung bình",
            f"{avg_cust_hist:,.0f}", "Mỗi ngày/cửa hàng (01/2013 - 07/2015)"
        ), unsafe_allow_html=True)

# TAB 2: Dự báo

with tab_predict:
    left_col, right_col = st.columns([1, 2.5], gap="large")

    with left_col:
        st.markdown("### Thiết lập dự báo")

        store_list = sorted(df["Store"].unique())
        store_id = st.selectbox("Chọn ID cửa hàng", store_list)

        start_date = st.date_input("Ngày bắt đầu", value=pd.to_datetime("2015-08-01"))
        end_date = st.date_input("Ngày kết thúc", value=pd.to_datetime("2015-08-31"))

        store_df = df[df["Store"] == store_id].copy().sort_values("Date")

        st.markdown("---")
        use_promo = st.toggle("Bật chương trình Promo", value=True)

        days_on, days_off = 0, 0
        if use_promo:
            promo_type = st.radio(
                "Chế độ Promo",
                ["TỰ ĐỘNG GỢI Ý", "TÙY CHỈNH THỦ CÔNG"],
                horizontal=True,
                label_visibility="collapsed"
            )
            if promo_type == "TỰ ĐỘNG GỢI Ý":
                streaks = store_df["Promo"].groupby(
                    (store_df["Promo"] != store_df["Promo"].shift()).cumsum()
                ).agg(["first", "count"])
                try:
                    days_on = int(streaks[streaks["first"] == 1]["count"].mode().iloc[0])
                    days_off = int(streaks[streaks["first"] == 0]["count"].mode().iloc[0])
                except Exception:
                    days_on, days_off = 5, 9
                st.info(f"Gợi ý chu kỳ: {days_on} ngày BẬT, {days_off} ngày TẮT.")
            else:
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    days_on = st.number_input("Số ngày BẬT", min_value=1, value=5)
                with col_p2:
                    days_off = st.number_input("Số ngày TẮT", min_value=1, value=9)

        if st.button("Chạy mô hình", use_container_width=True, type="primary"):
            if model is None:
                st.error("Chưa load được MLlib Model, không thể chạy dự báo.")
            elif start_date > end_date:
                st.error("Ngày kết thúc phải lớn hơn ngày bắt đầu!")
            elif len(store_df) < 20:
                st.warning("Cửa hàng không đủ dữ liệu lịch sử (>20 ngày).")
            else:
                with st.spinner("Đang chạy mô hình dự báo (MLlib Batch Prediction)..."):
                    future_dates = pd.date_range(start=start_date, end=end_date)
                    hist_df_full = store_df.copy()

                    _tmp = hist_df_full.copy()
                    _tmp["_dow"] = _tmp["Date"].dt.dayofweek + 1
                    cust_dow_avg = _tmp.groupby("_dow")["Customers"].mean().to_dict()

                    valid_hist = hist_df_full[hist_df_full["Customers"] > 0].copy()
                    if not valid_hist.empty:
                        valid_hist['DayOfWeek'] = valid_hist['Date'].dt.dayofweek + 1
                        dynamic_basket = valid_hist.groupby(
                            ['DayOfWeek', 'Promo']
                        ).apply(
                            lambda x: x['Sales'].sum() / x['Customers'].sum()
                        ).to_dict()
                    else:
                        dynamic_basket = {}

                    def create_future_df(f_dates, p_pattern):
                        df_f = pd.DataFrame({"Date": f_dates})
                        df_f["Store"] = store_id
                        df_f["Sales"] = np.nan
                        df_f["Customers"] = np.nan
                        df_f["Promo"] = p_pattern
                        df_f["StateHoliday"] = 0
                        df_f["SchoolHoliday"] = 0
                        df_f["Year"] = df_f["Date"].dt.year
                        df_f["Month"] = df_f["Date"].dt.month
                        df_f["WeekOfYear"] = df_f["Date"].dt.isocalendar().week.astype(int)
                        _dow_f = df_f["Date"].dt.dayofweek + 1
                        df_f["IsWeekend"] = _dow_f.isin([6, 7]).astype(int)
                        df_f["Promo_Weekend"] = df_f["Promo"] * df_f["IsWeekend"]
                        df_f["Promo_Month"] = df_f["Promo"] * df_f["Month"]
                        df_f["_DOW_tmp"] = _dow_f
                        df_f = pd.get_dummies(df_f, columns=["_DOW_tmp"], prefix="DOW")
                        for d in range(1, 8):
                            if f"DOW_{d}" not in df_f.columns:
                                df_f[f"DOW_{d}"] = 0
                        for col in mllib_features:
                            if col not in df_f.columns:
                                df_f[col] = (
                                    hist_df_full[col].iloc[-1]
                                    if col in hist_df_full.columns else 0
                                )
                        return df_f

                    def run_forecast_scenario(full_df, future_start_idx):
                        sales_median = hist_df_full["Sales"].median()
                        for i in range(future_start_idx, len(full_df)):
                            fill_lag_rolling(full_df, i, cust_dow_avg, hist_df_full)
                            full_df.loc[i, "Sales"] = sales_median

                        X_future = full_df.loc[future_start_idx:].copy()
                        X_future = X_future.reindex(columns=mllib_features, fill_value=0)
                        for col in X_future.columns[X_future.isnull().any()].tolist():
                            fill_val = (
                                hist_df_full[col].median()
                                if col in hist_df_full.columns else 0
                            )
                            X_future[col] = X_future[col].fillna(fill_val)

                        all_preds = mllib_predict_batch(
                            model, spark_session, X_future, mllib_features)

                        for j, i in enumerate(range(future_start_idx, len(full_df))):
                            pred_sales = max(0, float(all_preds[j]))
                            current_dow = full_df.loc[i, "Date"].dayofweek + 1
                            current_promo = full_df.loc[i, "Promo"]
                            avg_basket = dynamic_basket.get(
                                (current_dow, current_promo), 10)
                            pred_cust = int(pred_sales / avg_basket) if pred_sales > 0 else 0
                            full_df.loc[i, "Sales"] = pred_sales
                            full_df.loc[i, "Customers"] = pred_cust
                        return full_df

                    df_future_base = create_future_df(future_dates, 0)
                    full_df = pd.concat(
                        [hist_df_full, df_future_base], ignore_index=True)
                    full_df = full_df.sort_values("Date").reset_index(drop=True)
                    future_start_idx = full_df[
                        full_df["Date"] == pd.Timestamp(start_date)].index[0]

                    full_df = run_forecast_scenario(full_df, future_start_idx)
                    df_future = full_df[
                        full_df["Date"] >= pd.Timestamp(start_date)].copy()

                    if use_promo:
                        cycle_pattern = [1] * days_on + [0] * days_off
                        full_pattern = (
                            cycle_pattern * (
                                len(future_dates) // len(cycle_pattern) + 1)
                        )[:len(future_dates)]

                        df_future_promo = create_future_df(future_dates, full_pattern)
                        full_df_promo = pd.concat(
                            [hist_df_full, df_future_promo], ignore_index=True)
                        full_df_promo = full_df_promo.sort_values("Date").reset_index(drop=True)
                        full_df_promo = run_forecast_scenario(
                            full_df_promo, future_start_idx)
                        df_promo_predicted = full_df_promo[
                            full_df_promo["Date"] >= pd.Timestamp(start_date)].copy()

                        st.session_state['df_chart_display'] = df_promo_predicted
                        st.session_state['baseline_report'] = df_future
                        st.session_state['bi_report'] = df_promo_predicted
                    else:
                        st.session_state['df_chart_display'] = df_future
                        st.session_state['baseline_report'] = df_future
                        st.session_state['bi_report'] = None

                st.toast("Dự báo thành công!")

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown("### Biểu đồ doanh thu dự báo")

        if st.session_state.get('df_chart_display') is not None:
            df_baseline = st.session_state['baseline_report']
            df_promo = st.session_state['bi_report']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_baseline['Date'], y=df_baseline['Sales'],
                mode='lines+markers',
                line=dict(color='#F59E0B', width=2, dash='dash'),
                marker=dict(size=4),
                name='Dự báo (Không Promo)'
            ))

            if df_promo is not None:
                fig.add_trace(go.Scatter(
                    x=df_promo['Date'], y=df_promo['Sales'],
                    fill='tonexty', mode='lines+markers',
                    line=dict(color='#E30613', width=3),
                    fillcolor='rgba(227, 6, 19, 0.1)',
                    marker=dict(size=6, color='white',
                                line=dict(width=2, color='#E30613')),
                    name='Dự báo (Có Promo)'
                ))
                promo_days = df_promo[df_promo['Promo'] == 1]
                if not promo_days.empty:
                    fig.add_trace(go.Scatter(
                        x=promo_days['Date'], y=promo_days['Sales'],
                        mode='markers',
                        marker=dict(size=10, color='#10B981', symbol='circle'),
                        name='Ngày kích hoạt Promo'
                    ))

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(showgrid=False, tickformat="%d/%m", color="#6B7280",
                           tickfont=dict(weight='bold')),
                yaxis=dict(showgrid=True, gridcolor='#E5E7EB', gridwidth=1,
                           griddash='dash', color="#6B7280"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom",
                            y=1.02, xanchor="right", x=1),
                height=350, hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Phân tích & Đề xuất")

            if df_promo is not None:
                total_baseline = df_baseline['Sales'].sum()
                total_promo = df_promo['Sales'].sum()
                uplift_value = total_promo - total_baseline
                uplift_pct = (
                    (uplift_value / total_baseline) * 100
                    if total_baseline > 0 else 0
                )

                avg_basket_size = (df_baseline['Sales'].sum()
                                   / df_baseline['Customers'].sum())
                CUSTOMERS_PER_STAFF = 100

                temp_df = df_promo.copy()
                temp_df['Sales_Uplift'] = temp_df['Sales'] - df_baseline['Sales']
                temp_df['Extra_Customers'] = temp_df['Sales_Uplift'] / avg_basket_size
                temp_df['Extra_Staff_Needed'] = np.ceil(
                    temp_df['Extra_Customers'] / CUSTOMERS_PER_STAFF
                ).clip(lower=0)

                promo_days_df = temp_df[temp_df['Promo'] == 1]
                avg_extra_staff = (
                    int(np.ceil(promo_days_df['Extra_Staff_Needed'].mean()))
                    if not promo_days_df.empty else 0
                )

                st.success(f"""
**BÁO CÁO KINH DOANH: HIỆU QUẢ PROMO & NHÂN SỰ**

- **Tổng Sales dự kiến (Không Promo):** {total_baseline:,.0f} €
- **Tổng Sales dự kiến (Có Promo):** {total_promo:,.0f} €

---

- **Doanh thu tăng thêm (Uplift):** +{uplift_value:,.0f} € (+{uplift_pct:.2f}%)

- **Khuyến nghị Nhân sự:** Chỉ tính riêng những ngày BẬT KHUYẾN MÃI, cửa hàng cần bổ sung trung bình **{avg_extra_staff} nhân sự part-time/ngày**.
                """)
            else:
                st.info("Cửa hàng giữ nguyên chiến lược hiện tại (Không Promo).")
        else:
            st.info("Hãy thiết lập thông số và nhấn 'Chạy mô hình' để xem kết quả.")

# TAB 3: Thống kê dữ liệu được dự báo ở TAB 2
with tab_analytics:
    if st.session_state.get('df_chart_display') is None:
        st.info("Chưa có dữ liệu dự báo. Vui lòng quay lại tab 'Dự Báo' để chạy trước.")
    else:
        df_chart = st.session_state['df_chart_display']
        df_baseline = st.session_state['baseline_report']

        total_sales = df_chart['Sales'].sum()
        avg_sales = df_chart['Sales'].mean()
        total_customers = df_chart['Customers'].sum()

        st.markdown("### Báo cáo Tổng quan (BI Report)")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(label="Tổng Doanh Thu Ước Tính", value=f"€{total_sales:,.0f}")
        kpi2.metric(label="Doanh Thu Trung Bình/Ngày", value=f"€{avg_sales:,.0f}")
        kpi3.metric(label="Tổng Lượt Khách", value=f"{total_customers:,.0f}")

        baseline_sales = df_baseline['Sales'].sum()
        if st.session_state.get('bi_report') is not None and baseline_sales > 0:
            lift = total_sales - baseline_sales
            lift_pct = (lift / baseline_sales) * 100
            kpi4.metric(label="Hiệu Quả Promo (Lift)",
                        value=f"+{lift_pct:.2f}%", delta=f"+€{lift:,.0f}")
        else:
            kpi4.metric(label="Hiệu Quả Promo",
                        value="N/A", delta="Kịch bản không có Promo")

        st.markdown("---")

        if st.session_state.get('bi_report') is not None:
            st.markdown("#### So sánh: Kịch bản Promo vs Baseline")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=df_baseline['Date'], y=df_baseline['Sales'],
                mode='lines', name='Baseline (Không Promo)',
                line=dict(color='#9CA3AF', width=2, dash='dash')
            ))
            fig_comp.add_trace(go.Scatter(
                x=df_chart['Date'], y=df_chart['Sales'],
                mode='lines', name='Kịch bản chạy Promo',
                line=dict(color='#E30613', width=2)
            ))
            fig_comp.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(showgrid=False, tickformat="%d/%m", color="#9CA3AF"),
                yaxis=dict(showgrid=True, gridcolor='#374151', gridwidth=1,
                           griddash='dash', color="#9CA3AF"),
                legend=dict(orientation="h", yanchor="bottom",
                            y=1.02, xanchor="right", x=1),
                height=400, hovermode='x unified'
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Bảng Dữ liệu Dự báo Chi tiết")

        display_df = df_chart[['Date', 'Store', 'Promo', 'Sales', 'Customers']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Sales'] = display_df['Sales'].round(2)
        display_df['Customers'] = display_df['Customers'].round(0)
        display_df = display_df.rename(columns={
            'Date': 'Ngày', 'Store': 'Cửa Hàng', 'Promo': 'Có Promo?',
            'Sales': 'Dự Báo Doanh Thu (€)', 'Customers': 'Ước Tính Lượt Khách'
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Đẩy kết quả dự báo lên HDFS")
        st.caption("Kết quả dự báo được lưu trên HDFS để phân tích hoặc truy vấn về sau.")

        # ── Tự sinh tên file theo Store ID + khoảng thời gian dự báo ──────
        _stores = sorted(df_chart["Store"].unique().tolist())
        if len(_stores) == 1:
            _store_tag = f"S{int(_stores[0])}"
        else:
            _store_tag = f"S{int(_stores[0])}-S{int(_stores[-1])}_{len(_stores)}stores"
        _d_from = df_chart["Date"].min().strftime("%Y%m%d")
        _d_to   = df_chart["Date"].max().strftime("%Y%m%d")
        _auto_filename = f"forecast_{_store_tag}_{_d_from}_{_d_to}.csv"

        st.markdown(
            f"**Tên file dự báo:** `{_auto_filename}`  \n"
            f"(Cửa hàng: {_store_tag} · Từ {df_chart['Date'].min():%d/%m/%Y} "
            f"đến {df_chart['Date'].max():%d/%m/%Y})"
        )

        with st.expander("Cấu hình HDFS để upload", expanded=False):
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                upload_hdfs_host = st.text_input(
                    "HDFS Host", value="localhost", key="upload_hdfs_host")
                upload_hdfs_port = st.text_input(
                    "HDFS Port", value="9000", key="upload_hdfs_port")
            with u_col2:
                upload_hdfs_dir = st.text_input(
                    "Thư mục đích trên HDFS",
                    value="/user/project/rossmann/forecasts",
                    key="upload_hdfs_dir"
                )
            upload_hdfs_path = f"{upload_hdfs_dir.rstrip('/')}/{_auto_filename}"
            st.code(
                f"hdfs dfs -put -f /tmp/{_auto_filename} "
                f"hdfs://{upload_hdfs_host}:{upload_hdfs_port}{upload_hdfs_path}"
            )

        if st.button("Upload kết quả lên HDFS", type="primary",
                     key="upload_hdfs_btn"):
            try:
                import subprocess

                export_df = df_chart[['Date', 'Store', 'Promo',
                                      'Sales', 'Customers']].copy()
                export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')
                # Dùng thư mục tạm của hệ điều hành (Windows không có /tmp)
                import tempfile
                tmp_path = os.path.join(tempfile.gettempdir(), _auto_filename)
                export_df.to_csv(tmp_path, index=False)

                hdfs_dest = (f"hdfs://{upload_hdfs_host}:"
                             f"{upload_hdfs_port}{upload_hdfs_path}")

                hdfs_cmd = _find_hdfs_cmd()
                if hdfs_cmd is None:
                    st.error("Không tìm thấy lệnh `hdfs`. Kiểm tra Hadoop đã cài "
                             "và đặt biến môi trường HADOOP_HOME.")
                    st.stop()

                result = subprocess.run(
                    [hdfs_cmd, "dfs", "-put", "-f", tmp_path, hdfs_dest],
                    capture_output=True, text=True, timeout=60
                )

                if result.returncode == 0:
                    st.session_state["forecast_on_hdfs"] = True
                    st.session_state["forecast_hdfs_uri"] = hdfs_dest
                    st.session_state["forecast_row_count"] = len(export_df)
                    st.success(f"""
**Upload thành công!**

- **File:** `{_auto_filename}`
- **Số dòng:** {len(export_df):,}
- **HDFS path:** `{hdfs_dest}`

File đã được lưu trên HDFS thành công.
                    """)
                else:
                    st.error(f"HDFS lỗi: {result.stderr}")
                    st.markdown("""
**Kiểm tra nhanh:**
- HDFS đang chạy? -> `start-dfs.sh`
- Thư mục đích đã tồn tại? -> `hdfs dfs -mkdir -p /user/project/rossmann/`
- Biến môi trường HADOOP_HOME đã set?
                    """)
            except FileNotFoundError:
                st.error("Không tìm thấy lệnh `hdfs`. Kiểm tra Hadoop đã cài và PATH đã set.")
            except subprocess.TimeoutExpired:
                st.error("Timeout khi kết nối HDFS. Kiểm tra HDFS có đang chạy không.")
            except Exception as e:
                st.error(f"Lỗi không xác định: {e}")



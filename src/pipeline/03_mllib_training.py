import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, lit

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

spark = (
    SparkSession.builder
    .appName("Rossmann_03_MLlib")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

HDFS_ROOT = "hdfs://localhost:9000/user/project/rossmann"

features_str = "\n".join(spark.sparkContext.textFile(f"{HDFS_ROOT}/features.json").collect())
features = json.loads(features_str)
print("Features:", len(features))

sdf = spark.read.csv(f"{HDFS_ROOT}/df_features.csv", header=True, inferSchema=True)
print("Loaded df_features:", sdf.count(), "rows,", len(sdf.columns), "cols")

total       = sdf.count()
train_count = int(total * 0.8)

w   = Window.orderBy(lit(1))
sdf = sdf.withColumn("_row_id", row_number().over(w))

train_sdf = sdf.filter(F.col("_row_id") <= train_count).drop("_row_id")
test_sdf  = sdf.filter(F.col("_row_id") >  train_count).drop("_row_id")

train_sdf.cache()
test_sdf.cache()
print(f"Số mẫu train : {train_sdf.count():,}")
print(f"Số mẫu test  : {test_sdf.count():,}")

assembler = VectorAssembler(inputCols=features, outputCol="features_vec", handleInvalid="skip")
scaler    = StandardScaler(inputCol="features_vec", outputCol="features_scaled",
                           withStd=True, withMean=False)

evaluator_rmse = RegressionEvaluator(labelCol="Sales", predictionCol="prediction", metricName="rmse")
evaluator_mae  = RegressionEvaluator(labelCol="Sales", predictionCol="prediction", metricName="mae")
evaluator_r2   = RegressionEvaluator(labelCol="Sales", predictionCol="prediction", metricName="r2")

#Phân tích kế hoạch thực thi
print("EXPLAIN: Kế hoạch tiền xử lý trên train_sdf")
print()
_explain_model = Pipeline(stages=[assembler, scaler]).fit(train_sdf)
_explain_model.transform(train_sdf).explain(mode="formatted")

tune_sdf, _ = train_sdf.randomSplit([0.1, 0.9], seed=42)
tune_sdf.cache()
print(f"Số mẫu tune  : {tune_sdf.count():,}  (10% train)")

print(f"\n{'='*55}\n  Tuning: RandomForestRegressor\n{'='*55}")
rf_tune = RandomForestRegressor(featuresCol="features_scaled", labelCol="Sales", seed=42)
pipeline_rf_tune = Pipeline(stages=[assembler, scaler, rf_tune])
rf_param_grid = (
    ParamGridBuilder()
    .addGrid(rf_tune.numTrees,            [50, 100])
    .addGrid(rf_tune.maxDepth,            [6, 8])
    .addGrid(rf_tune.minInstancesPerNode, [2])
    .build()
)
cv_rf = CrossValidator(estimator=pipeline_rf_tune, estimatorParamMaps=rf_param_grid,
                       evaluator=evaluator_rmse, numFolds=2, seed=42, parallelism=1)
print("Đang chạy CrossValidator RF (4 tổ hợp × 2 folds = 8 fits)...")
cv_rf_model = cv_rf.fit(tune_sdf)

best_rf_stage    = cv_rf_model.bestModel.stages[-1]
best_rf_numTrees = best_rf_stage.getNumTrees
best_rf_maxDepth = best_rf_stage.getMaxDepth()
best_rf_minInst  = best_rf_stage.getMinInstancesPerNode()
print(f"  RF best numTrees           : {best_rf_numTrees}")
print(f"  RF best maxDepth           : {best_rf_maxDepth}")
print(f"  RF best minInstancesPerNode: {best_rf_minInst}")

print(f"\n{'='*55}\n  Tuning: GBTRegressor\n{'='*55}")
gbt_tune = GBTRegressor(featuresCol="features_scaled", labelCol="Sales", seed=42)
pipeline_gbt_tune = Pipeline(stages=[assembler, scaler, gbt_tune])
gbt_param_grid = (
    ParamGridBuilder()
    .addGrid(gbt_tune.maxIter,  [50, 100])
    .addGrid(gbt_tune.maxDepth, [4, 6])
    .addGrid(gbt_tune.stepSize, [0.1])
    .build()
)
cv_gbt = CrossValidator(estimator=pipeline_gbt_tune, estimatorParamMaps=gbt_param_grid,
                        evaluator=evaluator_rmse, numFolds=2, seed=42, parallelism=1)
print("Đang chạy CrossValidator GBT (4 tổ hợp × 2 folds = 8 fits)...")
cv_gbt_model = cv_gbt.fit(tune_sdf)

best_gbt_stage    = cv_gbt_model.bestModel.stages[-1]
best_gbt_maxIter  = best_gbt_stage.getMaxIter()
best_gbt_maxDepth = best_gbt_stage.getMaxDepth()
best_gbt_stepSize = best_gbt_stage.getStepSize()
print(f"  GBT best maxIter  : {best_gbt_maxIter}")
print(f"  GBT best maxDepth : {best_gbt_maxDepth}")
print(f"  GBT best stepSize : {best_gbt_stepSize}")

lr = LinearRegression(featuresCol="features_scaled", labelCol="Sales",
                      maxIter=100, regParam=0.01, elasticNetParam=0.0)
rf = RandomForestRegressor(featuresCol="features_scaled", labelCol="Sales",
                           numTrees=best_rf_numTrees, maxDepth=best_rf_maxDepth,
                           minInstancesPerNode=best_rf_minInst, seed=42)
gbt = GBTRegressor(featuresCol="features_scaled", labelCol="Sales",
                   maxIter=best_gbt_maxIter, maxDepth=best_gbt_maxDepth,
                   stepSize=best_gbt_stepSize, subsamplingRate=0.8, seed=42)

model_configs = {
    "LinearRegression":      lr,
    "RandomForestRegressor": rf,
    "GBTRegressor":          gbt,
}
mllib_results = {}
mllib_models  = {}
for name, estimator in model_configs.items():
    print(f"\n{'='*55}\n  Training: {name}\n{'='*55}")
    pipeline = Pipeline(stages=[assembler, scaler, estimator])
    fitted   = pipeline.fit(train_sdf)
    preds    = fitted.transform(test_sdf)

    rmse = evaluator_rmse.evaluate(preds)
    mae  = evaluator_mae.evaluate(preds)
    r2   = evaluator_r2.evaluate(preds)
    mllib_results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    mllib_models[name]  = fitted
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  R²   : {r2:.4f}")

results_df = pd.DataFrame(mllib_results).T.sort_values("RMSE")
print("\nModel Comparison (Spark MLlib):")
print(results_df.to_string())

best_model_name  = results_df["RMSE"].idxmin()
best_mllib_model = mllib_models[best_model_name]
print(f"\nModel tốt nhất: {best_model_name}")

def report_importance(fitted_pipeline, feature_names, model_name, top_n=20):
    stage = fitted_pipeline.stages[-1]
    print(f"\n{'='*55}\n  Feature importance: {model_name}\n{'='*55}")
    if hasattr(stage, "featureImportances"):
        vals = stage.featureImportances.toArray()
        ranked = sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)
        for nm, v in ranked[:top_n]:
            print(f"  {nm:30s} {v:.5f}")
        return ranked
    elif hasattr(stage, "coefficients"):
        vals = stage.coefficients.toArray()
        ranked = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)
        for nm, v in ranked[:top_n]:
            print(f"  {nm:30s} {v:+.5f}")
        return ranked
    return []


importance_tables = {}
for name, fitted in mllib_models.items():
    importance_tables[name] = report_importance(fitted, features, name)

best_stage = best_mllib_model.stages[-1]
if hasattr(best_stage, "featureImportances"):
    top = importance_tables[best_model_name][:15][::-1]
    names_top = [t[0] for t in top]
    vals_top  = [t[1] for t in top]
    plt.figure(figsize=(10, 7))
    plt.barh(names_top, vals_top, color="teal")
    plt.title(f"Top 15 Feature Importance – {best_model_name}", weight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    
x, width = np.arange(2), 0.25
names = list(mllib_results.keys())
fig, ax = plt.subplots(figsize=(12, 6))
for idx, name in enumerate(names):
    vals = [mllib_results[name]["MAE"], mllib_results[name]["RMSE"]]
    bars = ax.bar(x + idx * width, vals, width, label=name)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{int(val)}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + width * (len(names) - 1) / 2)
ax.set_xticklabels(["MAE", "RMSE"])
ax.set_title("So sánh MAE và RMSE của các mô hình Spark MLlib")
ax.set_ylabel("Sai số")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

MODEL_SAVE_PATH = f"{HDFS_ROOT}/models/{best_model_name}"
best_mllib_model.write().overwrite().save(MODEL_SAVE_PATH)
print(f"Đã lưu model tại: {MODEL_SAVE_PATH}")

spark.stop()
print("[03] Done.")
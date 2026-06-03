# inspect_pkl

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------------------

#              ------ 2026/5/26 欄位完整顯示  --------

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# --------------------------------------------------------------

"""def inspect_pkl(tables:dict) -> None:
    pkl_path = base_dir/".cache"/"特徵工程完成後中繼資料.pkl")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("PKL type:", type(data))
    print()
    
    # 如果是dict (最常見)
    if isinstance(data, dict):
        print('Tables inside PKL:')
        for k, v in data.items():
            print('-'*40)
            print("key:", k)
            print("type:", type(v))
            if isinstance(v, pd.DataFrame):   
                print("shape", v.shape)
                print("columns", v.columns.tolist())
    
    # 如果直接是DataFrame
    elif isinstance(v, pd.DataFrame):
        print("This PKL is a single DataFrame")
        print(data.shape)
        print(data.columns.to_list())
    
    else:
        print("Unknown Structure")"""

# =====================================================================================================================

# ----------                                       選擇欄位有Label 作為建模主表                                   ------

#  =====================================================================================================================

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report, # precison,recall, F1-score
    roc_auc_score,         # ROC curve
    confusion_matrix,      # Confusion Matrix
    precision_recall_curve   # 找最適threshold
)
from sklearn.impute import SimpleImputer
import joblib

# =========================
#       1. 基本設定
# =========================
base_dir = Path(__file__).parent
# PKL_PATH = base_dir/".cache"/"特徵工程完成後中繼資料.pkl"
# MODEL_PATH = base_dir/".cache"/"hawkeye_xgb_model.pkl"
# FEATURE_IMPORTANCE_PATH = base_dir/".cache"/"hawkeye_feature_importance.csv"

# 2026/6/2 加入2025/7資料
PKL_PATH = base_dir/".cache"/"特徵工程完成後中繼資料_202507.pkl"
MODEL_PATH = base_dir/".cache"/"hawkeye_xgb_model_202507.pkl"
FEATURE_IMPORTANCE_PATH = base_dir/".cache"/"hawkeye_feature_importance_202507.csv"

#PKL_PATH = Path(r"D:\數據分析科\鷹眼\data\.cache\特徵工程完成後中繼資料.pkl")
#MODEL_PATH = Path(r"D:\數據分析科\鷹眼\data\.cache\hawkeye_xgb_model.pkl")
#FEATURE_IMPORTANCE_PATH = Path(r"D:\數據分析科\鷹眼\data\.cache\hawkeye_feature_importance.csv")

print('特徵路徑',FEATURE_IMPORTANCE_PATH.exists())
LABEL_COL = "label"

# 若你的主表 key 不是這個，改這裡
#MAIN_TABLE_KEY = "訓練或預測資料集"

# 因為需要重新切割交易日期，故選擇該張報表 2026/3/10
MAIN_TABLE_KEY = "特徵"

# 明顯不該直接進模型的欄位
DROP_COLS = [LABEL_COL, "交易日期", "帳號",
              "客戶身分證或統編", "交易日期時間", "轉入或轉出交易",
              "交易分別行", "交易筆次", "Session", "Flow",
              "開戶日期" ]
'''
DROP_COLS = [
    LABEL_COL,
    "acct_nbr_ori",
    "AccNo",
    "IssueDt",
    "DATA_DT",
    "customer_id",
    "cust_id",
]
'''
RANDOM_STATE = 42
#TEST_SIZE = 0.2


# =========================
# 2. 輔助函式
# =========================
def print_basic_info(df: pd.DataFrame) -> None:   # -> None (型別註記)不會印出任何資料
    print("=" * 60)
    print("Data shape:", df.shape)
    print("Columns count:", len(df.columns))
    print("First 10 columns:", df.columns.tolist()[:10])
    print("=" * 60)

    if LABEL_COL in df.columns:
        print("Label distribution:")
        print(df[LABEL_COL].value_counts(dropna=False))
        print(df[LABEL_COL].value_counts(normalize=True, dropna=False))
        print("=" * 60)


def find_main_table(tables: dict) -> pd.DataFrame:
    print("Available table keys:")
    for k, v in tables.items():
        if isinstance(v, pd.DataFrame):
            print(f"- {k}: shape={v.shape}")
        else:
            print(f"- {k}: type={type(v)}")

    # 檢查是否為DataFrame
    if MAIN_TABLE_KEY in tables:
        df = tables[MAIN_TABLE_KEY]
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{MAIN_TABLE_KEY} 不是 DataFrame")
        return df

    raise KeyError(
        f"找不到主表 key: {MAIN_TABLE_KEY}，請先看上方 keys 後修改 MAIN_TABLE_KEY"
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if LABEL_COL not in df.columns:
        raise KeyError(f"找不到標籤欄位：{LABEL_COL}") # 無key值

    y = df[LABEL_COL]
    # 只保留 0 / 1
    valid_mask = y.isin([0, 1])
    df = df.loc[valid_mask]
    y = y.loc[valid_mask].astype(int)

    # 先移除不該進模型的欄位
    X = df.drop(columns=DROP_COLS, errors="ignore")

    # 先只留數值與布林欄位
    X = X.select_dtypes(include=["number", "bool"])
    X = reduce_memory_usage(X)

    # 布林轉成 0/1
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("int8")

    # 去掉全空欄
    X = X.dropna(axis = 1, how = "all")


    #  2026/3 降型別，省記憶體
    float_cols = X.select_dtypes(include=["float64"]).columns
    int_cols = X.select_dtypes(include=["int64"]).columns

    if len(float_cols) > 0:
        X[float_cols] = X[float_cols].astype("float32")

    if len(int_cols) > 0:
        X[int_cols] = X[int_cols].astype("int32")

    print("Prepared X shape:", X.shape)
    print("Prepared y shape:", y.shape)
    print("Positive count", int((y == 1).sum()))
    print("Negative count", int((y == 0).sum()))
    return X, y

# 2026/2 記憶體超出 先註解掉,XGBoost本身可處理空值Nan

def split_and_impute(
        X_train: pd.DataFrame,  # 型別提示 也可以只寫x
        X_test: pd.DataFrame
        ):
        #test_size = TEST_SIZE,
        #random_state = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series,pd.Series,SimpleImputer]:

        # 缺值補中位數
        imputer = SimpleImputer(strategy = "median")
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns = X_train.columns,
            index = X_train.index
        )
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test),
            columns = X_test.columns,
            index = X_test.index
        )

        print("X_train_imputed shape:", X_train_imputed.shape)
        print("X_test_imputed shape:", X_test_imputed.shape)
        # print("y_train Positive:", int((y_train == 1).sum()))
        # print("y_test Positive:", int((y_train == 1).sum()))

        return X_train_imputed, X_test_imputed, imputer  #為了保持後續新資料補植方式一致，顧回傳imputer


# KS Statistic 分析 區分好壞客戶比例 [max(|累積壞客戶比例 - 累積好客戶比例|)]

# =============================

# 0.1        幾乎沒用
# 0.2-0.3    普通
# 0.4-0.6    不錯
# 0.6+       很好
# 銀行多數為0.45~0.65
# =============================


# AUC  分析 區分模型好壞
# =============================

# 0.9-1      優秀
# 0.8-0.9      良好
# 0.7-0.8      普通
# 0.5-0.7      差
#0.5以下       WORSE THAN GUESSING

# =============================

def calc_ks(y_true: pd.Series, y_score: pd.Series) -> float:  # y_true:真實標籤、y_prob:模型預測機率
    tmp = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False) # 照預測機率排序,fraud 最前面

    # 計算總數
    total_bad = (tmp["y"] == 1).sum()
    total_good = (tmp["y"] == 0).sum()
    # 防止極端情形 (避免分母為0)
    if total_bad == 0 or total_good == 0:
        return 0.0

    # 累積比例
    tmp["cum_bad"] = (tmp["y"] == 1).cumsum() / total_bad
    tmp["cum_good"] = (tmp["y"] == 0).cumsum() / total_good
    ks = (tmp["cum_bad"] - tmp["cum_good"]).abs().max()
    return float(ks)

# SKY 2026/3/11 降低記憶體
def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_type):

            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min >= -128 and c_max <= 127:
                    df[col] = df[col].astype("int8")
                elif c_min >= -32768 and c_max <= 32767:
                    df[col] = df[col].astype("int16")
                elif c_min >= -2147483648 and c_max <= 2147483647:
                    df[col] = df[col].astype("int32")

            elif str(col_type)[:5] == "float":
                df[col] = df[col].astype("float32")

    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage before:", round(start_mem,2), "MB")
    print("Memory usage after :", round(end_mem,2), "MB")
    print("Reduced:", round(100*(start_mem-end_mem)/start_mem,2), "%")
    return df

# =========================
# 3. 主流程
# =========================
def main() -> None:
    # 讀 pkl
    print("Loading pkl...")
    with open(PKL_PATH, "rb") as f:
        tables = pickle.load(f)

    # 檢視是否正確讀取到正確的表
    print("tables.keys:", tables.keys())
    print("MAIN_TABLE_KEY:", MAIN_TABLE_KEY)
    print("df.columns:", tables[MAIN_TABLE_KEY].columns.tolist())


    if not isinstance(tables, dict):
        raise TypeError("pkl 內容不是 dict，請先確認 pickle 結構")
    # 找主表
    df = find_main_table(tables)
    print_basic_info(df)

    # ====== 時間切割 ======
    # train:2025/8 ~ 2025/12
    # test: 2026/1 ~ 2026/1

    # 轉換日期型態
    df["交易日期"] = pd.to_datetime(df["交易日期"], errors = "coerce")

    # test 格式
    print("交易日期 原始 dtype:", df["交易日期"].dtype)
    print("交易日期 原始前20筆:")
    print(df["交易日期"].head(20).tolist())
    print("交易日期 原始去重前20筆:")
    print(df["交易日期"].astype("string").drop_duplicates().head(20).tolist())

# --------------------------------------------------
#          ★★★  記得修改日期 ★★★
    split_date = pd.Timestamp("2026-01-01")
# --------------------------------------------------

    print("交易日期 轉換後前20筆:")
    print(df["交易日期"].head(20).tolist())
    print("交易日期 NaT 筆數:", df["交易日期"].isna().sum())
    print("交易日期 最小值:", df["交易日期"].min())
    print("交易日期 最大值:", df["交易日期"].max())
    print("交易日期月份分佈:")
    print(df["交易日期"].dt.to_period("M").value_counts().sort_index())

    train_df = df[df["交易日期"] < split_date]
    test_df = df[df["交易日期"] >= split_date]

    print("Train_period:", train_df["交易日期"].min(), "to", train_df["交易日期"].max())
    print("Test_period:", test_df["交易日期"].min(), "to", test_df["交易日期"].max())
    print("Train label distribution:")
    print(train_df[LABEL_COL].value_counts(dropna=False))
    print("\nTest label distribution:")
    print(test_df[LABEL_COL].value_counts(dropna=False))
    print('\n',test_df.shape)
    print(
        '\n',test_df.groupby(
            ['帳號','交易日期時間']).size().sort_values(ascending = False).reset_index(name = '重複筆數').head(20)
    )
    print("=" * 60)

    # raise ValueError(值存在，但內容不合法): 主動丟出錯誤或停止程式  / raise keyError: 存取的key不存在
    if train_df.empty or test_df.empty:
        raise ValueError("train_df 或 test_df 為空，請確認split_date")

    # 準備 X / y
    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)
    print("Prepared X_train shape:", X_train.shape)
    print("Prepared X_test shape:", X_test.shape)
    print("Prepared y_train positive", int((y_train == 1).sum()))
    print("Prepared y_test positive:", int((y_test == 1).sum()))
    print("=" * 60)


    # 補值中位數

    X_train_imputed,X_test_imputed,imputer = split_and_impute(X_train, X_test)  #因記憶體超量 先註解掉
    # X_train_imputed = X_train
    # X_test_imputed =X_test
    # imputer = None

#    if y.nunique() < 2:
#        raise ValueError("Label 只有單一類別，無法建模")

    # 處理類別不平衡
    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())

    if pos_count == 0:
        raise ValueError("訓練集沒有正樣本，無法訓練")

    scale_pos_weight = neg_count / pos_count
    print("scale_pos_weight:", scale_pos_weight)
    print("=" * 60)

    # 建模
    model = XGBClassifier(
        n_estimators = 300,
        max_depth = 6,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 1,
        reg_lambda = 1.0,
        objective = "binary:logistic",
        eval_metric = "aucpr",              #類別極度不平衡使用 'aucpr'
        random_state = RANDOM_STATE,
        scale_pos_weight = scale_pos_weight,
        n_jobs = -1,  # CPU算力全開

    )

    print("Training XGBOOST_XGclassifier...")
    model.fit(X_train_imputed, y_train)
    # AML leakage test
    # y_train_shuffle = np.random.permutation(y_train)
    #model.fit(X_train_imputed, y_train_shuffle) # SKY 2026/3/10

    # 預測
    #y_pred = model.predict(X_test_imputed)
    y_prob = model.predict_proba(X_test_imputed)[:, 1]

    # SKY 2026/4/22檢視為什麼產出交易明細都是重複
    print("len(X_test_imputed):", len(X_test_imputed))
    print("len(y_prob):", len(y_prob))
    print("len(test_df):", len(test_df))


    for threshold in [0.8, 0.9, 0.95, 0.99,0.995,0.999]:
        y_pred_temp = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_temp)
        tn, fp, fn, tp = cm.ravel()
        model_list = tp+fp
        print("threshold:", threshold)
        print("Confusion Matrix",cm)
        print(classification_report(y_test, y_pred_temp, digits=4))
        print("模型名單數", model_list)
        print("命中數", tp)
        print("-" * 60)



    # 評估指標
    auc = roc_auc_score(y_test, y_prob)
    ks = calc_ks(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred_temp)     #混淆矩陣(真實,預測) , 參數不可對調位置

    #sample_idx = np.random.choice(len(y_test), size = 100000, replace = False)
    #precision, recall, thresholds = precision_recall_curve(y_test.iloc[sample_idx], y_prob[sample_idx]) #PR CURVE
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)  # PR CURVE

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.savefig("precision-recall_curve.png", bbox_inches="tight")
    plt.show()

    # 計算 F1

    f1 = 2*(precision * recall)/(precision + recall + 1e-9)
    # 找最適threshold
    best_idx = f1.argmax()
    best_threshold = thresholds[best_idx]
    print("best_threshold (F1):", best_threshold)
    print("F1", f1[best_idx])

    # 模擬不同的t,n
    for t in [0.8, 0.9, 0.95, 0.99,0.995,0.999]:
        n = (y_prob >= t).sum()
        #n = (y_prob[sample_idx] >= t).sum()
        print(f'threshold = {t}, 名單數 = {n}')

    print("AUC:", round(auc, 6))
    print("KS :", round(ks, 6))
    print("=" * 60)
    print("Confusion Matrix:", cm)
    print("=" * 60)
    print("Classification Report:")
    print(classification_report(y_test, y_pred_temp, digits = 4))
    #print(classification_report(y_test.iloc[sample_idx], y_pred_temp[sample_idx], digits=4))
    print("=" * 60)
    print("precision", precision)
    print("\nrecall", recall)
    print("\nthresholds")  # numpy.ndarray格式 不能使用apply(lambda x: f"{x:.9f}")調整格式，另外生成list格式一定要加 ,[f'{x:.9f}' for x in thresholds]
    print("=" * 60)
    # 預測分佈機率
    print("Pred prob summary:")

    print(pd.Series(y_prob).describe().apply(lambda x: f"{x:.9f}"))
    #print(pd.Series(y_prob)[sample_idx].describe())
    print("=" * 60)

    # 特徵重要度
    feature_importance = pd.DataFrame({
        "feature": X_train_imputed.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending = False)

    print("Top 15 Feature Importances:")
    print(feature_importance.head(15))
    print("=" * 60)

    # 檢視異常名單明細 (y_pred == 1)
    for threshold in [0.8, 0.9, 0.95, 0.99,0.995,0.999]:
        #threshold = best_threshold
        raws = y_prob >= threshold
        cols = ['帳號', '交易日期時間', '帳戶餘額','當日交易總轉出', 'Session 連續轉入或轉出次數']
        model_list_df = test_df.loc[raws, cols]
        model_list_df['fraud_prob'] = y_prob[raws]        # 詐欺機率
        model_list_df['y_pred'] = (y_prob[raws] >= threshold).astype(int)        # 預測結果
        model_list_df['y_true'] = y_test.loc[raws].values # 真實標籤
        #model_list_df['Flow 平均交易時間差_new'] = pd.to_timedelta(model_list_df['Flow 平均交易時間差'], unit='s')  # 轉換為分:秒


        # 取出模型名單   #'Flow 平均交易時間差_new', '單筆交易餘額占比'
        cols_new = ['帳號', '交易日期時間','帳戶餘額','當日交易總轉出','Session 連續轉入或轉出次數']

        model_list_df = model_list_df.sort_values("fraud_prob", ascending = False)
        tx_list_df  = model_list_df[cols_new]
        print('\n交易資訊',tx_list_df.head(20))
        print('交易名單數', len(tx_list_df))
        print("=" * 70)

        # 依序將 0.8, 0.9, 0.95, 0.99,0.995,0.999產出至xlsx檔案
        threshold_str = str(threshold).replace('.', '')
        #model_list_df.to_excel(f'交易資訊模型名單_{threshold_str}.xlsx', index=False)
        model_list_df.to_excel(f'交易資訊模型名單_{threshold_str}_202507.xlsx', index=False) #2026/6/2 新增
        # 歸納為帳戶層
        acct_list_df = (model_list_df
                      .sort_values('fraud_prob', ascending = False)
                      .groupby('帳號', as_index=False).first()
                      .sort_values('fraud_prob', ascending = False))

        print('\n帳戶資訊', acct_list_df.head(20))
        print('帳戶名單數', len(acct_list_df))
        print("=" * 70)

        # 依序將 0.8, 0.9, 0.95, 0.99,0.995,0.999產出至xlsx檔案
        threshold_str = str(threshold).replace('.', '')
        #acct_list_df.to_excel(f'帳戶資訊模型名單_{threshold_str}.xlsx', index=False)
        acct_list_df.to_excel(f'帳戶資訊模型名單_{threshold_str}_202507.xlsx', index=False) #2026/6/2 新增
    # 統計
    # 存檔
    joblib.dump(
        {
        # model
        "model" : model,
        "imputer" : imputer,
        # metadata
        "features" : X_train_imputed.columns.to_list(),
        "split_date" : split_date,


        # 2026/5/28 新增 為了在notebook讀取使用
        # train
        "y_train" : y_train,
        "train_prob": model.predict_proba(X_train_imputed)[:,1], # 前面沒命令過，與y_brob 一樣要先命令
        # test
        "y_test" : y_test,
        "test_prob" : y_prob,
        "X_test" : X_test_imputed,
        # 交易日期, 帳號
        "test_meta" : test_df[['帳號','交易日期時間','交易金額','交易分行別','交易通路','label','約轉設定通路']].reset_index(drop = True),
        "train_meta": train_df[['帳號','交易日期時間','交易金額','交易分行別','交易通路', 'label' ,'約轉設定通路']].reset_index(drop = True)
        }, MODEL_PATH)

    # =======================================

    # 2026/5/26 檢查pkl內容
    pkl_path = base_dir / ".cache" / "hawkeye_xgb_model_202507.pkl"
    with open(pkl_path, 'rb') as f:
        data = joblib.load(f)
    print(type(data))
    print(data.keys())  # 應該是dict類型
    print(type(data['features']))
    print(data['features'])
    #print(data['model'][data['features']['label'] == 1].head(10))
    print(type(data['model']))
    print(data['model'])
    pd.set_option('display.float_format', '{:.6f}'.format)
    print(data['model'].feature_importances_)

    # =======================================
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index = False, encoding = "utf-8-sig")
    print("Model saved to:", MODEL_PATH)
    print("Feature importance saved to:", FEATURE_IMPORTANCE_PATH)

# 控制程式入口
if __name__ == "__main__":   # 啟動開關
    main()

































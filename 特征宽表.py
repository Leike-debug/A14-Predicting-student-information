import os
import glob
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# ================== 配置区域 ==================
DATA_PATH = r"D:\github\A14-Predicting-student-information\data\translated"
OUTPUT_PATH = r"D:\github\A14-Predicting-student-information\data\features"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
os.chdir(DATA_PATH)

# ================== 高鲁棒性通用工具函数 ==================
def load_data(keyword):
    """【优化 1】合并读取：解决同名分表被漏读的问题"""
    files = glob.glob(f"*{keyword}*.csv")
    if not files:
        print(f"  [跳过] 未找到包含 '{keyword}' 的数据表。")
        return None
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if '学号_统一' in df.columns:
                dfs.append(df)
        except Exception as e:
            print(f"  [读取异常] {f}: {e}")
            
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

def find_col(df, keywords):
    """【优化 2】收紧匹配规则：优先精确匹配，再采用带黑名单的正则包含匹配"""
    for col in df.columns:
        if str(col).upper().strip() in keywords:
            return col
    
    black_list = ['UPDATE', 'CREATE', '更新', '创建', '操作']
    for col in df.columns:
        col_str = str(col).upper().strip()
        if any(k in col_str for k in keywords) and not any(b in col_str for b in black_list):
            return col
    return None

def calc_burstiness(series):
    """【优化 4】增加限制：过滤低频噪声，保证行为阵发性计算的稳定性"""
    if len(series) < 5: 
        return 0.0
    intervals = series.dropna().sort_values().diff().dt.total_seconds().dropna()
    if len(intervals) < 2: 
        return 0.0
    mean_int, std_int = intervals.mean(), intervals.std()
    if std_int + mean_int == 0: 
        return 0.0
    return (std_int - mean_int) / (std_int + mean_int)

def calc_entropy(series):
    """计算时间分布的信息熵"""
    counts = series.value_counts()
    if len(counts) == 0: return 0.0
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

def calc_max_consecutive_days(dates_series):
    """【新增】计算行为连续性：最长连续活跃天数"""
    dates = pd.to_datetime(dates_series, errors='coerce').dt.floor('D').drop_duplicates().sort_values().reset_index(drop=True)
    if len(dates) == 0: 
        return 0
    day_diff = dates.diff().dt.days
    streak_id = (~(day_diff == 1)).cumsum()
    return int(streak_id.value_counts().max())

# ================== 核心提取模块 ==================

def extract_category_1_economic():
    print("[处理中] 类别1：隐性经济压力与生存状态...")
    df = load_data("一卡通交易")
    if df is None: return pd.DataFrame(columns=['学号_统一'])
    
    amt_col = find_col(df, ['交易额', '金额', 'AMOUNT', 'MONEY'])
    type_col = find_col(df, ['商户类型', '交易类型', 'TYPE'])
    
    if amt_col:
        df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        is_canteen = df[type_col].str.contains('食堂|餐饮|餐', na=False) if type_col else pd.Series(True, index=df.index)
        
        canteen_spend = df[is_canteen].groupby('学号_统一')[amt_col].sum()
        total_spend = df.groupby('学号_统一')[amt_col].sum()
        spend_std = df.groupby('学号_统一')[amt_col].std().fillna(0)
        
        feat = pd.DataFrame({
            '经济_食堂消费占比(恩格尔)': (canteen_spend / (total_spend + 1e-5)).fillna(0),
            '经济_消费波动率': spend_std
        }).reset_index()
        return feat
    return pd.DataFrame(columns=['学号_统一'])

def extract_category_2_study():
    print("[处理中] 类别2：学习投入、死线拖延与连贯性...")
    dfs = []
    
    df_hw = load_data("作业提交")
    if df_hw is not None:
        sub_time = find_col(df_hw, ['提交时间', 'SUBMIT'])
        ddl_time = find_col(df_hw, ['截止时间', 'DEADLINE'])
        if sub_time and ddl_time:
            df_hw[sub_time] = pd.to_datetime(df_hw[sub_time], errors='coerce')
            df_hw[ddl_time] = pd.to_datetime(df_hw[ddl_time], errors='coerce')
            df_hw['is_rush'] = ((df_hw[ddl_time] - df_hw[sub_time]).dt.total_seconds() < 7200).astype(int)
            hw_feat = df_hw.groupby('学号_统一').agg(
                学习_死线突击次数=('is_rush', 'sum'),
                学习_作业总数=('学号_统一', 'count')
            ).reset_index()
            hw_feat['学习_死线突击率'] = hw_feat['学习_死线突击次数'] / (hw_feat['学习_作业总数'] + 1e-5)
            dfs.append(hw_feat)

    df_ol = load_data("线上学习")
    if df_ol is not None:
        login_col = find_col(df_ol, ['登录', '访问', 'LOGIN'])
        post_col = find_col(df_ol, ['发帖', '讨论', 'POST'])
        if login_col and post_col:
            df_ol[login_col] = pd.to_numeric(df_ol[login_col], errors='coerce')
            df_ol[post_col] = pd.to_numeric(df_ol[post_col], errors='coerce')
            ol_feat = df_ol.groupby('学号_统一').agg(
                学习_线上交互深度=(post_col, 'sum')
            ).reset_index()
            dfs.append(ol_feat)
            
    df_lib = load_data("图书馆打卡")
    if df_lib is not None:
        time_col = find_col(df_lib, ['时间', 'TIME', 'DATE'])
        # 🔥【修复的核心点】：使用标准的 ('列名', '函数') 元组格式！
        agg_funcs = {'学习_图书馆打卡总频次': ('学号_统一', 'count')}
        if time_col:
            agg_funcs['学习_最长连续自习天数'] = (time_col, calc_max_consecutive_days)
            
        lib_feat = df_lib.groupby('学号_统一').agg(**agg_funcs).reset_index()
        dfs.append(lib_feat)

    if dfs:
        return reduce(lambda left, right: pd.merge(left, right, on='学号_统一', how='outer'), dfs)
    return pd.DataFrame(columns=['学号_统一'])

def extract_category_3_sports():
    print("[处理中] 类别3：运动阵发性与意志力连贯性...")
    df = load_data("跑步打卡")
    if df is not None:
        time_col = find_col(df, ['时间', 'TIME', 'DATE'])
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col]).sort_values(['学号_统一', time_col])
            
            feat = df.groupby('学号_统一').agg(
                体育_运动阵发指数=(time_col, calc_burstiness),
                体育_总运动次数=(time_col, 'count'),
                体育_最长连续运动天数=(time_col, calc_max_consecutive_days)
            ).reset_index()
            return feat
    return pd.DataFrame(columns=['学号_统一'])

def extract_category_4_grades():
    print("[处理中] 类别4：学业轨迹与破窗效应...")
    df = load_data("成绩")
    if df is not None:
        score_col = find_col(df, ['分数', '成绩', 'SCORE'])
        term_col = find_col(df, ['学期', 'TERM'])
        if score_col:
            df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
            df['is_fail'] = (df[score_col] < 60).astype(int)
            
            feat = df.groupby('学号_统一').agg(
                学业_挂科总门数=('is_fail', 'sum'),
                学业_平均分=(score_col, 'mean')
            ).reset_index()
            
            if term_col:
                term_gpa = df.groupby(['学号_统一', term_col])[score_col].mean().reset_index()
                volatility = term_gpa.groupby('学号_统一')[score_col].std().reset_index(name='学业_成绩波动率方差')
                feat = pd.merge(feat, volatility, on='学号_统一', how='left')
            return feat
    return pd.DataFrame(columns=['学号_统一'])

def extract_category_5_extracurricular():
    print("[处理中] 类别5：课外拓展与综合效能...")
    df = load_data("学科竞赛")
    if df is not None:
        level_col = find_col(df, ['级别', '等级', 'LEVEL'])
        if level_col:
            weight_map = {'国家级': 5, '省级': 3, '市级': 2, '校级': 1, '院级': 0.5}
            df['竞赛权重'] = df[level_col].map(lambda x: weight_map.get(str(x).strip(), 1))
            feat = df.groupby('学号_统一')['竞赛权重'].sum().reset_index(name='拓展_核心胜任力总分')
            return feat
    return pd.DataFrame(columns=['学号_统一'])

def extract_category_6_life():
    print("[处理中] 类别6：作息剥夺与时空精细熵...")
    dfs = []
    
    df_net = load_data("上网统计")
    if df_net is not None:
        time_col = find_col(df_net, ['时长', 'DURATION'])
        start_col = find_col(df_net, ['上线时间', '开始时间', 'START'])
        if time_col and start_col:
            df_net[time_col] = pd.to_numeric(df_net[time_col], errors='coerce')
            df_net[start_col] = pd.to_datetime(df_net[start_col], errors='coerce')
            df_net['is_midnight'] = df_net[start_col].dt.hour.isin([0, 1, 2, 3, 4, 5])
            
            midnight_dur = df_net[df_net['is_midnight']].groupby('学号_统一')[time_col].sum()
            total_dur = df_net.groupby('学号_统一')[time_col].sum()
            
            midnight_days = df_net[df_net['is_midnight']].groupby('学号_统一').agg(
                生活_最长连续熬夜天数=(start_col, calc_max_consecutive_days)
            )
            
            net_feat = pd.DataFrame({
                '生活_作息剥夺率(深夜上网)': (midnight_dur / (total_dur + 1e-5)).fillna(0)
            }).reset_index()
            net_feat = pd.merge(net_feat, midnight_days, on='学号_统一', how='left')
            dfs.append(net_feat)

    df_door = load_data("门禁数据")
    if df_door is not None:
        time_col = find_col(df_door, ['时间', 'TIME'])
        if time_col:
            df_door[time_col] = pd.to_datetime(df_door[time_col], errors='coerce')
            df_door['time_bin'] = df_door[time_col].dt.hour * 2 + df_door[time_col].dt.minute // 30
            
            door_feat = df_door.groupby('学号_统一').agg(
                生活_晚归频次=('time_bin', lambda x: x.isin([46, 47, 0, 1, 2, 3, 4]).sum()),
                生活_时空精细节律熵=('time_bin', calc_entropy)
            ).reset_index()
            dfs.append(door_feat)

    if dfs:
        return reduce(lambda left, right: pd.merge(left, right, on='学号_统一', how='outer'), dfs)
    return pd.DataFrame(columns=['学号_统一'])

# ================== 跨域交叉特征构建 ==================
def build_cross_domain_features(master_df):
    print("[处理中] 融合阶段：构建高解释性跨域耦合特征...")
    
    if '学习_图书馆打卡总频次' in master_df.columns and '学业_平均分' in master_df.columns:
        pct_lib = master_df['学习_图书馆打卡总频次'].rank(pct=True, na_option='bottom')
        pct_gpa = master_df['学业_平均分'].rank(pct=True, na_option='bottom')
        master_df['交叉_伪勤奋背离指数'] = pct_lib - pct_gpa
        
    risk_cols = ['学业_挂科总门数', '生活_作息剥夺率(深夜上网)', '学习_死线突击率']
    exist_cols = [c for c in risk_cols if c in master_df.columns]
    
    if exist_cols:
        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(master_df[exist_cols].fillna(0))
        master_df['交叉_破窗效应高危指数'] = scaled_vals.sum(axis=1)

    return master_df

# ================== 引擎主控 ==================
def main():
    print("="*60)
    print("🚀 核心群体行为特征提取引擎 (高鲁棒性版)")
    print("="*60)

    f1 = extract_category_1_economic()
    f2 = extract_category_2_study()
    f3 = extract_category_3_sports()
    f4 = extract_category_4_grades()
    f5 = extract_category_5_extracurricular()
    f6 = extract_category_6_life()

    features_list = [f for f in [f1, f2, f3, f4, f5, f6] if not f.empty]
    
    if not features_list:
        print("[错误] 未提取到任何有效特征，请检查源数据。")
        return

    print("\n[处理中] 聚合全体样本大宽表 (Master Table)...")
    master_df = reduce(lambda x, y: pd.merge(x, y, on='学号_统一', how='outer'), features_list)

    num_cols = master_df.select_dtypes(include=[np.number]).columns
    master_df[num_cols] = master_df[num_cols].fillna(0)

    master_df = build_cross_domain_features(master_df)

    output_file = os.path.join(OUTPUT_PATH, "高频行为特征大宽表_Master.csv")
    master_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("="*60)
    print(f"🎉 特征工程执行完毕！输出路径: {output_file}")
    print(f"📊 特征矩阵维度: {master_df.shape[0]} 名学生, {master_df.shape[1]-1} 项精选核心指标")
    print("✨ 已全面提升容错率、指标精确度及极值抵抗能力。")

if __name__ == "__main__":
    main()
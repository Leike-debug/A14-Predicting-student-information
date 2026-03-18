import pandas as pd
import numpy as np
import os
import glob
from functools import reduce
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# ================== 配置路径 ==================
INPUT_DIR = r"D:\github\A14-Predicting-student-information\data\translated" 
OUTPUT_DIR = r"D:\github\A14-Predicting-student-information\data\features"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================== 基础工具 ==================
def normalize_time(series):
    return pd.to_datetime(series, errors='coerce')

def normalize_id(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

def find_col(df, keywords):
    for col in df.columns:
        if any(k in str(col).upper() for k in keywords):
            return col
    return None

def calc_max_consecutive_days(date_series):
    dates = pd.to_datetime(date_series).dt.floor('D').dropna().drop_duplicates().sort_values()
    if len(dates) <= 1: return len(dates)
    return (dates.diff().dt.days != 1).cumsum().value_counts().max()


# ================== 核心：带“学号自动捕获”的强力路由 ==================
def load_and_route_files():
    print("[阶段1] 扫描并严格按文档架构路由源数据表...")
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    grouped_dfs = {
        'basic': [], 'study': [], 'sports': [], 
        'score': [], 'activity': [], 'life': []
    }
    
    for f in files:
        fname = os.path.basename(f)
        try:
            df = pd.read_csv(f, low_memory=False, on_bad_lines='skip', dtype=str)
            if df.empty: continue
            
            # 🔥 核心修复：自动捕获学号列，防止因列名不叫“学号_统一”而被静默丢弃！
            id_col = None
            for col in df.columns:
                if any(x in str(col).upper() for x in ['学号', 'XH', 'STUDENT', 'XUEHAO']):
                    id_col = col
                    break
                    
            if not id_col:
                print(f"  [跳过警告] {fname}: 找不到包含'学号'的列！")
                continue
                
            # 强制重命名为统一下板
            df.rename(columns={id_col: '学号_统一'}, inplace=True)
            df['学号_统一'] = normalize_id(df['学号_统一'])
            # 剔除真的没有学号数据的空行
            df = df[df['学号_统一'].notna() & (df['学号_统一'] != '') & (df['学号_统一'] != 'nan')]
            
            if df.empty: continue
            
            df['来源文件'] = fname 
            
            # 🚀 严格按照截图路由
            if any(k in fname for k in ['基本信息', '异动', '去向', '课程信息']):
                grouped_dfs['basic'].append(df)
            elif any(k in fname for k in ['体育课', '选课', '上课', '作业', '考试', '签到', '课堂', '线上', '讨论', '图书', '考勤']):
                grouped_dfs['study'].append(df)
            elif any(k in fname for k in ['体能考核', '体测', '日常锻炼', '跑步']):
                grouped_dfs['sports'].append(df)
            elif any(k in fname for k in ['成绩', '四六级', '测评', '奖学金']):
                grouped_dfs['score'].append(df)
            elif any(k in fname for k in ['竞赛', '社团']):
                grouped_dfs['activity'].append(df)
            elif any(k in fname for k in ['门禁', '上网']):
                grouped_dfs['life'].append(df)
            else:
                grouped_dfs['basic'].append(df)
                
        except Exception as e:
            print(f"  [读取报错] {fname}: {e}")
            
    print("\n[路由结果最终核对]：")
    for k, v in grouped_dfs.items():
        files_in_category = [df['来源文件'].iloc[0] for df in v]
        print(f"  -> {k} 类别成功装入 {len(v)} 张表:")
        for fn in files_in_category:
            print(f"     |- {fn}")
            
    return grouped_dfs


# ================== 1 基本信息 (🔥 彻底解决合并内存爆炸) ==================
def process_basic_features(dfs):
    if not dfs: return
    
    # 用 concat 替代 merge，彻底杜绝同名列相乘导致的 MemoryError
    combined_df = pd.concat(dfs, ignore_index=True)
    
    cols_to_drop = [c for c in combined_df.columns if '来源文件' in c or '时间' in c or '代码' in c]
    feat = combined_df.drop(columns=cols_to_drop, errors='ignore')
    
    # 按照学号聚合去重
    feat = feat.groupby('学号_统一').first().reset_index()

    out_path = os.path.join(OUTPUT_DIR, "初步特征宽表_1_基本信息.csv")
    feat.to_csv(out_path, index=False, encoding='utf-8-sig')


# ================== 2 学习记录 ==================
def process_study_features(dfs):
    if not dfs: return
    
    time_series_list = []
    for df in dfs:
        t_col = find_col(df, ['时间', '日期', 'DATE', 'TIME'])
        if t_col:
            tmp = df[['学号_统一', t_col]].rename(columns={t_col: '时间'})
            time_series_list.append(tmp)
            
    if not time_series_list: return
    
    df = pd.concat(time_series_list, ignore_index=True)
    df['学号_统一'] = normalize_id(df['学号_统一'])
    df['时间'] = normalize_time(df['时间'])
    df = df.dropna(subset=['时间']).sort_values(['学号_统一', '时间'])

    df['hour'] = df['时间'].dt.hour
    df['weekday'] = df['时间'].dt.dayofweek
    df['month'] = df['时间'].dt.month
    df['diff_hours'] = df.groupby('学号_统一')['时间'].diff().dt.total_seconds() / 3600

    feat = df.groupby('学号_统一').agg(
        学习_交互总次数=('时间', 'count'),
        学习_活跃总天数=('时间', lambda x: x.dt.date.nunique()),
        学习_最长连续活跃天数=('时间', calc_max_consecutive_days),
        学习_交互平均间隔_小时=('diff_hours', 'mean'),
        学习_交互最大间隔_小时=('diff_hours', 'max'),
        学习_间隔标准差=('diff_hours', 'std'),
        学习_晨间时段占比_6至9点=('hour', lambda x: x.isin([6,7,8,9]).mean()),
        学习_日间时段占比_10至17点=('hour', lambda x: x.isin([10,11,12,13,14,15,16,17]).mean()),
        学习_夜间时段占比_18至22点=('hour', lambda x: x.isin([18,19,20,21,22]).mean()),
        学习_深夜时段占比_23至5点=('hour', lambda x: x.isin([23,0,1,2,3,4,5]).mean()),
        学习_周末活跃占比=('weekday', lambda x: (x >= 5).mean()),
        学习_期末死线突击比例=('month', lambda x: x.isin([6, 7, 12, 1]).mean())
    ).reset_index().fillna(0)
    
    out_path = os.path.join(OUTPUT_DIR, "初步特征宽表_2_学习记录.csv")
    feat.to_csv(out_path, index=False, encoding='utf-8-sig')


# ================== 3 体育锻炼 ==================
def process_sports_features(dfs):
    if not dfs: return
    
    time_series_list = []
    for df in dfs:
        t_col = find_col(df, ['时间', '日期', 'DATE'])
        if t_col:
            tmp = df[['学号_统一', t_col]].rename(columns={t_col: '时间'})
            time_series_list.append(tmp)
            
    if not time_series_list: return
    df = pd.concat(time_series_list, ignore_index=True)
    df['学号_统一'] = normalize_id(df['学号_统一'])
    df['时间'] = normalize_time(df['时间'])
    df = df.dropna(subset=['时间']).sort_values(['学号_统一', '时间'])

    df['diff_days'] = df.groupby('学号_统一')['时间'].diff().dt.total_seconds() / (3600*24)
    df['hour'] = df['时间'].dt.hour
    df['weekday'] = df['时间'].dt.dayofweek

    feat = df.groupby('学号_统一').agg(
        体育_运动总次数=('时间', 'count'),
        体育_活跃运动天数=('时间', lambda x: x.dt.date.nunique()),
        体育_最长连续坚持天数=('时间', calc_max_consecutive_days),
        体育_平均运动间隔天数=('diff_days', 'mean'),
        体育_最大运动间隔天数=('diff_days', 'max'),
        体育_打卡阵发方差=('diff_days', 'std'),
        体育_晨练时段占比_5至8点=('hour', lambda x: x.isin([5,6,7,8]).mean()),
        体育_夜跑时段占比_18至22点=('hour', lambda x: x.isin([18,19,20,21,22]).mean()),
        体育_周末运动占比=('weekday', lambda x: (x >= 5).mean())
    ).reset_index().fillna(0)

    out_path = os.path.join(OUTPUT_DIR, "初步特征宽表_3_体育锻炼.csv")
    feat.to_csv(out_path, index=False, encoding='utf-8-sig')


# ================== 4 成绩与荣誉 ==================
def process_score_features(dfs):
    if not dfs: return
    
    score_list = []
    for df in dfs:
        s_col = find_col(df, ['成绩', '分数', 'SCORE', '总分', '测评'])
        if s_col:
            tmp = df[['学号_统一', s_col]].rename(columns={s_col: '分数'})
            score_list.append(tmp)
            
    if not score_list: return
    df = pd.concat(score_list, ignore_index=True)
    df['学号_统一'] = normalize_id(df['学号_统一'])
    df['分数'] = pd.to_numeric(df['分数'], errors='coerce')
    df = df.dropna(subset=['分数'])

    feat = df.groupby('学号_统一').agg(
        成绩_总记录数=('分数', 'count'),
        成绩_平均分=('分数', 'mean'),
        成绩_最高分=('分数', 'max'),
        成绩_最低分=('分数', 'min'),
        成绩_中位数=('分数', 'median'),
        成绩_偏科方差=('分数', 'std'),
        成绩_优秀率_大于85分=('分数', lambda x: (x >= 85).mean()), 
        成绩_良好率_70至84分=('分数', lambda x: ((x >= 70) & (x < 85)).mean()),
        成绩_及格边缘率_60至69分=('分数', lambda x: ((x >= 60) & (x < 70)).mean()),
        成绩_挂科率_低于60分=('分数', lambda x: (x < 60).mean())
    ).reset_index().fillna(0)

    out_path = os.path.join(OUTPUT_DIR, "初步特征宽表_4_成绩与荣誉.csv")
    feat.to_csv(out_path, index=False, encoding='utf-8-sig')


# ================== 5 课外拓展 ==================
def process_activity_features(dfs):
    if not dfs: return
    
    df = pd.concat(dfs, ignore_index=True)
    df['学号_统一'] = normalize_id(df['学号_统一'])
    
    feat = df.groupby('学号_统一').agg(
        拓展_参与总次数=('学号_统一', 'count'),
        拓展_参与的跨域种类数=('来源文件', 'nunique') 
    ).reset_index().fillna(0)

    out_path = os.path.join(OUTPUT_DIR, "初步特征宽表_5_课外拓展.csv")
    feat.to_csv(out_path, index=False, encoding='utf-8-sig')


# ================== 6 生活轨迹 ==================
def process_life_features(dfs):
    if not dfs: return
    
    life_list = []
    for df in dfs:
        t_col = find_col(df, ['时间', '日期', '上线'])
        loc_col = find_col(df, ['地点', '位置', '终端', '门禁'])
        if t_col:
            tmp = df[['学号_统一', t_col]].rename(columns={t_col: '时间'})
            if loc_col: tmp['地点'] = df[loc_col]
            else: tmp['地点'] = '未知'
            life_list.append(tmp)
            
    if not life_list: return
    df = pd.concat(life_list, ignore_index=True)
    df['学号_统一'] = normalize_id(df['学号_统一'])
    df['时间'] = normalize_time(df['时间'])
    df = df.dropna(subset=['时间']).sort_values(['学号_统一', '时间'])
    
    df['diff_hours'] = df.groupby('学号_统一')['时间'].diff().dt.total_seconds() / 3600
    df['hour'] = df['时间'].dt.hour
    df['weekday'] = df['时间'].dt.dayofweek
    
    # 时空共现预警
    df['time_bin'] = df['时间'].dt.floor('2min')
    group_sizes = df.groupby(['地点', 'time_bin']).size().reset_index(name='crowd_size')
    df = pd.merge(df, group_sizes, on=['地点', 'time_bin'], how='left')
    df['is_social'] = (df['crowd_size'] > 1).astype(int)
    
    feat = df.groupby('学号_统一').agg(
        生活_出行上网总次数=('时间', 'count'),
        生活_活跃天数=('时间', lambda x: x.dt.date.nunique()),
        生活_行为平均间隔_小时=('diff_hours', 'mean'),
        生活_极端最大间隔_小时=('diff_hours', 'max'),
        生活_时间节律标准差=('hour', 'std'),
        生活_日间正常活跃占比_8至18点=('hour', lambda x: x.isin(range(8, 19)).mean()),
        生活_夜间活跃占比_19至23点=('hour', lambda x: x.isin([19,20,21,22,23]).mean()),
        生活_凌晨修仙占比_0至5点=('hour', lambda x: x.isin([0,1,2,3,4,5]).mean()),
        生活_周末外溜占比=('weekday', lambda x: (x >= 5).mean()),
        生活_隐性社交孤岛预警率=('is_social', lambda x: 1 - x.mean())
    ).reset_index().fillna(0)

    out_path = os.path.join(OUTPUT_DIR, "初步特征宽表_6_生活轨迹.csv")
    feat.to_csv(out_path, index=False, encoding='utf-8-sig')




# ================== 主流程 ==================
def main():
    print("="*60)
    print("🚀 防漏防爆版：全源数据特征抽取与降维引擎启动")
    print("="*60)
    
    grouped_dfs = load_and_route_files()
    
    print("\n[阶段2] 开始提取六大维度底层特征矩阵...")
    process_basic_features(grouped_dfs['basic'])
    process_study_features(grouped_dfs['study'])
    process_sports_features(grouped_dfs['sports'])
    process_score_features(grouped_dfs['score'])
    process_activity_features(grouped_dfs['activity'])
    process_life_features(grouped_dfs['life'])



if __name__ == "__main__":
    main()
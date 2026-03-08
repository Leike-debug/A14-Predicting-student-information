import os
import pandas as pd
import numpy as np

# 忽略 pandas 的链式赋值警告
pd.options.mode.chained_assignment = None

# 设置数据工作目录
DATA_PATH = r"D:\github\A14-Predicting-student-information\data"
os.chdir(DATA_PATH)

def find_col(df, keyword):
    """根据关键字进行列名的模糊匹配"""
    for col in df.columns:
        if keyword.upper() in col.upper():
            return col
    return None

def load_data(filename, id_keyword):
    """加载原始数据并标准化学生主键列"""
    filepath = f"翻译后_{filename}.csv"
    if not os.path.exists(filepath):
        print(f"[提示] 文件未找到，跳过处理: {filepath}")
        return None
    
    df = pd.read_csv(filepath, low_memory=False)
    id_col = find_col(df, id_keyword)
    
    if id_col:
        df.rename(columns={id_col: '学号_统一'}, inplace=True)
        # 清理学号字段：转换为字符串格式，去除可能存在的.0后缀及首尾空格
        df['学号_统一'] = df['学号_统一'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        return df
    else:
        print(f"[警告] 数据集 '{filename}' 未匹配到主键标识。")
        return None

# ================== 1. 基本信息类 ==================
def process_basic_features():
    print("[处理中] 类别1：基本信息与静态维度...")
    df_base = load_data("学生基本信息", "XH")
    if df_base is None:
        return

    # 关联学籍异动统计
    df_yd = load_data("学籍异动", "XH")
    if df_yd is not None:
        yd_feat = df_yd.groupby('学号_统一').size().reset_index(name='基础_累计异动次数')
        df_base = pd.merge(df_base, yd_feat, on='学号_统一', how='left')

    # 关联毕业去向
    df_by = load_data("毕业去向", "SID")
    if df_by is not None:
        target_col = find_col(df_by, "BYQXMC")
        if target_col:
            by_feat = df_by.drop_duplicates('学号_统一')[['学号_统一', target_col]]
            df_base = pd.merge(df_base, by_feat, on='学号_统一', how='left')

    df_base.fillna(0).to_csv("高级特征_1_基本信息.csv", index=False, encoding='utf-8-sig')

# ================== 2. 学习记录类 ==================
def process_study_features():
    print("[处理中] 类别2：学习记录与过程评价...")
    master = pd.DataFrame(columns=['学号_统一'])

    # 考勤状态分析 (重点提取负面状态：旷课/迟到)
    df_kq = load_data("考勤汇总表", "XH")
    if df_kq is not None:
        status_col = find_col(df_kq, "ZT")
        if status_col:
            kq_feat = df_kq.groupby('学号_统一').agg(
                学习_累计旷课次数=(status_col, lambda x: (x == '旷课').sum()),
                学习_迟到早退次数=(status_col, lambda x: x.isin(['迟到', '早退']).sum())
            ).reset_index()
            master = pd.merge(master, kq_feat, on='学号_统一', how='outer')

    # 图书馆时序偏好分析 (提取周末自习指标)
    df_lib = load_data("图书馆打卡记录", "cardld")
    if df_lib is not None:
        time_col = find_col(df_lib, "visittime")
        if time_col:
            df_lib[time_col] = pd.to_datetime(df_lib[time_col], errors='coerce')
            df_lib['is_weekend'] = df_lib[time_col].dt.dayofweek.isin([5, 6]).astype(int)
            lib_feat = df_lib.groupby('学号_统一').agg(
                学习_周末自习打卡次数=('is_weekend', 'sum'),
                学习_自习打卡总数=('is_weekend', 'count')
            ).reset_index()
            lib_feat['学习_周末自习占比'] = lib_feat['学习_周末自习打卡次数'] / (lib_feat['学习_自习打卡总数'] + 1e-5)
            master = pd.merge(master, lib_feat, on='学号_统一', how='outer')

    # 作业提交时序与质量分析
    df_hw = load_data("学生作业提交记录", "CREATER_LOGIN_NAME")
    if df_hw is not None:
        time_col = find_col(df_hw, "ANSWER_TIME")
        score_col = find_col(df_hw, "SCORE")
        if time_col and score_col:
            df_hw[time_col] = pd.to_datetime(df_hw[time_col], errors='coerce')
            df_hw[score_col] = pd.to_numeric(df_hw[score_col], errors='coerce')
            hw_feat = df_hw.groupby('学号_统一').agg(
                学习_作业平均得分=(score_col, 'mean'),
                学习_深夜提交作业次数=(time_col, lambda x: (x.dt.hour >= 23).sum())
            ).reset_index()
            master = pd.merge(master, hw_feat, on='学号_统一', how='outer')

    master.fillna(0).to_csv("高级特征_2_学习记录.csv", index=False, encoding='utf-8-sig')

# ================== 3. 体育锻炼类 ==================
def process_sports_features():
    print("[处理中] 类别3：体育锻炼与体质考核...")
    master = pd.DataFrame(columns=['学号_统一'])

    # 跑步锻炼稳定性分析 (计算两次跑步的平均间隔时间)
    df_run = load_data("跑步打卡", "USERNUM")
    if df_run is not None:
        time_col = find_col(df_run, "PUNCH_DAY")
        if time_col:
            df_run[time_col] = pd.to_datetime(df_run[time_col], errors='coerce')
            df_run = df_run.dropna(subset=[time_col]).sort_values(['学号_统一', time_col])
            df_run['date_diff'] = df_run.groupby('学号_统一')[time_col].diff().dt.days
            run_feat = df_run.groupby('学号_统一').agg(
                体育_打卡总次数=('学号_统一', 'count'),
                体育_打卡平均间隔天数=('date_diff', 'mean')
            ).reset_index()
            master = pd.merge(master, run_feat, on='学号_统一', how='outer')

    # 体测成绩评估
    df_tc = load_data("体测数据", "XH")
    if df_tc is not None:
        score_col = find_col(df_tc, "ZF")
        if score_col:
            df_tc[score_col] = pd.to_numeric(df_tc[score_col], errors='coerce')
            tc_feat = df_tc.groupby('学号_统一')[score_col].last().reset_index(name='体育_最新体测总分')
            master = pd.merge(master, tc_feat, on='学号_统一', how='outer')

    master.fillna(0).to_csv("高级特征_3_体育锻炼.csv", index=False, encoding='utf-8-sig')

# ================== 4. 成绩与荣誉类 ==================
def process_grades_features():
    print("[处理中] 类别4：学业成绩与荣誉奖项...")
    df_cj = load_data("学生成绩", "XH")
    if df_cj is None: return

    score_col = find_col(df_cj, "KCCJ")
    credit_col = find_col(df_cj, "XF")
    
    if score_col and credit_col:
        df_cj[score_col] = pd.to_numeric(df_cj[score_col], errors='coerce')
        df_cj[credit_col] = pd.to_numeric(df_cj[credit_col], errors='coerce')
        df_cj = df_cj.dropna(subset=[score_col, credit_col])
        
        # 计算学分加权指标及挂科风险权重
        df_cj['加权得分'] = df_cj[score_col] * df_cj[credit_col]
        df_cj['是否挂科'] = (df_cj[score_col] < 60).astype(int)
        df_cj['挂科学分'] = df_cj['是否挂科'] * df_cj[credit_col]
        
        cj_feat = df_cj.groupby('学号_统一').agg(
            学业_总修读学分=(credit_col, 'sum'),
            学业_加权总得分=('加权得分', 'sum'),
            学业_挂科总学分=('挂科学分', 'sum'),
            学业_绝对挂科门数=('是否挂科', 'sum')
        ).reset_index()
        
        cj_feat['学业_学分加权平均分'] = cj_feat['学业_加权总得分'] / (cj_feat['学业_总修读学分'] + 1e-5)
        cj_feat['学业_挂科学分占比'] = cj_feat['学业_挂科总学分'] / (cj_feat['学业_总修读学分'] + 1e-5)
        
        # 字段精简
        cj_feat = cj_feat[['学号_统一', '学业_总修读学分', '学业_学分加权平均分', '学业_挂科学分占比', '学业_绝对挂科门数']]

        # 关联奖学金经济激励指标
        df_jxj = load_data("奖学金获奖", "XSBH")
        if df_jxj is not None:
            amt_col = find_col(df_jxj, "FFJE")
            if amt_col:
                df_jxj[amt_col] = pd.to_numeric(df_jxj[amt_col], errors='coerce')
                jxj_feat = df_jxj.groupby('学号_统一')[amt_col].sum().reset_index(name='学业_奖学金累计总额')
                cj_feat = pd.merge(cj_feat, jxj_feat, on='学号_统一', how='left')

        cj_feat.fillna(0).to_csv("高级特征_4_成绩与荣誉.csv", index=False, encoding='utf-8-sig')

# ================== 5. 课外拓展类 ==================
def process_extra_features():
    print("[处理中] 类别5：课外实践与拓展素质...")
    master = pd.DataFrame(columns=['学号_统一'])

    df_js = load_data("学科竞赛", "XHHGH")
    if df_js is not None:
        level_col = find_col(df_js, "HJJB")
        if level_col:
            # 竞赛等级定级量化模型
            weight_mapping = {'国家级': 5.0, '省级': 3.0, '市级': 2.0, '校级': 1.0, '院级': 0.5}
            df_js['竞赛权重得分'] = df_js[level_col].map(lambda x: weight_mapping.get(str(x).strip(), 0.5) if pd.notnull(x) else 0)
            
            js_feat = df_js.groupby('学号_统一').agg(
                拓展_竞赛参与总次数=('学号_统一', 'count'),
                拓展_竞赛综合权重得分=('竞赛权重得分', 'sum')
            ).reset_index()
            master = pd.merge(master, js_feat, on='学号_统一', how='outer')

    master.fillna(0).to_csv("高级特征_5_课外拓展.csv", index=False, encoding='utf-8-sig')

# ================== 6. 生活轨迹类 ==================
def process_life_features():
    print("[处理中] 类别6：生活轨迹与规律性预警...")
    master = pd.DataFrame(columns=['学号_统一'])

    # 门禁异常作息指标计算 (统计23:00以后回寝记录)
    df_mj = load_data("门禁数据", "IDSERTAL")
    if df_mj is not None:
        time_col = find_col(df_mj, "LOGINTIME")
        if time_col:
            df_mj[time_col] = pd.to_datetime(df_mj[time_col], errors='coerce')
            mj_feat = df_mj.groupby('学号_统一').agg(
                生活_门禁刷卡总次数=('学号_统一', 'count'),
                生活_晚归异常次数=(time_col, lambda x: (x.dt.hour >= 23).sum())
            ).reset_index()
            master = pd.merge(master, mj_feat, on='学号_统一', how='outer')

    # 网络使用强度分析
    df_net = load_data("上网统计", "XSBH")
    if df_net is not None:
        time_col = find_col(df_net, "SWLJSC")
        if time_col:
            df_net[time_col] = pd.to_numeric(df_net[time_col], errors='coerce')
            net_feat = df_net.groupby('学号_统一').agg(
                生活_网络使用累计时长=(time_col, 'sum'),
                生活_单次网络平均使用时长=(time_col, 'mean')
            ).reset_index()
            master = pd.merge(master, net_feat, on='学号_统一', how='outer')

    master.fillna(0).to_csv("高级特征_6_生活轨迹.csv", index=False, encoding='utf-8-sig')

# ================== 主程序执行 ==================
if __name__ == "__main__":
    print("开始执行全维度高级特征工程脚本...\n" + "="*45)
    process_basic_features()
    process_study_features()
    process_sports_features()
    process_grades_features()
    process_extra_features()
    process_life_features()
    print("="*45 + "\n特征工程处理完成。目标输出文件已生成在指定目录。")
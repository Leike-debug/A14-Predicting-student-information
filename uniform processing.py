import os
import glob
import pandas as pd
from docx import Document
from DOCX_parsing import parse_comma_separated_docx

class DataTranslator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.translated_data = {}
        
    def find_file_pairs(self):
        xlsx_dir = os.path.join(self.data_path, 'excel type')
        xlsx_files = glob.glob(os.path.join(xlsx_dir, '*.xlsx'))
        xlsx_names = [os.path.basename(f).replace('.xlsx', '') for f in xlsx_files]
        
        docx_dir = os.path.join(self.data_path, 'doc type')
        docx_files = glob.glob(os.path.join(docx_dir, '*.docx'))
        docx_names = [os.path.basename(f).replace('.docx', '') for f in docx_files]  
        
        pairs = []
        for name in set(xlsx_names) & set(docx_names):
            pairs.append({
                'name': name,
                'xlsx': os.path.join(xlsx_dir, f'{name}.xlsx'),
                'docx': os.path.join(docx_dir, f'{name}.docx')
            })
        print(f"找到 {len(pairs)} 对匹配的文件")
        return pairs

    def translate_all(self):
        pairs = self.find_file_pairs()
        for pair in pairs:
            print(f"\n处理: {pair['name']}")
            try:
                df = pd.read_excel(pair['xlsx'])
            except Exception as e:
                print(f"❌ 读取Excel失败: {e}")
                continue
                
            mapping = parse_comma_separated_docx(pair['docx'])
            
            if mapping:
                # ================= 1. 翻译表头 =================
                new_columns = {}
                for col in df.columns:
                    clean_col = str(col).strip()
                    if clean_col in mapping:
                        new_columns[col] = f"{clean_col}({mapping[clean_col]})"
                df.rename(columns=new_columns, inplace=True)
                print(f"  ✅ 成功翻译了 {len(new_columns)} 个表头字段")
                
                # ================= 2. 全自动生成“代号含义”辅助列 =================
                # 把纯数字的代号拿出来，比如 {'1': '因病休学', '001': '必修'}
                val_dict = {k: v for k, v in mapping.items() if str(k).isdigit()}
                
                if val_dict:
                    # 遍历现在的所有列
                    for col in df.columns:
                        valid_vals = df[col].dropna().astype(str)
                        if valid_vals.empty: 
                            continue
                        
                        # 核心：处理Pandas自动把整数 1 变成浮点数 '1.0' 的情况
                        sample_vals = valid_vals.head(30).apply(lambda x: x[:-2] if x.endswith('.0') else x)
                        
                        # 判断：如果这列抽样的数据，有30%以上都能在代号字典里找到
                        matched_count = sample_vals.isin(val_dict.keys()).sum()
                        if matched_count > 0 and matched_count >= len(sample_vals) * 0.3:
                            
                            # 确定这是代码列，开始生成辅助列
                            helper_col_name = f"{col}_含义"
                            
                            def map_value(x):
                                if pd.isna(x): return x
                                sx = str(x).strip()
                                if sx.endswith('.0'): sx = sx[:-2] # 去除 .0
                                return val_dict.get(sx, x) # 翻译，找不到的保留原数字
                                
                            # 插入到原列的右侧
                            df.insert(df.columns.get_loc(col) + 1, helper_col_name, df[col].apply(map_value))
                            print(f"  ✨ 自动识别并生成了辅助列: [{helper_col_name}]")
                
                # ================= 3. 保存最终数据 =================
                output_path = os.path.join(self.data_path, f'翻译后_{pair["name"]}.csv')
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                self.translated_data[pair['name']] = df
                print(f"  💾 已保存: {output_path}")
            else:
                print(f"  ⚠️ 没有从docx提取到有效的字典映射")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    DATA_FOLDER = r"D:\github\A14-Predicting-student-information\data"  
    translator = DataTranslator(data_path=DATA_FOLDER)
    translator.translate_all()
    print("\n🎉 批量处理任务完美完成！")
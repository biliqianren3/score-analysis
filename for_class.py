import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import io
from PIL import Image
import hashlib
import easyocr

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

# ============ 辅助函数 ============

@st.cache_resource
def get_ocr_reader():
    """
    初始化并返回 easyocr Reader 实例。
    语言列表：简体中文 + 英文（可根据需要调整）
    """
    return easyocr.Reader(['ch_sim', 'en'])  # 如需纯英文可改为 ['en']

def validate_and_clean_data(df):
    """验证和清理数据（严格模式：缺失必需列时报错）"""
    try:
        cleaned_df = df.copy()
        # ... 函数体 ...
        return cleaned_df
    except Exception as e:
        st.error(f"数据验证失败: {e}")
        return None

def process_uploaded_file(uploaded_file, file_id, overwrite=False):
    """
    处理上传的文件，转换为系统内部长表格式，并合并到 dashboard_data。
    """
    # --- 读取文件 ---
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return

    # --- 检查必需列 ---
    required = ['学号', '姓名', '考试名称']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"文件缺少必需列：{missing}，请确保列名正确或先进行列映射。")
        return

    # --- 自动检测科目 ---
    detected = detect_subjects_from_columns(df.columns)
    st.caption(f"文件检测到：物理/历史={detected['main']}，选考={detected['optional']}")

    if detected['main'] and detected['main'] != st.session_state.physics_or_history:
        st.warning(f"检测到物理/历史科目为【{detected['main']}】，但当前设置为【{st.session_state.physics_or_history}】。")
    if set(detected['optional']) and set(detected['optional']) != set(st.session_state.selected_two):
        st.warning(f"检测到选考科目为 {detected['optional']}，但当前设置为 {st.session_state.selected_two}。")

    # --- 根据格式转换 ---
    format_type = detect_format_type(df)
    subjects = ["语文", "数学", "英语", st.session_state.physics_or_history] + st.session_state.selected_two

    if format_type == '宽表格式':
        processed_df = convert_wide_to_long(df, subjects)
        if processed_df is None or processed_df.empty:
            st.error("❌ 转换后无有效成绩数据。请检查科目设置是否与文件列名匹配。")
            return
    elif format_type == '长表格式':
        processed_df = df.copy()
    else:
        st.error("无法识别文件格式，请确保文件包含正确的列名")
        return

    # --- 验证和清理数据 ---
    cleaned_data = validate_and_clean_data(processed_df)
    if cleaned_data is None:
        return

    # --- 添加来源文件信息 ---
    cleaned_data['来源文件'] = uploaded_file.name
    cleaned_data['来源文件标识'] = file_id

    # --- 定义关键字段（用于去重） ---
    key_cols = ['学号', '姓名', '考试名称', '科目', '分数类型']
    added_count = 0  # 初始化新增记录数

    # --- 合并到 dashboard_data ---
    if overwrite:
        # 覆盖模式：直接合并（调用者已删除旧数据）
        if st.session_state.dashboard_data.empty:
            st.session_state.dashboard_data = cleaned_data
        else:
            st.session_state.dashboard_data = pd.concat(
                [st.session_state.dashboard_data, cleaned_data],
                ignore_index=True
            )
        added_count = len(cleaned_data)
    else:
        # 追加模式：检查重复记录，仅添加不存在的记录
        if st.session_state.dashboard_data.empty:
            st.session_state.dashboard_data = cleaned_data
            added_count = len(cleaned_data)
        else:
            # 基于关键字段去重，保留第一次出现的行（即原 dashboard_data 中的行优先）
            merged = pd.concat([st.session_state.dashboard_data, cleaned_data], ignore_index=True)
            merged_deduplicated = merged.drop_duplicates(subset=key_cols, keep='first')
            added_count = len(merged_deduplicated) - len(st.session_state.dashboard_data)
            st.session_state.dashboard_data = merged_deduplicated
            if added_count < len(cleaned_data):
                st.warning(f"检测到 {len(cleaned_data) - added_count} 条重复记录（基于学号、姓名、考试、科目、类型），已自动跳过。")

    # --- 记录文件元数据 ---
    # 从 file_id 中提取哈希
    parts = file_id.split('_')
    file_hash = parts[-1] if len(parts) >= 3 else file_id
    st.session_state.file_metadata[file_id] = {
        'filename': uploaded_file.name,
        'file_hash': file_hash,
        'physics_history': st.session_state.physics_or_history,
        'selected_two': st.session_state.selected_two.copy(),
        'record_count': len(cleaned_data),
        'timestamp': datetime.now().isoformat()
    }
    if file_id not in st.session_state.imported_files:
        st.session_state.imported_files.append(file_id)
    if uploaded_file.name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(uploaded_file.name)

    # --- 重置筛选状态 ---
    st.session_state.filtered_data = pd.DataFrame()
    st.session_state.selected_exam = '全部'
    st.session_state.selected_subjects = ['全部']
    st.session_state.min_score = 0
    if st.session_state.subject_max_scores:
        st.session_state.max_score = max(st.session_state.subject_max_scores.values())
    else:
        st.session_state.max_score = 150

    st.success(f"✅ 成功处理 {len(cleaned_data)} 条记录（新增 {added_count} 条）")

def get_file_identifier(uploaded_file):
    # 使用文件名、大小和内容哈希组合
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    return f"{uploaded_file.name}_{uploaded_file.size}_{file_hash}"

def ocr_image_to_dataframe(image_bytes):
    """
    使用 easyocr 对图片进行 OCR，尝试提取表格数据并返回 DataFrame。
    返回 (DataFrame, 状态信息)
    """
    reader = get_ocr_reader()
    
    # 将字节流转换为 PIL Image
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return None, f"图片打开失败: {e}"
    
    # 转为 numpy 数组（easyocr 接受 numpy 数组）
    image_np = np.array(image)
    
    # 执行 OCR（detail=1 返回详细信息，包括边界框）
    results = reader.readtext(image_np, detail=1, paragraph=False)
    
    if not results:
        return None, "未识别到任何文字"
    
    # 按垂直位置（y 坐标）分组，形成行
    # 每个结果格式：(bbox, text, confidence)
    # bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 四个角坐标
    # 按左上角 y 坐标排序（从上到下）
    sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
    
    lines = []
    current_line = []
    current_y = None
    threshold = 15  # 垂直距离阈值（像素），可根据图片分辨率调整
    
    for bbox, text, conf in sorted_results:
        # 取左上角和右下角 y 坐标的平均值作为该文本行的中心 y
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        if current_y is None or abs(y_center - current_y) > threshold:
            # 新行开始，保存上一行
            if current_line:
                # 行内按 x 坐标排序（从左到右）
                current_line.sort(key=lambda x: x[0][0][0])
                line_text = ' '.join([item[1] for item in current_line])
                lines.append(line_text)
            current_line = [(bbox, text, conf)]
            current_y = y_center
        else:
            # 属于同一行
            current_line.append((bbox, text, conf))
    
    # 处理最后一行
    if current_line:
        current_line.sort(key=lambda x: x[0][0][0])
        line_text = ' '.join([item[1] for item in current_line])
        lines.append(line_text)
    
    if len(lines) < 2:
        return None, "未能识别出足够行数（可能表格结构复杂）"
    
    # 将第一行作为表头，后续行作为数据（按空格分割）
    header = lines[0].split()
    data_rows = []
    for line in lines[1:]:
        cols = line.split()
        if len(cols) == len(header):
            data_rows.append(cols)
        else:
            # 如果列数不匹配，可尝试智能处理或跳过
            # 这里简单忽略该行
            pass
    
    if not data_rows:
        return None, "解析失败：无法匹配表头列数"
    
    df = pd.DataFrame(data_rows, columns=header)
    return df, "识别成功"

def detect_subjects_from_columns(columns):
    """
    从列名中推断文件中存在的科目及其分数类型。
    返回一个字典：{'main': 物理或历史科目名, 'optional': 四选二科目列表}
    """
    main_candidates = ['物理', '历史']
    optional_candidates = ['政治', '地理', '化学', '生物']
    
    detected_main = None
    for m in main_candidates:
        if any(m in col for col in columns):
            detected_main = m
            break
    
    detected_optional = []
    for o in optional_candidates:
        # 如果存在包含该科目名的列，则认为该科目存在（如“化学原始分”、“化学赋分”）
        if any(o in col for col in columns):
            detected_optional.append(o)
    
    return {'main': detected_main, 'optional': detected_optional}

def detect_format_type(df):
    columns = df.columns.tolist()
    common_subjects = ['语文', '数学', '英语', '物理', '历史', '化学', '生物', '政治', '地理']

    # 长表格式特征：包含“科目”列和“分数”/“成绩”列
    if '科目' in columns and ('分数' in columns or '成绩' in columns):
        return '长表格式'

    # 宽表格式特征：任意列名包含常见科目名
    if any(any(subj in col for subj in common_subjects) for col in columns):
        return '宽表格式'

    return '未知格式'

def convert_wide_to_long(df, subjects):
    """
    智能转换宽表为长表，支持列名带后缀（成绩、原始分、赋分、校排名等）
    并在开头打印列名以便调试。
    """

    processed_data = []
    # 获取当前选科设置
    main_subjects = ["语文", "数学", "英语", st.session_state.physics_or_history]
    optional_subjects = st.session_state.selected_two
    all_subjects = main_subjects + optional_subjects

    # 遍历每一行
    for idx, row in df.iterrows():
        student_id = row['学号']
        student_name = row['姓名']
        exam_name = row['考试名称'] if '考试名称' in df.columns else '未知考试'

        # 遍历所有科目（根据侧边栏设置）
        for subject in all_subjects:
            # 可能的列名模式
            score_patterns = [f"{subject}成绩", f"{subject}原始分", f"{subject}赋分"]
            rank_patterns = [f"{subject}校排名", f"{subject}排名"]

            # 查找分数列
            found_scores = []
            for pattern in score_patterns:
                if pattern in df.columns and pd.notna(row.get(pattern)):
                    # 判断分数类型
                    if "赋分" in pattern:
                        score_type = "赋分"
                    else:
                        score_type = "原始分"  # 包括"成绩"和"原始分"
                    found_scores.append({
                        '分数': float(row[pattern]),
                        '分数类型': score_type,
                        '列名': pattern
                    })

            # 查找排名列
            rank_value = None
            for pattern in rank_patterns:
                if pattern in df.columns and pd.notna(row.get(pattern)):
                    rank_value = int(row[pattern]) if pd.notna(row[pattern]) else None
                    break

            # 为每个找到的分数生成一条记录
            for score_info in found_scores:
                processed_data.append({
                    '学号': student_id,
                    '姓名': student_name,
                    '考试名称': exam_name,
                    '科目': subject,
                    '分数': score_info['分数'],
                    '分数类型': score_info['分数类型'],
                    '校排名': rank_value  # 同一科目所有分数类型共享一个排名
                })

    return pd.DataFrame(processed_data)

def get_filtered_by_score_type(data, score_type):
    """根据分数类型筛选数据，若无类型列则返回原数据"""
    if '分数类型' in data.columns:
        filtered = data[data['分数类型'] == score_type]
        if filtered.empty:
            st.warning(f"当前数据中没有 {score_type} 类型的数据，将使用全部数据。")
            return data
        return filtered
    else:
        # 无类型列，提示用户（可选）
        st.info("当前数据未包含分数类型信息，将使用所有分数进行分析（可能混合原始分和赋分）。")
        return data

def build_columns(physics_history, selected_two):
    """根据科目设置构建表格列名"""
    columns = ["学号", "姓名"]
    # 语数外
    for subj in ["语文", "数学", "英语"]:
        columns.append(f"{subj}成绩")
        columns.append(f"{subj}校排名")
    # 物理/历史
    columns.append(f"{physics_history}成绩")
    columns.append(f"{physics_history}校排名")
    # 四选二科目（原始分、赋分、班排名）
    for subj in selected_two:
        columns.append(f"{subj}原始分")
        columns.append(f"{subj}赋分")
        columns.append(f"{subj}校排名")
    return columns

def create_blank_df(student_count, columns):
    """创建空白数据表格"""
    df = pd.DataFrame(index=range(student_count), columns=columns)
    df["学号"] = [f"S{1001 + i}" for i in range(student_count)]
    df["姓名"] = ""
    for col in columns[2:]:
        df[col] = np.nan
    return df

def process_pasted_data(df):
    """处理粘贴数据（科目检测、转换、导入）"""
    # 自动检测科目
    detected = detect_subjects_from_columns(df.columns)
    st.caption(f"检测到：物理/历史={detected['main']}，选考={detected['optional']}")

    # 提示科目不匹配（可选自动更新）
    if detected['main'] and detected['main'] != st.session_state.physics_or_history:
        st.warning(f"检测到物理/历史科目为【{detected['main']}】，但当前设置为【{st.session_state.physics_or_history}】。")
        # 可以在这里加按钮更新，但为简化，让用户手动调整

    if set(detected['optional']) and set(detected['optional']) != set(st.session_state.selected_two):
        st.warning(f"检测到选考科目为 {detected['optional']}，但当前设置为 {st.session_state.selected_two}。")

    # 根据格式转换
    format_type = detect_format_type(df)
    subjects = ["语文", "数学", "英语", st.session_state.physics_or_history] + st.session_state.selected_two

    if format_type == '宽表格式':
        processed_df = convert_wide_to_long(df, subjects)
    elif format_type == '长表格式':
        processed_df = df.copy()
    else:
        st.error("无法识别数据格式，请确保包含正确的列名")
        return

    if processed_df is not None:
        cleaned_data = validate_and_clean_data(processed_df)
        if cleaned_data is not None:
            cleaned_data['来源'] = '粘贴数据'
            if st.session_state.dashboard_data.empty:
                st.session_state.dashboard_data = cleaned_data
            else:
                st.session_state.dashboard_data = pd.concat(
                    [st.session_state.dashboard_data, cleaned_data],
                    ignore_index=True
                )
            st.success(f"✅ 成功导入 {len(cleaned_data)} 条记录")
            # 清除临时状态
            st.session_state.paste_temp_df = None
            st.session_state.show_mapping = False
            st.session_state.mapping_done = False
            st.rerun()

# ============ 初始化session_state ============
def initialize_session_state():
    """初始化所有session_state变量"""
    defaults = {
        'dashboard_data': pd.DataFrame(),
        'filtered_data': pd.DataFrame(),
        'manual_data': pd.DataFrame(),
        'manual_mode': False,
        'current_exam': "第一次月考",
        'exam_date': datetime.now().date(),
        'physics_or_history': "物理",
        'selected_two': ["化学", "生物"],
        'custom_subjects': ["语文", "数学", "英语", "物理", "化学", "生物"],
        'chart_config': {
            'theme': 'plotly_white',
            'height': 400,
            'animation': True
        },
        'current_filters': {},
        'selected_exam': '全部',
        'selected_subjects': ['全部'],
        'min_score': 0,
        'max_score': 150,
        'data_loaded': True,
        'processed_df': pd.DataFrame(),
        'file_processed': False,
        'uploaded_files': [],
        'subject_max_scores': {
            '语文': 150,
            '数学': 150,
            '英语': 150,
            '物理': 100,
            '历史': 100,
            '化学': 100,
            '生物': 100,
            '政治': 100,
            '地理': 100},
        'paste_temp_df': None,          
        'show_mapping': False,           
        'mapping_done': False,            
        'file_metadata': {},          # 文件元数据字典
        'imported_files': [],         # 已导入文件标识列表
        'active_file': '全部'       # 当前激活的文件名（用于显示过滤）
        }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def calculate_scores_by_type(data, score_type):
    """根据分数类型计算总分和排名"""
    if data.empty:
        return pd.DataFrame()
    
    if '分数类型' not in data.columns:
        if score_type == '原始分':
            filtered = data
        else:
            # 没有赋分数据时返回空
            return pd.DataFrame()
    else:
        filtered = data[data['分数类型'] == score_type]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # 分组求和
    scores = filtered.groupby(['学号', '姓名', '考试名称'])['分数'].sum().reset_index()
    scores = scores.rename(columns={'分数': '总分'})
    
    # 计算排名
    scores['校排名'] = scores.groupby('考试名称')['总分'].rank(method='min', ascending=False).astype(int)
    return scores

def merge_scores(raw_scores, scaled_scores):
    """合并原始分和赋分总分表，缺失值保留NaN"""
    if raw_scores.empty and scaled_scores.empty:
        return pd.DataFrame()
    
    if raw_scores.empty:
        # 只有赋分数据
        merged = scaled_scores.copy()
        merged['原始分总分'] = np.nan
        merged['原始分校排名'] = None
        merged = merged.rename(columns={'总分': '赋分总分', '校排名': '赋分校排名'})
        return merged
    elif scaled_scores.empty:
        # 只有原始分数据
        merged = raw_scores.copy()
        merged['赋分总分'] = np.nan
        merged['赋分校排名'] = None
        merged = merged.rename(columns={'总分': '原始分总分', '校排名': '原始分校排名'})
        return merged
    else:
        # 两者都有，外连接保留所有学生，缺失值自然为NaN
        merged = pd.merge(
            raw_scores[['学号', '姓名', '考试名称', '总分', '校排名']],
            scaled_scores[['学号', '姓名', '考试名称', '总分', '校排名']],
            on=['学号', '姓名', '考试名称'],
            suffixes=('_原始', '_赋分'),
            how='outer'
        )
        merged = merged.rename(columns={
            '总分_原始': '原始分总分',
            '校排名_原始': '原始分校排名',
            '总分_赋分': '赋分总分',
            '校排名_赋分': '赋分校排名'
        })
        return merged

def create_student_wide_table(data, physics_history, selected_two):
    """从长表生成包含各科明细的宽表（动态适应选科）"""
    if data.empty:
        return pd.DataFrame()
    
    has_score_type = '分数类型' in data.columns
    has_class_rank = '校排名' in data.columns
    
    main_subjects = ["语文", "数学", "英语", physics_history]
    optional_subjects = selected_two
    
    grouped = data.groupby(['学号', '姓名', '考试名称'])
    rows = []
    
    for (student_id, student_name, exam_name), group in grouped:
        row = {
            '学号': student_id,
            '姓名': student_name,
            '考试名称': exam_name,
        }
        
        # 主科
        for subj in main_subjects:
            # 分数
            if has_score_type:
                score_rows = group[(group['科目'] == subj) & (group['分数类型'] == '原始分')]
            else:
                score_rows = group[group['科目'] == subj]
            row[f"{subj}成绩"] = score_rows.iloc[0]['分数'] if not score_rows.empty else np.nan
            
            # 班排名
            if has_class_rank:
                rank_rows = group[group['科目'] == subj]
                valid_ranks = rank_rows['校排名'].dropna()
                row[f"{subj}校排名"] = valid_ranks.iloc[0] if not valid_ranks.empty else np.nan
            else:
                row[f"{subj}校排名"] = np.nan
        
        # 选考科目（原始分、赋分、班排名）
        for subj in optional_subjects:
            # 原始分
            if has_score_type:
                raw_rows = group[(group['科目'] == subj) & (group['分数类型'] == '原始分')]
            else:
                raw_rows = group[group['科目'] == subj]
            row[f"{subj}原始分"] = raw_rows.iloc[0]['分数'] if not raw_rows.empty else np.nan
            
            # 赋分
            if has_score_type:
                scaled_rows = group[(group['科目'] == subj) & (group['分数类型'] == '赋分')]
            else:
                scaled_rows = pd.DataFrame()  
            row[f"{subj}赋分"] = scaled_rows.iloc[0]['分数'] if not scaled_rows.empty else np.nan
            
            # 班排名
            if has_class_rank:
                rank_rows = group[group['科目'] == subj]
                valid_ranks = rank_rows['校排名'].dropna()
                row[f"{subj}校排名"] = valid_ranks.iloc[0] if not valid_ranks.empty else np.nan
            else:
                row[f"{subj}校排名"] = np.nan
        
        rows.append(row)
    
    wide_df = pd.DataFrame(rows)
    return wide_df

def generate_example_df(physics_history, selected_two):
    """根据当前选科生成示例数据表格（用于无数据时展示）"""
    # 基础示例学生数据
    students = [
        ('S1001', '赵睿杰'),
        ('S1002', '郭弘昌'),
        ('S1003', '章鑫杰'),
        ('S1004', '严凡'),
        ('S1005', '黄文静'),
    ]
    exam_name = '三校联考'
    
    rows = []
    for student_id, student_name in students:
        row = {
            '学号': student_id,
            '姓名': student_name,
            '考试名称': exam_name,
        }
        
        # 原始分总分、赋分总分、赋分校排名（模拟数据）
        idx = students.index((student_id, student_name))
        raw_total = [592.5, 589.0, 564.5, 549.5, 539.0][idx]
        scaled_total = [643.5, 627.0, 623.5, 598.5, 594.0][idx]
        scaled_rank = [28, 64, 78, 161, 181][idx]
        row['原始分总分'] = raw_total
        row['赋分总分'] = scaled_total
        row['赋分校排名'] = scaled_rank
        
        # 主科（语文、数学、英语、物理/历史）
        main_scores = {
            '语文': [118, 116.5, 125, 108, 117],
            '数学': [129, 113, 112, 106, 108],
            '英语': [135.5, 121.5, 126.5, 117.5, 114],
            physics_history: [79, 88, 83, 84, 79]  # 物理或历史
        }
        main_ranks = {
            '语文': [158, 208, 19, 498, 190],
            '数学': [20, 122, 126, 198, 172],
            '英语': [19, 267, 156, 354, 433],
            physics_history: [151, 31, 101, 84, 151]
        }
        for subj in ['语文', '数学', '英语', physics_history]:
            row[f"{subj}成绩"] = main_scores[subj][idx]
            row[f"{subj}校排名"] = main_ranks[subj][idx]
        
        # 四选二科目（原始分、赋分、班排名）
        optional_data = {
            '化学': {
                '原始分': [55, 77, 55, 58, 46],
                '赋分': [88, 95, 88, 89, 82],
                '校排名': [233, 19, 233, 180, 439]
            },
            '生物': {
                '原始分': [76, 73, 63, 76, 75],
                '赋分': [94, 93, 89, 94, 94],
                '校排名': [46, 76, 226, 46, 46]
            },
            '政治': {
                '原始分': [70, 68, 75, 72, 69],
                '赋分': [85, 84, 88, 86, 83],
                '校排名': [100, 120, 80, 110, 130]
            },
            '地理': {
                '原始分': [65, 63, 68, 66, 62],
                '赋分': [82, 80, 85, 83, 78],
                '校排名': [150, 170, 140, 160, 180]
            }
        }
        for subj in selected_two:
            if subj in optional_data:
                row[f"{subj}原始分"] = optional_data[subj]['原始分'][idx]
                row[f"{subj}赋分"] = optional_data[subj]['赋分'][idx]
                row[f"{subj}校排名"] = optional_data[subj]['校排名'][idx]
            else:
                row[f"{subj}原始分"] = np.nan
                row[f"{subj}赋分"] = np.nan
                row[f"{subj}校排名"] = np.nan
        
        rows.append(row)
    
    example_df = pd.DataFrame(rows)
    # 确保列顺序合理
    base_cols = ['学号', '姓名', '考试名称', '原始分总分', '赋分总分', '赋分校排名']
    main_cols = []
    for subj in ['语文', '数学', '英语', physics_history]:
        main_cols.extend([f"{subj}成绩", f"{subj}校排名"])
    optional_cols = []
    for subj in selected_two:
        optional_cols.extend([f"{subj}原始分", f"{subj}赋分", f"{subj}校排名"])
    ordered_cols = base_cols + main_cols + optional_cols
    return example_df[ordered_cols]

initialize_session_state()

# ============ 侧边栏配置 ============
with st.sidebar:
    st.header("🎛️ 控制面板")
    
    # ============ 输入模式选择（始终可见） ============
    st.subheader("📝 输入模式")
    input_mode = st.radio(
        "选择数据输入方式",
        ["文件上传", "图片识别", "手动输入"],
        horizontal=True,
        key="input_mode_radio"
    )
    st.session_state.manual_mode = (input_mode == "手动输入")
    st.session_state.image_mode = (input_mode == "图片识别")  # 新增标记
    
    st.divider()
    
    # ============ 文件上传模式 ============
    if input_mode == "文件上传":
        with st.expander("📁 文件上传设置", expanded=True):
            # 科目设置
            st.markdown("##### 🔀 物理/历史（二选一）")
            physics_or_history = st.radio(
                "选择物理或历史",
                ["物理", "历史"],
                horizontal=True,
                key="physics_history_radio"
            )
            st.session_state.physics_or_history = physics_or_history
            
            st.markdown("##### 🎲 四选二科目")
            four_choices = ["政治", "地理", "化学", "生物"]
            selected_two = st.multiselect(
                "选择2门科目",
                options=four_choices,
                default=st.session_state.selected_two,
                max_selections=2,
                help="必须选择2门科目"
            )
            
            if len(selected_two) < 2:
                st.warning("请选择2门科目")
            else:
                st.session_state.selected_two = selected_two
            
            # 生成科目列表
            subjects = ["语文", "数学", "英语"]
            subjects.append(physics_or_history)
            subjects.extend(selected_two)
            st.session_state.custom_subjects = subjects
            st.success(f"✅ 当前科目：{', '.join(subjects)}")
            
            st.divider()
            st.markdown("##### 📥 下载模板")
            template_buffer = io.BytesIO()
            template_df = pd.DataFrame({
                '学号': ['S001', 'S002'],
                '姓名': ['张三', '李四'],
                '考试名称': ['第一次月考', '第一次月考'],
                '考试日期': [datetime(2025,3,2), datetime(2025,3,2)],
                '原始分总分': [592.5, 592.5],
                '赋分总分': [643.5, 643.5],
                '赋分校排名': [28, 28],
                '语文成绩': [118, 118],
                '语文校排名': [158, 158],
                '数学成绩': [129, 129],
                '数学校排名': [20, 20],
                '英语成绩': [135.5, 135.5],
                '英语校排名': [19, 19],
                '物理成绩': [79, 79],
                '物理校排名': [151, 151],
                '化学原始分': [55, 55],
                '化学赋分': [88, 88],
                '化学校排名': [233, 233],
                '生物原始分': [76, 76],
                '生物赋分': [94, 94],
                '生物校排名': [46, 46],
            })
            template_df.to_excel(template_buffer, index=False)
            st.download_button(
                label="下载Excel模板",
                data=template_buffer.getvalue(),
                file_name="成绩模板.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="下载包含标准列名的模板文件"
            )
            
            st.divider()
            st.markdown("##### 📤 上传文件")
            if len(selected_two) == 2:
                uploaded_file = st.file_uploader(
                    "上传成绩文件",
                    type=['xlsx', 'xls', 'csv'],
                    help="支持Excel和CSV格式"
                )
                
                if uploaded_file:
                    # 生成文件标识
                    file_bytes = uploaded_file.getvalue()
                    file_hash = hashlib.md5(file_bytes).hexdigest()
                    file_id = f"{uploaded_file.name}_{uploaded_file.size}_{file_hash}"
                    
                    # 检查是否已存在
                    if file_id in st.session_state.imported_files:
                        # 文件已存在，询问用户操作
                        st.warning(f"文件 '{uploaded_file.name}' 已导入过。请选择操作：")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("覆盖已有数据", key=f"overwrite_{file_id}"):
                                # 删除该文件之前导入的所有记录
                                st.session_state.dashboard_data = st.session_state.dashboard_data[
                                    st.session_state.dashboard_data.get('来源文件', '') != file_id
                                ]
                                # 继续处理文件（重新导入）
                                process_uploaded_file(uploaded_file, file_id, overwrite=True)
                                st.rerun()
                        with col2:
                            if st.button("追加新数据", key=f"append_{file_id}"):
                                # 直接追加，但可增加去重检查
                                process_uploaded_file(uploaded_file, file_id, overwrite=False)
                                st.rerun()
                        with col3:
                            if st.button("取消", key=f"cancel_{file_id}"):
                                st.stop()  # 停止执行，不处理
                        st.stop()  # 等待用户选择，不继续执行后续代码
                    else:
                        # 新文件，直接处理
                        process_uploaded_file(uploaded_file, file_id, overwrite=False)
                        st.stop()
                    
                    # 检查是否已经处理过当前文件
                    if (st.session_state.get('file_processed') and 
                        st.session_state.get('current_file_name') == uploaded_file.name):
                        df = st.session_state.processed_df
                    else:
                        try:
                            if uploaded_file.name.endswith('.csv'):
                                df = pd.read_csv(uploaded_file)
                            else:
                                df = pd.read_excel(uploaded_file)
                        except Exception as e:
                            st.error(f"文件读取失败: {e}")
                            st.stop()
                        
                        st.session_state.file_processed = False
                        st.session_state.current_file_name = uploaded_file.name
                        
                        # 检查必需列
                        required = ['学号', '姓名', '考试名称']
                        missing = [col for col in required if col not in df.columns]
                        
                        if missing:
                            st.warning(f"文件缺少必需列：{missing}。请手动映射列名。")
                            with st.form(key='column_mapping_form'):
                                available_cols = df.columns.tolist()
                                mapping = {}
                                for std_col in required:
                                    default_index = 0
                                    for i, col in enumerate(available_cols):
                                        if std_col in col or col in std_col:
                                            default_index = i
                                            break
                                    mapping[std_col] = st.selectbox(
                                        f"选择对应 '{std_col}' 的列",
                                        ['无'] + available_cols,
                                        index=default_index + 1,
                                        key=f"map_{std_col}"
                                    )
                                submitted = st.form_submit_button("应用映射")
                                
                                if submitted:
                                    rename_dict = {}
                                    for std_col, file_col in mapping.items():
                                        if file_col != '无':
                                            rename_dict[file_col] = std_col
                                    if rename_dict:
                                        df = df.rename(columns=rename_dict)
                                        still_missing = [col for col in required if col not in df.columns]
                                        if still_missing:
                                            st.error(f"仍然缺少列：{still_missing}，无法处理。")
                                            st.stop()
                                        else:
                                            st.success("列映射成功！")
                                            st.session_state.processed_df = df
                                            st.session_state.file_processed = True
                                            st.rerun()
                                    else:
                                        st.error("未选择任何映射，请至少映射所有必需列。")
                                        st.stop()
                                else:
                                    st.stop()
                        else:
                            st.session_state.processed_df = df
                            st.session_state.file_processed = True
                    
                    # 自动检测科目
                    detected = detect_subjects_from_columns(df.columns)
                    st.caption(f"文件检测到：物理/历史={detected['main']}，选考={detected['optional']}")
                    
                    # 提示科目不匹配并提供一键更新
                    if detected['main'] and detected['main'] != st.session_state.physics_or_history:
                        st.warning(f"检测到文件中的物理/历史科目为【{detected['main']}】，但当前设置为【{st.session_state.physics_or_history}】。")
                        if st.button(f"将物理/历史更新为 {detected['main']}"):
                            st.session_state.physics_or_history = detected['main']
                            st.rerun()
                    
                    if set(detected['optional']) and set(detected['optional']) != set(st.session_state.selected_two):
                        st.warning(f"检测到文件中的选考科目为 {detected['optional']}，但当前设置为 {st.session_state.selected_two}。")
                        if st.button(f"将选考科目更新为 {detected['optional']}"):
                            st.session_state.selected_two = detected['optional']
                            st.rerun()
                    
                    format_type = detect_format_type(df)
                    
                    if format_type == '宽表格式':
                        processed_df = convert_wide_to_long(df, subjects)
                        if processed_df is None or processed_df.empty:
                            st.error("❌ 转换后无有效成绩数据。")
                            st.info(f"可能原因：当前科目设置与文件列名不匹配。")
                            st.info(f"文件检测到的物理/历史：{detected['main']}，选考：{detected['optional']}")
                            st.info(f"当前设置的物理/历史：{st.session_state.physics_or_history}，选考：{st.session_state.selected_two}")
                            st.info("请确保侧边栏科目设置与文件实际科目一致，或点击上方按钮自动更新设置后重新上传。")
                            st.stop()
                    elif format_type == '长表格式':
                        processed_df = df.copy()
                    else:
                        st.error("无法识别文件格式，请确保文件包含正确的列名")
                        processed_df = None
                    
                    if processed_df is not None:
                        cleaned_data = validate_and_clean_data(processed_df)
                        if cleaned_data is not None:
                            cleaned_data['来源文件'] = uploaded_file.name
                            if st.session_state.dashboard_data.empty:
                                st.session_state.dashboard_data = cleaned_data
                            else:
                                st.session_state.dashboard_data = pd.concat(
                                    [st.session_state.dashboard_data, cleaned_data],
                                    ignore_index=True
                                )
                            
                            if 'file_metadata' not in st.session_state:
                                st.session_state.file_metadata = {}
                            st.session_state.file_metadata[uploaded_file.name] = {
                                'physics_history': st.session_state.physics_or_history,
                                'selected_two': st.session_state.selected_two.copy()
                            }
                            st.session_state.uploaded_files.append(uploaded_file.name)
                            st.success(f"✅ 成功处理 {len(cleaned_data)} 条记录")
                            st.session_state.file_processed = False
                            st.session_state.processed_df = pd.DataFrame()
            else:
                st.info("⚠️ 请先完成四选二科目的选择（需选择2门），然后才能上传文件。")
    
    # ============ 图片识别 ============
    elif input_mode == "图片识别":
        with st.expander("📷 图片识别", expanded=True):
            # 科目设置（同文件上传）
            st.markdown("##### 🔀 物理/历史（二选一）")
            physics_or_history = st.radio(
                "选择物理或历史",
                ["物理", "历史"],
                horizontal=True,
                key="image_physics_history"
            )
            st.session_state.physics_or_history = physics_or_history

            st.markdown("##### 🎲 四选二科目")
            four_choices = ["政治", "地理", "化学", "生物"]
            selected_two = st.multiselect(
                "选择2门科目",
                options=four_choices,
                default=st.session_state.selected_two,
                max_selections=2,
                help="必须选择2门科目"
            )
            if len(selected_two) < 2:
                st.warning("请选择2门科目")
            else:
                st.session_state.selected_two = selected_two

            st.divider()
            st.markdown("##### 📸 上传成绩单图片")
            uploaded_image = st.file_uploader(
                "上传图片 (支持 jpg, png, bmp)",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=False
            )

            if uploaded_image is not None:
                # 显示图片预览
                st.image(uploaded_image, caption="已上传图片", use_container_width=True)
                
                if st.button("🔍 识别图片", type="primary"):
                    with st.spinner("正在识别中..."):
                        image_bytes = uploaded_image.read()
                        df, msg = ocr_image_to_dataframe(image_bytes)
                    
                    if df is not None:
                        st.success(msg)
                        st.dataframe(df.head())
                        
                        # 用户确认或修正识别结果
                        st.markdown("#### 请确认或修正数据")
                        # 提供可编辑表格（类似手动输入模式）
                        # 此处可简化，直接让用户确认后导入
                        if st.button("✅ 确认导入"):
                            
                            # 先检查必需列
                            required = ['学号', '姓名']
                            missing = [col for col in required if col not in df.columns]
                            if missing:
                                st.error(f"识别结果缺少列：{missing}")
                            else:
                                # 转换宽表为长表（参考 convert_wide_to_long）
                                # 需要知道哪些列是科目
                                subject_columns = [col for col in df.columns if col not in ['学号', '姓名', '考试名称']]
                                # 假设考试名称未知，需用户输入
                                exam_name = st.text_input("请输入本次考试名称", value=st.session_state.current_exam)
                                if exam_name:
                                    # 构建长表
                                    long_rows = []
                                    for _, row in df.iterrows():
                                        for subj in subject_columns:
                                            if pd.notna(row[subj]):
                                                long_rows.append({
                                                    '学号': row['学号'],
                                                    '姓名': row['姓名'],
                                                    '科目': subj,
                                                    '分数': float(row[subj]),
                                                    '分数类型': '原始分',  # 默认原始分
                                                    '考试名称': exam_name
                                                })
                                    long_df = pd.DataFrame(long_rows)
                                    cleaned_data = validate_and_clean_data(long_df)
                                    if cleaned_data is not None:
                                        # 合并到 dashboard_data
                                        if st.session_state.dashboard_data.empty:
                                            st.session_state.dashboard_data = cleaned_data
                                        else:
                                            st.session_state.dashboard_data = pd.concat(
                                                [st.session_state.dashboard_data, cleaned_data],
                                                ignore_index=True
                                            )
                                        st.success(f"✅ 成功导入 {len(cleaned_data)} 条记录")
                                        st.rerun()
                    else:
                        st.error(f"识别失败：{msg}")
    
    # ============ 手动输入模式 ============
    else:
        with st.expander("✍️ 手动输入设置", expanded=True):
            # 考试信息
            exam_name = st.text_input(
                "考试名称",
                value=st.session_state.current_exam,
                key="exam_name_input"
            )
            st.session_state.current_exam = exam_name
            exam_date = st.date_input(
                "考试日期",
                value=st.session_state.exam_date,
                key="exam_date_input"
            )
            st.session_state.exam_date = exam_date
            st.divider()
            
            # 科目设置
            st.markdown("##### 🔀 物理/历史（二选一）")
            physics_or_history = st.radio(
                "选择物理或历史",
                ["物理", "历史"],
                horizontal=True,
                key="manual_physics_history"
            )
            st.session_state.physics_or_history = physics_or_history
            
            st.markdown("##### 🎲 四选二科目")
            four_choices = ["政治", "地理", "化学", "生物"]
            selected_two = st.multiselect(
                "选择2门科目",
                options=four_choices,
                default=st.session_state.selected_two,
                max_selections=2,
                help="必须选择2门科目"
            )
            if len(selected_two) < 2:
                st.warning("请选择2门科目")
            else:
                st.session_state.selected_two = selected_two
            
            subjects = ["语文", "数学", "英语"]
            subjects.append(physics_or_history)
            subjects.extend(selected_two)
            st.session_state.custom_subjects = subjects
    
    st.divider()
    
    # ============ 显示与分数设置（始终显示，默认折叠） ============
    with st.expander("⚙️ 显示与分数设置", expanded=False):
        theme = st.selectbox(
            "图表主题",
            ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
            index=0
        )
        st.session_state.chart_config['theme'] = theme
        
        st.divider()
        score_type_global = st.radio(
            "分析使用的分数类型",
            ["原始分", "赋分"],
            horizontal=True,
            key="global_score_type",
            help="选择用于图表统计的分数类型（若数据中无对应类型，则使用原始分）"
        )
    
        st.divider()

        # ========== 科目满分设置 ==========
        st.markdown("##### 📏 科目满分设置")
        # 获取当前所有科目（根据选科）
        current_main = ["语文", "数学", "英语", st.session_state.physics_or_history]
        current_optional = st.session_state.selected_two
        all_current_subjects = current_main + current_optional

        # 确保每个科目都有默认满分值
        for subj in all_current_subjects:
            if subj not in st.session_state.subject_max_scores:
                if subj in ["语文", "数学", "英语"]:
                    st.session_state.subject_max_scores[subj] = 150
                else:
                    st.session_state.subject_max_scores[subj] = 100

        # 为每个科目创建数字输入框
        cols = st.columns(2)  # 分两列显示，节省空间
        for i, subj in enumerate(all_current_subjects):
            with cols[i % 2]:
                new_max = st.number_input(
                    f"{subj}",
                    min_value=1,
                    max_value=200,
                    value=st.session_state.subject_max_scores[subj],
                    step=1,
                    key=f"max_{subj}"
                )
                st.session_state.subject_max_scores[subj] = new_max

        # 显示当前最大满分（用于图表范围）
        if st.session_state.subject_max_scores:
            max_full = max(st.session_state.subject_max_scores.values())
            st.caption(f"图表Y轴上限将统一为 {max_full}")
    
    # ============ 数据筛选 ============
    with st.expander("🔍 数据筛选", expanded=False):
        if not st.session_state.dashboard_data.empty:
            # 获取当前基础数据（根据 active_file）
            if st.session_state.active_file == '全部':
                base_df = st.session_state.dashboard_data
            else:
                base_df = st.session_state.dashboard_data[
                    st.session_state.dashboard_data.get('来源文件', '') == st.session_state.active_file
                ]
            
            if base_df.empty:
                st.info("当前选择文件无数据")
            else:
                # 计算最大满分
                max_full_slider = max(st.session_state.subject_max_scores.values()) if st.session_state.subject_max_scores else 150
                
                exams = ['全部'] + sorted(base_df['考试名称'].unique().tolist())
                selected_exam = st.selectbox("考试名称", exams, index=0, key="filter_exam")
                st.session_state.selected_exam = selected_exam
                
                subjects = ['全部'] + sorted(base_df['科目'].unique().tolist())
                selected_subjects = st.multiselect("科目筛选", subjects, default=['全部'], key="filter_subjects")
                st.session_state.selected_subjects = selected_subjects
                
                col1, col2 = st.columns(2)
                with col1:
                    min_score = st.slider("最低分", 0, max_full_slider, 0, key="filter_min")
                    st.session_state.min_score = min_score
                with col2:
                    max_score = st.slider("最高分", 0, max_full_slider, max_full_slider, key="filter_max")
                    st.session_state.max_score = max_score
                
                if st.button("🔍 应用筛选", use_container_width=True, type="primary"):
                    filtered_df = base_df.copy()
                    if selected_exam != '全部':
                        filtered_df = filtered_df[filtered_df['考试名称'] == selected_exam]
                    if '全部' not in selected_subjects:
                        filtered_df = filtered_df[filtered_df['科目'].isin(selected_subjects)]
                    filtered_df = filtered_df[
                        (filtered_df['分数'] >= min_score) & 
                        (filtered_df['分数'] <= max_score)
                    ]
                    st.session_state.filtered_data = filtered_df
                    st.success(f"✅ 已筛选出 {len(filtered_df)} 条记录")
                    st.rerun()
                
                if not st.session_state.filtered_data.empty:
                    if st.button("🧹 清除筛选", use_container_width=True):
                        st.session_state.filtered_data = pd.DataFrame()
                        st.rerun()
        else:
            st.info("暂无数据，请先上传或输入数据。")
    
    # ============ 数据管理（默认折叠） ============
    with st.expander("🗃️ 数据管理", expanded=False):
        if st.button("🗑️ 清除所有数据", use_container_width=True, type="secondary"):
            st.session_state.dashboard_data = pd.DataFrame()
            st.session_state.filtered_data = pd.DataFrame()
            st.session_state.manual_data = pd.DataFrame()
            st.session_state.active_file = '全部'
            st.session_state.uploaded_files = []
            st.session_state.imported_files = []
            st.session_state.file_metadata = {}
            st.rerun()
        
        # 新增：清理重复文件按钮
        if st.button("🧹 清理重复文件（基于内容）", use_container_width=True):
            if not st.session_state.file_metadata:
                st.warning("没有文件元数据可清理")
            else:
                # 按哈希分组
                hash_groups = {}
                for file_id, meta in st.session_state.file_metadata.items():
                    file_hash = meta.get('file_hash')
                    if not file_hash:
                        # 兼容旧数据，尝试从 file_id 解析
                        parts = file_id.split('_')
                        file_hash = parts[-1] if len(parts) >= 3 else file_id
                    hash_groups.setdefault(file_hash, []).append((file_id, meta))
                
                to_delete = []
                for file_hash, group in hash_groups.items():
                    if len(group) > 1:
                        # 按时间戳降序排序（最新的在前）
                        group.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)
                        # 保留第一个，其余加入删除列表
                        to_delete.extend([item[0] for item in group[1:]])
                
                if not to_delete:
                    st.info("没有发现重复文件")
                else:
                    deleted_count = 0
                    for file_id in to_delete:
                        # 从 dashboard_data 中删除对应记录
                        st.session_state.dashboard_data = st.session_state.dashboard_data[
                            st.session_state.dashboard_data.get('来源文件标识', '') != file_id
                        ]
                        # 获取文件名（用于后续清理 uploaded_files）
                        filename = st.session_state.file_metadata[file_id]['filename']
                        # 删除元数据
                        del st.session_state.file_metadata[file_id]
                        # 从 imported_files 中移除
                        if file_id in st.session_state.imported_files:
                            st.session_state.imported_files.remove(file_id)
                        deleted_count += 1
                    
                    # 更新 uploaded_files：移除那些没有任何文件标识的文件名
                    # 收集所有剩余的文件名
                    remaining_filenames = {meta['filename'] for meta in st.session_state.file_metadata.values()}
                    st.session_state.uploaded_files = [fname for fname in st.session_state.uploaded_files if fname in remaining_filenames]
                    
                    # 如果当前激活的文件已被删除，则将 active_file 设为 '全部'
                    if st.session_state.active_file != '全部' and st.session_state.active_file not in remaining_filenames:
                        st.session_state.active_file = '全部'
                    
                    # 重置筛选状态
                    st.session_state.filtered_data = pd.DataFrame()
                    
                    st.success(f"已清理 {deleted_count} 个重复文件，数据已更新")
                    st.rerun()
        
        if st.session_state.uploaded_files:
            st.divider()
            st.subheader("📂 切换数据文件")
            
            # 选项包括“全部”和所有已上传的文件名
            options = ['全部'] + st.session_state.uploaded_files
            selected_view_file = st.selectbox(
                "选择要查看的文件",
                options=options,
                index=options.index(st.session_state.active_file) if st.session_state.active_file in options else 0,
                key='view_file_selector'
            )
            
            if selected_view_file != st.session_state.active_file:
                st.session_state.active_file = selected_view_file
                
                if selected_view_file == '全部':
                    # 显示所有数据，科目设置保持不变（或可重置为默认，视需求而定）
                    # 此处保持科目设置不变，因为“全部”可能混合不同科目的文件，用户需自行调整
                    pass
                else:
                    # 找到该文件对应的文件标识，还原科目设置
                    for file_id, meta in st.session_state.file_metadata.items():
                        if meta['filename'] == selected_view_file:
                            st.session_state.physics_or_history = meta['physics_history']
                            st.session_state.selected_two = meta['selected_two'].copy()
                            break
                
                # 切换文件后，清除筛选状态
                st.session_state.filtered_data = pd.DataFrame()
                st.session_state.selected_exam = '全部'
                st.session_state.selected_subjects = ['全部']
                st.session_state.min_score = 0
                if st.session_state.subject_max_scores:
                    st.session_state.max_score = max(st.session_state.subject_max_scores.values())
                else:
                    st.session_state.max_score = 150
                
                st.rerun()
            
            elif selected_view_file == '全部':
                st.session_state.current_view_file = '全部'

# ============ 主页面布局 ============
st.title("📊 班级成绩分析系统")
st.markdown("### 支持文件上传和手动输入双模式")

# 模式标识
current_mode = st.session_state.get('input_mode_radio', '文件上传')
mode_text_map = {
    "文件上传": "📁 文件上传模式",
    "图片识别": "📋 图片识别模式",
    "手动输入": "✍️ 手动输入模式"
}
mode_text = mode_text_map.get(current_mode, "📁 文件上传模式")
st.markdown(f'<div class="mode-indicator">{mode_text}</div>', unsafe_allow_html=True)

# 显示数据加载状态
if not st.session_state.dashboard_data.empty:
    total_records = len(st.session_state.dashboard_data)
    total_students = st.session_state.dashboard_data['学号'].nunique()
    total_exams = st.session_state.dashboard_data['考试名称'].nunique()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总记录数", total_records)
    with col2:
        st.metric("学生总数", total_students)
    with col3:
        st.metric("考试场次", total_exams)

# 根据 active_file 获取基础数据
if st.session_state.active_file == '全部':
    base_data = st.session_state.dashboard_data
else:
    base_data = st.session_state.dashboard_data[
        st.session_state.dashboard_data.get('来源文件', '') == st.session_state.active_file
    ]

# 显示总成绩表格
st.markdown("### 🏆 学生总分排名（原始分/赋分）")

if not base_data.empty:
    # 获取当前选科（从 session_state）
    physics_history = st.session_state.physics_or_history
    selected_two = st.session_state.selected_two

    # 生成宽表
    wide_df = create_student_wide_table(base_data, physics_history, selected_two)

    # 计算两种总分（用于合并）
    raw_scores = calculate_scores_by_type(base_data, '原始分')
    scaled_scores = calculate_scores_by_type(base_data, '赋分')
    
    if not wide_df.empty:
        # 合并原始分总分
        if not raw_scores.empty:
            wide_df = wide_df.merge(
                raw_scores[['学号', '姓名', '考试名称', '总分', '校排名']],
                on=['学号', '姓名', '考试名称'],
                how='left'
            ).rename(columns={'总分': '原始分总分', '校排名': '原始分校排名'})
        else:
            wide_df['原始分总分'] = np.nan
            wide_df['原始分校排名'] = np.nan
        
        # 合并赋分总分
        if not scaled_scores.empty:
            wide_df = wide_df.merge(
                scaled_scores[['学号', '姓名', '考试名称', '总分', '校排名']],
                on=['学号', '姓名', '考试名称'],
                how='left'
            ).rename(columns={'总分': '赋分总分', '校排名': '赋分校排名'})
        else:
            wide_df['赋分总分'] = np.nan
            wide_df['赋分校排名'] = np.nan
        
        # 排序
        wide_df = wide_df.sort_values('原始分总分', ascending=False)
        
        # 动态构建 column_config
        col_config = {
            '学号': st.column_config.TextColumn("学号", width="small"),
            '姓名': st.column_config.TextColumn("姓名", width="small"),
            '考试名称': st.column_config.TextColumn("考试名称", width="small"),
            '原始分总分': st.column_config.NumberColumn("原始分总分", width="small", format="%.1f"),
            '原始分校排名': st.column_config.NumberColumn("原始分校排名", width="small", format="%d"),
            '赋分总分': st.column_config.NumberColumn("赋分总分", width="small", format="%.1f"),
            '赋分校排名': st.column_config.NumberColumn("赋分校排名", width="small", format="%d"),
        }
        
        # 添加主科成绩和排名列
        main_subjects = ["语文", "数学", "英语", physics_history]
        for subj in main_subjects:
            col_config[f"{subj}成绩"] = st.column_config.NumberColumn(f"{subj}成绩", width="small", format="%.1f")
            col_config[f"{subj}校排名"] = st.column_config.NumberColumn(f"{subj}校排名", width="small", format="%d")
        
        # 添加选考科目列
        for subj in selected_two:
            col_config[f"{subj}原始分"] = st.column_config.NumberColumn(f"{subj}原始分", width="small", format="%.1f")
            col_config[f"{subj}赋分"] = st.column_config.NumberColumn(f"{subj}赋分", width="small", format="%.1f")
            col_config[f"{subj}校排名"] = st.column_config.NumberColumn(f"{subj}校排名", width="small", format="%d")
        
        st.dataframe(
            wide_df,
            use_container_width=True,
            hide_index=True,
            column_config=col_config
        )
        
        # 统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("学生总数", wide_df['学号'].nunique())
        with col2:
            avg_raw = wide_df['原始分总分'].mean()
            st.metric("原始分平均", f"{avg_raw:.1f}" if pd.notna(avg_raw) else "N/A")
        with col3:
            avg_scaled = wide_df['赋分总分'].mean()
            st.metric("赋分平均", f"{avg_scaled:.1f}" if pd.notna(avg_scaled) else "N/A")
        with col4:
            st.metric("考试场次", wide_df['考试名称'].nunique())
    else:
        st.info("暂无数据，请先上传或输入成绩数据。")
else:
    # 无数据时，根据当前选科生成示例表格
    physics_history = st.session_state.physics_or_history
    selected_two = st.session_state.selected_two
    example_df = generate_example_df(physics_history, selected_two)
    
    st.info("📝 暂无数据，请先上传或输入成绩数据。以下是示例表格：")
    
    # 动态构建列配置
    col_config = {
        '学号': st.column_config.TextColumn("学号", width="small"),
        '姓名': st.column_config.TextColumn("姓名", width="small"),
        '考试名称': st.column_config.TextColumn("考试名称", width="small"),
        '原始分总分': st.column_config.NumberColumn("原始分总分", width="small", format="%.1f"),
        '赋分总分': st.column_config.NumberColumn("赋分总分", width="small", format="%.1f"),
        '赋分校排名': st.column_config.NumberColumn("赋分校排名", width="small", format="%d"),
    }
    # 主科
    for subj in ['语文', '数学', '英语', physics_history]:
        col_config[f"{subj}成绩"] = st.column_config.NumberColumn(f"{subj}成绩", width="small", format="%.1f")
        col_config[f"{subj}校排名"] = st.column_config.NumberColumn(f"{subj}校排名", width="small", format="%d")
    # 选考科目
    for subj in selected_two:
        col_config[f"{subj}原始分"] = st.column_config.NumberColumn(f"{subj}原始分", width="small", format="%.1f")
        col_config[f"{subj}赋分"] = st.column_config.NumberColumn(f"{subj}赋分", width="small", format="%.1f")
        col_config[f"{subj}校排名"] = st.column_config.NumberColumn(f"{subj}校排名", width="small", format="%d")
    
    st.dataframe(
        example_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config
    )

st.divider()

# ============ 手动输入界面 ============
if st.session_state.manual_mode:
    st.header("✍️ 手动输入成绩")
    st.markdown(f"### 当前考试：{st.session_state.current_exam}")

    # 获取当前科目设置
    physics_history = st.session_state.physics_or_history
    selected_two = st.session_state.selected_two

    if not selected_two or len(selected_two) < 2:
        st.warning("请在侧边栏完成科目设置")
        st.stop()

    # 学生数量选择
    student_count = st.number_input("学生数量", min_value=1, max_value=100, value=5, step=1)

    # 构建列名
    columns = build_columns(physics_history, selected_two)

    # 初始化或更新可编辑表格数据（当列变化或学生数量变化时重建）
    if ("editable_df" not in st.session_state or 
        list(st.session_state.editable_df.columns) != columns or 
        len(st.session_state.editable_df) != student_count):
        st.session_state.editable_df = create_blank_df(student_count, columns)

    # 获取当前科目满分设置
    subject_max_scores = st.session_state.subject_max_scores

    # 基础列配置（学号、姓名）
    col_config = {
        "学号": st.column_config.TextColumn("学号", width="small", required=True),
        "姓名": st.column_config.TextColumn("姓名", width="small", required=True),
    }

    # 添加语数外及物理/历史的成绩列和排名列
    main_subjects = ["语文", "数学", "英语", physics_history]
    for subj in main_subjects:
        max_score = subject_max_scores.get(subj, 150)  # 默认150
        # 成绩列
        col_config[f"{subj}成绩"] = st.column_config.NumberColumn(
            f"{subj}成绩",
            min_value=0,
            max_value=max_score,
            step=0.5,
            format="%.1f",
            width="small"
        )
        # 排名列（通常为正整数，无上限，但可设min=1）
        col_config[f"{subj}校排名"] = st.column_config.NumberColumn(
            f"{subj}校排名",
            min_value=1,
            step=1,
            format="%d",
            width="small"
        )

    # 添加选考科目的原始分、赋分、校排名
    for subj in selected_two:
        max_score = subject_max_scores.get(subj, 100)  # 默认100
        # 原始分
        col_config[f"{subj}原始分"] = st.column_config.NumberColumn(
            f"{subj}原始分",
            min_value=0,
            max_value=max_score,
            step=0.5,
            format="%.1f",
            width="small"
        )
        # 赋分（通常满分也是100，但用户可能调整）
        col_config[f"{subj}赋分"] = st.column_config.NumberColumn(
            f"{subj}赋分",
            min_value=0,
            max_value=max_score,
            step=0.5,
            format="%.1f",
            width="small"
        )
        # 排名
        col_config[f"{subj}校排名"] = st.column_config.NumberColumn(
            f"{subj}校排名",
            min_value=1,
            step=1,
            format="%d",
            width="small"
        )
    
    # 在手动输入界面增加批量粘贴功能
    with st.expander("📋 批量粘贴数据（从Excel复制）", expanded=False):
        st.markdown("将Excel表格直接粘贴到下方，即可快速填充到上面的表格中。")
        paste_text = st.text_area("粘贴区域", height=150, key="bulk_paste_area")
        if st.button("填充到表格", key="fill_table"):
            if paste_text.strip():
                try:
                    import io
                    # 假设从Excel复制默认是制表符分隔
                    paste_df = pd.read_csv(io.StringIO(paste_text), sep='\t', engine='python')
                    # 检查列数是否匹配
                    expected_cols = len(st.session_state.editable_df.columns)
                    if paste_df.shape[1] != expected_cols:
                        st.error(f"粘贴数据的列数 ({paste_df.shape[1]}) 与表格列数 ({expected_cols}) 不匹配，无法填充。")
                    else:
                        # 假设粘贴的数据不含列名行，且列顺序与表格一致
                        new_data = paste_df.values.tolist()
                        # 确保行数不超过当前表格行数
                        if len(new_data) > len(st.session_state.editable_df):
                            st.warning(f"粘贴数据行数 ({len(new_data)}) 超过表格行数 ({len(st.session_state.editable_df)})，多余行将被忽略。")
                            new_data = new_data[:len(st.session_state.editable_df)]
                        # 逐行填充
                        for i, row_data in enumerate(new_data):
                            for j, val in enumerate(row_data):
                                col_name = st.session_state.editable_df.columns[j]
                                try:
                                    # 尝试转换为数字（如果是数字列）
                                    if pd.isna(val) or val == '':
                                        st.session_state.editable_df.at[i, col_name] = np.nan
                                    else:
                                        st.session_state.editable_df.at[i, col_name] = float(val) if str(val).replace('.','',1).replace('-','',1).isdigit() else val
                                except:
                                    st.session_state.editable_df.at[i, col_name] = val
                        st.success(f"已填充 {len(new_data)} 行数据，请检查并修改。")
                        st.rerun()
                except Exception as e:
                    st.error(f"解析失败: {e}")
            else:
                st.warning("请先粘贴数据。")
    
    # 显示可编辑表格
    st.markdown("#### 编辑成绩数据（双击单元格修改）")
    edited_df = st.data_editor(
        st.session_state.editable_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config=col_config  # 传入动态配置
    )
    st.session_state.editable_df = edited_df

    # 确认更新按钮
    if st.button("✅ 确认更新", type="primary", use_container_width=True):
        df_input = st.session_state.editable_df.copy()

        # 数据验证
        if df_input["学号"].isnull().any() or df_input["姓名"].isnull().any():
            st.error("学号和姓名不能为空")
        else:
            new_rows = []
            exam_name = st.session_state.current_exam
            exam_date = st.session_state.exam_date

            for idx, row in df_input.iterrows():
                student_id = str(row["学号"])
                student_name = str(row["姓名"])

                # 处理语数外及物理/历史（只有原始分）
                for subj in ["语文", "数学", "英语", physics_history]:
                    score_col = f"{subj}成绩"
                    rank_col = f"{subj}校排名"
                    if pd.notna(row.get(score_col)):
                        new_rows.append({
                            "学号": student_id,
                            "姓名": student_name,
                            "科目": subj,
                            "分数": float(row[score_col]),
                            "分数类型": "原始分",
                            "校排名": row.get(rank_col) if pd.notna(row.get(rank_col)) else None,
                            "考试名称": exam_name,
                            "考试日期": exam_date
                        })

                # 处理四选二科目（原始分和赋分）
                for subj in selected_two:
                    # 原始分
                    raw_score_col = f"{subj}原始分"
                    scaled_score_col = f"{subj}赋分"
                    rank_col = f"{subj}校排名"
                    
                    if pd.notna(row.get(raw_score_col)):
                        new_rows.append({
                            "学号": student_id,
                            "姓名": student_name,
                            "科目": subj,
                            "分数": float(row[raw_score_col]),
                            "分数类型": "原始分",
                            "校排名": row.get(rank_col) if pd.notna(row.get(rank_col)) else None,
                            "考试名称": exam_name,
                            "考试日期": exam_date
                        })
                    if pd.notna(row.get(scaled_score_col)):
                        new_rows.append({
                            "学号": student_id,
                            "姓名": student_name,
                            "科目": subj,
                            "分数": float(row[scaled_score_col]),
                            "分数类型": "赋分",
                            "校排名": row.get(rank_col) if pd.notna(row.get(rank_col)) else None,
                            "考试名称": exam_name,
                            "考试日期": exam_date
                        })

            # 创建 DataFrame
            new_data = pd.DataFrame(new_rows)
            cleaned_data = validate_and_clean_data(new_data)

            if cleaned_data is not None:
                key_cols = ['学号', '姓名', '考试名称', '科目', '分数类型']  # 提前定义
                if st.session_state.dashboard_data.empty:
                    st.session_state.dashboard_data = cleaned_data
                else:
                    combined = pd.concat([st.session_state.dashboard_data, cleaned_data], ignore_index=True)
                    combined_deduplicated = combined.drop_duplicates(subset=key_cols, keep='first')
                    added_count = len(combined_deduplicated) - len(st.session_state.dashboard_data)
                    st.session_state.dashboard_data = combined_deduplicated
                    st.info(f"新增 {added_count} 条记录，跳过 {len(cleaned_data) - added_count} 条重复记录")
                st.success(f"✅ 成功录入 {len(cleaned_data)} 条成绩记录")
                st.rerun()

# ============ 数据分析界面 ============
if st.session_state.data_loaded and not st.session_state.dashboard_data.empty:
    st.divider()
    st.header("📈 数据分析")
    
    # 获取基础数据（根据当前激活的文件）
    if st.session_state.active_file == '全部':
        base_data = st.session_state.dashboard_data
    else:
        base_data = st.session_state.dashboard_data[
            st.session_state.dashboard_data['来源文件'] == st.session_state.active_file
        ]

    # 再应用侧边栏筛选
    if not st.session_state.filtered_data.empty:
        current_data = st.session_state.filtered_data  # 优先使用已应用的筛选
    else:
        current_data = base_data  # 否则使用基础数据
    
    if current_data is not None and not current_data.empty:
        # 创建分析标签页
        tab1, tab2, tab3, tab4 = st.tabs(["📊 整体分析", "🎯 个人分析", "📈 趋势分析", "📋 详细数据"])
        
        with tab1:
            score_type = st.session_state.get('global_score_type', '原始分')
            data_analysis = get_filtered_by_score_type(current_data, score_type)
            
            # 计算所有科目满分的最大值（用于统一Y轴）
            if st.session_state.subject_max_scores:
                max_full = max(st.session_state.subject_max_scores.values())
            else:
                max_full = 150
            
            st.markdown("### 班级整体表现")
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.histogram(
                    data_analysis,
                    x='分数',
                    nbins=20,
                    title=f'成绩分布 ({score_type})',
                    color_discrete_sequence=['#636EFA'],
                    template=st.session_state.chart_config['theme']
                )
                # 直方图Y轴也可设置（可选）
                # fig1.update_yaxes(range=[0, max_full])  # 如果需要统一显示范围可取消注释
                st.plotly_chart(fig1, use_container_width=True)
        
            with col2:
                subject_avg = data_analysis.groupby('科目')['分数'].mean().reset_index()
                fig2 = px.bar(
                    subject_avg,
                    x='科目',
                    y='分数',
                    title=f'各科平均分 ({score_type})',
                    color='分数',
                    template=st.session_state.chart_config['theme']
                )
                fig2.update_yaxes(range=[0, max_full])  # ✅ 正确位置：fig2 定义之后
                st.plotly_chart(fig2, use_container_width=True)
    
            # 各科成绩箱线图
            st.markdown("### 各科成绩分布")
            fig3 = px.box(
                data_analysis,
                x='科目',
                y='分数',
                title=f'各科成绩箱线图 ({score_type})',
                template=st.session_state.chart_config['theme']
            )
            fig3.update_yaxes(range=[0, max_full])  # ✅ 正确位置：fig3 定义之后
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            score_type = st.session_state.get('global_score_type', '原始分')
            data_analysis = get_filtered_by_score_type(current_data, score_type)

            # 计算统一Y轴上限
            if st.session_state.subject_max_scores:
                max_full = max(st.session_state.subject_max_scores.values())
            else:
                max_full = 150

            student_list = sorted(data_analysis['姓名'].unique().tolist())
            selected_student = st.selectbox("选择学生", student_list, key="student_selector")
            # ... 后续代码保持不变

            if selected_student:
                student_data = data_analysis[data_analysis['姓名'] == selected_student]

                # 个人成绩卡片
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_score = student_data['分数'].mean()
                    st.metric("平均分", f"{avg_score:.1f}")
                with col2:
                    total_score = student_data['分数'].sum()
                    st.metric("总分", f"{total_score:.1f}")
                with col3:
                    best_idx = student_data['分数'].idxmax()
                    best_subject = student_data.loc[best_idx, '科目']
                    best_score = student_data['分数'].max()
                    st.metric("最佳科目", best_subject, f"{best_score:.1f}")
                with col4:
                    worst_idx = student_data['分数'].idxmin()
                    worst_subject = student_data.loc[worst_idx, '科目']
                    worst_score = student_data['分数'].min()
                    st.metric("待提高科目", worst_subject, f"{worst_score:.1f}")

                # 个人成绩表格（显示该生的所有分数类型）
                st.markdown("#### 各科成绩及班排名")
                student_detail = current_data[current_data['姓名'] == selected_student].copy()
                if not student_detail.empty:
                    display_cols = ['科目', '分数']
                    table_config = {
                        '科目': st.column_config.TextColumn("科目", width="medium"),
                        '分数': st.column_config.NumberColumn("分数", width="small", format="%.1f")
                    }
                    if '分数类型' in student_detail.columns:
                        display_cols.append('分数类型')
                        table_config['分数类型'] = st.column_config.TextColumn("类型", width="small")
                    if '校排名' in student_detail.columns:
                        display_cols.append('校排名')
                        table_config['校排名'] = st.column_config.NumberColumn("校排名", width="small", format="%d")

                    # 去重并排序
                    sort_cols = ['科目']
                    if '分数类型' in display_cols:
                        sort_cols.append('分数类型')
                    display_detail = student_detail[display_cols].drop_duplicates().sort_values(sort_cols)
                    st.dataframe(display_detail, use_container_width=True, hide_index=True, column_config=table_config)

                # 个人成绩柱状图（使用筛选后的分数类型）
                st.markdown(f"#### {selected_student} 各科成绩 ({score_type})")
                student_subjects = student_data.groupby('科目')['分数'].mean().reset_index()
                student_subjects = student_subjects.sort_values('分数', ascending=False)
                fig_bar = px.bar(
                    student_subjects,
                    x='科目',
                    y='分数',
                    title=f'{selected_student} 各科成绩 ({score_type})',
                    color='分数',
                    template=st.session_state.chart_config['theme']
                )
                fig_bar.update_yaxes(range=[0, max_full])
                st.plotly_chart(fig_bar, use_container_width=True)

                # 能力雷达图（对比班级平均，班级平均基于相同分数类型）
                st.markdown("### 能力雷达图")
                class_avg = data_analysis.groupby('科目')['分数'].mean().reset_index()
                # 合并学生成绩和班级平均
                comparison_df = pd.merge(
                    student_subjects,
                    class_avg,
                    on='科目',
                    suffixes=('_学生', '_班级')
                )
                if not comparison_df.empty:
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=comparison_df['分数_学生'].tolist(),
                        theta=comparison_df['科目'].tolist(),
                        fill='toself',
                        name=selected_student,
                        line_color='blue'
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=comparison_df['分数_班级'].tolist(),
                        theta=comparison_df['科目'].tolist(),
                        fill='toself',
                        name='班级平均',
                        line_color='orange'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max_full]
                            )
                        ),
                        height=400,
                        template=st.session_state.chart_config['theme'],
                        showlegend=True
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("无法绘制雷达图：数据不足")
        
        with tab3:
            st.markdown("### 趋势分析")
            
            if '考试名称' in current_data.columns and current_data['考试名称'].nunique() > 1:
                # 平均分趋势
                exam_trend = current_data.groupby(['考试名称', '科目'])['分数'].mean().reset_index()
                
                fig_trend = px.line(
                    exam_trend,
                    x='考试名称',
                    y='分数',
                    color='科目',
                    title='各科平均分趋势',
                    markers=True,
                    template=st.session_state.chart_config['theme'])
                
                fig_trend.update_layout(height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # 总分趋势
                student_trend = current_data.groupby(['姓名', '考试名称'])['分数'].sum().reset_index()
                
                # 选择要显示的学生
                top_students = student_trend.groupby('姓名')['分数'].mean().nlargest(5).index.tolist()
                top_data = student_trend[student_trend['姓名'].isin(top_students)]
                
                fig_total_trend = px.line(
                    top_data,
                    x='考试名称',
                    y='分数',
                    color='姓名',
                    title='前5名学生总分趋势',
                    markers=True,
                    template=st.session_state.chart_config['theme']
                )
                fig_total_trend.update_layout(height=400)
                st.plotly_chart(fig_total_trend, use_container_width=True)
            else:
                st.info("需要多场考试数据才能进行趋势分析")
        
        with tab4:
            st.markdown("### 详细数据查看")
            st.dataframe(
                current_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    '学号': st.column_config.TextColumn("学号", width="small"),
                    '姓名': st.column_config.TextColumn("姓名", width="small"),
                    '科目': st.column_config.TextColumn("科目", width="small"),
                    '分数': st.column_config.NumberColumn("分数", width="small", format="%.1f"),
                    '考试名称': st.column_config.TextColumn("考试名称", width="small"),
                    '考试日期': st.column_config.DateColumn("考试日期", width="small")
                }
            )
            st.caption(f"当前全局分析分数类型为：{st.session_state.get('global_score_type', '原始分')}，但此表格显示所有数据。")
            
            # 数据统计
            st.markdown("#### 数据统计摘要")
            summary_stats = current_data.groupby('科目')['分数'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
            summary_stats.columns = ['科目', '记录数', '平均分', '标准差', '最低分', '最高分']
            st.dataframe(summary_stats, use_container_width=True, hide_index=True)

# ============ 数据导出功能 ============
st.divider()
st.markdown("### 💾 数据导出")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    if not st.session_state.dashboard_data.empty:
        # 实时计算两种总分
        raw_scores = calculate_scores_by_type(st.session_state.dashboard_data, '原始分')
        scaled_scores = calculate_scores_by_type(st.session_state.dashboard_data, '赋分')
        merged_scores = merge_scores(raw_scores, scaled_scores)  # 合并函数已存在
        
        if not merged_scores.empty:
            csv_data = merged_scores.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载总分表 (CSV)",
                data=csv_data,
                file_name=f"总分表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )

with export_col2:
    if not st.session_state.dashboard_data.empty:
        detail_csv = st.session_state.dashboard_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载详细数据 (CSV)",
            data=detail_csv,
            file_name=f"详细成绩数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with export_col3:
    if not st.session_state.dashboard_data.empty:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            st.session_state.dashboard_data.to_excel(writer, sheet_name='详细成绩', index=False)
            
            # 写入正确总分表
            raw_scores = calculate_scores_by_type(st.session_state.dashboard_data, '原始分')
            scaled_scores = calculate_scores_by_type(st.session_state.dashboard_data, '赋分')
            merged_scores = merge_scores(raw_scores, scaled_scores)
            if not merged_scores.empty:
                merged_scores.to_excel(writer, sheet_name='总分排名', index=False)
            else:
                # 若无总分数据，创建一个空表或提示
                pd.DataFrame().to_excel(writer, sheet_name='总分排名', index=False)
        
        excel_data = excel_buffer.getvalue()
        st.download_button(
            label="📥 下载Excel文件",
            data=excel_data,
            file_name=f"成绩分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ============ 页面底部 ============
st.divider()
st.caption(f"📊 成绩分析系统 | 当前模式：{'✍️ 手动输入' if st.session_state.manual_mode else '📁 文件上传'} | 数据记录：{len(st.session_state.dashboard_data)} 条 | 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============ 自定义CSS样式 ============
st.markdown("""
<style>
    /* ========== 模式指示器 ========== */
    .mode-indicator {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .mode-indicator.manual {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .mode-indicator.upload {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }

    /* ========== 基础卡片样式（仅定义边框、圆角、内边距，不设背景色） ========== */
    .stMetric {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    .stForm {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #dee2e6;
        border-radius: 8px;
    }

    /* ========== 按钮样式 ========== */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* ========== 标题样式 ========== */
    h1, h2, h3 {
        color: #2c3e50;
    }

    /* ========== 选项卡样式 ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 16px;
    }

    /* ========== 全局黑色背景覆盖（所有卡片默认黑色背景、白色文字） ========== */
    .stMetric,
    [data-testid="stForm"],
    [data-testid="stDataFrame"] {
        background-color: #000000;
        color: #ffffff;
    }

    /* 卡片内标签、数值、表格单元格文字默认白色 */
    .stMetric label,
    .stMetric [data-testid="stMetricValue"],
    [data-testid="stForm"] label,
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {
        color: #ffffff;
    }

    /* 表格单元格边框深灰色（适合黑色背景） */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {
        border-color: #444444;
    }

    /* ========== 侧边栏还原为浅色（使用稳定 data-testid） ========== */
    [data-testid="stSidebar"] .stMetric,
    [data-testid="stSidebar"] [data-testid="stForm"],
    [data-testid="stSidebar"] [data-testid="stDataFrame"] {
        background-color: #f8f9fa;   /* 浅色背景 */
        color: #2c3e50;               /* 深色文字（继承标题色） */
        border-color: #dee2e6;        /* 还原边框色 */
    }

    /* 侧边栏内标签、数值、表格单元格文字深色 */
    [data-testid="stSidebar"] .stMetric label,
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stForm"] label,
    [data-testid="stSidebar"] [data-testid="stDataFrame"] td,
    [data-testid="stSidebar"] [data-testid="stDataFrame"] th {
        color: #2c3e50;
    }

    /* 侧边栏表格边框还原为浅灰色 */
    [data-testid="stSidebar"] [data-testid="stDataFrame"] td,
    [data-testid="stSidebar"] [data-testid="stDataFrame"] th {
        border-color: #dee2e6;
    }

</style>
""", unsafe_allow_html=True)


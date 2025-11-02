import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind

# === 1️⃣ 读取CSV文件 ===
file_path = "Durf survey - JS - 1 article_October 27, 2025_07.28.csv"
df = pd.read_csv(file_path)
print("✅ 文件读取成功！")
print("原始数据维度：", df.shape)

# === 2️⃣ 自动筛选存在的列 ===
existing_cols = df.columns.tolist()
q1_to_q26 = [f"Q{i}" for i in range(1, 27) if f"Q{i}" in existing_cols]
matrix_qs = [f"Q{i}_{j}" for i in range(27, 32) for j in range(1, 6) if f"Q{i}_{j}" in existing_cols]
cols_to_use = ["ResponseId"] + q1_to_q26 + matrix_qs
df = df[cols_to_use]
print(f"✅ 成功选取 {len(cols_to_use)} 列（跳过缺失列）")

# === 3️⃣ 删除全空行 ===
df_clean = df.dropna(how="all", subset=q1_to_q26 + matrix_qs)
print("✅ 删除全空行后维度：", df_clean.shape)

# === 4️⃣ 定义 Likert 文本映射规则 ===
likert_map = {
    "Strongly Disagree": 0.0,
    "Disagree": 0.166,
    "Somewhat Disagree": 0.333,
    "Neither Agree Nor Disagree": 0.5,
    "Somewhat Agree": 0.666,
    "Agree": 0.833,
    "Strongly Agree": 1.0
}

# === 5️⃣ 清理文本（Q1–Q26）===
for col in q1_to_q26:
    if col in df_clean.columns:
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\r|\n", "", regex=True)
            .str.title()
            .map(likert_map)
        )

# === 6️⃣ 数值型问题（Q27–Q31）转为浮点数 ===
for col in matrix_qs:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# === 7️⃣ 删除全空行（映射后）===
df_clean = df_clean.dropna(how="all", subset=q1_to_q26 + matrix_qs)
print(f"✅ 转换后剩余样本：{df_clean.shape[0]}")

# === 8️⃣ 归一化所有非空列（仅数值部分）===
scaler = MinMaxScaler()
numeric_cols = df_clean.columns[1:]  # 除 ResponseId 外全部
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols].fillna(0))
df_clean[numeric_cols] = df_clean[numeric_cols].round(3)  # ✅ 保留三位小数

print("✅ 所有题目归一化完成，共处理列数：", len(numeric_cols))

# === 9️⃣ 计算平均分指标 ===

# 质量指标（Q1–Q14 + Q27–Q31 所有子项）
quality_cols = [f"Q{i}" for i in range(1, 15) if f"Q{i}" in df_clean.columns]
matrix_cols = [f"Q{i}_{j}" for i in range(27, 32) for j in range(1, 6) if f"Q{i}_{j}" in df_clean.columns]
all_quality_cols = quality_cols + matrix_cols

# 同理心指标（Q15–Q18 + Q21–Q26）
empathy_cols = [f"Q{i}" for i in list(range(15, 19)) + list(range(21, 27)) if f"Q{i}" in df_clean.columns]

# 检查列存在情况
print(f"📊 质量维度列数量: {len(all_quality_cols)}")
print(f"💬 同理心维度列数量: {len(empathy_cols)}")

# 计算平均分（忽略缺失值）
df_clean["overall_quality_score"] = df_clean[all_quality_cols].mean(axis=1, skipna=True)
df_clean["overall_empathy_score"] = df_clean[empathy_cols].mean(axis=1, skipna=True)

# 保留三位小数
df_clean["overall_quality_score"] = df_clean["overall_quality_score"].round(3)
df_clean["overall_empathy_score"] = df_clean["overall_empathy_score"].round(3)

# ===  🔟 导出结果 ===
output_path = "qualtrics_cleaned_normalized.csv"
df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
print("✅ 已导出清洗 & 归一化后的数据：", output_path)
print(df_clean.head(10))

# =============================================
# 🎯 🎯 🎯 新增：t-test 分析部分
# =============================================

# === 1️⃣ 根据行号划分两组 ===
# 注意：Python 行号从 0 开始
avatar_data = df_clean.iloc[1:5]   # 第 2-5 行
chat_data   = df_clean.iloc[5:9]   # 第 6-9 行

print(f"✅ avatar_based_chatbot 样本量: {len(avatar_data)}")
print(f"✅ chat_based_chatbot 样本量: {len(chat_data)}")

# === 2️⃣ 提取两个关键指标 ===
avatar_quality = avatar_data["overall_quality_score"].dropna()
chat_quality   = chat_data["overall_quality_score"].dropna()

avatar_empathy = avatar_data["overall_empathy_score"].dropna()
chat_empathy   = chat_data["overall_empathy_score"].dropna()

# === 3️⃣ 独立样本 t 检验（Welch’s t-test）===
t_quality, p_quality = ttest_ind(avatar_quality, chat_quality, equal_var=False)
t_empathy, p_empathy = ttest_ind(avatar_empathy, chat_empathy, equal_var=False)

# === 4️⃣ 输出结果 ===
print("\n📊 ===== T-TEST RESULTS =====")
print(f"🎯 Overall Quality Score:")
print(f"   Avatar Mean = {avatar_quality.mean():.3f}, Chat Mean = {chat_quality.mean():.3f}")
print(f"   t = {t_quality:.3f}, p = {p_quality:.4f}")

print(f"\n💬 Overall Empathy Score:")
print(f"   Avatar Mean = {avatar_empathy.mean():.3f}, Chat Mean = {chat_empathy.mean():.3f}")
print(f"   t = {t_empathy:.3f}, p = {p_empathy:.4f}")

# === 5️⃣ 显著性判断 ===
alpha = 0.05
if p_quality < alpha:
    print("✅ 对话质量（overall_quality_score）存在显著差异")
else:
    print("❌ 对话质量差异不显著")

if p_empathy < alpha:
    print("✅ 同理心水平（overall_empathy_score）存在显著差异")
else:
    print("❌ 同理心水平差异不显著")

# === 可选：导出结果 ===
with open("t_test_results.txt", "w", encoding="utf-8") as f:
    f.write(f"T-test Results\n\n")
    f.write(f"Quality: t={t_quality:.3f}, p={p_quality:.4f}\n")
    f.write(f"Empathy: t={t_empathy:.3f}, p={p_empathy:.4f}\n")
print("📝 已导出结果文件：t_test_results.txt")

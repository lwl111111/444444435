import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap  # 导入SHAP库
import matplotlib.pyplot as plt

# 加载预训练的XGBoost模型
model = joblib.load('vote排名前6.pkl')

# 更新后的特征范围定义
feature_ranges = {
    "Sex": {"type": "categorical", "options": [0, 1]},
    'Long-standing illness or disability': {"type": "categorical", "options": [0, 1]},
    "Age": {"type": "numerical"},
    'Number of non-cancer illnesses': {"type": "numerical"},
    'Number of medications taken': {"type": "numerical"},
    "Systolic Blood Pressure": {"type": "numerical"},
    'Cholesterol ratio': {"type": "numerical"},
    "Plasma GDF15": {"type": "numerical"},
    "Plasma MMP12": {"type": "numerical"},
    "Plasma NTproBNP": {"type": "numerical"},
    "Plasma AGER": {"type": "numerical"},
    "Plasma PRSS8": {"type": "numerical"},
    "Plasma PSPN": {"type": "numerical"},
    "Plasma WFDC2": {"type": "numerical"},
    "Plasma LPA": {"type": "numerical"},
    "Plasma CXCL17": {"type": "numerical"},
    "Plasma GAST": {"type": "numerical"},
    "Plasma RGMA": {"type": "numerical"},
    "Plasma EPHA4": {"type": "numerical"},
}

# Streamlit界面标题
st.title("10-Year MACE Risk Prediction")

# 创建两个列，显示输入项
col1, col2 = st.columns(2)

feature_values = []

# 通过 feature_ranges 保持顺序
for i, (feature, properties) in enumerate(feature_ranges.items()):
    if properties["type"] == "numerical":
        # 数值型输入框
        if i % 2 == 0:
            with col1:
                value = st.number_input(
                    label=f"{feature}",
                    value=0.0,  # 默认值为0
                    key=f"{feature}_input"
                )
        else:
            with col2:
                value = st.number_input(
                    label=f"{feature}",
                    value=0.0,  # 默认值
                    key=f"{feature}_input"
                )
    elif properties["type"] == "categorical":
        if feature == "Sex":
            with col1:  # 将"Sex"放在第一个列中
                value = st.radio(
                    label="Sex",
                    options=[0, 1],  # 0 = Female, 1 = Male
                    format_func=lambda x: "Female" if x == 0 else "Male",
                    key=f"{feature}_input"
                )
        elif feature == 'Long-standing illness or disability':
            with col2:  # 将"Long-standing illness or disability"放在第二个列中
                value = st.radio(
                    label="Long-standing illness or disability",
                    options=[0, 1],  # 0 = No, 1 = Yes
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key=f"{feature}_input"
                )
    feature_values.append(value)

# 将特征值转换为模型输入格式
features = np.array([feature_values])

if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 获取 MACE 类的概率
    mace_probability = predicted_proba[1] * 100
    text = f"Predicted probability of MACE in the next 10 years: {mace_probability:.2f}%."
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.text(
        0.5, 0.1, text,
        fontsize=18,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=1200)
    st.image("prediction_text.png")

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))
    shap_values_single = shap_values[0]  # 只取当前输入的SHAP（适合XGBoost/sklearn二分类）

    # -------- 只展示前5+14 other features的 SHAP 瀑布图 --------
    abs_shap = np.abs(shap_values_single)
    top5_idx = np.argsort(abs_shap)[-5:][::-1]  # 前5 index，按贡献从大到小
    all_idx = np.arange(len(shap_values_single))
    other_idx = np.setdiff1d(all_idx, top5_idx)
    n_other = len(other_idx)
    feature_names_all = list(pd.DataFrame([feature_values], columns=feature_ranges.keys()).columns)
    other_name = f"{n_other} other features"

    feature_names_6 = [feature_names_all[i] for i in top5_idx] + [other_name]
    shap_values_6 = list(shap_values_single[top5_idx]) + [shap_values_single[other_idx].sum()]
    feature_values_6 = [feature_values[i] for i in top5_idx] + ['-']

    # 构造SHAP Explanation对象
    expl = shap.Explanation(
        values=np.array(shap_values_6),
        base_values=explainer.expected_value,
        data=np.array(feature_values_6),
        feature_names=feature_names_6
    )

    # 绘制瀑布图
    plt.figure(figsize=(8, 4))
    shap.plots.waterfall(expl, show=False)
    plt.tight_layout()
    plt.savefig("shap_waterfall_top6.png", bbox_inches='tight', dpi=400)
    st.image("shap_waterfall_top6.png")

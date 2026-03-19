# Streamlit app for Churn Prediction (Final UI Improved)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from sklearn.preprocessing import StandardScaler


# ------------------ BACKGROUND + UI STYLING ------------------ #
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Main container */
        .main {{
            background-color: rgba(255, 255, 255, 0.92);
            padding: 25px;
            border-radius: 12px;
        }}

        /* 🔥 DARK TEXT */
        html, body, [class*="css"] {{
            color: #1B2631 !important;
        }}

        h1, h2, h3 {{
            color: #154360 !important;
        }}

        label {{
            color: #1B2631 !important;
            font-weight: 500;
        }}

        /* Buttons */
        div.stButton > button {{
            width: 100%;
            height: 3em;
            border-radius: 10px;
            font-size: 16px;
            background-color: #2E86C1;
            color: white;
            font-weight: bold;
        }}

        /* Metric */
        [data-testid="stMetricValue"] {{
            color: #154360 !important;
        }}
        </style>
    """, unsafe_allow_html=True)


# ------------------ LOAD DATA ------------------ #
@st.cache_data
def load_data(path='Customer_churn_dataset.csv'):
    return pd.read_csv(path)


# ------------------ PREPROCESSOR ------------------ #
@st.cache_resource
def build_preprocessor(df):
    telco = df.copy()
    telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
    telco.dropna(how='any', inplace=True)

    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    telco['tenure_bin'] = pd.cut(telco['tenure'], bins=bins, labels=labels, include_lowest=True)

    X_template = telco.drop(columns=['customerID', 'Churn', 'tenure'])
    X_template = pd.get_dummies(X_template, drop_first=True)

    scaler = StandardScaler()
    scaler.fit(X_template)

    return {
        'template_columns': list(X_template.columns),
        'scaler': scaler,
        'tenure_bins': (bins, labels),
        'sample_df': telco
    }


# ------------------ INPUT PREPROCESS ------------------ #
def preprocess_input(user_input, prep):
    df_in = pd.DataFrame([user_input])

    bins, labels = prep['tenure_bins']
    df_in['tenure_bin'] = pd.cut(df_in['tenure'], bins=bins, labels=labels, include_lowest=True)

    df_in = df_in.drop(columns=['tenure'])
    df_in_enc = pd.get_dummies(df_in, drop_first=True)

    df_in_enc = df_in_enc.reindex(columns=prep['template_columns'], fill_value=0)

    X_scaled = prep['scaler'].transform(df_in_enc)
    return X_scaled


# ------------------ LOAD MODEL ------------------ #
@st.cache_resource
def load_model(path='Model/ada_boost_churn_model.pkl'):
    return joblib.load(path)


# ------------------ MAIN APP ------------------ #
def main():
    st.set_page_config(page_title='Churn Prediction', layout='wide')

    # BACKGROUND IMAGE PATH
    set_background("assets/bg.png")

    # HEADER
    st.markdown("""
        <h1 style='text-align: center;'>
            📡 Telecom Customer Churn Prediction
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align: center; font-size:18px;'>
            Predict whether a customer is likely to churn using Machine Learning
        </p>
    """, unsafe_allow_html=True)

    # LOAD
    df = load_data()
    prep = build_preprocessor(df)
    model = load_model()

    sample = prep['sample_df']

    st.markdown("## 📊 Customer Details")

    # ------------------ FORM ------------------ #
    with st.form('input_form'):

        tenure = st.slider(
            'Tenure (months)',
            min_value=int(sample['tenure'].min()),
            max_value=int(sample['tenure'].max()),
            value=12
        )

        user_input = {'tenure': tenure}

        col1, col2, col3 = st.columns(3)

        cols_to_ask = [c for c in sample.columns if c not in ['customerID', 'Churn', 'tenure', 'tenure_bin']]

        for i, col in enumerate(cols_to_ask):
            container = [col1, col2, col3][i % 3]

            with container:
                if sample[col].dtype == 'object' or sample[col].dtype.name == 'category':
                    opts = sorted(sample[col].dropna().unique().tolist())
                    user_input[col] = st.selectbox(col, opts)
                else:
                    minv = float(sample[col].min())
                    maxv = float(sample[col].max())
                    default = float(sample[col].median())

                    user_input[col] = st.number_input(col, value=default, min_value=minv, max_value=maxv)

        submitted = st.form_submit_button('🚀 Predict Churn')

    # ------------------ RESULT ------------------ #
    if submitted:
        X_in = preprocess_input(user_input, prep)
        pred_proba = model.predict_proba(X_in)[0][1]
        pred_class = model.predict(X_in)[0]

        st.markdown("### 🎯 Prediction Result")

        if pred_class == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

        st.metric(label="Churn Probability", value=f"{pred_proba:.2%}")

        if st.checkbox('Show debug info'):
            st.write("Input:", user_input)
            st.write("Processed shape:", X_in.shape)

    st.markdown("---")
    st.markdown("💡 Built with Streamlit | ML Model: AdaBoost")


# ------------------ RUN ------------------ #
if __name__ == '__main__':
    main()
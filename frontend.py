import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("AutoML System")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file is not None:
    with open(f"datasets/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File Uploaded Successfully")

selected_models = st.multiselect("Select Models to Train", ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "HistGradientBoosting"])

if st.button("Train Models"):
    if not selected_models:
        st.error("Please select at least one model.")
    else:
        progress_bar = st.progress(0)
        response = requests.post(
            "http://127.0.0.1:8000/train/",
            data={"filename": uploaded_file.name, "models": selected_models}
        )
        progress_bar.progress(100)

        data = response.json()
        results = data.get("results", {})  # ✅ Extract results dictionary
        best_model = data.get("best_model", "")

        if not results:
            st.error("No results found. Please try again.")
        else:
            df_results = pd.DataFrame(results).T
            df_results.reset_index(inplace=True)
            df_results.rename(columns={'index': 'Model'}, inplace=True)
            df_results = df_results.round(2)

            sns.set_theme(style="whitegrid")
            metrics_melted = df_results.melt(id_vars=['Model'], var_name='Metric', value_name='Value')

            plt.figure(figsize=(14, 8))
            sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_melted, palette="viridis")
            plt.title('Model Performance Comparison Across Metrics')
            plt.ylabel('Score')
            plt.xlabel('Model')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            st.pyplot(plt)
            st.write("### Model Performance Metrics")
            st.dataframe(df_results)

            # ✅ Highlight Best Model
            st.write(f"🏆 **Best Model:** `{best_model}`")

            st.session_state.best_model = best_model


# ✅ Download Best Model
if "best_model" in st.session_state:
    st.subheader("Download Best Model")
    st.write(f"Selected Model: `{st.session_state.best_model}`")
    if st.button("Download Best Model"):
        model_url = f"http://127.0.0.1:8000/download/{st.session_state.best_model}"
        st.markdown(f"[Click here to download {st.session_state.best_model}]( {model_url} )", unsafe_allow_html=True)

# utils/visualizer.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def visualize_data(data):
    st.subheader("📄 Data Preview")
    st.dataframe(data)

    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        st.subheader("⚙️ Chart Configuration")
        chart_type = st.selectbox("📈 Select chart type:", ["Line Chart", "Bar Chart", "Area Chart", "Custom Plot"])
        selected_cols = st.multiselect("📌 Select numeric columns to visualize:", numeric_cols, default=numeric_cols[:2])

        if selected_cols:
            st.subheader("📊 Chart Output")
            plot_data = data[selected_cols]

            if chart_type == "Line Chart":
                st.line_chart(plot_data)
            elif chart_type == "Bar Chart":
                st.bar_chart(plot_data)
            elif chart_type == "Area Chart":
                st.area_chart(plot_data)
            elif chart_type == "Custom Plot":
                plot_custom(data, selected_cols)
        else:
            st.warning("⚠️ Please select at least one column.")
    else:
        st.warning("⚠️ No numeric columns detected in your dataset.")

def plot_custom(data, selected_cols):
    st.markdown("### 📅 Time Series / Custom Line Plot")
    fig, ax = plt.subplots(figsize=(10, 6))

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        for col in selected_cols:
            ax.plot(data["Date"], data[col], label=col)
        ax.set_xlabel("Date")
    else:
        for col in selected_cols:
            ax.plot(data.index, data[col], label=col)
        ax.set_xlabel("Index")

    ax.set_ylabel("Value")
    ax.set_title("Custom Plot")
    ax.legend()
    st.pyplot(fig)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt 

# Load model & data
model = joblib.load("best_model.pkl")
df = pd.read_csv("student_habits_performance.csv")
def load_css():
    with open("Styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
# Sidebar navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["Prediction", "Categorical Graphs", "Heatmap", "Scatter Plots"]
)

# ---------------- PAGE 1 ----------------
if page == "Prediction":
    st.title("🎯 Student Performance Predictor")
    st. markdown(
    "<p style='text-align: center; color: white;'>Predict your academic performance based on your daily habits.</p>",
    unsafe_allow_html=True
    )

    study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 2.0)
    attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
    mental_health = st.slider("Mental Health (1-10)", 1, 10, 5)
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 6.5)

    part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
    ptj_encoded = 1 if part_time_job == "Yes" else 0

    if st.button("Predict Exam Score"):
        inp_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
        prediction = model.predict(inp_data)[0]
        prediction = max(0, min(100, prediction))

        # 🎯 Prediction
        st.success(f"Predicted Exam Score: {prediction:.2f}")

        # 📊 Basic Feedback
        if prediction > 80:
            st.success("Excellent! Maintain your current habits.")
        elif prediction > 50:
            st.warning("Good, but improving study hours and sleep can help.")
        else:
            st.error("Low performance. Focus more on study and reduce distractions.")

        # 🔥 INSIGHTS SECTION (YOUR SIGNATURE FEATURE)
        st.subheader("📌 Insights & Suggestions")

        if attendance < 60:
            st.warning("Low attendance is negatively affecting your performance.")

        if study_hours < 3:
            st.warning("Increasing study hours can significantly improve your score.")

        if sleep_hours < 6:
            st.warning("Lack of sleep may reduce concentration and performance.")

        if mental_health < 5:
            st.warning("Improving mental well-being can positively impact your results.")

        if ptj_encoded == 1 and study_hours < 3:
            st.info("Balancing part-time work and studies better can help improve performance.")

        # 🔥 ADVANCED REASONING
        if prediction < 50 and attendance < 60:
            st.error("Your low score is strongly linked to poor attendance.")

        elif prediction < 50 and study_hours < 3:
            st.error("Your score is low mainly due to insufficient study time.")
# ---------------- PAGE 2 ----------------
elif page == "Categorical Graphs":
    st.title("📊 Categorical Data Visualization")

    categorical_cols = [
        "gender", "part_time_job", "diet_quality",
        "parental_education_level", "internet_quality",
        "extracurricular_participation"
    ]

    for col in categorical_cols:
        st.subheader(f"{col} Distribution")

        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x=col)
        plt.xticks(rotation=45)

        st.pyplot(plt)

# ---------------- PAGE 3 ----------------
elif page == "Heatmap":
    st.title("📊 Correlation Heatmap")

    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

    st.pyplot(plt)

# ---------------- PAGE 4 ----------------
elif page == "Scatter Plots":
    st.title("📈 Scatter Plots (Feature vs Exam Score)")

    num_features = [
        'age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'attendance_percentage', 'sleep_hours', 'exercise_frequency',
        'mental_health_rating'
    ]

    for feature in num_features:
        st.subheader(f"{feature} vs Exam Score")

        plt.figure()
        sns.scatterplot(data=df, x=feature, y="exam_score")

        st.pyplot(plt)

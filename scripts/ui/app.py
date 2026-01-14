import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="Student Performance ML App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()
DATA_CLEANED_PATH = (PROJECT_ROOT / "data" / "cleaned" / "student_performance_cleaned.csv").resolve()
DATA_RAW_PATH = (PROJECT_ROOT / "data" / "raw" / "StudentPerformanceFactors.csv").resolve()
CORRELATION_IMAGE_PATH = (PROJECT_ROOT / "data" / "images" / "correlation_matrix.png").resolve()
TOP_CORRELATIONS_IMAGE_PATH = (PROJECT_ROOT / "data" / "images" / "top_correlations.png").resolve()
EXAM_SCORE_DISTRIBUTION_IMAGE_PATH = (PROJECT_ROOT / "data" / "images" / "exam_score_distribution.png").resolve()
MODELS_DIR = (PROJECT_ROOT / "models").resolve()
MODEL_PATH = (MODELS_DIR / "model_14-01-2026_14-40-49.pkl").resolve()
FEATURE_INFO_PATH = (MODELS_DIR / "feature_info.json").resolve()

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/lainguyn123/student-performance-factors"

@st.cache_data
def load_data():
    """Load cleaned dataset"""
    try:
        if not DATA_CLEANED_PATH.exists():
            st.error(f"Data file not found at {DATA_CLEANED_PATH}")
            st.info(f"Current working directory: {Path.cwd()}")
            st.info(f"Project root: {PROJECT_ROOT}")
            return None
        df = pd.read_csv(DATA_CLEANED_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info(f"Looking for file at: {DATA_CLEANED_PATH}")
        return None

@st.cache_data
def load_raw_data():
    """Load raw dataset for display"""
    try:
        df = pd.read_csv(DATA_RAW_PATH)
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        if not MODEL_PATH.exists():
            st.error(f"Model file not found at {MODEL_PATH}")
            available_models = list(MODELS_DIR.glob("*.pkl"))
            if available_models:
                st.info(f"Available model files: {[m.name for m in available_models]}")
            return None
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(f"Looking for model at: {MODEL_PATH}")
        return None

@st.cache_data
def load_feature_info():
    """Load model metadata"""
    try:
        if not FEATURE_INFO_PATH.exists():
            st.error(f"Feature info file not found at {FEATURE_INFO_PATH}")
            return None
        with open(FEATURE_INFO_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading feature info: {e}")
        st.info(f"Looking for file at: {FEATURE_INFO_PATH}")
        return None

def prepare_features_for_prediction(input_data):
    """Prepare input features for model prediction with feature engineering"""
    df = pd.DataFrame([input_data])
    
    if 'Hours_Studied' in df.columns and 'Sleep_Hours' in df.columns:
        df["study_efficiency"] = df["Hours_Studied"] / df["Sleep_Hours"].replace(0, 0.5)
        df["study_sleep_ratio"] = df["Hours_Studied"] / (df["Sleep_Hours"] + 1)
    
    if 'Hours_Studied' in df.columns and 'Tutoring_Sessions' in df.columns:
        df["effective_study_hours"] = df["Hours_Studied"] + 2 * df["Tutoring_Sessions"]
        df["tutoring_study_ratio"] = df["Tutoring_Sessions"] / (df["Hours_Studied"] + 1)
    
    if 'Attendance' in df.columns and 'Hours_Studied' in df.columns:
        df["attendance_study_interaction"] = df["Attendance"] * df["Hours_Studied"] / 100
    
    if 'Previous_Scores' in df.columns and 'Hours_Studied' in df.columns:
        df["previous_study_interaction"] = df["Previous_Scores"] * df["Hours_Studied"] / 100
    
    if 'Motivation_Level' in df.columns and 'Hours_Studied' in df.columns:
        df["motivation_study_interaction"] = df["Motivation_Level"] * df["Hours_Studied"]
    
    if 'Parental_Involvement' in df.columns and 'Parental_Education_Level' in df.columns:
        df["parental_support_score"] = df["Parental_Involvement"] * (df["Parental_Education_Level"] + 1)
    
    if 'Teacher_Quality' in df.columns and 'School_Type' in df.columns:
        df["school_quality_score"] = df["Teacher_Quality"] * (df["School_Type"] + 1)
    
    key_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours']
    for feature in key_features:
        if feature in df.columns:
            df[f"{feature}_squared"] = df[feature] ** 2
    
    if 'Hours_Studied' in df.columns:
        df["study_hours_category"] = pd.cut(df["Hours_Studied"], bins=5, labels=False, duplicates='drop')
        df["study_hours_category"] = df["study_hours_category"].fillna(0).astype(float)
    
    if 'Sleep_Hours' in df.columns:
        df["sleep_category"] = pd.cut(df["Sleep_Hours"], bins=5, labels=False, duplicates='drop')
        df["sleep_category"] = df["sleep_category"].fillna(0).astype(float)
    
    df = df.fillna(0)
    return df

if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home"

st.sidebar.title("📊 Navigation")

pages = {
    "🏠 Home": "🏠 Home",
    "📈 Data Overview": "📈 Data Overview",
    "🔍 Exploratory Data Analysis": "🔍 Exploratory Data Analysis",
    "🤖 Model Results": "🤖 Model Results",
    "📊 Predictions": "📊 Predictions"
}

for page_name, page_value in pages.items():
    is_active = st.session_state.current_page == page_value
    if st.sidebar.button(page_name, use_container_width=True, type="primary" if is_active else "secondary"):
        st.session_state.current_page = page_value
        st.rerun()

page = st.session_state.current_page

df_cleaned = load_data()
df_raw = load_raw_data()

if page == "🏠 Home":
    st.title("🎓 Student Performance Prediction - Machine Learning App")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the Student Performance ML Application
    
    This application provides insights into student performance factors and demonstrates 
    machine learning models for predicting exam scores.
    
    **Dataset Information:**
    - **Source**: [Kaggle - Student Performance Factors]({})
    - **Records**: 6,607
    - **Features**: 20 original features + 2 engineered features
    - **Target Variable**: Exam Score
    
    **Features Include:**
    - Study habits (Hours Studied, Sleep Hours, Tutoring Sessions)
    - School factors (Attendance, Teacher Quality, School Type)
    - Personal factors (Gender, Motivation Level, Learning Disabilities)
    - Family factors (Parental Involvement, Family Income, Parental Education)
    - Environmental factors (Internet Access, Access to Resources, Distance from Home)
    
    **Use the navigation sidebar to explore:**
    - 📈 **Data Overview**: Basic statistics and data information
    - 🔍 **Exploratory Data Analysis**: Visualizations and insights
    - 🤖 **Model Results**: Machine learning model performance
    - 📊 **Predictions**: Make predictions on new data
    """.format(KAGGLE_DATASET_URL))
    
    st.markdown("---")
    
    if df_cleaned is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df_cleaned):,}")
        
        with col2:
            st.metric("Total Features", len(df_cleaned.columns))
        
        with col3:
            if 'Exam_Score' in df_cleaned.columns:
                avg_score = df_cleaned['Exam_Score'].mean()
                st.metric("Average Exam Score", f"{avg_score:.1f}")
        
        with col4:
            if 'Exam_Score' in df_cleaned.columns:
                max_score = df_cleaned['Exam_Score'].max()
                st.metric("Maximum Exam Score", f"{max_score:.1f}")

elif page == "📈 Data Overview":
    st.title("📈 Data Overview")
    st.markdown("---")
    
    if df_cleaned is None:
        st.error("Unable to load data. Please check the data file path.")
    else:
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Dataset Source**: [Kaggle]({KAGGLE_DATASET_URL})")
            st.info(f"**Total Records**: {len(df_cleaned):,}")
            st.info(f"**Total Features**: {len(df_cleaned.columns)}")
        
        with col2:
            st.info(f"**Memory Usage**: {df_cleaned.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.info(f"**Missing Values**: {df_cleaned.isnull().sum().sum()}")
            st.info(f"**Duplicate Rows**: {df_cleaned.duplicated().sum()}")
        
        st.markdown("---")
        
        st.subheader("Data Preview")
        st.dataframe(df_cleaned.head(10), use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df_cleaned.columns,
            'Data Type': df_cleaned.dtypes,
            'Non-Null Count': df_cleaned.count(),
            'Null Count': df_cleaned.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Statistical Summary")
        st.dataframe(df_cleaned.describe(), use_container_width=True)
        
        st.markdown("---")
        
        if 'Exam_Score' in df_cleaned.columns:
            st.subheader("Target Variable: Exam Score")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean", f"{df_cleaned['Exam_Score'].mean():.2f}")
                st.metric("Median", f"{df_cleaned['Exam_Score'].median():.2f}")
                st.metric("Std Deviation", f"{df_cleaned['Exam_Score'].std():.2f}")
            
            with col2:
                st.metric("Min", f"{df_cleaned['Exam_Score'].min():.2f}")
                st.metric("Max", f"{df_cleaned['Exam_Score'].max():.2f}")
                st.metric("Range", f"{df_cleaned['Exam_Score'].max() - df_cleaned['Exam_Score'].min():.2f}")

elif page == "🔍 Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")
    
    if df_cleaned is None:
        st.error("Unable to load data. Please check the data file path.")
    else:
        st.subheader("Correlation Matrix")
        st.markdown("""
        The correlation matrix shows relationships between numerical features. 
        This helps identify which features are most correlated with the target variable (Exam Score).
        """)
        
        if CORRELATION_IMAGE_PATH.exists():
            st.image(str(CORRELATION_IMAGE_PATH), use_container_width=True)
        else:
            st.warning("Correlation matrix image not found. Please run 02_eda.py to generate it.")
        
        st.markdown("---")
        
        if 'Exam_Score' in df_cleaned.columns:
            st.subheader("Top Features Correlated with Exam Score")
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            correlations = df_cleaned[numeric_cols].corr()['Exam_Score'].abs().sort_values(ascending=False)
            correlations = correlations[correlations.index != 'Exam_Score']
            
            top_corr_df = pd.DataFrame({
                'Feature': correlations.index,
                'Correlation with Exam Score': correlations.values
            })
            st.dataframe(top_corr_df.head(10), use_container_width=True)
            
            if TOP_CORRELATIONS_IMAGE_PATH.exists():
                st.image(str(TOP_CORRELATIONS_IMAGE_PATH), use_container_width=True)
            else:
                st.warning("Top correlations image not found. Please run 02_eda.py to generate it.")
        
        st.markdown("---")
        
        st.subheader("Feature Distributions")
        
        if 'Exam_Score' in df_cleaned.columns:
            if EXAM_SCORE_DISTRIBUTION_IMAGE_PATH.exists():
                st.image(str(EXAM_SCORE_DISTRIBUTION_IMAGE_PATH), use_container_width=True)
            else:
                st.warning("Exam score distribution image not found. Please run 02_eda.py to generate it.")
        
        st.markdown("*Additional distribution plots will be added as EDA progresses.*")

elif page == "🤖 Model Results":
    st.title("🤖 Model Results")
    st.markdown("---")
    
    model = load_model()
    feature_info = load_feature_info()
    
    if model is None or feature_info is None:
        st.error("Model or feature information not found. Please ensure the model file exists.")
    else:
        st.subheader("Model Information")
        st.info(f"**Model Type**: LightGBM Regressor")
        st.info(f"**Model File**: {MODEL_PATH.name}")
        
        st.markdown("---")
        
        st.subheader("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        test_metrics = feature_info.get('test_metrics', {})
        cv_metrics = feature_info.get('cv_metrics', {})
        
        with col1:
            rmse = test_metrics.get('rmse', 0)
            st.metric("RMSE (Test)", f"{rmse:.4f}", help="Root Mean Squared Error")
            st.caption(f"CV: {cv_metrics.get('rmse_mean', 0):.4f} ± {cv_metrics.get('rmse_std', 0):.4f}")
        
        with col2:
            mae = test_metrics.get('mae', 0)
            st.metric("MAE (Test)", f"{mae:.4f}", help="Mean Absolute Error")
        
        with col3:
            r2 = test_metrics.get('r2', 0)
            st.metric("R² Score (Test)", f"{r2:.4f}", help="Coefficient of Determination")
            st.caption(f"CV: {cv_metrics.get('r2_mean', 0):.4f} ± {cv_metrics.get('r2_std', 0):.4f}")
        
        with col4:
            mse = test_metrics.get('mse', 0)
            st.metric("MSE (Test)", f"{mse:.4f}", help="Mean Squared Error")
        
        st.markdown("---")
        
        st.subheader("Hyperparameters")
        best_params = feature_info.get('best_params', {})
        if best_params:
            params_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Value'])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("Feature Importance")
        feature_importance = feature_info.get('feature_importance', [])
        if feature_importance:
            importance_df = pd.DataFrame(feature_importance)
            importance_df = importance_df.sort_values('importance', ascending=False).head(15)
            
            st.bar_chart(importance_df.set_index('feature')['importance'])
            
            with st.expander("View All Features"):
                st.dataframe(pd.DataFrame(feature_importance).sort_values('importance', ascending=False), 
                           use_container_width=True, hide_index=True)

elif page == "📊 Predictions":
    st.title("📊 Make Predictions")
    st.markdown("---")
    
    model = load_model()
    feature_info = load_feature_info()
    
    if model is None or feature_info is None:
        st.error("Model not found. Please ensure the model file exists.")
    else:
        st.subheader("Input Student Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hours_studied = st.number_input("Hours Studied", min_value=0, max_value=50, value=20)
            attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=85)
            sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=12, value=7)
            previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=75)
            tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=2)
        
        with col2:
            physical_activity = st.number_input("Physical Activity", min_value=0, max_value=10, value=3)
            motivation_level = st.selectbox("Motivation Level", ["High", "Low", "Medium"], index=2)
            internet_access = st.selectbox("Internet Access", ["Yes", "No"], index=0)
            extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"], index=0)
            learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"], index=0)
        
        with col3:
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            parental_involvement = st.selectbox("Parental Involvement", ["High", "Low", "Medium"], index=2)
            access_to_resources = st.selectbox("Access to Resources", ["High", "Low", "Medium"], index=2)
            peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"], index=2)
            family_income = st.selectbox("Family Income", ["High", "Low", "Medium"], index=2)
        
        col4, col5 = st.columns(2)
        with col4:
            teacher_quality = st.selectbox("Teacher Quality", ["High", "Low", "Medium"], index=2)
            school_type = st.selectbox("School Type", ["Private", "Public"], index=1)
        with col5:
            parental_education = st.selectbox("Parental Education Level", ["College", "High School", "Postgraduate"], index=1)
            distance_from_home = st.selectbox("Distance from Home", ["Far", "Moderate", "Near"], index=2)
        
        st.markdown("---")
        
        if st.button("🔮 Predict Exam Score", type="primary"):
            gender_map = {"Male": 0, "Female": 1}
            yes_no_map = {"Yes": 1, "No": 0}
            
            motivation_map = {"High": 0, "Low": 1, "Medium": 2}
            parental_map = {"High": 0, "Low": 1, "Medium": 2}
            access_map = {"High": 0, "Low": 1, "Medium": 2}
            income_map = {"High": 0, "Low": 1, "Medium": 2}
            teacher_map = {"High": 0, "Low": 1, "Medium": 2}
            school_map = {"Private": 0, "Public": 1}
            peer_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
            education_map = {"College": 0, "High School": 1, "Postgraduate": 2}
            distance_map = {"Far": 0, "Moderate": 1, "Near": 2}
            
            input_data = {
                'Hours_Studied': hours_studied,
                'Attendance': attendance,
                'Sleep_Hours': sleep_hours,
                'Previous_Scores': previous_scores,
                'Tutoring_Sessions': tutoring_sessions,
                'Physical_Activity': physical_activity,
                'Motivation_Level': motivation_map[motivation_level],
                'Internet_Access': yes_no_map[internet_access],
                'Extracurricular_Activities': yes_no_map[extracurricular],
                'Learning_Disabilities': yes_no_map[learning_disabilities],
                'Gender': gender_map[gender],
                'Parental_Involvement': parental_map[parental_involvement],
                'Access_to_Resources': access_map[access_to_resources],
                'Peer_Influence': peer_map[peer_influence],
                'Family_Income': income_map[family_income],
                'Teacher_Quality': teacher_map[teacher_quality],
                'School_Type': school_map[school_type],
                'Parental_Education_Level': education_map[parental_education],
                'Distance_from_Home': distance_map[distance_from_home]
            }
            
            try:
                features_df = prepare_features_for_prediction(input_data)
                required_features = feature_info.get('features', [])
                
                missing_features = [f for f in required_features if f not in features_df.columns]
                if missing_features:
                    for feat in missing_features:
                        features_df[feat] = 0
                
                features_df = features_df[required_features]
                
                prediction = model.predict(features_df)[0]
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Exam Score", f"{prediction:.2f}")
                with col2:
                    rmse = feature_info.get('test_metrics', {}).get('rmse', 0)
                    lower_bound = max(0, prediction - rmse)
                    upper_bound = min(100, prediction + rmse)
                    st.metric("Confidence Range", f"{lower_bound:.2f} - {upper_bound:.2f}")
                with col3:
                    st.metric("Model RMSE", f"{rmse:.4f}")
                
            
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.exception(e)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Dataset")
st.sidebar.markdown(f"[Kaggle Dataset]({KAGGLE_DATASET_URL})")

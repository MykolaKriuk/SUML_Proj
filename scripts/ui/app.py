import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Student Performance ML App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "student_performance_cleaned.csv"
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "StudentPerformanceFactors.csv"
CORRELATION_IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "correlation_matrix.png"

# Kaggle dataset link
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/lainguyn123/student-performance-factors"

# Cache data loading
@st.cache_data
def load_data():
    """Load cleaned dataset"""
    try:
        df = pd.read_csv(DATA_CLEANED_PATH)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_CLEANED_PATH}")
        return None

@st.cache_data
def load_raw_data():
    """Load raw dataset for display"""
    try:
        df = pd.read_csv(DATA_RAW_PATH)
        return df
    except FileNotFoundError:
        return None

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home"

# Sidebar navigation
st.sidebar.title("📊 Navigation")

# Navigation buttons
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

# Get current page from session state
page = st.session_state.current_page

# Load data
df_cleaned = load_data()
df_raw = load_raw_data()

# Home Page
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
    
    # Quick stats cards
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

# Data Overview Page
elif page == "📈 Data Overview":
    st.title("📈 Data Overview")
    st.markdown("---")
    
    if df_cleaned is None:
        st.error("Unable to load data. Please check the data file path.")
    else:
        # Dataset info
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
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df_cleaned.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df_cleaned.columns,
            'Data Type': df_cleaned.dtypes,
            'Non-Null Count': df_cleaned.count(),
            'Null Count': df_cleaned.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)
        
        st.markdown("---")
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df_cleaned.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Target variable distribution
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

# Exploratory Data Analysis Page
elif page == "🔍 Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")
    
    if df_cleaned is None:
        st.error("Unable to load data. Please check the data file path.")
    else:
        # Correlation Matrix
        st.subheader("Correlation Matrix")
        st.markdown("""
        The correlation matrix shows relationships between numerical features. 
        This helps identify which features are most correlated with the target variable (Exam Score).
        """)
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        corr = df_cleaned[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
        ax.set_title("Correlation Matrix of Student Performance Features")
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Top correlations with target
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
            
            # Visualization of top correlations
            fig, ax = plt.subplots(figsize=(10, 6))
            top_10 = correlations.head(10)
            ax.barh(range(len(top_10)), top_10.values)
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels(top_10.index)
            ax.set_xlabel('Absolute Correlation with Exam Score')
            ax.set_title('Top 10 Features Correlated with Exam Score')
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Distribution plots
        st.subheader("Feature Distributions")
        
        if 'Exam_Score' in df_cleaned.columns:
            # Exam Score distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_cleaned['Exam_Score'], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Exam Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Exam Scores')
            st.pyplot(fig)
            plt.close()
        
        # Additional distribution plots can be added here
        st.markdown("*Additional distribution plots will be added as EDA progresses.*")

# Model Results Page
elif page == "🤖 Model Results":
    st.title("🤖 Model Results")
    st.markdown("---")
    
    st.info("""
    **Model Results Section**
    
    This section will display:
    - Model performance metrics (RMSE, MAE, R², etc.)
    - Model comparison charts
    - Feature importance plots
    - Cross-validation results
    - Learning curves
    
    *Model training scripts are not yet implemented. Results will appear here once models are trained.*
    """)
    
    # Placeholder for model selection
    st.subheader("Model Selection")
    model_type = st.selectbox(
        "Select Model Type",
        ["Linear Regression", "Random Forest", "Gradient Boosting", "Neural Network", "All Models"]
    )
    
    st.markdown("---")
    
    # Placeholder metrics
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RMSE", "N/A", help="Root Mean Squared Error")
    
    with col2:
        st.metric("MAE", "N/A", help="Mean Absolute Error")
    
    with col3:
        st.metric("R² Score", "N/A", help="Coefficient of Determination")
    
    with col4:
        st.metric("MAPE", "N/A", help="Mean Absolute Percentage Error")
    
    st.markdown("---")
    
    # Placeholder for feature importance
    st.subheader("Feature Importance")
    st.info("Feature importance visualization will appear here once models are trained.")
    
    st.markdown("---")
    
    # Placeholder for model comparison
    st.subheader("Model Comparison")
    st.info("Model comparison charts will appear here once multiple models are trained.")

# Predictions Page
elif page == "📊 Predictions":
    st.title("📊 Make Predictions")
    st.markdown("---")
    
    st.info("""
    **Prediction Interface**
    
    This section will allow you to:
    - Input student features
    - Get predicted exam scores
    - View prediction confidence intervals
    - Compare predictions across different models
    
    *Prediction functionality will be available once models are trained and saved.*
    """)
    
    st.markdown("---")
    
    # Placeholder input form
    st.subheader("Input Student Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hours_studied = st.number_input("Hours Studied", min_value=0, max_value=50, value=20)
        attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=85)
        sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=12, value=7)
        previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=75)
    
    with col2:
        tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=2)
        physical_activity = st.number_input("Physical Activity", min_value=0, max_value=10, value=3)
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"], index=0)
    
    with col3:
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"], index=0)
        learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"], index=0)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Exam Score", type="primary"):
        st.success("""
        **Prediction functionality will be implemented once models are trained.**
        
        The predicted exam score and confidence interval will appear here.
        """)
        
        # Placeholder result
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Exam Score", "N/A")
        with col2:
            st.metric("Confidence Interval", "N/A")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Dataset")
st.sidebar.markdown(f"[Kaggle Dataset]({KAGGLE_DATASET_URL})")

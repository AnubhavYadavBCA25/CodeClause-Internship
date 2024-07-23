import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

st.set_page_config(layout="wide")

############## Home Page ##############
def Home():
    st.title("🫀Heart Disease Prediction Web App")
    st.divider()

    # Introduction
    st.subheader("Introduction", divider='rainbow')
    st.write("This is a web app to show the visualization and prediction of heart disease.")
    st.write("The dataset used in this project is the Cleveland Heart Disease dataset which is available in the Kaggle Platform.")
    st.write("The dataset consists of 14 columns and 303 rows.")
    st.write("Link to the dataset: [Heart Failure Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)")
    st.divider()

    # Objective
    st.subheader("Objective", divider='rainbow')
    st.write("The objective of this project is to build a machine learning model that can predict the presence of heart disease in a patient.")
    st.divider()

    # Technologies Used
    st.subheader("Technologies Used", divider='rainbow')
    st.markdown("""
    - Python (Programming Language)
    - Streamlit (Web App Framework)
    - Pandas (Data Manipulation)
    - Matplotlib and Seaborn (Data Visualization)
    - Plotly (Interactive Data Visualization)
    - Scikit-learn (Machine Learning)
    - Joblib (Model Serialization)
    - VS Code (Code Editor) 
    """)

    # Footer
    st.markdown("---")
    st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <div style="text-align:center;">
        <p>Made with ❤️ by Anubhav Yadav</p>
        <p>Follow me on 
            <a href="https://linkedin.com/in/anubhav-yadav-srm" target="_blank"><i class="fab fa-linkedin"></i>LinkedIn</a> | 
            <a href="https://github.com/AnubhavYadavBCA25" target="_blank"><i class="fab fa-github"></i>GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True
)

############## Visualizations Page ##############
def Visualizations():
    st.header("📊Visualizations", divider='rainbow')
    st.subheader("Here are some visualizations of the dataset.")
    st.divider()

    # Load the dataset
    st.subheader("Dataset", divider='rainbow')
    df = pd.read_csv(r"Golden Level (Task 3)\notebooks\data\heart.csv")
    st.dataframe(df)
    st.divider()

    # Data Statistics for Numerical Features
    st.subheader("Data Statistics for Numerical Features", divider='rainbow')
    st.write(df.describe().transpose())

    # Data Statistics for Categorical Features
    st.subheader("Data Statistics for Categorical Features", divider='rainbow')
    st.write(df.describe(include='object').transpose())
    st.divider()

    # Data Visualization

    # Distribution of Categorical Features
    st.subheader("Distribution of Categorical Features", divider='rainbow')
    seleted_feature_pie = st.selectbox("Select Feature:", df.select_dtypes(include='object').columns, key='pie')
    fig = px.pie(df, names=seleted_feature_pie, title=f"Distribution of {seleted_feature_pie}", hole=0.3)
    st.plotly_chart(fig)
    st.divider()

    # Distribution of Numerical Features
    st.subheader("Distribution of Numerical Features", divider='rainbow')
    seleted_feature_hist = st.selectbox("Select Feature:", df.select_dtypes(include='number').columns, key='hist')
    fig = px.histogram(df, x=seleted_feature_hist, title=f"Distribution of {seleted_feature_hist}", color='HeartDisease', color_discrete_map={0:'green',1:'red'})
    st.plotly_chart(fig)
    st.divider()

    # Countplot of Categorical Features
    st.subheader("Countplot of Categorical Features", divider='rainbow')
    seleted_feature_bar = st.selectbox("Select Feature:", df.select_dtypes(include='object').columns, key='bar')
    feature_count = df[seleted_feature_bar].value_counts().reset_index()
    feature_count.columns = [seleted_feature_bar, 'Count']
    fig = px.bar(feature_count, x=seleted_feature_bar, y='Count', title=f"Countplot of {seleted_feature_bar}", color='Count', color_continuous_scale='rainbow')
    st.plotly_chart(fig)
    st.divider()

    # Scatterplot of Numerical Features
    st.subheader("Scatterplot of Numerical Features", divider='rainbow')
    y = st.selectbox("Select Y-axis Feature:", df[['RestingBP', 'Cholesterol', 'MaxHR']].columns, key='scatter_x')
    fig = px.scatter(df, x=df['Age'], y=y, title=f"Scatterplot of Age vs {y}", trendline='ols')
    fig.data[1].line.color = 'red'
    st.plotly_chart(fig)
    st.divider()

    # Correlation Heatmap
    st.subheader("Correlation Heatmap Only For Numerical Features", divider='rainbow')
    df_corr = df.drop(df.select_dtypes(include='object'), axis=1)
    fig = go.Figure(data=go.Heatmap(
    z=df_corr.corr().values,
    x=df_corr.corr().columns,
    y=df_corr.corr().index,
    colorscale='Viridis',
    text=df_corr.corr().round(2).astype(str).values,
    hoverinfo='text'
))
    for i, row in enumerate(df_corr.corr().values):
        for j, value in enumerate(row):
            fig.add_annotation(x=df_corr.corr().columns[j], y=df_corr.corr().index[i],
                            text=str(round(value, 2)),
                            showarrow=False, font=dict(color="white"))
    fig.update_layout(title_text='Correlation Heatmap')
    st.plotly_chart(fig)

# Prediction Page
def Prediction():
    st.header("🪄Prediction", divider='rainbow')






pg = st.navigation([
    st.Page(Home, title="Home", icon="🏠"),
    st.Page(Visualizations, title="Visualization", icon="📊"),
    st.Page(Prediction, title="Prediction", icon="🪄")
])
pg.run()
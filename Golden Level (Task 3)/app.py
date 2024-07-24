import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib


st.set_page_config(layout="wide")
st.sidebar.title("📚Menu")

############## Home Page ##############
def Home():
    st.title("🫀Heart Disease Prediction Web App")
    st.divider()

    # Introduction
    st.subheader("Introduction 📚", divider='rainbow')
    st.write("This is a web app to show the visualizations and prediction of heart disease.")
    st.write("The dataset used in this project is the Cleveland Heart Disease dataset which is available in the Kaggle Platform.")
    st.write("The dataset consists of 12 features and 918 records.")
    st.write("Link to the dataset: [Heart Failure Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)")
    st.divider()

    # Objective
    st.subheader("Objective 🎯", divider='rainbow')
    st.write("The objective of this project is to build a machine learning model that can predict the presence of heart disease in a patient.")
    st.divider()

    # Technologies Used
    st.subheader("Technologies Used 🛠️", divider='rainbow')
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
    st.subheader("Here are Some Visualizations of the Dataset.")
    st.divider()

    # Load the dataset
    st.subheader("Dataset", divider='rainbow')
    df = pd.read_csv(r"Golden Level (Task 3)\notebooks\data\heart.csv")
    st.dataframe(df)
    st.write("*Note: You can download the dataset by clicking on the 'Download as CSV' button on upper right hand side.*")
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
    st.write("*Note: Click on the legend to toggle the values.*")
    st.divider()

    # Distribution of Numerical Features
    st.subheader("Distribution of Numerical Features", divider='rainbow')
    seleted_feature_hist = st.selectbox("Select Feature:", df.select_dtypes(include='number').columns, key='hist')
    fig = px.histogram(df, x=seleted_feature_hist, title=f"Distribution of {seleted_feature_hist}", color='HeartDisease', color_discrete_map={0:'green',1:'red'})
    st.plotly_chart(fig)
    st.write("*Note: Click on the legend to toggle the values and you can zoom on the graph*")
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

############### Prediction Page ###############
def Prediction():
    st.header("🪄Prediction", divider='rainbow')
    st.subheader("Predict the Presence of Heart Disease in a Patient")
    st.divider()

    # Load the model and scaler
    model = joblib.load(r"Golden Level (Task 3)\notebooks\artifacts\rfc_model.pkl")
    scaler = joblib.load(r"Golden Level (Task 3)\notebooks\artifacts\scaler.pkl")
    
    # Input Form
    st.subheader("Input Form", divider='rainbow')    

    # Age Column
    Age = st.number_input("**Age**", min_value=0, max_value=100, value=50)
    st.divider()

    # Sex Column
    Sex = st.radio("**Enter Your Gender**",['Male','Female'])
    st.divider()

    # Chest Pain Type Column
    ChestPainType = st.selectbox("**Chest Pain Type**",['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'])
    st.divider()

    # Resting Blood Pressure Column
    RestingBP = st.number_input("**Resting Blood Pressure (in mmHg)**", min_value=0, value=200)
    st.divider()

    # Cholesterol Column
    Cholesterol = st.number_input("**Cholesterol (in mm/dl)**", min_value=0, value=600)
    st.divider()

    # Fasting Blood Sugar Column
    FastingBloodSugar = st.radio("**Fasting Blood Sugar > 120 mg/dl**",['Yes','No'])
    st.divider()

    # Resting ECG Column
    RestingECG = st.selectbox("**Resting ECG**",['Normal','ST-T wave abnormality','Left ventricular hypertrophy'])
    st.divider()

    # Maximum Heart Rate Achieved Column
    MaxHR = st.number_input("**Maximum Heart Rate Achieved**", min_value=0, value=120, max_value=300)
    st.divider()

    # Exercise Induced Angina Column
    ExerciseInducedAngina = st.radio("**Exercise Induced Angina**",['Yes','No'])
    st.divider()

    # ST Depression Induced by Exercise Relative to Rest Column
    STDepression = st.number_input("**ST Depression Induced by Exercise Relative to Rest**", min_value=0.0, value=2.0, max_value=6.0, step=0.1)
    st.divider()

    # Slope of the Peak Exercise ST Segment Column
    Slope = st.selectbox("**Slope of the Peak Exercise ST Segment**",['Upsloping','Flat','Downsloping'])
    st.divider()

    # Mapping the values
    # Sex conversion
    sex_mapping = {'Male': 1, 'Female': 0}
    Sex_num = sex_mapping[Sex]

    # Chest Pain Type conversion
    cp_mapping = {'Typical Angina': 46, 'Atypical Angina': 173, 'Non-anginal Pain': 203, 'Asymptomatic': 496}
    ChestPainType_num = cp_mapping[ChestPainType]

    # Fasting Blood Sugar conversion
    fbs_mapping = {'Yes': 1, 'No': 0}
    FastingBloodSugar_num = fbs_mapping[FastingBloodSugar]

    # Resting ECG conversion
    restecg_mapping = {'Normal': 1, 'ST-T wave abnormality': 2, 'Left ventricular hypertrophy': 0}
    RestingECG_num = restecg_mapping[RestingECG]

    # Exercise Induced Angina conversion
    exang_mapping = {'Yes': 1, 'No': 0}
    ExerciseInducedAngina_num = exang_mapping[ExerciseInducedAngina]

    # Slope conversion
    slope_mapping = {'Upsloping': 2, 'Flat': 1, 'Downsloping': 0}
    Slope_num = slope_mapping[Slope]

    # Predict Button
    if st.button("Predict"):
        # Data Preprocessing
        data = pd.DataFrame({
            'Age': [Age],
            'Sex': [Sex_num],
            'ChestPainType': [ChestPainType_num],
            'RestingBP': [RestingBP],
            'Cholesterol': [Cholesterol],
            'FastingBS': [FastingBloodSugar_num],
            'RestingECG': [RestingECG_num],
            'MaxHR': [MaxHR],
            'ExerciseAngina': [ExerciseInducedAngina_num],
            'Oldpeak': [STDepression],
            'ST_Slope': [Slope_num]
        })

        # Scaling the data
        data_scaled = scaler.transform(data)

        # Prediction
        prediction = model.predict(data_scaled)
        if prediction[0] == 1:
            st.error("You have a high risk of heart disease.")
        else:
            st.success("You have a low risk of heart disease.")
        

# Navigation
pg = st.navigation([
    st.Page(Home, title="Home", icon="🏠"),
    st.Page(Visualizations, title="Visualization", icon="📊"),
    st.Page(Prediction, title="Prediction", icon="🪄")
])
pg.run()
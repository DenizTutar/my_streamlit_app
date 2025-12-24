import streamlit as st
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression






st.title("MY Streamlit App")


df = pd.read_csv('bank-full.csv', sep=';')

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])


name = st.text_input("Enter your name")
st.write(f"Hello, {name}! Welcome to the app.") 

st.success('This is a streamlit application for feature selection methods')
#st.text("This is a streamlit application for feature selection methods.")

X = df.drop("y", axis=1)
y = df["y"]

scaler = MinMaxScaler()
X_scaled_array = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled_array, columns=X.columns)

#st.subheader("rows of the dataset:")
st.dataframe(df.head())

#value = st.slider("Select your age:", 0, 100)
#st.write(f"You selected: {value} years old.")


#option = st.selectbox("Choose a model", ["k-NN", "SVM", "Decision Tree"])
#features = st.multiselect("Select features", df.columns, format_func=str.title)
#st.write(f"You selected: {', '.join(features)}")

#st.sidebar.title("Settings")
#option = st.sidebar.selectbox("Choose an option:", ["A", "B", "C"])

#OPTION
method = st.selectbox(
    "Choose Feature Selection Method:",
    ("Chi-Square (Filter)", "RFE (Wrapper)", "Random Forest (Embedded)")
)

st.divider()



#CHI-SQUARE
if method == "Chi-Square (Filter)":
    st.subheader("Chi-Square")    
    
    selector = SelectKBest(chi2, k=5)
    selector.fit(X_scaled_array, y)
    selected_cols = X.columns[selector.get_support()]
    st.success(f"Choosen features: {list(selected_cols)}")
    
    scores_df = pd.DataFrame({'Skore': selector.scores_}, index=X.columns)
    st.bar_chart(scores_df.sort_values('Skore', ascending=False))

#RFE
elif method == "RFE (Wrapper)":
    st.subheader("RFE")
    





    with st.spinner('Model training...'):
        model_lr = LogisticRegression(max_iter=1000)
        rfe = RFE(model_lr, n_features_to_select=5)
        rfe.fit(X_scaled_array, y)
    selected_cols = X.columns[rfe.get_support()]
    st.success(f"choosen features: {list(selected_cols)}")
    
    #GRAPH
    ranking_df = pd.DataFrame({'Rank': rfe.ranking_}, index=X.columns)
    
    
    #SORT
    st.bar_chart(ranking_df.sort_values('Rank'))
#RANDOM FOREST
elif method == "Random Forest (Embedded)":
    st.subheader("Random Forest")
    with st.spinner('Training trees...'):
        model_rf = RandomForestClassifier()
        model_rf.fit(X_scaled_array, y)
    
    importances = model_rf.feature_importances_
    importance_df = pd.DataFrame({'importance': importances}, index=X.columns)
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    #TOP5
    top_5 = importance_df.head(5).index.tolist()
    st.success(f"choosen features: {top_5}")
    
    #GRAPH
    st.write("Feature Importance Chart:")
    st.bar_chart(importance_df)






#st.subheader("DataFrame Preview")
#st.dataframe(df)

#st.table(df)
#st.table(df.head())

#st.json({"name": "Alice", "age": 22})

#if st.button("Click me"):
#st.write("Button clicked!")

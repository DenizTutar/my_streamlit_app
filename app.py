import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Feature Selection App", layout="wide")

st.title("Feature Selection Analysis App")
st.markdown("This application applies the methods from the **Feature Selection Methods** lecture notes on the Bank Marketing dataset.")

st.sidebar.header("User Input")
name = st.sidebar.text_input("Enter your name", "Student")
st.sidebar.write(f"Hello, **{name}**!")

st.sidebar.divider()

st.sidebar.subheader("Method Selection")
method = st.sidebar.radio(
    "Select a feature selection method:",
    ("Chi-Square (Filter)", "RFE (Wrapper)", "Random Forest (Embedded)")
)

st.sidebar.subheader("Visualization")
chart_type = st.sidebar.selectbox(
    "Chart to Display:",
    ("Method Scores", "Correlation Matrix", "Target Variable Distribution", "Boxplot", "Pie Chart")
)

df = pd.read_csv('bank-full.csv', sep=';')

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop("y", axis=1)
y = df["y"]

scaler = MinMaxScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns)

if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

st.divider()

if method == "Chi-Square (Filter)":
    st.header("Chi-Square (Filter Method)")
    st.info("This method looks at the statistical relationship between features and the target variable.")
    
    k_value = 5
    selector = SelectKBest(chi2, k=k_value)
    selector.fit(X_scaled_df, y)
    
    selected_cols = X.columns[selector.get_support()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"Top {k_value} features:")
        st.write(list(selected_cols))
        
    with col2:
        if chart_type == "Method Scores":
            st.subheader("Feature Scores")
            scores_df = pd.DataFrame({'Score': selector.scores_}, index=X.columns)
            st.bar_chart(scores_df.sort_values('Score', ascending=False))
        elif chart_type == "Correlation Matrix":
            st.subheader("Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(X_scaled_df.corr(), annot=False, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        elif chart_type == "Target Variable Distribution":
            st.subheader("Target Variable Distribution")
            st.bar_chart(y.value_counts())
        elif chart_type == "Boxplot":
            st.subheader(f"Boxplot: {selected_cols[0]}")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=y, y=X[selected_cols[0]], ax=ax, palette="Set2")
            st.pyplot(fig)
        elif chart_type == "Pie Chart":
            st.subheader("Target Variable Distribution (Pie)")
            fig, ax = plt.subplots()
            y.value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, cmap='Pastel1')
            ax.set_ylabel('')
            st.pyplot(fig)

elif method == "RFE (Wrapper)":
    st.header("Recursive Feature Elimination (Wrapper Method)")
    st.info("This method trains a model and recursively eliminates the weakest features.")
    
    with st.spinner('Training model, please wait...'):
        model_lr = LogisticRegression(max_iter=1000)
        rfe = RFE(model_lr, n_features_to_select=5)
        rfe.fit(X_scaled_df, y)
    
    selected_cols = X.columns[rfe.get_support()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("Selected Features:")
        st.write(list(selected_cols))
        
    with col2:
        if chart_type == "Method Scores":
            st.subheader("Feature Ranking (Lower = Better)")
            ranking_df = pd.DataFrame({'Rank': rfe.ranking_}, index=X.columns)
            st.bar_chart(ranking_df.sort_values('Rank').head(10))
        elif chart_type == "Correlation Matrix":
            st.subheader("Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(X_scaled_df.corr(), annot=False, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        elif chart_type == "Target Variable Distribution":
            st.subheader("Target Variable Distribution")
            st.bar_chart(y.value_counts())
        elif chart_type == "Boxplot":
            st.subheader(f"Boxplot: {selected_cols[0]}")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=y, y=X[selected_cols[0]], ax=ax, palette="Set2")
            st.pyplot(fig)
        elif chart_type == "Pie Chart":
            st.subheader("Target Variable Distribution (Pie)")
            fig, ax = plt.subplots()
            y.value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, cmap='Pastel1')
            ax.set_ylabel('')
            st.pyplot(fig)

elif method == "Random Forest (Embedded)":
    st.header("Random Forest (Embedded Method)")
    st.info("This method calculates feature importance during the training of a tree-based model.")
    
    with st.spinner('Building trees...'):
        model_rf = RandomForestClassifier()
        model_rf.fit(X_scaled_df, y)
    
    importances = model_rf.feature_importances_
    importance_df = pd.DataFrame({'Importance': importances}, index=X.columns)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    top_5 = importance_df.head(5).index.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("Top 5 Important Features:")
        st.write(top_5)
        
    with col2:
        if chart_type == "Method Scores":
            st.subheader("Feature Importance Scores")
            st.bar_chart(importance_df)
        elif chart_type == "Correlation Matrix":
            st.subheader("Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(X_scaled_df.corr(), annot=False, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        elif chart_type == "Target Variable Distribution":
            st.subheader("Target Variable Distribution")
            st.bar_chart(y.value_counts())
        elif chart_type == "Boxplot":
            st.subheader(f"Boxplot: {top_5[0]}")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=y, y=X[top_5[0]], ax=ax, palette="Set2")
            st.pyplot(fig)
        elif chart_type == "Pie Chart":
            st.subheader("Target Variable Distribution (Pie)")
            fig, ax = plt.subplots()
            y.value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, cmap='Pastel1')
            ax.set_ylabel('')
            st.pyplot(fig)
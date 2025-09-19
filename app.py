# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # For nicer, interactive charts
from collections import Counter
import re

# Set the page configuration
st.set_page_config(
    page_title="MCA eConsultation AI-Analyst",
    page_icon="📊",
    layout="wide"
)

# Load the final analyzed data
@st.cache_data  # This caches the data so it's only loaded once
def load_data():
    df = pd.read_csv('final_analyzed_comments.csv')
    return df

df = load_data()

# Title of the dashboard
st.title("📊 MCA eConsultation AI-Analyst Dashboard")
st.markdown("""
**AI-Powered Analysis of Public Feedback on Draft Legislations**
""")

# Sidebar for filters
st.sidebar.header("Filters")
stakeholder_type = st.sidebar.multiselect(
    "Select Stakeholder Type:",
    options=df['Stakeholder'].unique(),
    default=df['Stakeholder'].unique()
)

sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment:",
    options=df['sentiment'].unique(),
    default=df['sentiment'].unique()
)

# Filter data based on selections
filtered_df = df[
    (df['Stakeholder'].isin(stakeholder_type)) &
    (df['sentiment'].isin(sentiment_filter))
]

# Main Dashboard Overview
st.header("📈 Overview Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Comments", len(filtered_df))
with col2:
    st.metric("Positive Feedback", len(filtered_df[filtered_df['sentiment'] == 'Positive']))
with col3:
    st.metric("Negative Feedback", len(filtered_df[filtered_df['sentiment'] == 'Negative']))
with col4:
    st.metric("Neutral Feedback", len(filtered_df[filtered_df['sentiment'] == 'Neutral']))

# Section 1: Sentiment Distribution
st.header("🧮 Sentiment Distribution")

fig1 = px.pie(filtered_df, names='sentiment', title='Proportion of Sentiments',
             color='sentiment', color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'orange'})
st.plotly_chart(fig1, use_container_width=True)

# Section 2: Word Cloud
st.header("☁️ Word Cloud of Feedback")
st.image('wordcloud.png', use_column_width=True)
st.caption("This visual highlights the most frequently used words across all comments.")

# Section 3: Detailed Comment Analysis
st.header("🔍 Detailed Comment Analysis")
st.markdown("""
Browse the analyzed feedback below. Use the filters in the sidebar to focus on specific stakeholder groups or sentiments.
""")

for index, row in filtered_df.iterrows():
    # Choose a color based on sentiment
    if row['sentiment'] == 'Positive':
        border_color = 'green'
    elif row['sentiment'] == 'Negative':
        border_color = 'red'
    else:
        border_color = 'orange'

    # Create an expandable box for each comment
    with st.expander(f"**{row['Stakeholder']}** - _{row['sentiment']}_ Sentiment"):
        st.markdown(f"""
        **Original Comment:**  
        {row['Comment']}  

        **AI-Generated Summary:**  
        *{row['summary']}*  

        **Sentiment Confidence Score:** `{row['compound_score']:.3f}`
        """)

# Section 4: Raw Data (Optional Tab)
st.header("📋 Raw Data")
if st.checkbox("Show raw data table"):
    st.dataframe(filtered_df[['Stakeholder', 'Comment', 'summary', 'sentiment', 'compound_score']])

# Footer
st.markdown("---")
st.markdown("*Powered by AI & built for the Smart India Hackathon.*")

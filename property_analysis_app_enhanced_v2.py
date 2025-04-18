
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    sa3_df = pd.read_excel('Suburb Excel and Radar January 2025.xlsx', sheet_name='SA3')
    sa3_df.columns = sa3_df.columns.str.strip()
    
    # Generate mock demographic data
    np.random.seed(42)
    demographics = pd.DataFrame({
        'SA3': sa3_df['SA3'],
        'Median Income': np.random.randint(40000, 120000, size=len(sa3_df)),
        'Population Density': np.random.uniform(100, 5000, size=len(sa3_df)),
        'Median Age': np.random.randint(25, 60, size=len(sa3_df)),
        'Education Level (Index)': np.random.uniform(0, 1, size=len(sa3_df)),
    })
    
    return sa3_df.merge(demographics, on='SA3', how='left')

# Load data
data = load_data()

# Sidebar filters
st.sidebar.header('Filters')
selected_sa3 = st.sidebar.multiselect('Select SA3 Regions', data['SA3'].unique())
price_range = st.sidebar.slider('Price Range', 
                              float(data['Median'].min()), 
                              float(data['Median'].max()),
                              (float(data['Median'].min()), float(data['Median'].max())))

# Filter data based on selections
filtered_data = data.copy()
if selected_sa3:
    filtered_data = filtered_data[filtered_data['SA3'].isin(selected_sa3)]
filtered_data = filtered_data[
    (filtered_data['Median'] >= price_range[0]) & 
    (filtered_data['Median'] <= price_range[1])
]

# Main content
st.title('Enhanced Property Market Analysis Dashboard')

# Tabs for different analyses
tab1, tab2, tab3 = st.tabs(['Sociomap Analysis', 'Market Metrics', 'Detailed Statistics'])

with tab1:
    st.header('Sociomap Analysis')
    
    # Sociomap: Property Price vs. Demographics
    fig = px.scatter(filtered_data, 
                    x='Median Income',
                    y='Median',
                    size='Population Density',
                    color='Education Level (Index)',
                    hover_data=['SA3', 'Median Age'],
                    title='Property Prices vs. Demographics')
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    corr_cols = ['Median', '12M Price Change', 'Yield', 'Median Income', 
                 'Population Density', 'Median Age', 'Education Level (Index)']
    corr_matrix = filtered_data[corr_cols].corr()
    
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_cols,
                    y=corr_cols,
                    title='Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header('Market Metrics')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price Change Distribution
        fig = px.histogram(filtered_data,
                          x='12M Price Change',
                          title='12-Month Price Change Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Yield vs Price
        fig = px.scatter(filtered_data,
                        x='Median',
                        y='Yield',
                        title='Yield vs. Property Price')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Market Performance Radar Chart
        categories = ['Buy Affordability', 'Rent Affordability', 
                     'Sales Turnover', 'Rent Turnover', 'Yield']
        
        fig = go.Figure()
        for sa3 in filtered_data['SA3']:
            values = filtered_data[filtered_data['SA3'] == sa3][categories].values[0]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                name=sa3
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            title='Market Performance Radar'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header('Detailed Statistics')
    
    # Summary statistics
    st.subheader('Summary Statistics')
    st.dataframe(filtered_data.describe())
    
    # Detailed metrics table
    st.subheader('Detailed Metrics by SA3')
    detailed_cols = ['SA3', 'Median', '12M Price Change', 'Yield', 
                    'Buy Affordability', 'Rent Affordability',
                    'Median Income', 'Population Density', 'Median Age']
    st.dataframe(filtered_data[detailed_cols])
    
    # Download button for detailed data
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="property_analysis.csv",
        mime="text/csv"
    )

# Footer with key insights
st.markdown("---")
st.subheader("Key Insights")

# Calculate and display key metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Median Property Price", 
              f"${filtered_data['Median'].median():,.0f}",
              f"{filtered_data['12M Price Change'].mean():.1f}%")
with col2:
    st.metric("Average Yield",
              f"{filtered_data['Yield'].mean():.2f}%")
with col3:
    st.metric("Median Income",
              f"${filtered_data['Median Income'].median():,.0f}")

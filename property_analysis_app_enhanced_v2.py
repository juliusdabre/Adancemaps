
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
    
    # Calculate 2-month price change (mock data for demonstration)
    np.random.seed(42)
    sa3_df['2M Price Change'] = sa3_df['12M Price Change'] / 6 + np.random.normal(0, 0.5, len(sa3_df))
    
    # Generate mock demographic data
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
tab1, tab2, tab3, tab4 = st.tabs(['Price Change Heatmaps', 'Sociomap Analysis', 'Market Metrics', 'Detailed Statistics'])

with tab1:
    st.header('Price Change Analysis')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 2-Month Price Change Heatmap
        fig = px.density_heatmap(filtered_data,
                                x='Median',
                                y='2M Price Change',
                                title='2-Month Price Change vs. Median Price',
                                labels={'Median': 'Median Price ($)',
                                       '2M Price Change': '2-Month Price Change (%)'},
                                marginal_x='box',
                                marginal_y='violin')
        st.plotly_chart(fig, use_container_width=True)
        
        # Top/Bottom Performers Table
        st.subheader('Top/Bottom Performers (2-Month Change)')
        col1a, col1b = st.columns(2)
        
        with col1a:
            st.write('Top 5 Growth Areas')
            top_5 = filtered_data.nlargest(5, '2M Price Change')[['SA3', '2M Price Change', 'Median']]
            st.dataframe(top_5)
            
        with col1b:
            st.write('Bottom 5 Growth Areas')
            bottom_5 = filtered_data.nsmallest(5, '2M Price Change')[['SA3', '2M Price Change', 'Median']]
            st.dataframe(bottom_5)
    
    with col2:
        # Price Change Comparison
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=filtered_data['2M Price Change'],
            name='2-Month',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        fig.add_trace(go.Box(
            y=filtered_data['12M Price Change'],
            name='12-Month',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        fig.update_layout(
            title='Price Change Distribution: 2-Month vs 12-Month',
            yaxis_title='Price Change (%)',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Analysis
        st.subheader('Price Change Correlation Analysis')
        corr_data = filtered_data[['2M Price Change', '12M Price Change', 'Yield', 'Median']].corr()
        fig = px.imshow(corr_data,
                       labels=dict(color="Correlation"),
                       title='Price Change Correlations')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header('Sociomap Analysis')
    
    # Sociomap: Property Price vs. Demographics
    fig = px.scatter(filtered_data, 
                    x='Median Income',
                    y='Median',
                    size='Population Density',
                    color='2M Price Change',  # Updated to show 2M price change
                    hover_data=['SA3', 'Median Age'],
                    title='Property Prices vs. Demographics (Color: 2M Price Change)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    corr_cols = ['Median', '2M Price Change', '12M Price Change', 'Yield', 
                 'Median Income', 'Population Density', 'Median Age', 
                 'Education Level (Index)']
    corr_matrix = filtered_data[corr_cols].corr()
    
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_cols,
                    y=corr_cols,
                    title='Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header('Market Metrics')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price Change Distribution
        fig = px.histogram(filtered_data,
                          x=['2M Price Change', '12M Price Change'],
                          title='Price Change Distribution',
                          barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
        
        # Yield vs Price
        fig = px.scatter(filtered_data,
                        x='Median',
                        y='Yield',
                        color='2M Price Change',  # Updated to show 2M price change
                        title='Yield vs. Property Price (Color: 2M Price Change)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Market Performance Radar Chart
        categories = ['Buy Affordability', 'Rent Affordability', 
                     'Sales Turnover', 'Rent Turnover', 'Yield']
        
        fig = go.Figure()
        for sa3 in filtered_data['SA3'].head():  # Show top 5 for clarity
            values = filtered_data[filtered_data['SA3'] == sa3][categories].values[0]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                name=sa3
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            title='Market Performance Radar (Top 5 Areas)'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header('Detailed Statistics')
    
    # Summary statistics
    st.subheader('Summary Statistics')
    st.dataframe(filtered_data.describe())
    
    # Detailed metrics table
    st.subheader('Detailed Metrics by SA3')
    detailed_cols = ['SA3', 'Median', '2M Price Change', '12M Price Change', 
                    'Yield', 'Buy Affordability', 'Rent Affordability',
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
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Median Property Price", 
              f"${filtered_data['Median'].median():,.0f}",
              f"{filtered_data['12M Price Change'].mean():.1f}%")
with col2:
    st.metric("2M Price Change",
              f"{filtered_data['2M Price Change'].mean():.1f}%")
with col3:
    st.metric("Average Yield",
              f"{filtered_data['Yield'].mean():.2f}%")
with col4:
    st.metric("Median Income",
              f"${filtered_data['Median Income'].median():,.0f}")

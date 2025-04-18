
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(layout="wide", page_title="Property Market Analysis Dashboard")

# Title and description
st.title("Property Market Analysis Dashboard")
st.markdown("### Comprehensive analysis of property markets with socio-economic indicators")

# Load and prepare data
@st.cache_data
def load_data():
    # Load the original data
    sa3_df = pd.read_excel('Suburb Excel and Radar January 2025.xlsx', sheet_name='SA3')
    sa3_df.columns = sa3_df.columns.str.strip()
    
    # Add mock demographic data
    np.random.seed(42)
    sa3_df['Median Income'] = np.random.randint(40000, 120000, size=len(sa3_df))
    sa3_df['Population Density'] = np.random.uniform(100, 5000, size=len(sa3_df))
    sa3_df['Median Age'] = np.random.randint(25, 60, size=len(sa3_df))
    sa3_df['Education Level (Index)'] = np.random.uniform(0, 1, size=len(sa3_df))
    
    # Generate mock monthly price changes
    months = pd.date_range(end='2025-01-01', periods=12, freq='M')
    monthly_changes = pd.DataFrame(index=sa3_df['SA3'])
    
    for i in range(12):
        monthly_changes[months[i].strftime('%Y-%m')] = (
            sa3_df['12M Price Change'] / 12 + np.random.normal(0, 0.2, len(sa3_df))
        )
    
    return sa3_df, monthly_changes, months

# Load the data
sa3_df, monthly_changes, months = load_data()

# Sidebar for filtering
st.sidebar.header("Filters")
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=sa3_df['SA3'].unique(),
    default=sa3_df['SA3'].head().tolist()
)

# If no region is selected, use all regions
if not selected_regions:
    selected_regions = sa3_df['SA3'].tolist()

# Filter the dataframes
filtered_sa3 = sa3_df[sa3_df['SA3'].isin(selected_regions)]
filtered_monthly = monthly_changes.loc[selected_regions]

# Create two columns for the first row
col1, col2 = st.columns(2)

with col1:
    st.subheader("12-Month Price Change Trends")
    # Create price change trend plot
    fig_trend = go.Figure()
    for region in selected_regions:
        fig_trend.add_trace(go.Scatter(
            x=months,
            y=monthly_changes.loc[region],
            name=region,
            mode='lines+markers'
        ))
    fig_trend.update_layout(
        height=400,
        xaxis_title="Month",
        yaxis_title="Monthly Price Change (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.subheader("Socio-Economic Indicators")
    # Create scatter plot of Income vs Population Density
    fig_scatter = px.scatter(
        filtered_sa3,
        x='Median Income',
        y='Population Density',
        size='12M Price Change',
        color='Median Age',
        hover_name='SA3',
        height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Summary Statistics
st.subheader("Summary Statistics")
summary_cols = [
    'SA3', '12M Price Change', 'Median Income', 
    'Population Density', 'Median Age', 'Education Level (Index)'
]
st.dataframe(
    filtered_sa3[summary_cols].style.format({
        'Median Income': '${:,.0f}',
        'Population Density': '{:,.1f}',
        'Education Level (Index)': '{:.2f}',
        '12M Price Change': '{:.1f}%'
    })
)

# Additional Metrics
col3, col4, col5 = st.columns(3)

with col3:
    st.metric(
        "Average Median Income",
        f"${filtered_sa3['Median Income'].mean():,.0f}",
        f"{filtered_sa3['Median Income'].std():,.0f} σ"
    )

with col4:
    st.metric(
        "Average Price Change",
        f"{filtered_sa3['12M Price Change'].mean():.1f}%",
        f"{filtered_sa3['12M Price Change'].std():.1f}% σ"
    )

with col5:
    st.metric(
        "Average Population Density",
        f"{filtered_sa3['Population Density'].mean():,.1f}",
        f"{filtered_sa3['Population Density'].std():,.1f} σ"
    )

# Correlation Analysis
st.subheader("Correlation Analysis")
correlation_cols = ['12M Price Change', 'Median Income', 'Population Density', 'Median Age']
correlation_matrix = filtered_sa3[correlation_cols].corr()

fig_corr = px.imshow(
    correlation_matrix,
    text=correlation_matrix.round(2),
    aspect="auto",
    color_continuous_scale="RdBu"
)
fig_corr.update_layout(height=400)
st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit - Property Market Analysis Tool")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="🚀 Marketing Campaign Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .animated-title {
        animation: slideInDown 1s ease-out;
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    @keyframes slideInDown {
        from {
            transform: translateY(-100px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        margin: 0 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Data generation function (from your original code)
@st.cache_data
def generate_campaign_data(n=1000):
    channels = ['Email','Social','Search','Display','Affiliate']
    segments = ['New Customer','Returning','Loyal','At‑Risk']
    start = datetime(2024,1,1)
    records = []
    np.random.seed(42)
    
    for i in range(n):
        camp_id = f"C{i+1:04d}"
        channel = np.random.choice(channels, p=[0.2,0.25,0.3,0.15,0.1])
        segment = np.random.choice(segments, p=[0.3,0.4,0.2,0.1])
        launch = start + timedelta(days=np.random.randint(0,180))
        duration = np.random.randint(7,45)
        end = launch + timedelta(days=duration)
        budget = np.round(np.random.uniform(5_000, 100_000), 2)
        spend = np.round(budget * np.random.uniform(0.7,1.0), 2)
        impressions = int(spend * np.random.uniform(10, 50))
        clicks = int(impressions * np.random.uniform(0.01, 0.15))
        conversions = int(clicks * np.random.uniform(0.05, 0.3))
        ctr = clicks / impressions if impressions > 0 else 0
        conv_rate = conversions / clicks if clicks > 0 else 0
        revenue = np.round(conversions * np.random.uniform(50, 200), 2)
        roi = (revenue - spend) / spend if spend > 0 else 0
        
        records.append({
            'campaign_id': camp_id,
            'channel': channel,
            'segment': segment,
            'launch_date': launch,
            'end_date': end,
            'budget': budget,
            'spend': spend,
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'ctr': ctr,
            'conv_rate': conv_rate,
            'revenue': revenue,
            'roi': roi
        })
    
    df = pd.DataFrame(records)
    df['launch_month'] = df['launch_date'].dt.to_period('M').astype(str)
    return df

# Load data
df_mkt = generate_campaign_data(1000)

# Main header with animation
st.markdown("""
<div class="main-header">
    <h1 class="animated-title">🚀 Marketing Campaign Analytics Dashboard</h1>
    <p style="text-align: center; color: white; font-size: 1.2rem; margin-top: 1rem;">
        Advanced insights into your marketing performance with interactive visualizations
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("### 🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Filters
selected_channels = st.sidebar.multiselect(
    "📺 Select Channels",
    options=df_mkt['channel'].unique(),
    default=df_mkt['channel'].unique()
)

selected_segments = st.sidebar.multiselect(
    "👥 Select Customer Segments",
    options=df_mkt['segment'].unique(),
    default=df_mkt['segment'].unique()
)

# Date range
min_date = df_mkt['launch_date'].min()
max_date = df_mkt['launch_date'].max()
date_range = st.sidebar.date_input(
    "📅 Campaign Launch Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Animation toggle
animate_charts = st.sidebar.checkbox("✨ Enable Chart Animations", value=True)

# Filter data
filtered_df = df_mkt[
    (df_mkt['channel'].isin(selected_channels)) &
    (df_mkt['segment'].isin(selected_segments)) &
    (df_mkt['launch_date'] >= pd.Timestamp(date_range[0])) &
    (df_mkt['launch_date'] <= pd.Timestamp(date_range[1]))
]

# Key metrics with animated counters
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_campaigns = len(filtered_df)
    st.metric("🎯 Total Campaigns", f"{total_campaigns:,}")

with col2:
    total_spend = filtered_df['spend'].sum()
    st.metric("💰 Total Spend", f"${total_spend:,.0f}")

with col3:
    total_revenue = filtered_df['revenue'].sum()
    st.metric("📈 Total Revenue", f"${total_revenue:,.0f}")

with col4:
    avg_roi = filtered_df['roi'].mean()
    st.metric("🔄 Average ROI", f"{avg_roi:.1%}")

with col5:
    total_conversions = filtered_df['conversions'].sum()
    st.metric("🎉 Total Conversions", f"{total_conversions:,}")

st.markdown("---")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Overview", "🎯 Channel Analysis", "👥 Segment Insights", "📈 Advanced Analytics"])

with tab1:
    st.markdown("### 🌟 Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D Budget Distribution
        fig_budget = px.histogram(
            filtered_df, 
            x='budget', 
            nbins=30,
            title='💰 Campaign Budget Distribution',
            color_discrete_sequence=['#667eea']
        )
        fig_budget.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font_size=16,
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        if animate_charts:
            fig_budget.update_traces(opacity=0.8)
            fig_budget.update_layout(transition_duration=500)
        
        st.plotly_chart(fig_budget, use_container_width=True)
    
    with col2:
        # ROI vs Revenue Bubble Chart
        fig_bubble = px.scatter(
            filtered_df,
            x='revenue',
            y='roi',
            size='conversions',
            color='channel',
            hover_data=['campaign_id', 'spend'],
            title='🎯 ROI vs Revenue (Bubble Size = Conversions)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_bubble.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font_size=16,
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        if animate_charts:
            fig_bubble.update_traces(opacity=0.7)
        
        st.plotly_chart(fig_bubble, use_container_width=True)
    
    # Monthly Campaign Launches with Animation
    monthly_data = filtered_df.groupby('launch_month').agg({
        'campaign_id': 'count',
        'spend': 'sum',
        'revenue': 'sum'
    }).reset_index()
    monthly_data.columns = ['month', 'launches', 'spend', 'revenue']
    
    fig_monthly = go.Figure()
    
    # Add bars for launches
    fig_monthly.add_trace(go.Bar(
        x=monthly_data['month'],
        y=monthly_data['launches'],
        name='Campaigns Launched',
        marker_color='#667eea',
        opacity=0.8
    ))
    
    # Add line for spend
    fig_monthly.add_trace(go.Scatter(
        x=monthly_data['month'],
        y=monthly_data['spend']/1000,  # Scale down for visibility
        mode='lines+markers',
        name='Spend (K$)',
        line=dict(color='#f093fb', width=3),
        yaxis='y2'
    ))
    
    fig_monthly.update_layout(
        title='📅 Monthly Campaign Launches & Spend Trends',
        xaxis_title='Month',
        yaxis_title='Number of Campaigns',
        yaxis2=dict(title='Spend (K$)', overlaying='y', side='right'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)

with tab2:
    st.markdown("### 🎯 Channel Deep Dive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sunburst Chart
        channel_segment_data = filtered_df.groupby(['channel', 'segment'])['conversions'].sum().reset_index()
        fig_sunburst = px.sunburst(
            channel_segment_data,
            path=['channel', 'segment'],
            values='conversions',
            title='🌅 Channel → Segment Conversion Flow',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_sunburst.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    with col2:
        # Radar Chart for Channel Performance
        channel_metrics = filtered_df.groupby('channel').agg({
            'ctr': 'mean',
            'conv_rate': 'mean',
            'roi': 'mean',
            'spend': 'mean'
        }).reset_index()
        
        # Normalize metrics for radar chart
        for col in ['ctr', 'conv_rate', 'roi', 'spend']:
            channel_metrics[f'{col}_norm'] = (channel_metrics[col] - channel_metrics[col].min()) / (channel_metrics[col].max() - channel_metrics[col].min())
        
        fig_radar = go.Figure()
        
        for channel in channel_metrics['channel']:
            channel_data = channel_metrics[channel_metrics['channel'] == channel]
            fig_radar.add_trace(go.Scatterpolar(
                r=[channel_data['ctr_norm'].iloc[0], 
                   channel_data['conv_rate_norm'].iloc[0], 
                   channel_data['roi_norm'].iloc[0], 
                   channel_data['spend_norm'].iloc[0]],
                theta=['CTR', 'Conversion Rate', 'ROI', 'Spend'],
                fill='toself',
                name=channel,
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title='📡 Channel Performance Radar',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Channel Spend Waterfall
    channel_spend = filtered_df.groupby('channel')['spend'].sum().sort_values(ascending=False)
    
    fig_waterfall = go.Figure(go.Waterfall(
        name="Channel Spend",
        orientation="v",
        measure=["relative"] * len(channel_spend),
        x=channel_spend.index,
        textposition="outside",
        text=[f"${v:,.0f}" for v in channel_spend.values],
        y=channel_spend.values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#667eea"}},
        decreasing={"marker": {"color": "#f093fb"}},
    ))
    
    fig_waterfall.update_layout(
        title="💰 Channel Spend Breakdown",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

with tab3:
    st.markdown("### 👥 Customer Segment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Violin Plot for ROI by Segment
        fig_violin = px.violin(
            filtered_df,
            x='segment',
            y='roi',
            box=True,
            title='🎻 ROI Distribution by Segment',
            color='segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_violin.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    
    with col2:
        # Treemap for Segment Revenue
        segment_revenue = filtered_df.groupby('segment')['revenue'].sum().reset_index()
        fig_treemap = px.treemap(
            segment_revenue,
            path=['segment'],
            values='revenue',
            title='🌳 Revenue by Segment',
            color='revenue',
            color_continuous_scale='Viridis'
        )
        fig_treemap.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    # Segment Performance Matrix
    segment_metrics = filtered_df.groupby('segment').agg({
        'ctr': 'mean',
        'conv_rate': 'mean',
        'roi': 'mean',
        'revenue': 'sum'
    }).reset_index()
    
    fig_scatter_matrix = px.scatter_matrix(
        segment_metrics,
        dimensions=['ctr', 'conv_rate', 'roi'],
        color='segment',
        title='📊 Segment Performance Matrix',
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig_scatter_matrix.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_scatter_matrix, use_container_width=True)

with tab4:
    st.markdown("### 📈 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation Heatmap
        numeric_cols = ['budget', 'spend', 'impressions', 'clicks', 'conversions', 'ctr', 'conv_rate', 'revenue', 'roi']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title='🔥 Correlation Heatmap',
            color_continuous_scale='RdBu_r'
        )
        fig_heatmap.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # CTR Distribution with Histogram
        fig_dist = px.histogram(
            filtered_df,
            x='ctr',
            nbins=30,
            title='📊 Click-Through Rate Distribution',
            color_discrete_sequence=['#667eea'],
            marginal='box'  # Adds box plot on top
        )
        fig_dist.update_layout(
            title='📊 Click-Through Rate Distribution',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            xaxis_title='Click-Through Rate',
            yaxis_title='Frequency'
        )
        fig_dist.update_traces(opacity=0.7)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Performance Comparison Table
    st.markdown("### 📋 Performance Comparison Table")
    
    comparison_data = filtered_df.groupby(['channel', 'segment']).agg({
        'spend': 'sum',
        'revenue': 'sum',
        'conversions': 'sum',
        'roi': 'mean',
        'ctr': 'mean',
        'conv_rate': 'mean'
    }).round(3).reset_index()
    
    # Style the dataframe
    styled_df = comparison_data.style.background_gradient(
        subset=['spend', 'revenue', 'conversions'],
        cmap='Blues'
    ).background_gradient(
        subset=['roi', 'ctr', 'conv_rate'],
        cmap='RdYlGn'
    )
    
    st.dataframe(styled_df, use_container_width=True)

# Real-time updates simulation
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; margin-top: 2rem;">
    <p>🚀 Advanced Marketing Analytics Dashboard | Built with Streamlit & Plotly</p>
    <p>📊 Real-time insights for data-driven marketing decisions</p>
</div>
""", unsafe_allow_html=True)

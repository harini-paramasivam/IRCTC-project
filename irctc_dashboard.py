import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
import warnings
warnings.filterwarnings('ignore')

# Define the IRCTC color theme palette for all visualizations
TEAL_BLUE = "#00838f"       # IRCTC header color (teal blue)
BRIGHT_BLUE = "#2196f3"     # Primary accent color
AMBER_ORANGE = "#ffa000"    # Highlight color for alerts
LIGHT_TEAL = "#b2ebf2"      # Sidebar color
SOFT_GREY = "#f5f7fa"       # Background color
GREEN_SUCCESS = "#4caf50"   # For positive metrics
ORANGE_ALERT = "#ff9800"    # For delays/negative metrics
PURPLE = "#9c27b0"          # For forecasts
LIGHT_GREEN = "#c8e6c9"     # For heatmap low values
YELLOW = "#fff9c4"          # For heatmap medium values
DEEP_ORANGE = "#ff7043"     # For heatmap high values
WHITE = "#ffffff"           # For cards and text

# Create a function to generate sample train data for demonstration
@st.cache_data
def load_sample_data():
    # Sample data with train numbers, names, routes, schedules, etc.
    n_samples = 500
    
    # Train types and weights
    train_types = ['Rajdhani Express', 'Shatabdi Express', 'Duronto Express', 
                  'Vande Bharat', 'Jan Shatabdi', 'Sampark Kranti', 
                  'Garib Rath', 'Superfast', 'Express', 'Passenger']
    type_weights = [0.08, 0.08, 0.06, 0.05, 0.07, 0.06, 0.05, 0.2, 0.25, 0.1]
    
    # Major cities/stations
    stations = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore', 
               'Hyderabad', 'Ahmedabad', 'Pune', 'Jaipur', 'Lucknow',
               'Patna', 'Bhopal', 'Chandigarh', 'Kochi', 'Guwahati']
    
    # Generate train numbers and names
    train_numbers = [f"{random.randint(10000, 99999)}" for _ in range(n_samples)]
    train_names = [f"{np.random.choice(train_types, p=type_weights)} {i+1}" for i in range(n_samples)]
    
    # Generate source and destination stations
    sources = np.random.choice(stations, n_samples)
    destinations = []
    for src in sources:
        available_stations = [s for s in stations if s != src]
        destinations.append(np.random.choice(available_stations))
    
    # Generate scheduled departures (distributed throughout the day)
    scheduled_deps = []
    for _ in range(n_samples):
        hour = random.randint(0, 23)
        minute = random.choice([0, 15, 30, 45])
        scheduled_deps.append(f"{hour:02d}:{minute:02d}")
    
    # Generate actual departures with delays
    actual_deps = []
    delays = []
    for sched in scheduled_deps:
        hour, minute = map(int, sched.split(':'))
        delay_minutes = np.random.choice([0, 0, 0, 0, 5, 10, 15, 20, 30, 45, 60, 90, 120], 
                                        p=[0.3, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])
        delays.append(delay_minutes)
        
        # Calculate actual departure time
        total_minutes = hour * 60 + minute + delay_minutes
        new_hour = (total_minutes // 60) % 24
        new_minute = total_minutes % 60
        actual_deps.append(f"{new_hour:02d}:{new_minute:02d}")
    
    # Generate delay reasons for delayed trains
    delay_reasons = ['On Time', 'Signal Issue', 'Track Maintenance', 'Weather Conditions', 
                    'Technical Fault', 'Congestion', 'Accident', 'Late Arrival of Link Train']
    reason_weights = [0.0, 0.2, 0.15, 0.15, 0.2, 0.15, 0.05, 0.1]  # First weight is placeholder, will be handled conditionally
    
    delay_explanations = []
    for delay in delays:
        if delay == 0:
            delay_explanations.append('On Time')
        else:
            delay_explanations.append(np.random.choice(delay_reasons[1:], p=reason_weights[1:]))
    
    # Generate train statuses
    statuses = []
    for delay in delays:
        if delay == 0:
            statuses.append('On Time')
        elif delay <= 15:
            statuses.append('Slight Delay')
        elif delay <= 30:
            statuses.append('Delayed')
        elif delay <= 60:
            statuses.append('Significant Delay')
        else:
            statuses.append('Severely Delayed')
    
    # Create zone information and train capacity
    zones = ['Northern', 'Southern', 'Eastern', 'Western', 'Central', 'North Eastern', 'South Eastern']
    capacities = [np.random.randint(500, 1500) for _ in range(n_samples)]
    booking_percentages = [min(100, max(50, random.normalvariate(85, 15))) for _ in range(n_samples)]
    
    # Calculate occupancy
    occupancies = [int(capacity * (booking_percentage/100)) for capacity, booking_percentage in zip(capacities, booking_percentages)]
    
    # Generate dates for the past week
    today = datetime.now().date()
    dates = [(today - timedelta(days=random.randint(0, 6))).strftime("%Y-%m-%d") for _ in range(n_samples)]
    
    # Structure the data
    data = {
        'train_number': train_numbers,
        'train_name': train_names,
        'source': sources,
        'destination': destinations,
        'scheduled_departure': scheduled_deps,
        'actual_departure': actual_deps,
        'delay_minutes': delays,
        'delay_reason': delay_explanations,
        'status': statuses,
        'zone': np.random.choice(zones, n_samples),
        'capacity': capacities,
        'occupancy': occupancies,
        'date': dates
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add dummy coordinates for map visualization
    # (These aren't real coordinates, just for demonstration)
    station_coords = {
        'Delhi': [28.7041, 77.1025],
        'Mumbai': [19.0760, 72.8777],
        'Chennai': [13.0827, 80.2707],
        'Kolkata': [22.5726, 88.3639],
        'Bangalore': [12.9716, 77.5946],
        'Hyderabad': [17.3850, 78.4867],
        'Ahmedabad': [23.0225, 72.5714],
        'Pune': [18.5204, 73.8567],
        'Jaipur': [26.9124, 75.7873],
        'Lucknow': [26.8467, 80.9462],
        'Patna': [25.5941, 85.1376],
        'Bhopal': [23.2599, 77.4126],
        'Chandigarh': [30.7333, 76.7794],
        'Kochi': [9.9312, 76.2673],
        'Guwahati': [26.1158, 91.7086]
    }
    
    df['source_lat'] = df['source'].map(lambda x: station_coords[x][0])
    df['source_lon'] = df['source'].map(lambda x: station_coords[x][1])
    df['dest_lat'] = df['destination'].map(lambda x: station_coords[x][0])
    df['dest_lon'] = df['destination'].map(lambda x: station_coords[x][1])
    
    # Add scheduled arrival times (for demonstration)
    df['scheduled_arrival'] = df.apply(lambda x: 
        f"{random.randint(0, 23):02d}:{random.choice([0, 15, 30, 45]):02d}", axis=1)
    
    return df

# Function to convert times to hours
def time_to_hours(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours + minutes/60

# Page configuration
st.set_page_config(
    page_title="IRCTC Real-Time Dashboard",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(f"""
<style>
    .stApp {{
        background-color: {SOFT_GREY};
    }}
    .main-header {{
        background-color: {TEAL_BLUE};
        color: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 131, 143, 0.2);
    }}
    .main-title {{
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 0;
    }}
    .sub-header {{
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 5px;
    }}
    .metric-card {{
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        text-align: center;
        height: 100%;
        transition: transform 0.2s;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
    }}
    .metric-card-on-time {{
        border-top: 5px solid {GREEN_SUCCESS};
    }}
    .metric-card-delayed {{
        border-top: 5px solid {ORANGE_ALERT};
    }}
    .metric-card-info {{
        border-top: 5px solid {BRIGHT_BLUE};
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: bold;
        color: {TEAL_BLUE};
    }}
    .metric-value-success {{
        color: {GREEN_SUCCESS};
    }}
    .metric-value-warning {{
        color: {ORANGE_ALERT};
    }}
    .metric-label {{
        font-size: 1rem;
        color: #555;
        margin-top: 8px;
        font-weight: 500;
    }}
    div[data-testid="stSidebar"] {{
        background-color: {LIGHT_TEAL};
        padding-top: 2rem;
    }}
    div[data-testid="stSidebar"] .block-container {{
        padding-top: 0;
    }}
    .sidebar-title {{
        color: {TEAL_BLUE};
        font-size: 1.2rem;
        font-weight: bold;
        padding-left: 10px;
    }}
    .chart-container {{
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        margin-bottom: 24px;
    }}
    .alert-box {{
        background-color: {AMBER_ORANGE};
        border-radius: 8px;
        padding: 12px 20px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(255, 160, 0, 0.2);
    }}
    .alert-title {{
        font-weight: bold;
        margin-bottom: 5px;
    }}
    .stSlider > div {{
        background-color: #e0e0e0;
    }}
    .stSlider > div > div > div {{
        background-color: {BRIGHT_BLUE};
    }}
    div.stButton > button {{
        background-color: {BRIGHT_BLUE};
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        border-radius: 30px;
        box-shadow: 0 2px 5px rgba(33, 150, 243, 0.3);
        transition: all 0.2s;
    }}
    div.stButton > button:hover {{
        background-color: {TEAL_BLUE};
        box-shadow: 0 4px 8px rgba(0, 131, 143, 0.4);
        transform: translateY(-2px);
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {TEAL_BLUE};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #e1f5fe;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: {TEAL_BLUE};
        border: 1px solid #e0e0e0;
        border-bottom: none;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {BRIGHT_BLUE};
        color: white;
        border: none;
    }}
    .irctc-logo {{
        display: flex;
        align-items: center;
        justify-content: center;
        color: {TEAL_BLUE};
        margin-bottom: 2rem;
        font-size: 2.5rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }}
    .route-line-success {{
        color: {GREEN_SUCCESS};
    }}
    .route-line-warning {{
        color: {ORANGE_ALERT};
    }}
</style>
""", unsafe_allow_html=True)

# Load data
df = load_sample_data()

# Convert delay minutes to categories for filtering
df['delay_category'] = pd.cut(
    df['delay_minutes'], 
    bins=[-1, 0, 15, 30, 60, float('inf')],
    labels=['On Time', 'Minor Delay (≤15m)', 'Moderate Delay (≤30m)', 
            'Significant Delay (≤60m)', 'Severe Delay (>60m)']
)

# Sidebar
with st.sidebar:
    st.markdown('<div class="irctc-logo"><h2>IRCTC</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🔍 Filter Trains</div>', unsafe_allow_html=True)
    
    # Date selector
    date_list = sorted(df['date'].unique(), reverse=True)
    selected_date = st.selectbox("Select Date", date_list)
    
    # Filter by train status
    status_options = ['All'] + sorted(df['status'].unique().tolist())
    selected_status = st.selectbox("Train Status", status_options)
    
    # Filter by zone
    zone_options = ['All'] + sorted(df['zone'].unique().tolist())
    selected_zone = st.selectbox("Railway Zone", zone_options)
    
    # Filter by delay category
    delay_options = ['All'] + sorted(df['delay_category'].unique().tolist())
    selected_delay = st.selectbox("Delay Category", delay_options)
    
    # Filter by train type
    train_types = df['train_name'].apply(lambda x: x.split()[0] + " " + x.split()[1] if len(x.split()) > 1 else x.split()[0])
    train_type_options = ['All'] + sorted(train_types.unique().tolist())
    selected_train_type = st.selectbox("Train Type", train_type_options)
    
    # Filter by source/destination
    station_options = ['All'] + sorted(df['source'].unique().tolist())
    selected_source = st.selectbox("Source Station", station_options)
    
    station_options = ['All'] + sorted(df['destination'].unique().tolist())
    selected_destination = st.selectbox("Destination Station", station_options)
    
    # Occupancy filter slider
    min_occupancy = 50  # Set a default minimum
    max_occupancy = 100
    occupancy_range = st.slider(
        "Occupancy Percentage", 
        min_occupancy, 
        max_occupancy, 
        (min_occupancy, max_occupancy)
    )
    
    # Apply filters button
    apply_filters = st.button("Apply Filters")

# Filter data based on sidebar selections
filtered_df = df.copy()

# Apply date filter
filtered_df = filtered_df[filtered_df['date'] == selected_date]

# Apply status filter
if selected_status != 'All':
    filtered_df = filtered_df[filtered_df['status'] == selected_status]

# Apply zone filter
if selected_zone != 'All':
    filtered_df = filtered_df[filtered_df['zone'] == selected_zone]

# Apply delay category filter
if selected_delay != 'All':
    filtered_df = filtered_df[filtered_df['delay_category'] == selected_delay]

# Apply train type filter
if selected_train_type != 'All':
    filtered_df = filtered_df[filtered_df['train_name'].str.startswith(selected_train_type)]

# Apply source/destination filters
if selected_source != 'All':
    filtered_df = filtered_df[filtered_df['source'] == selected_source]

if selected_destination != 'All':
    filtered_df = filtered_df[filtered_df['destination'] == selected_destination]

# Apply occupancy filter
filtered_df = filtered_df[
    (filtered_df['occupancy']/filtered_df['capacity']*100 >= occupancy_range[0]) & 
    (filtered_df['occupancy']/filtered_df['capacity']*100 <= occupancy_range[1])
]

# Main dashboard header
st.markdown(f"""
<div class="main-header">
    <div class="main-title">IRCTC Real-Time Train Monitoring Dashboard</div>
    <div class="sub-header">Last Updated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}</div>
</div>
""", unsafe_allow_html=True)

# Alert box for severely delayed trains
severely_delayed = filtered_df[filtered_df['delay_minutes'] > 60]
if not severely_delayed.empty:
    train_info = severely_delayed.iloc[0]
    st.markdown(f"""
    <div class="alert-box">
        <div class="alert-title">⚠️ Alert: {len(severely_delayed)} Trains Severely Delayed</div>
        <div>{train_info['train_number']} {train_info['train_name']} is delayed by {train_info['delay_minutes']} minutes due to {train_info['delay_reason']}</div>
    </div>
    """, unsafe_allow_html=True)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_trains = len(filtered_df)
    st.markdown(f"""
    <div class="metric-card metric-card-info">
        <div class="metric-value">{total_trains}</div>
        <div class="metric-label">Total Trains Running</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    on_time_count = len(filtered_df[filtered_df['delay_minutes'] == 0])
    on_time_percentage = int((on_time_count / total_trains) * 100) if total_trains > 0 else 0
    
    st.markdown(f"""
    <div class="metric-card metric-card-on-time">
        <div class="metric-value metric-value-success">{on_time_percentage}%</div>
        <div class="metric-label">Trains On Time</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_delay = filtered_df['delay_minutes'].mean() if not filtered_df.empty else 0
    
    st.markdown(f"""
    <div class="metric-card metric-card-delayed">
        <div class="metric-value metric-value-warning">{avg_delay:.1f} min</div>
        <div class="metric-label">Average Delay</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    delayed_count = len(filtered_df[filtered_df['delay_minutes'] > 0])
    
    st.markdown(f"""
    <div class="metric-card metric-card-delayed">
        <div class="metric-value">{delayed_count}</div>
        <div class="metric-label">Delayed Trains</div>
    </div>
    """, unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Delay Analysis", "🗺️ Geographic View", "📝 Train Details"])

with tab1:
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Delay Distribution by Train Type")
        
        # Calculate average delay by train type
        if not filtered_df.empty:
            train_types = filtered_df['train_name'].apply(lambda x: ' '.join(x.split()[:2]) if len(x.split()) > 1 else x)
            delay_by_type = filtered_df.groupby(train_types)['delay_minutes'].mean().reset_index()
            delay_by_type.columns = ['train_type', 'avg_delay']
            delay_by_type = delay_by_type.sort_values('avg_delay', ascending=False)
            
            fig = px.bar(
                delay_by_type,
                x='train_type',
                y='avg_delay',
                color='avg_delay',
                color_continuous_scale=[[0, GREEN_SUCCESS], [0.5, YELLOW], [1.0, ORANGE_ALERT]],
                labels={'train_type': 'Train Type', 'avg_delay': 'Average Delay (minutes)'}
            )
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                coloraxis_showscale=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available with the current filters")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with row1_col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Delay Reasons")
        
        # Create pie chart for delay reasons
        if not filtered_df.empty:
            delayed_trains = filtered_df[filtered_df['delay_minutes'] > 0]
            if not delayed_trains.empty:
                reason_counts = delayed_trains['delay_reason'].value_counts().reset_index()
                reason_counts.columns = ['reason', 'count']
                
                fig = px.pie(
                    reason_counts, 
                    names='reason', 
                    values='count',
                    hole=.4,
                    color_discrete_sequence=[BRIGHT_BLUE, AMBER_ORANGE, PURPLE, 
                                            "#00acc1", "#5e35b1", "#00897b", "#ffc107"]
                )
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='white',
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No delayed trains in the current selection")
        else:
            st.info("No data available with the current filters")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Full width delay heatmap
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Delay Intensity by Hour and Zone")
    
    # Convert scheduled departure to hour
    if not filtered_df.empty:
        filtered_df['departure_hour'] = filtered_df['scheduled_departure'].apply(
            lambda x: int(x.split(':')[0])
        )
        
        # Create heatmap data
        heatmap_data = filtered_df.groupby(['zone', 'departure_hour'])['delay_minutes'].mean().reset_index()
        
        if not heatmap_data.empty:
            heatmap_pivot = pd.pivot_table(
                heatmap_data, 
                values='delay_minutes',
                index='zone', 
                columns='departure_hour'
            ).fillna(0)
            
            fig = px.imshow(
                heatmap_pivot,
                color_continuous_scale=[LIGHT_GREEN, YELLOW, DEEP_ORANGE],
                labels=dict(x="Hour of Day", y="Railway Zone", color="Avg Delay (min)"),
                x=[f"{hour:02d}:00" for hour in range(24) if hour in heatmap_pivot.columns],
                aspect="auto"
            )
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='white',
                height=400,
                coloraxis_colorbar=dict(
                    title="Avg Delay (min)",
                    thicknessmode="pixels", 
                    thickness=15,
                    lenmode="pixels",
                    len=300,
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data to generate heatmap")
    else:
        st.info("No data available with the current filters")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Delay trend over hours
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Delay Trend Throughout Day")
    
    if not filtered_df.empty:
        # Calculate average delay by hour
        hourly_delay = filtered_df.groupby('departure_hour')['delay_minutes'].mean().reset_index()
        hourly_delay.columns = ['hour', 'avg_delay']
        
        if not hourly_delay.empty:
            # Calculate forecast line (simulated)
            forecast_hours = list(range(24))
            forecast_hours = [h for h in forecast_hours if h in hourly_delay['hour'].values]
            np.random.seed(42)  # For reproducibility
            
            hourly_delay_dict = dict(zip(hourly_delay['hour'], hourly_delay['avg_delay']))
            forecast_delays = [max(0, hourly_delay_dict.get(h, 0) + np.random.normal(0, 3)) for h in forecast_hours]
            
            fig = go.Figure()
            
            # Add actual delay line
            fig.add_trace(go.Scatter(
                x=[f"{h:02d}:00" for h in hourly_delay['hour']],
                y=hourly_delay['avg_delay'],
                mode='lines+markers',
                name='Actual Delay',
                line=dict(color=ORANGE_ALERT, width=3),
                marker=dict(size=8)
            ))
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=[f"{h:02d}:00" for h in forecast_hours],
                y=forecast_delays,
                mode='lines',
                name='Forecast',
                line=dict(color=PURPLE, width=2, dash='dash')
            ))
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=300,
                xaxis=dict(title="Hour of Day"),
                yaxis=dict(title="Average Delay (minutes)"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data to generate trend chart")
    else:
        st.info("No data available with the current filters")
        
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Train Routes Map")
    
    if not filtered_df.empty:
        # Create routes for map (limit for performance)
        route_df = filtered_df[['source', 'destination', 'source_lat', 'source_lon', 
                              'dest_lat', 'dest_lon', 'train_number', 'delay_minutes']].head(50)
        
        fig = go.Figure()
        
        # Create base map of India
        fig.add_trace(go.Scattergeo(
            lon = [68, 97],
            lat = [8, 37],
            mode = 'markers',
            marker=dict(size=0.1, color='white'),  # Invisible markers to set map bounds
            showlegend=False,
        ))
        
        # Add station markers for sources
        fig.add_trace(go.Scattergeo(
            lon = route_df['source_lon'].tolist(),
            lat = route_df['source_lat'].tolist(),
            text = route_df['source'].tolist(),
            mode = 'markers',
            marker = dict(
                size = 8,
                color = BRIGHT_BLUE,
                line = dict(width=1, color='white')
            ),
            name = 'Source Stations'
        ))
        
        # Add station markers for destinations
        fig.add_trace(go.Scattergeo(
            lon = route_df['dest_lon'].tolist(),
            lat = route_df['dest_lat'].tolist(),
            text = route_df['destination'].tolist(),
            mode = 'markers',
            marker = dict(
                size = 8,
                color = TEAL_BLUE,
                line = dict(width=1, color='white')
            ),
            name = 'Destination Stations'
        ))
        
        # Add lines for routes, colored by delay
        for i, row in route_df.iterrows():
            color = GREEN_SUCCESS if row['delay_minutes'] == 0 else ORANGE_ALERT
            width = 1 if row['delay_minutes'] == 0 else 2
            
            fig.add_trace(
                go.Scattergeo(
                    lon = [row['source_lon'], row['dest_lon']],
                    lat = [row['source_lat'], row['dest_lat']],
                    mode = 'lines',
                    line = dict(width=width, color=color),
                    name = f"Train {row['train_number']}",
                    showlegend=False
                )
            )
        
        fig.update_geos(
            scope = 'asia',
            resolution = 50,
            showcoastlines = True, coastlinecolor = "#444",
            showland = True, landcolor = "#F5F5F5",
            showocean = True, oceancolor = "LightBlue",
            showlakes = True, lakecolor = "LightBlue",
            showrivers = True, rivercolor = "LightBlue",
            fitbounds = "locations",
            visible = True
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='white',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available with the current filters")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Top Busy Routes")
        
        if not filtered_df.empty:
            # Calculate route counts
            route_counts = filtered_df.groupby(['source', 'destination']).size().reset_index()
            route_counts.columns = ['source', 'destination', 'count']
            route_counts = route_counts.sort_values('count', ascending=False).head(10)
            
            if not route_counts.empty:
                # Create combined route name
                route_counts['route'] = route_counts['source'] + ' to ' + route_counts['destination']
                
                fig = px.bar(
                    route_counts,
                    y='route',
                    x='count',
                    orientation='h',
                    color='count',
                    color_continuous_scale=[[0, LIGHT_GREEN], [0.5, BRIGHT_BLUE], [1.0, TEAL_BLUE]],
                    labels={'route': 'Route', 'count': 'Number of Trains'}
                )
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    coloraxis_showscale=False,
                    height=350,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data to generate route analysis")
        else:
            st.info("No data available with the current filters")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Zones by On-Time Performance")
        
        if not filtered_df.empty:
            # Calculate on-time percentage by zone
            zone_performance = filtered_df.groupby('zone').apply(
                lambda x: (x['delay_minutes'] == 0).sum() / len(x) * 100 if len(x) > 0 else 0
            ).reset_index()
            zone_performance.columns = ['zone', 'on_time_percentage']
            zone_performance = zone_performance.sort_values('on_time_percentage')
            
            if not zone_performance.empty:
                fig = px.bar(
                    zone_performance,
                    y='zone',
                    x='on_time_percentage',
                    orientation='h',
                    color='on_time_percentage',
                    color_continuous_scale=[[0, ORANGE_ALERT], [0.7, YELLOW], [1.0, GREEN_SUCCESS]],
                    labels={'zone': 'Railway Zone', 'on_time_percentage': 'On-Time Percentage (%)'},
                    range_color=[0, 100]
                )
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    coloraxis_showscale=False,
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data to generate zone performance analysis")
        else:
            st.info("No data available with the current filters")
            
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    # Train details table
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Live Train Status")
    
    if not filtered_df.empty:
        # Prepare data for display
        display_df = filtered_df[['train_number', 'train_name', 'source', 'destination', 
                               'scheduled_departure', 'actual_departure', 'delay_minutes', 
                               'status', 'delay_reason']].copy()
        
        # Add color coding to status column
        def color_status(val):
            if val == 'On Time':
                return f'<span style="color:{GREEN_SUCCESS}; font-weight:bold">{val}</span>'
            elif val == 'Slight Delay':
                return f'<span style="color:{AMBER_ORANGE}; font-weight:bold">{val}</span>'
            else:
                return f'<span style="color:{ORANGE_ALERT}; font-weight:bold">{val}</span>'
        
        # Sort by delay minutes for better UX
        display_df = display_df.sort_values('delay_minutes', ascending=False)
        
        # Use pandas styling for better display
        styled_df = display_df.copy()
        styled_df['status_colored'] = styled_df['status'].apply(color_status)
        
        # Replace the original status column with colored version
        styled_df = styled_df.drop(columns=['status'])
        styled_df = styled_df.rename(columns={'status_colored': 'Status'})
        
        # Rename columns for better display
        styled_df = styled_df.rename(columns={
            'train_number': 'Train Number',
            'train_name': 'Train Name',
            'source': 'Source',
            'destination': 'Destination',
            'scheduled_departure': 'Scheduled',
            'actual_departure': 'Actual',
            'delay_minutes': 'Delay (min)',
            'delay_reason': 'Reason'
        })
        
        # Display the table with HTML styling
        st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("No data available with the current filters")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional train details charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Most Delayed Trains")
        
        if not filtered_df.empty:
            # Get top delayed trains
            top_delayed = filtered_df.sort_values('delay_minutes', ascending=False).head(10)
            top_delayed['train_label'] = top_delayed['train_number'] + ' - ' + top_delayed['train_name'].str.split().str[:2].str.join(' ')
            
            if not top_delayed.empty:
                fig = px.bar(
                    top_delayed,
                    y='train_label',
                    x='delay_minutes',
                    orientation='h',
                    color='delay_minutes',
                    color_continuous_scale=[[0, YELLOW], [1.0, ORANGE_ALERT]],
                    labels={'train_label': 'Train', 'delay_minutes': 'Delay (minutes)'}
                )
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    coloraxis_showscale=False,
                    height=350,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No delayed trains in the current selection")
        else:
            st.info("No data available with the current filters")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Highest Occupancy Trains")
        
        if not filtered_df.empty:
            # Calculate occupancy percentage
            occupancy_df = filtered_df.copy()
            occupancy_df['occupancy_pct'] = (occupancy_df['occupancy'] / occupancy_df['capacity'] * 100).round(1)
            top_occupancy = occupancy_df.sort_values('occupancy_pct', ascending=False).head(10)
            top_occupancy['train_label'] = top_occupancy['train_number'] + ' - ' + top_occupancy['train_name'].str.split().str[:2].str.join(' ')
            
            if not top_occupancy.empty:
                fig = px.bar(
                    top_occupancy,
                    y='train_label',
                    x='occupancy_pct',
                    orientation='h',
                    color='occupancy_pct',
                    color_continuous_scale=[[0, LIGHT_GREEN], [0.7, BRIGHT_BLUE], [1.0, AMBER_ORANGE]],
                    labels={'train_label': 'Train', 'occupancy_pct': 'Occupancy (%)'},
                    range_color=[50, 100]
                )
                
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    coloraxis_showscale=False,
                    height=350,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data to generate occupancy analysis")
        else:
            st.info("No data available with the current filters")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional information for selected train
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Train-Specific Details")
    
    if not filtered_df.empty:
        # Create a dropdown to select a specific train
        train_options = filtered_df['train_number'] + ' - ' + filtered_df['train_name']
        selected_train = st.selectbox("Select a train for detailed information", train_options)
        
        if selected_train:
            train_number = selected_train.split(' - ')[0]
            train_details = filtered_df[filtered_df['train_number'] == train_number].iloc[0]
            
            # Display train details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <h4>{train_details['train_name']}</h4>
                <p><strong>Train Number:</strong> {train_details['train_number']}</p>
                <p><strong>Route:</strong> {train_details['source']} to {train_details['destination']}</p>
                <p><strong>Zone:</strong> {train_details['zone']}</p>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <h4>Schedule</h4>
                <p><strong>Scheduled Departure:</strong> {train_details['scheduled_departure']}</p>
                <p><strong>Actual Departure:</strong> {train_details['actual_departure']}</p>
                <p><strong>Delay:</strong> {train_details['delay_minutes']} minutes</p>
                <p><strong>Status:</strong> {train_details['status']}</p>
                """, unsafe_allow_html=True)
                
            with col3:
                occupancy = int(train_details['occupancy'])
                capacity = int(train_details['capacity'])
                occupancy_pct = round((occupancy / capacity) * 100, 1)
                
                st.markdown(f"""
                <h4>Capacity</h4>
                <p><strong>Total Capacity:</strong> {capacity} passengers</p>
                <p><strong>Current Occupancy:</strong> {occupancy} passengers</p>
                <p><strong>Occupancy Rate:</strong> {occupancy_pct}%</p>
                """, unsafe_allow_html=True)
                
                # Create a progress bar for occupancy
                occupancy_color = GREEN_SUCCESS
                if occupancy_pct > 85:
                    occupancy_color = AMBER_ORANGE
                if occupancy_pct > 95:
                    occupancy_color = ORANGE_ALERT
                    
                st.markdown(f"""
                <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; width: 100%;">
                    <div style="background-color: {occupancy_color}; width: {min(occupancy_pct, 100)}%; height: 20px; border-radius: 10px;"></div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No data available with the current filters")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #555;">
    <p>IRCTC Real-Time Train Monitoring Dashboard | Data last updated: May 17, 2025</p>
    <p>This is a demonstration dashboard with simulated data.</p>
</div>
""", unsafe_allow_html=True)
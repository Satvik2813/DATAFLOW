import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="DATAFLOW",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Functions ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

@st.cache_resource
def load_resources():
    try:
        with open('model/cloud_cost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('model/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

# --- Custom CSS (Dark Theme, Glassmorphism, Background) ---
def set_bg_and_style():
    bg_b64 = get_base64_of_bin_file('assets/background.png')
    favicon_b64 = get_base64_of_bin_file('assets/favicon.png')
    
    css = f"""
    <style>
    /* Background Image - Enhanced clarity */
    .stApp {{
        background-image: linear-gradient(rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.9)), url("data:image/png;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Dark Theme Optimization & Typography */
    html, body, .stApp {{
        color: #E2E8F0;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }}
    
    /* Hide top header bar and footer */
    header {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Main Content Container (Modern Glassmorphism) */
    .block-container {{
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 32px;
        padding: 50px !important; /* Increased padding */
        margin-top: 3rem;
        margin-bottom: 3rem;
        max-width: 1100px !important; /* Larger main container */
        box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.6), 0 0 40px rgba(59, 130, 246, 0.1);
    }}
    
    /* Header Container styling with Favicon */
    .header-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: clamp(0.5rem, 3vw, 1.5rem);
        margin-bottom: 0.5rem;
        flex-wrap: nowrap;
        width: 100%;
    }}
    
    .header-logo {{
        width: clamp(40px, 10vw, 60px);
        height: clamp(40px, 10vw, 60px);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        flex-shrink: 0;
    }}
    
    h1 {{
        color: #FFFFFF;
        font-weight: 800; /* Bold */
        letter-spacing: 0.01em; /* Slightly increased */
        font-size: clamp(28px, 8vw, 52px) !important; /* Responsive font scaling */
        white-space: nowrap; /* Prevent word breaking */
        text-align: center;
        margin: 0;
        text-shadow: 0 4px 12px rgba(0,0,0,0.4);
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .subtitle {{
        color: rgba(148, 163, 184, 0.85); /* Slightly reduced opacity */
        text-align: center;
        font-size: 22px; /* 22px */
        margin-bottom: 3.5rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }}
    
    /* Input Fields Styling - Glassmorphism & Spacing */
    div[data-testid="stNumberInput"] > div {{
        margin-bottom: 22px !important; /* Spaced out rows */
    }}
    
    div[data-testid="stNumberInput"] input,
    .stNumberInput input,
    [data-baseweb="input"] input {{
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 16px !important; /* Padding 16px */
        font-size: 20px !important; /* 20px text */
        height: 60px !important; /* Exact height */
        width: 100% !important;
        transition: all 0.3s ease !important;
    }}
    
    div[data-testid="stNumberInput"] input:focus,
    .stNumberInput input:focus {{
        border-color: rgba(255, 255, 255, 0.3) !important;
        background: rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-1px);
    }}
    
    /* Plus/Minus Stepper Buttons - Minimal & Transparent */
    div[data-testid="stNumberInput"] button,
    [data-baseweb="input"] button {{
        width: 38px !important;
        height: 100% !important;
        background: rgba(255, 255, 255, 0.05) !important;
        color: transparent !important; /* Hide original +/- */
        position: relative;
    }}
    
    [data-baseweb="button"] svg {{
        display: none !important; /* Hide original SVG paths */
    }}
    
    /* Down Arrow (-) Replacement */
    div[data-testid="stNumberInput"] button:first-of-type::before {{
        content: "▼";
        color: rgba(255, 255, 255, 0.40) !important; /* Semi-transparent */
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 14px !important;
        transition: color 0.2s ease;
    }}
    div[data-testid="stNumberInput"] button:first-of-type:hover::before {{
        color: rgba(255, 255, 255, 0.6) !important; /* Opacity on Hover */
    }}
    
    /* Up Arrow (+) Replacement */
    div[data-testid="stNumberInput"] button:last-of-type::before {{
        content: "▲";
        color: rgba(255, 255, 255, 0.40) !important; /* Semi-transparent */
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 14px !important;
        transition: color 0.2s ease;
    }}
    div[data-testid="stNumberInput"] button:last-of-type:hover::before {{
        color: rgba(255, 255, 255, 0.6) !important; /* Opacity on Hover */
    }}
    
    /* Labels - Professional readability */
    div[data-testid="stWidgetLabel"] p,
    .stNumberInput label p,
    .stNumberInput label {{
        color: #E2E8F0 !important;
        font-weight: 600 !important; /* Semi-Bold */
        font-size: 20px !important; /* 20px exact */
        margin-bottom: 8px !important; /* Small spacing under label */
        letter-spacing: 0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    /* spacing between grid columns */
    div[data-testid="column"] {{
        padding: 0 20px; /* Column gap spacing */
    }}
    
    /* Predict Button Styling (Gradient with subtle glow) */
    .stButton > button {{
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; /* 12px Radius */
        height: 64px !important; /* 64px height */
        font-size: 20px !important; /* 20px text */
        font-weight: 600;
        letter-spacing: 0.02em;
        width: 100%;
        margin-top: 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
        text-transform: uppercase;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.5), 0 0 25px rgba(59, 130, 246, 0.3); /* Soft Glow */
        background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%);
        border: 1px solid rgba(255,255,255,0.2);
    }}
    
    .stButton > button:active {{
        transform: translateY(1px);
    }}
    
    /* Success Container Styling - Premium Dashboard Look */
    .success-container {{
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        margin-top: 3.5rem;
        margin-bottom: 3.5rem;
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        box-shadow: 0 20px 40px -10px rgba(16, 185, 129, 0.15), inset 0 1px 0 rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }}
    
    .success-container::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10B981, #34D399, #10B981);
    }}
    
    .success-title {{
        color: #6EE7B7;
        font-size: 1.15rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }}
    
    .success-value {{
        color: #FFFFFF;
        font-size: 4.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
        line-height: 1.1;
        letter-spacing: -0.02em;
    }}
    
    /* Metric Cards - Sleek UI */
    .metric-card {{
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 0.75rem;
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-4px);
    }}
    
    .metric-label {{ 
        color: #94A3B8; 
        font-size: 1.35rem; 
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .metric-val {{ 
        color: #F8FAFC; 
        font-weight: 800;
        font-size: 1.85rem;
        background: rgba(15, 23, 42, 0.5);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        max-width: 100%;
        word-break: break-word;
        overflow-wrap: break-word;
    }}
    
    @keyframes slideUpFade {{
        from {{ opacity: 0; transform: translateY(30px) scale(0.98); }}
        to {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return favicon_b64

# --- Main App ---
def main():
    favicon_b64 = set_bg_and_style()

    # Header with Favicon
    st.markdown(f"""
        <div class="header-container">
            <img src="data:image/png;base64,{favicon_b64}" class="header-logo" alt="DataFlow Logo">
            <h1>DATAFLOW</h1>
        </div>
        <div class='subtitle'>Professional Cloud Infrastructure Cost Predictor</div>
    """, unsafe_allow_html=True)

    # Input Section (Two Columns) with precise layout
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<h4 style='color: #E2E8F0; margin-bottom: 1.5rem; font-weight: 600; font-size: 24px;'>Core Metrics</h4>", unsafe_allow_html=True)
        cpu_usage = st.number_input("⚡ CPU Usage (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        network_traffic = st.number_input("🌐 Network Traffic (MB/s)", min_value=0.0, value=100.0, step=1.0)
        num_executed_instructions = st.number_input("⚙️ Executed Instructions", min_value=0.0, value=1000000.0, step=1000.0, format="%.0f")

    with col2:
        st.markdown("<h4 style='color: #E2E8F0; margin-bottom: 1.5rem; font-weight: 600; font-size: 24px;'>System Metrics</h4>", unsafe_allow_html=True)
        memory_usage = st.number_input("💾 Memory Usage (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        execution_time = st.number_input("⏱️ Execution Time (s)", min_value=0.0, value=120.0, step=1.0)
        energy_efficiency = st.number_input("🌱 Energy Efficiency", min_value=0.0, value=1.0, step=0.01)

    # Predict Button
    predict_clicked = st.button("Predict Cloud Cost", use_container_width=True)

    if predict_clicked:
        model, scaler = load_resources()
        
        if model is not None and scaler is not None:
            # Prepare inputs - Order must match model training exactly
            input_features = np.array([[
                cpu_usage,
                memory_usage,
                network_traffic,
                execution_time,
                num_executed_instructions,
                energy_efficiency
            ]])
            
            feature_names = [
                'cpu_usage', 'memory_usage', 'network_traffic',
                'execution_time', 'num_executed_instructions', 'energy_efficiency'
            ]
            
            input_df = pd.DataFrame(input_features, columns=feature_names)

            try:
                # Modern spinner text
                with st.spinner("Analyzing infrastructure telemetry & projecting operation cost..."):
                    import time
                    time.sleep(0.8) # Slightly longer sleep for better UX feel
                    
                    scaled_input = scaler.transform(input_df)
                    prediction = model.predict(scaled_input)[0]
                
                # Format prediction
                formatted_cost = f"${max(0, prediction):,.2f}"
                
                # Display Result prominently
                st.markdown(
                    f"""
                    <div class="success-container">
                        <div class="success-title">
                            <span style="font-size: 1.5rem;">📊</span> Projected Operation Cost
                        </div>
                        <div class="success-value">{formatted_cost}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                st.divider()
                
                # Visualization of input metrics
                st.markdown("<h3 style='color: #F8FAFC; margin-top: 1rem; margin-bottom: 2rem; font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em;'>Infrastructure Telemetry Summary</h3>", unsafe_allow_html=True)
                
                chart_data = pd.DataFrame(
                    {
                        "Resource": ["CPU Utilization", "Memory Utilization"],
                        "Percentage (%)": [cpu_usage, memory_usage]
                    }
                )
                
                col_chart1, col_chart2 = st.columns([1.2, 1], gap="large")
                
                with col_chart1:
                    # Enhanced chart design
                    st.markdown("<p style='color: #94A3B8; font-weight: 500; margin-bottom: 1rem;'>Resource Utilization Overview</p>", unsafe_allow_html=True)
                    st.bar_chart(chart_data.set_index("Resource"), color="#3B82F6", height=280)
                
                with col_chart2:
                    st.markdown("<p style='color: #94A3B8; font-weight: 500; margin-bottom: 1rem;'>Operational Parameters</p>", unsafe_allow_html=True)
                    # Metrics for other values with improved UI
                    st.markdown(f"""
                        <div class="metric-card">
                            <span class="metric-label">🌐 Network Traffic</span>
                            <span class="metric-val">{network_traffic:,.0f} MB/s</span>
                        </div>
                        <div class="metric-card">
                            <span class="metric-label">⏱️ Execution Time</span>
                            <span class="metric-val">{execution_time:,.0f} s</span>
                        </div>
                        <div class="metric-card">
                            <span class="metric-label">⚙️ Instructions</span>
                            <span class="metric-val">{num_executed_instructions:,.0f}</span>
                        </div>
                        <div class="metric-card">
                            <span class="metric-label">🌱 Efficiency Score</span>
                            <span class="metric-val">{energy_efficiency:.2f}</span>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                # Fallback to array if DataFrame fails
                try:
                     with st.spinner("Attempting secondary projection method..."):
                         scaled_input_fallback = scaler.transform(input_features)
                         prediction = model.predict(scaled_input_fallback)[0]
                         formatted_cost = f"${max(0, prediction):,.2f}"
                         st.success(f"Projected Cost: {formatted_cost}")
                except Exception as e2:
                     st.error(f"Critical analysis failure: {e2}")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import joblib
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xgboost as xgb
from prophet import Prophet
from prophet.serialize import model_from_json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Air Quality Prediction & Health Impact",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.fc_out(attention_output)

        return output, attention_weights

class ImprovedBiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, output_dim=3, dropout=0.3):
        super(ImprovedBiLSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1,
                           batch_first=True, bidirectional=True)

        self.attention = MultiHeadAttention(hidden_dim * 2, num_heads)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out)
        output = attn_out[:, -1, :]
        output = torch.relu(self.fc1(output))
        output = self.dropout1(output)
        output = self.fc2(output)
        return output

@st.cache_resource
def load_models():
    MODEL_DIR = 'Saved_Model/'

    with open(MODEL_DIR + 'config.json', 'r') as f:
        config = json.load(f)

    feature_cols = joblib.load(MODEL_DIR + 'feature_cols.pkl')
    scaler_X = joblib.load(MODEL_DIR + 'scaler_X.pkl')
    scaler_y = joblib.load(MODEL_DIR + 'scaler_y.pkl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(MODEL_DIR + 'bilstm_attention.pt',
                           map_location=device, weights_only=False)

    lstm_model = ImprovedBiLSTMModel(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_heads=checkpoint['num_heads'],
        output_dim=checkpoint['output_dim'],
        dropout=checkpoint['dropout']
    )
    lstm_model.load_state_dict(checkpoint['state_dict'])
    lstm_model.to(device)
    lstm_model.eval()

    xgb_models = {}
    for pollutant in ['pm25', 'no2', 'o3']:
        model = xgb.Booster()
        model.load_model(MODEL_DIR + f'xgb_{pollutant}.json')
        xgb_models[pollutant] = model

    prophet_models = {}
    for pollutant in ['pm25', 'no2', 'o3']:
        with open(MODEL_DIR + f'prophet_{pollutant}.json', 'r') as f:
            prophet_models[pollutant] = model_from_json(json.load(f))

    ensemble_weights = np.load(MODEL_DIR + 'ensemble_weights.npy')

    return {
        'config': config,
        'feature_cols': feature_cols,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'lstm_model': lstm_model,
        'xgb_models': xgb_models,
        'prophet_models': prophet_models,
        'ensemble_weights': ensemble_weights,
        'device': device
    }

@st.cache_data
def load_historical_data():
    df = pd.read_csv('uk_air_quality_data_complete.csv')
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    return df

def create_features(df_current, config):
    LAG_HOURS = config['LAG_HOURS']
    ROLLING_WINDOWS = config['ROLLING_WINDOWS']

    features = {}

    for pollutant in ['pm25', 'no2', 'o3']:
        if pollutant in df_current.columns:
            for lag in LAG_HOURS:
                features[f'{pollutant}_lag_{lag}h'] = df_current[pollutant].shift(lag).iloc[-1]

            for window in ROLLING_WINDOWS:
                features[f'{pollutant}_rolling_mean_{window}h'] = df_current[pollutant].rolling(window).mean().iloc[-1]
                features[f'{pollutant}_rolling_std_{window}h'] = df_current[pollutant].rolling(window).std().iloc[-1]

            features[f'{pollutant}_change_24h'] = df_current[pollutant].diff(24).iloc[-1]

    current_time = pd.Timestamp.now()
    features['hour'] = current_time.hour
    features['day_of_week'] = current_time.dayofweek
    features['month'] = current_time.month
    features['hour_sin'] = np.sin(2 * np.pi * current_time.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * current_time.hour / 24)
    features['day_sin'] = np.sin(2 * np.pi * current_time.dayofweek / 7)
    features['day_cos'] = np.cos(2 * np.pi * current_time.dayofweek / 7)
    features['month_sin'] = np.sin(2 * np.pi * current_time.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * current_time.month / 12)

    return pd.DataFrame([features])

def make_prediction(models, current_data):
    feature_cols = models['feature_cols']
    scaler_X = models['scaler_X']
    scaler_y = models['scaler_y']
    config = models['config']

    features_df = create_features(current_data, config)

    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0

    features_df = features_df[feature_cols]
    X_scaled = scaler_X.transform(features_df)

    device = models['device']
    lstm_model = models['lstm_model']
    X_seq = np.tile(X_scaled, (config['seq_length'], 1)).reshape(1, config['seq_length'], -1)
    X_tensor = torch.FloatTensor(X_seq).to(device)

    with torch.no_grad():
        lstm_pred_scaled = lstm_model(X_tensor).cpu().numpy()
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)[0]

    xgb_preds = []
    for pollutant in ['pm25', 'no2', 'o3']:
        dmatrix = xgb.DMatrix(X_scaled)
        pred = models['xgb_models'][pollutant].predict(dmatrix)[0]
        xgb_preds.append(pred)
    xgb_pred = np.array(xgb_preds)

    prophet_preds = []
    future_date = pd.Timestamp.now() + timedelta(hours=24)
    future_df = pd.DataFrame({'ds': [future_date]})

    for pollutant in ['pm25', 'no2', 'o3']:
        forecast = models['prophet_models'][pollutant].predict(future_df)
        prophet_preds.append(forecast['yhat'].values[0])
    prophet_pred = np.array(prophet_preds)

    weights = models['ensemble_weights']
    ensemble_pred = (weights[0] * lstm_pred +
                    weights[1] * xgb_pred +
                    weights[2] * prophet_pred)

    return {
        'lstm': {'pm25': lstm_pred[0], 'no2': lstm_pred[1], 'o3': lstm_pred[2]},
        'xgboost': {'pm25': xgb_pred[0], 'no2': xgb_pred[1], 'o3': xgb_pred[2]},
        'prophet': {'pm25': prophet_pred[0], 'no2': prophet_pred[1], 'o3': prophet_pred[2]},
        'ensemble': {'pm25': ensemble_pred[0], 'no2': ensemble_pred[1], 'o3': ensemble_pred[2]}
    }

def calculate_aqi(pm25, no2, o3):
    def pm25_to_aqi(pm25):
        if pm25 <= 12.0:
            return (50 / 12.0) * pm25
        elif pm25 <= 35.4:
            return 50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1)
        elif pm25 <= 55.4:
            return 100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5)
        elif pm25 <= 150.4:
            return 150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5)
        elif pm25 <= 250.4:
            return 200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5)
        else:
            return 300 + ((500 - 300) / (500.4 - 250.5)) * (pm25 - 250.5)

    def no2_to_aqi(no2_ppb):
        if no2_ppb <= 53:
            return (50 / 53) * no2_ppb
        elif no2_ppb <= 100:
            return 50 + ((100 - 50) / (100 - 54)) * (no2_ppb - 54)
        elif no2_ppb <= 360:
            return 100 + ((150 - 100) / (360 - 101)) * (no2_ppb - 101)
        elif no2_ppb <= 649:
            return 150 + ((200 - 150) / (649 - 361)) * (no2_ppb - 361)
        elif no2_ppb <= 1249:
            return 200 + ((300 - 200) / (1249 - 650)) * (no2_ppb - 650)
        else:
            return 300 + ((500 - 300) / (2049 - 1250)) * (no2_ppb - 1250)

    def o3_to_aqi(o3_ppb):
        if o3_ppb <= 54:
            return (50 / 54) * o3_ppb
        elif o3_ppb <= 70:
            return 50 + ((100 - 50) / (70 - 55)) * (o3_ppb - 55)
        elif o3_ppb <= 85:
            return 100 + ((150 - 100) / (85 - 71)) * (o3_ppb - 71)
        elif o3_ppb <= 105:
            return 150 + ((200 - 150) / (105 - 86)) * (o3_ppb - 86)
        elif o3_ppb <= 200:
            return 200 + ((300 - 200) / (200 - 106)) * (o3_ppb - 106)
        else:
            return 300

    aqi_pm25 = pm25_to_aqi(pm25)
    aqi_no2 = no2_to_aqi(no2)
    aqi_o3 = o3_to_aqi(o3)

    overall_aqi = max(aqi_pm25, aqi_no2, aqi_o3)

    if overall_aqi <= 50:
        category = "Good"
        color = "#00E400"
    elif overall_aqi <= 100:
        category = "Moderate"
        color = "#FFFF00"
    elif overall_aqi <= 150:
        category = "Unhealthy for Sensitive Groups"
        color = "#FF7E00"
    elif overall_aqi <= 200:
        category = "Unhealthy"
        color = "#FF0000"
    elif overall_aqi <= 300:
        category = "Very Unhealthy"
        color = "#8F3F97"
    else:
        category = "Hazardous"
        color = "#7E0023"

    return overall_aqi, category, color

def calculate_health_impact(pm25, no2, o3):
    RR_PM25_RESPIRATORY = 1.052
    RR_NO2_RESPIRATORY = 1.027
    RR_O3_RESPIRATORY = 1.010
    RR_PM25_CARDIOVASCULAR = 1.011

    BASELINE_RESPIRATORY = 800
    BASELINE_CARDIOVASCULAR = 500
    POPULATION = 1000000

    rr_pm25_resp = RR_PM25_RESPIRATORY ** (pm25 / 10)
    rr_no2_resp = RR_NO2_RESPIRATORY ** (no2 / 10)
    rr_o3_resp = RR_O3_RESPIRATORY ** (o3 / 10)
    rr_pm25_cardio = RR_PM25_CARDIOVASCULAR ** (pm25 / 10)

    af_resp = (rr_pm25_resp * rr_no2_resp * rr_o3_resp - 1) / (rr_pm25_resp * rr_no2_resp * rr_o3_resp)
    af_cardio = (rr_pm25_cardio - 1) / rr_pm25_cardio

    respiratory_admissions = (BASELINE_RESPIRATORY * af_resp * POPULATION) / 100000
    cardiovascular_admissions = (BASELINE_CARDIOVASCULAR * af_cardio * POPULATION) / 100000

    return {
        'respiratory_admissions': respiratory_admissions,
        'cardiovascular_admissions': cardiovascular_admissions,
        'total_admissions': respiratory_admissions + cardiovascular_admissions
    }

def send_email_alert(recipient_email, aqi, category, predictions, smtp_config):
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['sender_email']
        msg['To'] = recipient_email
        msg['Subject'] = f"AIR QUALITY ALERT: {category} (AQI: {aqi:.0f})"

        body = f"""
        AIR QUALITY ALERT

        Current AQI Status: {category}
        AQI Value: {aqi:.1f}

        Predicted Pollutant Levels (24-hour forecast):
        - PM2.5: {predictions['pm25']:.2f} µg/m³
        - NO2: {predictions['no2']:.2f} ppb
        - O3: {predictions['o3']:.2f} ppb

        Recommendations:
        """

        if aqi <= 50:
            body += "\n- Air quality is good. Enjoy outdoor activities!"
        elif aqi <= 100:
            body += "\n- Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion."
        elif aqi <= 150:
            body += "\n- Sensitive groups should reduce prolonged outdoor exertion."
        elif aqi <= 200:
            body += "\n- Everyone should reduce prolonged outdoor exertion."
        elif aqi <= 300:
            body += "\n- Everyone should avoid prolonged outdoor exertion. Sensitive groups should remain indoors."
        else:
            body += "\n- HEALTH WARNING: Everyone should avoid all outdoor exertion. Stay indoors!"

        body += f"\n\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        body += "\n\nThis is an automated alert from the Air Quality Prediction System."

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
        server.starttls()
        server.login(smtp_config['sender_email'], smtp_config['sender_password'])
        server.send_message(msg)
        server.quit()

        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def create_aqi_gauge(aqi, category):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"AQI: {category}", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#00E400'},
                {'range': [50, 100], 'color': '#FFFF00'},
                {'range': [100, 150], 'color': '#FF7E00'},
                {'range': [150, 200], 'color': '#FF0000'},
                {'range': [200, 300], 'color': '#8F3F97'},
                {'range': [300, 500], 'color': '#7E0023'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': aqi
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_pollutant_comparison(predictions):
    pollutants = ['PM2.5', 'NO2', 'O3']
    models = ['BiLSTM', 'XGBoost', 'Prophet', 'Ensemble']

    data = []
    for model_name, model_key in zip(models, ['lstm', 'xgboost', 'prophet', 'ensemble']):
        data.append([
            predictions[model_key]['pm25'],
            predictions[model_key]['no2'],
            predictions[model_key]['o3']
        ])

    fig = go.Figure()

    for i, model in enumerate(models):
        fig.add_trace(go.Bar(
            name=model,
            x=pollutants,
            y=data[i],
            text=[f'{val:.2f}' for val in data[i]],
            textposition='auto',
        ))

    fig.update_layout(
        title='Predicted Pollutant Levels by Model',
        xaxis_title='Pollutant',
        yaxis_title='Concentration',
        barmode='group',
        height=400
    )

    return fig

def create_historical_trends(df):
    df_agg = df.groupby(['datetime_utc', 'parameter'])['value'].mean().reset_index()
    df_pivot = df_agg.pivot(index='datetime_utc', columns='parameter', values='value')

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('PM2.5 Trend', 'NO2 Trend', 'O3 Trend'),
        vertical_spacing=0.1
    )

    if 'pm25' in df_pivot.columns:
        fig.add_trace(
            go.Scatter(x=df_pivot.index, y=df_pivot['pm25'],
                      name='PM2.5', line=dict(color='red')),
            row=1, col=1
        )

    if 'no2' in df_pivot.columns:
        fig.add_trace(
            go.Scatter(x=df_pivot.index, y=df_pivot['no2'],
                      name='NO2', line=dict(color='blue')),
            row=2, col=1
        )

    if 'o3' in df_pivot.columns:
        fig.add_trace(
            go.Scatter(x=df_pivot.index, y=df_pivot['o3'],
                      name='O3', line=dict(color='green')),
            row=3, col=1
        )

    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="µg/m³", row=1, col=1)
    fig.update_yaxes(title_text="ppb", row=2, col=1)
    fig.update_yaxes(title_text="ppb", row=3, col=1)

    return fig

st.title("Real-Time Air Quality Prediction & Health Impact Analysis")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")

    st.subheader("Alert System")
    enable_alerts = st.checkbox("Enable Email Alerts", value=False)

    if enable_alerts:
        recipient_email = st.text_input("Recipient Email", value="")
        aqi_threshold = st.slider("AQI Alert Threshold", 0, 500, 100)

        st.subheader("SMTP Configuration")
        smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
        smtp_port = st.number_input("SMTP Port", value=587)
        sender_email = st.text_input("Sender Email", value="")
        sender_password = st.text_input("Sender Password", type="password", value="")

    st.markdown("---")
    st.subheader("About")
    st.info("""
    This dashboard provides:
    - 24-hour air quality predictions
    - Real-time AQI monitoring
    - Health impact assessment
    - Automated alert system

    Models: BiLSTM+Attention, XGBoost, Prophet
    Optimization: Genetic Algorithm Ensemble
    """)

try:
    models = load_models()
    historical_data = load_historical_data()

    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Historical Analysis", "Health Impact", "Model Performance"])

    with tab1:
        st.header("24-Hour Air Quality Forecast")

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Manual Input (Optional)")
            use_manual = st.checkbox("Use Manual Input Values")

            if use_manual:
                manual_pm25 = st.number_input("Current PM2.5 (µg/m³)", value=15.0, min_value=0.0)
                manual_no2 = st.number_input("Current NO2 (ppb)", value=20.0, min_value=0.0)
                manual_o3 = st.number_input("Current O3 (ppb)", value=30.0, min_value=0.0)

        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Running prediction models..."):
                if use_manual:
                    current_data = pd.DataFrame({
                        'pm25': [manual_pm25] * 100,
                        'no2': [manual_no2] * 100,
                        'o3': [manual_o3] * 100
                    })
                else:
                    df_recent = historical_data.tail(1000)
                    df_pivot = df_recent.pivot_table(
                        values='value',
                        index='datetime_utc',
                        columns='parameter',
                        aggfunc='mean'
                    ).ffill()

                    available_cols = [col for col in ['pm25', 'no2', 'o3'] if col in df_pivot.columns]
                    if len(available_cols) < 3:
                        missing = set(['pm25', 'no2', 'o3']) - set(available_cols)
                        st.warning(f"Missing parameters in data: {missing}. Using default values.")
                        for col in ['pm25', 'no2', 'o3']:
                            if col not in df_pivot.columns:
                                df_pivot[col] = 0

                    current_data = df_pivot[['pm25', 'no2', 'o3']].fillna(0)

                predictions = make_prediction(models, current_data)

                st.session_state['predictions'] = predictions

        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            ensemble_pred = predictions['ensemble']

            aqi, category, color = calculate_aqi(
                ensemble_pred['pm25'],
                ensemble_pred['no2'],
                ensemble_pred['o3']
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("AQI", f"{aqi:.0f}", delta=category)
            with col2:
                st.metric("PM2.5", f"{ensemble_pred['pm25']:.2f} µg/m³")
            with col3:
                st.metric("NO2", f"{ensemble_pred['no2']:.2f} ppb")
            with col4:
                st.metric("O3", f"{ensemble_pred['o3']:.2f} ppb")

            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(create_aqi_gauge(aqi, category), use_container_width=True)

            with col2:
                st.plotly_chart(create_pollutant_comparison(predictions), use_container_width=True)

            if enable_alerts and aqi > aqi_threshold:
                if recipient_email and sender_email and sender_password:
                    smtp_config = {
                        'smtp_server': smtp_server,
                        'smtp_port': smtp_port,
                        'sender_email': sender_email,
                        'sender_password': sender_password
                    }

                    if send_email_alert(recipient_email, aqi, category, ensemble_pred, smtp_config):
                        st.success(f"Alert email sent to {recipient_email}")
                    else:
                        st.warning("Failed to send alert email")
                else:
                    st.warning("Please configure SMTP settings to enable alerts")

    with tab2:
        st.header("Historical Data Analysis")

        st.plotly_chart(create_historical_trends(historical_data), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Summary")
            df_summary = historical_data.groupby('parameter')['value'].describe()
            st.dataframe(df_summary)

        with col2:
            st.subheader("Recent Measurements")
            st.dataframe(historical_data.tail(20))

    with tab3:
        st.header("Health Impact Assessment")

        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            ensemble_pred = predictions['ensemble']

            health_impact = calculate_health_impact(
                ensemble_pred['pm25'],
                ensemble_pred['no2'],
                ensemble_pred['o3']
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Respiratory Admissions",
                    f"{health_impact['respiratory_admissions']:.1f}",
                    help="Estimated daily hospital admissions per 100,000 population"
                )

            with col2:
                st.metric(
                    "Cardiovascular Admissions",
                    f"{health_impact['cardiovascular_admissions']:.1f}",
                    help="Estimated daily hospital admissions per 100,000 population"
                )

            with col3:
                st.metric(
                    "Total Admissions",
                    f"{health_impact['total_admissions']:.1f}",
                    help="Total estimated daily hospital admissions per 100,000 population"
                )

            st.markdown("---")

            fig = go.Figure(data=[
                go.Bar(name='Respiratory', x=['Admissions'], y=[health_impact['respiratory_admissions']]),
                go.Bar(name='Cardiovascular', x=['Admissions'], y=[health_impact['cardiovascular_admissions']])
            ])

            fig.update_layout(
                title='Estimated Hospital Admissions per 100,000 Population',
                barmode='group',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Health Recommendations")

            aqi, category, color = calculate_aqi(
                ensemble_pred['pm25'],
                ensemble_pred['no2'],
                ensemble_pred['o3']
            )

            if aqi <= 50:
                st.success("Air quality is good. Ideal conditions for outdoor activities.")
            elif aqi <= 100:
                st.info("Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.")
            elif aqi <= 150:
                st.warning("Sensitive groups (children, elderly, respiratory conditions) should reduce prolonged outdoor exertion.")
            elif aqi <= 200:
                st.warning("Everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid outdoor activities.")
            elif aqi <= 300:
                st.error("Health alert: Everyone should avoid prolonged outdoor exertion. Sensitive groups should remain indoors.")
            else:
                st.error("Health warning: Everyone should avoid all outdoor exertion. Emergency conditions.")
        else:
            st.info("Generate a prediction first to view health impact assessment.")

    with tab4:
        st.header("Model Performance & Ensemble Weights")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ensemble Weights")
            weights = models['ensemble_weights']

            fig = go.Figure(data=[
                go.Bar(x=['BiLSTM', 'XGBoost', 'Prophet'],
                      y=weights,
                      text=[f'{w:.3f}' for w in weights],
                      textposition='auto')
            ])

            fig.update_layout(
                title='Optimized Ensemble Weights (Genetic Algorithm)',
                yaxis_title='Weight',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Model Configuration")
            config_data = {
                'Parameter': ['Sequence Length', 'Lag Hours', 'Rolling Windows', 'Total Features'],
                'Value': [
                    models['config']['seq_length'],
                    ', '.join(map(str, models['config']['LAG_HOURS'])),
                    ', '.join(map(str, models['config']['ROLLING_WINDOWS'])),
                    len(models['feature_cols'])
                ]
            }
            st.table(pd.DataFrame(config_data))

        st.subheader("Feature Importance")
        st.info(f"Total Features Used: {len(models['feature_cols'])}")

        feature_display = pd.DataFrame({
            'Feature Name': models['feature_cols'][:15]
        })
        st.dataframe(feature_display, use_container_width=True)

except Exception as e:
    st.error(f"Error loading models or data: {str(e)}")
    st.info("Please ensure all model files are in the 'Saved_Model/' directory and that 'uk_air_quality_data_complete.csv' is present in the project root.")

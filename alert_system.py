import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import io

class AirQualityAlertSystem:
    def __init__(self, config_file='alert_config_example.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        self.smtp_settings = self.config['smtp_settings']
        self.thresholds = self.config['alert_thresholds']
        self.notification_settings = self.config['notification_settings']

    def get_aqi_category(self, aqi):
        if aqi <= self.thresholds['good']:
            return "Good", "#00E400"
        elif aqi <= self.thresholds['moderate']:
            return "Moderate", "#FFFF00"
        elif aqi <= self.thresholds['unhealthy_sensitive']:
            return "Unhealthy for Sensitive Groups", "#FF7E00"
        elif aqi <= self.thresholds['unhealthy']:
            return "Unhealthy", "#FF0000"
        elif aqi <= self.thresholds['very_unhealthy']:
            return "Very Unhealthy", "#8F3F97"
        else:
            return "Hazardous", "#7E0023"

    def get_health_recommendations(self, aqi):
        category, _ = self.get_aqi_category(aqi)

        recommendations = {
            "Good": [
                "Air quality is good. Enjoy outdoor activities!",
                "Ideal conditions for exercise and outdoor work.",
                "No health precautions needed."
            ],
            "Moderate": [
                "Air quality is acceptable for most people.",
                "Unusually sensitive individuals should consider limiting prolonged outdoor exertion.",
                "General population can engage in normal outdoor activities."
            ],
            "Unhealthy for Sensitive Groups": [
                "Sensitive groups (children, elderly, respiratory conditions) should reduce prolonged outdoor exertion.",
                "General population can continue normal outdoor activities.",
                "Watch for symptoms like coughing or shortness of breath."
            ],
            "Unhealthy": [
                "Everyone should reduce prolonged outdoor exertion.",
                "Sensitive groups should avoid outdoor activities.",
                "Consider moving activities indoors.",
                "Close windows to prevent outdoor air from entering."
            ],
            "Very Unhealthy": [
                "Health alert: Everyone should avoid prolonged outdoor exertion.",
                "Sensitive groups should remain indoors with air filtration.",
                "Postpone outdoor activities if possible.",
                "Wear N95 masks if you must go outside."
            ],
            "Hazardous": [
                "EMERGENCY: Everyone should avoid all outdoor exertion.",
                "Stay indoors with windows and doors closed.",
                "Use air purifiers if available.",
                "Seek medical attention if experiencing health effects.",
                "Consider relocating to areas with better air quality if possible."
            ]
        }

        return recommendations.get(category, ["Monitor air quality closely."])

    def create_alert_visualization(self, predictions, health_impact):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        pollutants = ['pm25', 'no2', 'o3']
        pollutant_names = ['PM2.5', 'NO2', 'O3']
        values = [predictions[p] for p in pollutants]

        axes[0, 0].bar(pollutant_names, values, color=['red', 'blue', 'green'])
        axes[0, 0].set_title('Predicted Pollutant Levels', fontweight='bold')
        axes[0, 0].set_ylabel('Concentration')
        axes[0, 0].grid(axis='y', alpha=0.3)

        aqi_ranges = [50, 100, 150, 200, 300, 500]
        aqi_labels = ['Good', 'Moderate', 'Unhealthy\nSensitive', 'Unhealthy', 'Very\nUnhealthy', 'Hazardous']
        aqi_colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023']

        pm25_aqi = self.calculate_aqi(predictions['pm25'], predictions['no2'], predictions['o3'])
        current_category_idx = sum(pm25_aqi > threshold for threshold in aqi_ranges)

        axes[0, 1].barh(aqi_labels, aqi_ranges, color=aqi_colors, alpha=0.7)
        axes[0, 1].axvline(pm25_aqi, color='red', linewidth=3, linestyle='--', label=f'Current AQI: {pm25_aqi:.0f}')
        axes[0, 1].set_title('AQI Level', fontweight='bold')
        axes[0, 1].set_xlabel('AQI Value')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='x', alpha=0.3)

        admissions = ['Respiratory', 'Cardiovascular']
        admission_values = [
            health_impact['respiratory_admissions'],
            health_impact['cardiovascular_admissions']
        ]

        axes[1, 0].bar(admissions, admission_values, color=['orange', 'purple'])
        axes[1, 0].set_title('Estimated Hospital Admissions\n(per 100,000 population)', fontweight='bold')
        axes[1, 0].set_ylabel('Daily Admissions')
        axes[1, 0].grid(axis='y', alpha=0.3)

        categories = list(self.thresholds.keys())
        threshold_values = list(self.thresholds.values())

        axes[1, 1].plot(threshold_values, marker='o', linewidth=2, markersize=8, color='darkblue')
        axes[1, 1].fill_between(range(len(threshold_values)), threshold_values, alpha=0.3)
        axes[1, 1].set_xticks(range(len(categories)))
        axes[1, 1].set_xticklabels(categories, rotation=45, ha='right')
        axes[1, 1].set_title('AQI Threshold Scale', fontweight='bold')
        axes[1, 1].set_ylabel('AQI Value')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    def calculate_aqi(self, pm25, no2, o3):
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

        return max(pm25_to_aqi(pm25), no2_to_aqi(no2), o3_to_aqi(o3))

    def send_alert(self, predictions, health_impact, include_visualization=True):
        if not self.notification_settings['enabled']:
            return False, "Notifications are disabled"

        aqi = self.calculate_aqi(predictions['pm25'], predictions['no2'], predictions['o3'])

        if aqi < self.notification_settings['alert_on_threshold']:
            return False, f"AQI ({aqi:.0f}) below threshold ({self.notification_settings['alert_on_threshold']})"

        category, color = self.get_aqi_category(aqi)
        recommendations = self.get_health_recommendations(aqi)

        for recipient in self.notification_settings['recipient_emails']:
            try:
                msg = MIMEMultipart()
                msg['From'] = self.smtp_settings['sender_email']
                msg['To'] = recipient
                msg['Subject'] = f"AIR QUALITY ALERT: {category} - AQI {aqi:.0f}"

                body = f"""
AIR QUALITY ALERT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
CURRENT STATUS
================================================================================

Air Quality Index (AQI): {aqi:.1f}
Category: {category}

--------------------------------------------------------------------------------
PREDICTED POLLUTANT LEVELS (24-hour forecast)
--------------------------------------------------------------------------------

PM2.5: {predictions['pm25']:.2f} µg/m³
NO2:   {predictions['no2']:.2f} ppb
O3:    {predictions['o3']:.2f} ppb

--------------------------------------------------------------------------------
HEALTH IMPACT ASSESSMENT
--------------------------------------------------------------------------------

Estimated Hospital Admissions (per 100,000 population):
  - Respiratory:      {health_impact['respiratory_admissions']:.1f} admissions/day
  - Cardiovascular:   {health_impact['cardiovascular_admissions']:.1f} admissions/day
  - Total:            {health_impact['total_admissions']:.1f} admissions/day

--------------------------------------------------------------------------------
HEALTH RECOMMENDATIONS
--------------------------------------------------------------------------------

"""
                for i, rec in enumerate(recommendations, 1):
                    body += f"{i}. {rec}\n"

                body += f"""

--------------------------------------------------------------------------------
POLLUTION BREAKDOWN
--------------------------------------------------------------------------------

PM2.5 (Fine Particulate Matter):
  - Current Level: {predictions['pm25']:.2f} µg/m³
  - Primary Sources: Vehicle emissions, industrial processes, wood burning
  - Health Effects: Respiratory and cardiovascular problems

NO2 (Nitrogen Dioxide):
  - Current Level: {predictions['no2']:.2f} ppb
  - Primary Sources: Traffic emissions, power plants
  - Health Effects: Respiratory irritation, reduced lung function

O3 (Ozone):
  - Current Level: {predictions['o3']:.2f} ppb
  - Formation: Sunlight + NOx + VOCs
  - Health Effects: Breathing difficulties, aggravated asthma

================================================================================

This is an automated alert from the Air Quality Prediction System.
Data Source: OpenAQ API
Models: BiLSTM+Attention, XGBoost, Prophet (Ensemble)

To unsubscribe or modify alert settings, please update your configuration.

================================================================================
"""

                msg.attach(MIMEText(body, 'plain'))

                if include_visualization:
                    try:
                        img_buffer = self.create_alert_visualization(predictions, health_impact)

                        image = MIMEBase('application', 'octet-stream')
                        image.set_payload(img_buffer.read())
                        encoders.encode_base64(image)
                        image.add_header('Content-Disposition',
                                       f'attachment; filename=air_quality_alert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                        msg.attach(image)
                    except Exception as e:
                        print(f"Could not attach visualization: {e}")

                server = smtplib.SMTP(self.smtp_settings['smtp_server'],
                                     self.smtp_settings['smtp_port'])

                if self.smtp_settings.get('use_tls', True):
                    server.starttls()

                server.login(self.smtp_settings['sender_email'],
                           self.smtp_settings['sender_password'])

                server.send_message(msg)
                server.quit()

            except Exception as e:
                return False, f"Failed to send email to {recipient}: {str(e)}"

        return True, f"Alert sent to {len(self.notification_settings['recipient_emails'])} recipients"

    def test_connection(self):
        try:
            server = smtplib.SMTP(self.smtp_settings['smtp_server'],
                                 self.smtp_settings['smtp_port'])

            if self.smtp_settings.get('use_tls', True):
                server.starttls()

            server.login(self.smtp_settings['sender_email'],
                        self.smtp_settings['sender_password'])

            server.quit()

            return True, "SMTP connection successful"

        except Exception as e:
            return False, f"SMTP connection failed: {str(e)}"

if __name__ == "__main__":
    print("Air Quality Alert System - Test Module")
    print("=" * 60)

    try:
        alert_system = AirQualityAlertSystem()

        print("\nTesting SMTP connection...")
        success, message = alert_system.test_connection()

        if success:
            print(f"SUCCESS: {message}")

            test_predictions = {
                'pm25': 45.5,
                'no2': 38.2,
                'o3': 55.7
            }

            test_health_impact = {
                'respiratory_admissions': 12.5,
                'cardiovascular_admissions': 8.3,
                'total_admissions': 20.8
            }

            print("\nSending test alert...")
            success, message = alert_system.send_alert(test_predictions, test_health_impact)

            if success:
                print(f"SUCCESS: {message}")
            else:
                print(f"INFO: {message}")

        else:
            print(f"ERROR: {message}")
            print("\nPlease check your SMTP configuration in alert_config_example.json")

    except FileNotFoundError:
        print("ERROR: alert_config_example.json not found")
        print("Please create this file with your SMTP configuration")

    except Exception as e:
        print(f"ERROR: {str(e)}")

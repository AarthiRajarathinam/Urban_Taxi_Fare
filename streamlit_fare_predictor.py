"""
Taxi Fare Prediction Streamlit Application
==========================================
A user-friendly interface for predicting taxi fares using the trained Polynomial Regression model.

Features:
- Interactive input forms for trip details
- Real-time fare predictions
- Confidence interval display
- Model performance metrics
- Prediction history
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🚕 Taxi Fare Prediction",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .prediction-result {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin: 20px 0;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .info-box {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL CLASS DEFINITION (Required for unpickling)
# ============================================================================

class ManualPolynomialRegression:
    """Polynomial Regression (Degree 2) with interaction terms"""
    def __init__(self, degree=2):
        self.degree = degree
        self.coef_ = None
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        X_poly = np.hstack([np.ones((X.shape[0], 1)), X])
        
        for i in range(X.shape[1]):
            X_poly = np.hstack([X_poly, (X[:, i] ** self.degree).reshape(-1, 1)])
        
        self.coef_ = np.linalg.lstsq(X_poly, y, rcond=None)[0].flatten()
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        X_poly = np.hstack([np.ones((X.shape[0], 1)), X])
        
        for i in range(X.shape[1]):
            X_poly = np.hstack([X_poly, (X[:, i] ** self.degree).reshape(-1, 1)])
        
        return X_poly @ self.coef_

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_and_params():
    """Load the trained model and scaling parameters."""
    try:
        model = pickle.load(open('best_taxi_fare_model.pkl', 'rb'))
        scaling_params = pickle.load(open('model_scaling_params.pkl', 'rb'))
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, scaling_params, metadata
    except FileNotFoundError as e:
        st.error(f"❌ Error: Required model files not found. {str(e)}")
        st.stop()

def predict_fare(input_features, model, scaling_params):
    """
    Predict the taxi fare for given features.
    
    Parameters:
    -----------
    input_features : dict
        Dictionary with 10 required features
    model : object
        Trained model object
    scaling_params : dict
        Dictionary containing mean, std, and feature_names
        
    Returns:
    --------
    float : Predicted fare amount
    """
    # Create feature array in correct order
    feature_names = scaling_params['feature_names']
    X = np.array([[input_features[feat] for feat in feature_names]])
    
    # Scale features
    X_scaled = (X - scaling_params['mean']) / scaling_params['std']
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    return prediction

def get_feature_ranges():
    """Return realistic ranges for each feature."""
    return {
        'trip_duration_minutes': {
            'min': 1, 'max': 120, 'default': 15,
            'label': '⏱️ Trip Duration (minutes)',
            'help': 'How long the trip will take'
        },
        'trip_distance_miles': {
            'min': 0.1, 'max': 50.0, 'default': 2.5, 'step': 0.1,
            'label': '📍 Trip Distance (miles)',
            'help': 'Distance from pickup to dropoff'
        },
        'tip_amount': {
            'min': 0.0, 'max': 50.0, 'default': 2.0, 'step': 0.5,
            'label': '💵 Tip Amount ($)',
            'help': 'Expected tip amount'
        },
        'pickup_hour': {
            'min': 0, 'max': 23, 'default': 14,
            'label': '⏰ Pickup Hour (0-23)',
            'help': 'Hour of pickup (24-hour format)'
        },
        'dropoff_hour': {
            'min': 0, 'max': 23, 'default': 15,
            'label': '⏰ Dropoff Hour (0-23)',
            'help': 'Estimated hour of dropoff'
        },
        'pickup_latitude': {
            'min': 40.70, 'max': 40.80, 'default': 40.75, 'step': 0.01,
            'label': '🗺️ Pickup Latitude',
            'help': 'Latitude of pickup location (40.70-40.80 for NYC area)'
        },
        'dropoff_latitude': {
            'min': 40.70, 'max': 40.80, 'default': 40.76, 'step': 0.01,
            'label': '🗺️ Dropoff Latitude',
            'help': 'Latitude of dropoff location'
        },
        'pickup_dayofweek_num': {
            'min': 0, 'max': 6, 'default': 2,
            'label': '📅 Day of Week (0=Mon, 6=Sun)',
            'help': 'Day of the week (0=Monday, 6=Sunday)'
        },
        'payment_type': {
            'min': 1, 'max': 5, 'default': 1,
            'label': '💳 Payment Type (1-5)',
            'help': 'Payment method (1=Credit Card, 2=Cash, 3=Dispute, 4=Unknown, 5=Other)'
        },
        'improvement_surcharge': {
            'min': 0.0, 'max': 1.0, 'default': 0.3, 'step': 0.1,
            'label': '🏗️ Improvement Surcharge ($)',
            'help': 'MTA improvement surcharge'
        }
    }

def format_currency(value):
    """Format value as currency."""
    return f"${value:.2f}"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Title and Description
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚕 Taxi Fare Prediction System")
        st.markdown(
            "**Predict taxi fares using AI** • Enter trip details and get instant fare estimates"
        )
    with col2:
        st.image("https://img.icons8.com/color/96/000000/taxi.png", width=80)
    
    st.divider()
    
    # Load model
    with st.spinner("Loading model..."):
        model, scaling_params, metadata = load_model_and_params()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔮 Make Prediction", "📊 Model Performance", "ℹ️ Features Info", "💾 History"]
    )
    
    # ========================================================================
    # TAB 1: MAKE PREDICTION
    # ========================================================================
    with tab1:
        st.markdown("### 📋 Enter Trip Details")
        
        feature_ranges = get_feature_ranges()
        input_features = {}
        
        # Create input form with organized columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🛫 Trip Information**")
            input_features['trip_duration_minutes'] = st.slider(
                feature_ranges['trip_duration_minutes']['label'],
                min_value=feature_ranges['trip_duration_minutes']['min'],
                max_value=feature_ranges['trip_duration_minutes']['max'],
                value=feature_ranges['trip_duration_minutes']['default'],
                help=feature_ranges['trip_duration_minutes']['help']
            )
            
            input_features['trip_distance_miles'] = st.slider(
                feature_ranges['trip_distance_miles']['label'],
                min_value=feature_ranges['trip_distance_miles']['min'],
                max_value=feature_ranges['trip_distance_miles']['max'],
                value=feature_ranges['trip_distance_miles']['default'],
                step=feature_ranges['trip_distance_miles'].get('step', 1),
                help=feature_ranges['trip_distance_miles']['help']
            )
            
            input_features['tip_amount'] = st.slider(
                feature_ranges['tip_amount']['label'],
                min_value=feature_ranges['tip_amount']['min'],
                max_value=feature_ranges['tip_amount']['max'],
                value=feature_ranges['tip_amount']['default'],
                step=feature_ranges['tip_amount'].get('step', 0.5),
                help=feature_ranges['tip_amount']['help']
            )
            
            input_features['improvement_surcharge'] = st.slider(
                feature_ranges['improvement_surcharge']['label'],
                min_value=feature_ranges['improvement_surcharge']['min'],
                max_value=feature_ranges['improvement_surcharge']['max'],
                value=feature_ranges['improvement_surcharge']['default'],
                step=feature_ranges['improvement_surcharge'].get('step', 0.1),
                help=feature_ranges['improvement_surcharge']['help']
            )
        
        with col2:
            st.markdown("**⏰ Time & Location**")
            input_features['pickup_hour'] = st.slider(
                feature_ranges['pickup_hour']['label'],
                min_value=feature_ranges['pickup_hour']['min'],
                max_value=feature_ranges['pickup_hour']['max'],
                value=feature_ranges['pickup_hour']['default'],
                help=feature_ranges['pickup_hour']['help']
            )
            
            input_features['dropoff_hour'] = st.slider(
                feature_ranges['dropoff_hour']['label'],
                min_value=feature_ranges['dropoff_hour']['min'],
                max_value=feature_ranges['dropoff_hour']['max'],
                value=feature_ranges['dropoff_hour']['default'],
                help=feature_ranges['dropoff_hour']['help']
            )
            
            input_features['pickup_dayofweek_num'] = st.slider(
                feature_ranges['pickup_dayofweek_num']['label'],
                min_value=feature_ranges['pickup_dayofweek_num']['min'],
                max_value=feature_ranges['pickup_dayofweek_num']['max'],
                value=feature_ranges['pickup_dayofweek_num']['default'],
                help=feature_ranges['pickup_dayofweek_num']['help']
            )
            
            input_features['payment_type'] = st.slider(
                feature_ranges['payment_type']['label'],
                min_value=feature_ranges['payment_type']['min'],
                max_value=feature_ranges['payment_type']['max'],
                value=feature_ranges['payment_type']['default'],
                help=feature_ranges['payment_type']['help']
            )
        
        st.divider()
        
        st.markdown("**📍 Location Coordinates**")
        
        col1, col2 = st.columns(2)
        with col1:
            input_features['pickup_latitude'] = st.slider(
                feature_ranges['pickup_latitude']['label'],
                min_value=feature_ranges['pickup_latitude']['min'],
                max_value=feature_ranges['pickup_latitude']['max'],
                value=feature_ranges['pickup_latitude']['default'],
                step=feature_ranges['pickup_latitude'].get('step', 0.01),
                help=feature_ranges['pickup_latitude']['help']
            )
        
        with col2:
            input_features['dropoff_latitude'] = st.slider(
                feature_ranges['dropoff_latitude']['label'],
                min_value=feature_ranges['dropoff_latitude']['min'],
                max_value=feature_ranges['dropoff_latitude']['max'],
                value=feature_ranges['dropoff_latitude']['default'],
                step=feature_ranges['dropoff_latitude'].get('step', 0.01),
                help=feature_ranges['dropoff_latitude']['help']
            )
        
        st.divider()
        
        # Prediction Button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            predict_button = st.button("🔮 Predict Fare", use_container_width=True, type="primary")
        with col2:
            clear_button = st.button("🔄 Reset", use_container_width=True)
        
        # Handle predictions
        if predict_button:
            try:
                with st.spinner("Calculating fare..."):
                    predicted_fare = predict_fare(input_features, model, scaling_params)
                
                # Store in session state for history
                if 'predictions_history' not in st.session_state:
                    st.session_state.predictions_history = []
                
                prediction_record = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'trip_duration': input_features['trip_duration_minutes'],
                    'distance': input_features['trip_distance_miles'],
                    'tip': input_features['tip_amount'],
                    'predicted_fare': predicted_fare,
                    'input_features': input_features.copy()
                }
                st.session_state.predictions_history.append(prediction_record)
                
                # Display results with styling
                st.markdown("")
                st.markdown(f"""
                    <div class="prediction-result">
                        <h2>💰 Predicted Fare Amount</h2>
                        <h1>{format_currency(predicted_fare)}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional fare breakdown and information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Base Components",
                        f"{input_features['trip_distance_miles']:.2f} miles + "
                        f"{input_features['trip_duration_minutes']:.0f} min"
                    )
                
                with col2:
                    st.metric(
                        "Surcharges & Tips",
                        f"${input_features['improvement_surcharge'] + input_features['tip_amount']:.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Estimated Total",
                        format_currency(predicted_fare)
                    )
                
                # Confidence interval
                rmse = metadata['performance_metrics']['RMSE']
                mae = metadata['performance_metrics']['MAE']
                confidence_low = predicted_fare - (2 * rmse)
                confidence_high = predicted_fare + (2 * rmse)
                
                st.markdown(
                    f"""
                    <div class="info-box">
                    <b>95% Confidence Interval:</b> {format_currency(max(0, confidence_low))} - {format_currency(confidence_high)}
                    <br><b>Average Model Error (RMSE):</b> {format_currency(rmse)}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Summary of inputs
                with st.expander("📋 View Input Summary"):
                    summary_df = pd.DataFrame([input_features]).T
                    summary_df.columns = ['Value']
                    st.dataframe(summary_df, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
        
        if clear_button:
            st.rerun()
    
    # ========================================================================
    # TAB 2: MODEL PERFORMANCE
    # ========================================================================
    with tab2:
        st.markdown("### 📊 Model Performance Metrics")
        
        # Performance metrics display
        perf = metadata['performance_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{perf['R2_score']:.4f}")
            st.caption("Variance explained (higher is better)")
        
        with col2:
            st.metric("RMSE", format_currency(perf['RMSE']))
            st.caption("Average prediction error")
        
        with col3:
            st.metric("MAE", format_currency(perf['MAE']))
            st.caption("Typical prediction error")
        
        with col4:
            st.metric("MSE", f"{perf['MSE']:.4f}")
            st.caption("Mean squared error")
        
        st.divider()
        
        # Model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Model Information")
            st.write(f"**Model Name:** {metadata['model_name']}")
            st.write(f"**Model Type:** {metadata['model_type']}")
            st.write(f"**Task:** {metadata['task']}")
            st.write(f"**Training Date:** {metadata['training_date']}")
            st.write(f"**Version:** {metadata['version']}")
        
        with col2:
            st.markdown("### 🎯 Accuracy Assessment")
            accuracy_text = f"""
            - Explains **{perf['R2_score']*100:.2f}%** of fare variance
            - Average error: **±{format_currency(perf['RMSE'])}** per prediction
            - Typical error: **±{format_currency(perf['MAE'])}**
            - 95% CI: ±{format_currency(perf['RMSE'] * 2)} around prediction
            - Suitable for: Trip planning, fare estimation, UX
            """
            st.info(accuracy_text)
    
    # ========================================================================
    # TAB 3: FEATURES INFORMATION
    # ========================================================================
    with tab3:
        st.markdown("### ℹ️ Feature Descriptions")
        
        features_info = {
            'trip_duration_minutes': {
                'description': 'Duration of the trip in minutes',
                'range': '1-120 minutes',
                'impact': '⬆️ Higher duration → Higher fare'
            },
            'trip_distance_miles': {
                'description': 'Distance traveled in miles',
                'range': '0.1-50.0 miles',
                'impact': '⬆️ Higher distance → Higher fare'
            },
            'tip_amount': {
                'description': 'Tip amount given by passenger',
                'range': '$0-$50',
                'impact': '➡️ Tip does not affect base fare'
            },
            'pickup_hour': {
                'description': 'Hour of day when trip starts (24-hour format)',
                'range': '0-23 (midnight to 11pm)',
                'impact': '🕐 Peak hours may affect availability'
            },
            'dropoff_hour': {
                'description': 'Hour of day when trip ends',
                'range': '0-23',
                'impact': '🕐 Drop-off time impact captured'
            },
            'pickup_latitude': {
                'description': 'Geographic latitude of pickup location',
                'range': '40.70-40.80 (NYC area)',
                'impact': '📍 Location affects base fare'
            },
            'dropoff_latitude': {
                'description': 'Geographic latitude of dropoff location',
                'range': '40.70-40.80 (NYC area)',
                'impact': '📍 Destination zone matters'
            },
            'payment_type': {
                'description': 'Method of payment',
                'range': '1-5 (1:Card, 2:Cash, etc.)',
                'impact': '💳 Payment type captured for analysis'
            },
            'improvement_surcharge': {
                'description': 'MTA improvement surcharge',
                'range': '$0-$1',
                'impact': '🏗️ Fixed surcharge component'
            },
            'pickup_dayofweek_num': {
                'description': 'Day of the week (0=Monday, 6=Sunday)',
                'range': '0-6',
                'impact': '📅 Weekday vs weekend patterns'
            }
        }
        
        for feature_name, feature_info in features_info.items():
            with st.expander(f"📌 {feature_name.replace('_', ' ').title()}"):
                st.write(f"**Description:** {feature_info['description']}")
                st.write(f"**Range:** {feature_info['range']}")
                st.write(f"**Impact:** {feature_info['impact']}")
        
        st.divider()
        st.markdown("""
        ### 🔗 Feature Relationships
        
        **Key Drivers of Fare:**
        1. **Distance** - Primary driver of fare
        2. **Duration** - Time spent in vehicle
        3. **Location** - Pickup and dropoff zones
        4. **Time of Day** - Peak hour premiums
        5. **Day of Week** - Weekday vs weekend
        
        **Model Notes:**
        - All features are scaled using z-score normalization
        - Polynomial features capture non-linear relationships
        - Model trained on 156,400 NYC taxi trips
        """)
    
    # ========================================================================
    # TAB 4: PREDICTION HISTORY
    # ========================================================================
    with tab4:
        st.markdown("### 💾 Prediction History")
        
        if 'predictions_history' in st.session_state and len(st.session_state.predictions_history) > 0:
            history_df = pd.DataFrame(st.session_state.predictions_history)
            history_df = history_df[['timestamp', 'trip_duration', 'distance', 'tip', 'predicted_fare']]
            history_df.columns = ['Timestamp', 'Duration (min)', 'Distance (mi)', 'Tip ($)', 'Predicted Fare ($)']
            
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            fares = [p['predicted_fare'] for p in st.session_state.predictions_history]
            
            with col1:
                st.metric("Total Predictions", len(st.session_state.predictions_history))
            with col2:
                st.metric("Average Predicted Fare", format_currency(np.mean(fares)))
            with col3:
                st.metric("Highest Predicted Fare", format_currency(np.max(fares)))
            with col4:
                st.metric("Lowest Predicted Fare", format_currency(np.min(fares)))
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.predictions_history = []
                st.rerun()
        else:
            st.info("📭 No prediction history yet. Make a prediction in the 'Make Prediction' tab!")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.divider()
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.caption("🏆 Model: Polynomial Regression (Degree 2)")
    
    with footer_col2:
        st.caption(f"📅 Last Updated: {metadata['training_date']}")
    
    with footer_col3:
        st.caption("⚠️ For reference purposes - actual fares may vary")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()

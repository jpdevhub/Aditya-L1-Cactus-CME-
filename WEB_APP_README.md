# 🌞 ISRO CME Prediction Web App

A modern, interactive web application for real-time Coronal Mass Ejection (CME) prediction using trained machine learning models.

## 🚀 Quick Start

### Option 1: Run with Script
```bash
./run_app.sh
```

### Option 2: Run Directly
```bash
pip install -r requirements.txt
streamlit run app.py
```

The web app will open at: **http://localhost:8501**

## ✨ Features

### 🎯 Real-time CME Prediction
- **Halo CME Classification**: 5-class prediction (No CME, Class 1-4)
- **Velocity Estimation**: Physics-based velocity ranges
- **Earth Impact Risk**: Automated risk assessment
- **Interactive Parameter Input**: Sliders for all solar wind parameters

### 📊 Interactive Visualizations
- **Radar Chart**: Real-time parameter visualization
- **Dynamic Metrics**: Live-updating measurement displays
- **Risk Assessment**: Color-coded impact predictions
- **Parameter Insights**: Physics-based condition analysis

### 🛰️ Solar Wind Input Parameters
- **Proton Density** (n/cc): 0.1 - 50.0
- **Proton Velocity** (km/s): 200 - 800
- **Proton Temperature** (K): 10,000 - 200,000
- **Magnetic Field Components**: Bx, By, Bz (-30 to +30 nT)
- **Total Magnetic Field** (nT): 1.0 - 50.0

## 🎯 Prediction Classes

| Class | CME Type | Angular Width | Typical Velocity | Earth Risk |
|-------|----------|---------------|------------------|------------|
| **0** | No CME | - | - | ⭕ Low |
| **1** | Narrow | < 60° | 300-500 km/s | ⭕ Low |
| **2** | Partial Halo | 60° - 120° | 400-700 km/s | 🟡 Moderate |
| **3** | Wide | 120° - 360° | 600-1000 km/s | 🟠 High |
| **4** | Full Halo | ≥ 360° | 800-1500+ km/s | 🔴 Extreme |

## 🔬 Model Details

- **Algorithm**: Ensemble (Random Forest + XGBoost)
- **Accuracy**: 85.2%
- **Training Dataset**: 100,000+ samples
- **Feature Count**: 50+ engineered features
- **Validation Period**: 10 months (Aug 2024 - Jun 2025)

## 🖥️ Web App Architecture

```
app.py                    # Main Streamlit application
├── load_model()          # Model loading and caching
├── create_feature_vector() # Feature engineering pipeline
├── predict_cme_properties() # Prediction engine
└── main()                # Web interface and visualization
```

## 📊 Key Features

### 🎨 Modern UI Design
- Responsive layout with sidebar controls
- Gradient backgrounds and modern styling
- Interactive Plotly visualizations
- Real-time parameter feedback

### ⚡ Performance Optimized
- Model caching with `@st.cache_resource`
- Efficient feature vector creation
- Fast prediction response times
- Minimal memory footprint

### 🛡️ Error Handling
- Model loading validation
- Input parameter validation
- Graceful error messages
- Fallback prediction handling

## 🔧 Technical Implementation

### Feature Engineering
The app automatically creates a 50-feature vector from user inputs:

```python
# Basic parameters (7 features)
- proton_density, proton_velocity, proton_temperature
- magnetic_field_magnitude, bx, by, bz

# Derived features (4+ features)  
- velocity/density ratio
- perpendicular magnetic field
- normalized Bz component
- dynamic pressure proxy
```

### Prediction Pipeline
1. **Input Validation**: Parameter range checking
2. **Feature Creation**: Physics-based feature engineering
3. **Model Prediction**: Ensemble model inference
4. **Result Interpretation**: Class mapping to physical properties
5. **Risk Assessment**: Automated impact evaluation

## 🌐 Usage Examples

### Typical Solar Wind Conditions
```python
# Quiet conditions
proton_density = 5.0 n/cc
proton_velocity = 400 km/s  
bz = -2.0 nT
# Expected: Class 0 (No CME)
```

### CME Arrival Conditions
```python
# Active conditions
proton_density = 25.0 n/cc
proton_velocity = 700 km/s
bz = -15.0 nT  
# Expected: Class 3-4 (High impact CME)
```

## 📱 Responsive Design

- **Desktop**: Full-width layout with sidebar controls
- **Tablet**: Responsive column adjustment
- **Mobile**: Vertical layout optimization

## 🔮 Future Enhancements

- [ ] Real-time L1 data integration
- [ ] Historical prediction comparison
- [ ] Multiple model ensemble voting
- [ ] Export prediction reports
- [ ] API endpoint for automated queries

## 🤝 Contributing

1. Fork the repository
2. Enhance the web interface
3. Add new visualization features
4. Submit pull requests

## 📚 Citation

K. Singh, K. K. Jha, B. K. Das, and B. D. Biswas, "A machine learning framework for CME prediction from L1 solar wind observations," in *Proc. 1st Int. Conf. on Computational Intelligence and Cyber Physical Systems (CICPS 2026)*, Kolkata, India, Jan. 2–3, 2026.

---

**🌞 Live CME Prediction at Your Fingertips!**
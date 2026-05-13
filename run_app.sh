#!/bin/bash

# ISRO CME Prediction Web App Launcher
echo "🌞 Starting ISRO CME Prediction System Web App..."
echo "================================================"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Launch the web app
echo "🚀 Launching web app on http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo "================================================"

streamlit run app.py --server.port 8501 --server.address localhost
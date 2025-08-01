#!/bin/bash

echo "🚀 Starting IBM Salary Prediction Project"
echo "==========================================="

# Check if models exist
if [ ! -d "src/models" ]; then
    echo "📊 Generating sample data and training models..."
    cd src
    python3 ../data/sample_salary_data.py
    python3 ensemble_models.py
    cd ..
    echo "✅ Models trained successfully!"
else
    echo "✅ Models already exist"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
    echo "✅ Environment file created"
fi

echo ""
echo "🌐 Starting web application..."
echo "Access the application at: http://localhost:5000"
echo ""
echo "Available endpoints:"
echo "  • Home: http://localhost:5000/"
echo "  • Predict: http://localhost:5000/predict"
echo "  • Chat: http://localhost:5000/chat"
echo "  • About: http://localhost:5000/about"
echo "  • API: http://localhost:5000/api/predict"
echo "  • Health: http://localhost:5000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "==========================================="

cd web_app
python3 app.py
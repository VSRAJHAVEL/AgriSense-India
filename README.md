# AgriSense India - Crop Recommendation System

AgriSense India is an AI-powered agricultural intelligence platform designed to assist farmers and agricultural stakeholders in making data-driven decisions. By analyzing soil images and integrating market intelligence, the system provides precise crop recommendations tailored to specific soil types and environmental conditions.

## Key Features

*   **Soil Image Classification**: Utilizes a fine-tuned MobileNetV2 Convolutional Neural Network (CNN) to analyze user-uploaded soil images and classify them into primary Indian soil types (Alluvial, Black, Clay, Loamy, and Red).
*   **Intelligent Crop Recommendation**: Employs a Random Forest Machine Learning model to evaluate soil type, temperature, and rainfall to recommend the most suitable crops.
*   **Market Intelligence**: Integrates Minimum Support Price (MSP) data, expected yield parameters, and local market rates.
*   **Comprehensive Cultivation Data**: Provides end-to-end growing details including required NPK ratios, water requirements, optimal pH ranges, growing seasons, and suitable varieties for each crop.
*   **Responsive User Interface**: Features a modern, glassmorphic design architecture that provides a seamless experience across desktop and mobile devices.

## Technology Stack

*   **Core Backend**: Python 3.12, Flask, Gunicorn
*   **Computer Vision (Soil Classification)**: TensorFlow 2, Keras, CNN Architecture: **MobileNetV2** (Fine-tuned for Indian soil datasets)
*   **Machine Learning (Crop Recommendation)**: scikit-learn, Ensemble Method: **Random Forest Classifier**
*   **Data Processing & Engineering**: NumPy, Pandas, Joblib, Pillow (PIL)
*   **Frontend Architecture**: HTML5, Vanilla JavaScript, Custom CSS3 (Premium Glassmorphism UI)
*   **Deployment & Cloud**: Render (PaaS), Git

## Project Structure

*   `app.py`: The core Flask application handling routing and API endpoints.
*   `models/`: Directory containing the pre-trained MobileNetV2 soil classifier (.h5) and Random Forest crop recommender (.pkl) models.
*   `data/`: Contains `crop_details.json`, a comprehensive database of Indian agricultural statistics, market prices, and cultivation parameters.
*   `templates/`: HTML templates for the web interface.
*   `static/`: CSS and JavaScript assets powering the frontend design and logic.
*   `train_soil_model.py` / `train_crop_model.py`: Training scripts utilized to generate the initial ML models.

## Deployment

This application is configured for deployment on standard cloud platforms supporting Python WSGI applications. It utilizes `gunicorn` as the production WSGI HTTP server.

1.  Clone the repository.
2.  Install dependencies using `pip install -r requirements.txt`.
3.  Execute the application locally via `flask run` or in production using `gunicorn app:app`.

## Author

Developed by Rajhavel V S.

# Predictive Health Analytics Platform

## Project Overview
This project is a **Predictive Health Analytics Platform** designed to help users assess and improve their personal wellness. By leveraging **demographic, behavioral, and biometric data**—including age, gender, weight, height, daily activity, and dietary intake—the platform classifies users into **health risk categories** and provides personalized recommendations.

The application integrates **machine learning**, **interactive visualizations**, and **report generation** to empower users with actionable health insights.

---

## Features
- **Health Risk Classification** using KMeans clustering based on user data  
- **Personalized Wellness Reports** including body composition analysis (body fat %, muscle mass %, visceral fat %)  
- **Interactive Visualizations** such as correlation heatmaps, lifestyle comparisons, and gauge charts  
- **Caloric and Meal Planning** using BMR/TDEE calculations and Nutritionix API integration  
- **PDF Export** for personalized health reports  

---

## Technologies Used
- **Python** for backend logic and ML implementation  
- **Streamlit** for interactive web app deployment  
- **Pandas & NumPy** for data manipulation  
- **Scikit-learn** for machine learning (KMeans clustering)  
- **Plotly** for interactive charts and gauge visualizations  
- **FPDF** for PDF report generation  
- **APIs** (Nutritionix) for meal suggestions  

---

## Installation & Setup
1. **Clone the repository**  


2. **Create a virtual environment**
   python -m venv health_env

3. **Activate the virtual environment**

   Windows:
   health_env\Scripts\activate

4. **Install required packages**
   pip install streamlit pandas numpy scikit-learn fpdf plotly

5. **Run the streamlit app**
   streamlit run new_app.py

Usage

Input your age, gender, weight, height, daily activity, dietary intake, and other relevant health parameters

View your health risk cluster and personalized recommendations

Generate and download a PDF report with your wellness summary

Explore interactive visualizations for deeper insights into your health metrics

Key Highlights

Designed to be user-friendly and interactive

Integrates real-time ML predictions with visual analytics

Provides actionable diet and lifestyle suggestions

Fully deployed locally via Streamlit with simple setup instructions

```bash

streamlit==1.26.0
scikit-learn==1.3.2
pandas==2.2.1
numpy==1.26.0
scikit-learn==1.3.2
fpdf==1.7.2
plotly==5.20.0
requests==2.31.0


gne <your-repo-url>
cd <repo-folder>

Link to the app: https://predictive-health-analytics-copyright.streamlit.app

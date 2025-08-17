import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import base64
from groq import Groq

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Health Analyzer App",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💪",
    menu_items={
        'Get help': 'https://streamlit.io',
        'Report a bug': 'https://github.com/streamlit/streamlit/issues',
        'About': 'Streamlit is a powerful open-source app framework.',
    }
)

# --- Set up Groq API key and client ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Groq API key not found. Please add it to your `.streamlit/secrets.toml` file.")
    st.stop()

# Initialize the Groq client
client = Groq(api_key=groq_api_key)
# Choose a fast model from Groq
GROQ_MODEL = "llama3-8b-8192"

# Loading dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Final_health_dataset.csv")
    except FileNotFoundError:
        st.error("Error: The file 'Final_health_dataset.csv' was not found. Please ensure the CSV file is in the same directory as the app.")
        st.stop()

    # Preprocessing & renaming columns
    df = df.rename(columns={
        "Age": "Age",
        "Gender": "Gender",
        "Basal_Metabolic_Rate": "Calories_per_day",
        "Physical_Activity_Steps": "Steps_per_day",
        "Alcohol_Intake_Value": "Alcohol_grams_per_day",
        "Alcohol_Intake_Category": "Alcohol_Category",
        "Smoking_Status": "Smoker",
        "Percent_Body_Fat": "Body_Fat_Percent",
        "Muscle_Mass_Percent": "Muscle_Mass_Percent",
        "Visceral_Fat_Level": "Visceral_Fat_Level"
    })
    df["Gender"] = df["Gender"].map({0: "Male", 1: "Female"})
    df["Smoker"] = df["Smoker"].map({"No": 0, "Yes": 1})
    df["Gender_num"] = df["Gender"].map({"Male": 0, "Female": 1})

    return df

df = load_data()

features = ["Age", "Gender_num", "Calories_per_day", "Steps_per_day", "Alcohol_grams_per_day", "Smoker"]
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
cluster_map = {0: "Moderate Risk", 1: "Low Risk", 2: "High Risk"}

# --- Helper functions ---
def get_recommendations(data):
    tips = []
    if data["Calories_per_day"] > 2500:
        reduce = int((data["Calories_per_day"] - 2500) * 100 / data["Calories_per_day"])
        tips.append(f"Reduce calories by ~{reduce}% (<= 2500 kcal).")
    if data["Steps_per_day"] < 7000:
        tips.append("Increase activity to at least 7,000 steps/day.")
    if data["Alcohol_grams_per_day"] > 20:
        tips.append("Limit alcohol to <= 20g/day (approx. 1 drink).")
    if data["Smoker"] == 1:
        tips.append("Quit smoking for long-term health.")
    return tips or ["You're maintaining a healthy lifestyle."]

def draw_body_chart(user_input):
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=user_input.get("Body_Fat_Percent", 23.5),
            title={'text': "Body Fat %"},
            gauge={'bar': {'color': 'green'}}
        ))
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=user_input.get("Muscle_Mass_Percent", 35.0),
            title={'text': "Muscle Mass %"},
            gauge={'bar': {'color': 'green'}}
        ))
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=user_input.get("Visceral_Fat_Level", 10.0),
            title={'text': "Visceral Fat Level %"},
            gauge={'bar': {'color': 'green'}}
        ))
        st.plotly_chart(fig3, use_container_width=True)

def show_additional_charts(user_input):
    chart_data = pd.DataFrame({
        "Metric": ["Calories", "Steps", "Alcohol"],
        "User": [user_input["Calories_per_day"], user_input["Steps_per_day"], user_input["Alcohol_grams_per_day"]],
        "Ideal": [2500, 7000, 20]
    })

    st.subheader("Comparative Charts")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.line(chart_data, x="Metric", y=["User", "Ideal"], title="Line Chart Comparison")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(chart_data, names="Metric", values="User", hole=0.4, title="Your Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.bar(chart_data, x="Metric", y="User", color="Metric", title="Trend Analysis")
        st.plotly_chart(fig, use_container_width=True)

def generate_pdf(user_input, risk, tips):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Health Risk Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    for key, value in user_input.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, f"Predicted Risk: {risk}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, "Recommendations:", ln=True)
    for tip in tips:
        safe_tip = tip.encode("ascii", "ignore").decode("ascii")
        pdf.cell(200, 10, f"- {safe_tip}", ln=True)
    pdf.output("health_report.pdf")
    with open("health_report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<a href="data:application/pdf;base64,{base64_pdf}" download="Health_Report.pdf">Download your personalized health report</a>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# Calorie & Diet Planner helpers
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    return bmr

def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        "Sedentary (little to no exercise)": 1.2,
        "Lightly active (light exercise 1-3 days in a week)": 1.375,
        "Moderately active (moderate exercise 3-5 days in a week)": 1.55,
        "Very active (hard exercise 6-7 days in a week)": 1.725,
        "Super active (very hard exercise & physical job)": 1.9
    }
    return bmr * activity_multipliers.get(activity_level, 1.2)

# GLOBAL DICTIONARY: Moved here so it's accessible everywhere
food_db = {
    "oatmeal": {"protein": 4.0, "fat": 3.0, "carbs": 27.0, "calories": 150},
    "greek_yogurt": {"protein": 10.0, "fat": 0.0, "carbs": 4.0, "calories": 59},
    "large_egg": {"protein": 6.0, "fat": 5.0, "carbs": 0.6, "calories": 78},
    "mixed_berries": {"protein": 1.0, "fat": 0.5, "carbs": 12.0, "calories": 57},
    "apple": {"protein": 0.5, "fat": 0.3, "carbs": 25.0, "calories": 95},
    "pear": {"protein": 0.6, "fat": 0.2, "carbs": 27.0, "calories": 101},
    "banana": {"protein": 1.3, "fat": 0.4, "carbs": 27.0, "calories": 105},
    "orange": {"protein": 1.2, "fat": 0.2, "carbs": 15.0, "calories": 62},
    "chicken_breast": {"protein": 31.0, "fat": 3.6, "carbs": 0.0, "calories": 165},
    "lean_ground_beef": {"protein": 26.0, "fat": 15.0, "carbs": 0.0, "calories": 250},
    "lean_ground_turkey": {"protein": 25.0, "fat": 11.0, "carbs": 0.0, "calories": 200},
    "boiled_potatoes": {"protein": 2.0, "fat": 0.1, "carbs": 20.0, "calories": 87},
    "cooked_spinach": {"protein": 2.9, "fat": 0.4, "carbs": 3.6, "calories": 23},
    "cooked_green_beans": {"protein": 1.8, "fat": 0.2, "carbs": 7.6, "calories": 35},
    "cooked_carrots": {"protein": 0.9, "fat": 0.2, "carbs": 9.6, "calories": 41},
    "cooked_cauliflower": {"protein": 1.9, "fat": 0.3, "carbs": 5.0, "calories": 25},
    "cooked_bell_peppers": {"protein": 1.0, "fat": 0.3, "carbs": 6.0, "calories": 25},
    "cooked_asparagus": {"protein": 2.2, "fat": 0.2, "carbs": 3.9, "calories": 20},
    "butter": {"protein": 0.1, "fat": 81.0, "carbs": 0.1, "calories": 717},
    "protein_shake": {"protein": 28.0, "fat": 8.0, "carbs": 8.0, "calories": 200},
}

def generate_custom_meal_plans(target_calories, target_protein, target_fat, target_carbs):
    plans = {}
    
    shake_macros = food_db["protein_shake"]
    remaining_calories = target_calories - shake_macros["calories"]
    remaining_protein = target_protein - shake_macros["protein"]
    remaining_fat = target_fat - shake_macros["fat"]
    remaining_carbs = target_carbs - shake_macros["carbs"]

    # --- Plan 1: Chicken & Eggs ---
    plan1_p_dist = [0.25, 0.45, 0.30]
    plan1_f_dist = [0.20, 0.40, 0.40]
    plan1_c_dist = [0.25, 0.35, 0.40]
    
    plans["Chicken & Eggs"] = {
        "Meal 1: Breakfast": {
            "Food": "Oatmeal, Plain Greek Yogurt, 2 Eggs, Mixed Berries",
            "Weights": {
                "Oatmeal (cooked)": f"{round((remaining_carbs * plan1_c_dist[0] * 0.4) / food_db['oatmeal']['carbs'] * 100)} g",
                "Greek Yogurt": f"{round((remaining_protein * plan1_p_dist[0] * 0.4) / food_db['greek_yogurt']['protein'] * 100)} g",
                "Large Eggs": "2 eggs",
                "Mixed Berries": f"{round((remaining_carbs * plan1_c_dist[0] * 0.4) / food_db['mixed_berries']['carbs'] * 100)} g",
            }
        },
        "Meal 2: Lunch": {
            "Food": "Boneless Chicken Breast, Boiled Potatoes, Cooked Spinach, Butter",
            "Weights": {
                "Chicken Breast": f"{round((remaining_protein * plan1_p_dist[1]) / food_db['chicken_breast']['protein'] * 100)} g",
                "Boiled Potatoes": f"{round((remaining_carbs * plan1_c_dist[1]) / food_db['boiled_potatoes']['carbs'] * 100)} g",
                "Cooked Spinach": "150 g",
                "Butter": f"{round((remaining_fat * plan1_f_dist[1]) / food_db['butter']['fat'] * 100)} g",
            }
        },
        "Meal 3: Dinner": {
            "Food": "Boneless Chicken Breast, Boiled Potatoes, Cooked Green Beans, Butter, Apple",
            "Weights": {
                "Chicken Breast": f"{round((remaining_protein * plan1_p_dist[2]) / food_db['chicken_breast']['protein'] * 100)} g",
                "Boiled Potatoes": f"{round((remaining_carbs * plan1_c_dist[2] * 0.5) / food_db['boiled_potatoes']['carbs'] * 100)} g",
                "Cooked Green Beans": "150 g",
                "Butter": f"{round((remaining_fat * plan1_f_dist[2]) / food_db['butter']['fat'] * 100)} g",
                "Apple": "1 medium (180 g)"
            }
        },
        "Last Meal: Protein Shake": {
            "Food": "Protein shake (2 scoops protein powder + 125ml whole milk)",
            "Weights": "N/A"
        }
    }
    
    # --- Plan 2: Lean Beef & Eggs ---
    plan2_p_dist = [0.25, 0.45, 0.30]
    plan2_f_dist = [0.20, 0.40, 0.40]
    plan2_c_dist = [0.25, 0.35, 0.40]

    plans["Lean Beef & Eggs"] = {
        "Meal 1: Breakfast": {
            "Food": "Oatmeal, Plain Greek Yogurt, 2 Eggs, Pear",
            "Weights": {
                "Oatmeal (cooked)": f"{round((remaining_carbs * plan2_c_dist[0] * 0.4) / food_db['oatmeal']['carbs'] * 100)} g",
                "Greek Yogurt": f"{round((remaining_protein * plan2_p_dist[0] * 0.4) / food_db['greek_yogurt']['protein'] * 100)} g",
                "Large Eggs": "2 eggs",
                "Pear": "1 medium (180 g)",
            }
        },
        "Meal 2: Lunch": {
            "Food": "Lean Ground Beef, Boiled Potatoes, Cooked Carrots, Butter",
            "Weights": {
                "Lean Ground Beef": f"{round((remaining_protein * plan2_p_dist[1]) / food_db['lean_ground_beef']['protein'] * 100)} g",
                "Boiled Potatoes": f"{round((remaining_carbs * plan2_c_dist[1]) / food_db['boiled_potatoes']['carbs'] * 100)} g",
                "Cooked Carrots": "100 g",
                "Butter": f"{round((remaining_fat * plan2_f_dist[1]) / food_db['butter']['fat'] * 100)} g",
            }
        },
        "Meal 3: Dinner": {
            "Food": "Lean Ground Beef, Boiled Potatoes, Cooked Cauliflower, Butter, Orange",
            "Weights": {
                "Lean Ground Beef": f"{round((remaining_protein * plan2_p_dist[2]) / food_db['lean_ground_beef']['protein'] * 100)} g",
                "Boiled Potatoes": f"{round((remaining_carbs * plan2_c_dist[2] * 0.5) / food_db['boiled_potatoes']['carbs'] * 100)} g",
                "Cooked Cauliflower": "200 g",
                "Butter": f"{round((remaining_fat * plan2_f_dist[2]) / food_db['butter']['fat'] * 100)} g",
                "Orange": "1 medium (130 g)"
            }
        },
        "Last Meal: Protein Shake": {
            "Food": "Protein shake (2 scoops protein powder + 125ml whole milk)",
            "Weights": "N/A"
        }
    }

    # --- Plan 3: Lean Turkey & Eggs ---
    plan3_p_dist = [0.25, 0.45, 0.30]
    plan3_f_dist = [0.20, 0.40, 0.40]
    plan3_c_dist = [0.25, 0.35, 0.40]
    
    plans["Lean Turkey & Eggs"] = {
        "Meal 1: Breakfast": {
            "Food": "Oatmeal, Plain Greek Yogurt, 2 Eggs, Banana",
            "Weights": {
                "Oatmeal (cooked)": f"{round((remaining_carbs * plan3_c_dist[0] * 0.4) / food_db['oatmeal']['carbs'] * 100)} g",
                "Greek Yogurt": f"{round((remaining_protein * plan3_p_dist[0] * 0.4) / food_db['greek_yogurt']['protein'] * 100)} g",
                "Large Eggs": "2 eggs",
                "Banana": "1 medium (118 g)",
            }
        },
        "Meal 2: Lunch": {
            "Food": "Lean Ground Turkey, Boiled Potatoes, Cooked Bell Peppers, Butter",
            "Weights": {
                "Lean Ground Turkey": f"{round((remaining_protein * plan3_p_dist[1]) / food_db['lean_ground_turkey']['protein'] * 100)} g",
                "Boiled Potatoes": f"{round((remaining_carbs * plan3_c_dist[1]) / food_db['boiled_potatoes']['carbs'] * 100)} g",
                "Cooked Bell Peppers": "150 g",
                "Butter": f"{round((remaining_fat * plan3_f_dist[1]) / food_db['butter']['fat'] * 100)} g",
            }
        },
        "Meal 3: Dinner": {
            "Food": "Lean Ground Turkey, Boiled Potatoes, Cooked Asparagus, Butter, Strawberries",
            "Weights": {
                "Lean Ground Turkey": f"{round((remaining_protein * plan3_p_dist[2]) / food_db['lean_ground_turkey']['protein'] * 100)} g",
                "Boiled Potatoes": f"{round((remaining_carbs * plan3_c_dist[2] * 0.5) / food_db['boiled_potatoes']['carbs'] * 100)} g",
                "Cooked Asparagus": "150 g",
                "Butter": f"{round((remaining_fat * plan3_f_dist[2]) / food_db['butter']['fat'] * 100)} g",
                "Strawberries": "150 g"
            }
        },
        "Last Meal: Protein Shake": {
            "Food": "Protein shake (2 scoops protein powder + 125ml whole milk)",
            "Weights": "N/A"
        }
    }
    
    return plans

# Nutritionix API details (replace with your credentials)
NUTRITIONIX_APP_ID = "0afb1157"
NUTRITIONIX_API_KEY = "99be0e39954a5b45ecdf7a6399c8379d"
NUTRITIONIX_API_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"

# --- Streamlit app starts here ---
st.sidebar.title("Health Analytics")
page = st.sidebar.selectbox("Please select from our bulletproof options", [
    "Health Risk Analyzer", 
    "Calorie & Diet Planner", 
    "Meal Search", 
    "Health Coach"
])

top_row = st.container()
bottom_row = st.container()

with top_row:
    st.markdown("<h1 style='text-align: center;font-size: 4.0rem;'>Comprehensive Health Analytics Platform</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.image("img_final.png", use_container_width=True)

    st.markdown("## Welcome to Your Health Analyzer")
    st.markdown("<h3>Track. Understand. Improve.</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1em;'>Take charge of your health with a platform that analyzes your unique profile and gives instant, personalized insights. Just enter your age, weight, lifestyle habits, and activity level, and get a clear health risk rating with actionable tips to help you feel and live better.</p>", unsafe_allow_html=True)
    st.markdown("<h3>Smart Health, Powered by ML Models.</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1em;'>Behind the scenes, our advanced ML Models groups you into meaningful health categories and calculates your body composition and daily calorie needs. With interactive tools for tracking, diet planning, and progress monitoring, everything you need to understand and improve your health is in one place.</p>", unsafe_allow_html=True)
    st.markdown("<h3>Your Health, Made Simple.</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1em;'>From sleek dashboards to in-depth PDF reports and tailored meal suggestions, our platform turns complex health data into easy, everyday actions. No guesswork, no gimmicks—just the right insights to help you reach your goals.</p>", unsafe_allow_html=True)
    
    st.markdown("<b>Please, use the sidebar to navigate between sections.</b>", unsafe_allow_html=True)

with bottom_row:
    if page == "Health Risk Analyzer":
        st.title("Health Risk Analyzer")
        st.markdown("<p style='font-size: 1.1em;'>The app collects key health data such as age, gender, height, weight, daily calories, daily steps, alcohol intake, and smoking status. With a single click on <b>Analyze My Health</b>, it generates personalized health recommendations, displays body composition metrics like body fat %, muscle mass %, and visceral fat level, and presents comparative charts of your daily activity and lifestyle habits. Users can also download a detailed PDF containing their results and tailored advice.</p>", unsafe_allow_html=True)
        with st.form("health_form"):
            age = st.slider("Age", 18, 100, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            height = st.number_input("Height (cm)", 120, 250, 170)
            weight = st.number_input("Weight (kg)", 30, 200, 70)
            calories = st.number_input("Calories/day", 1000, 5000, 2500)
            steps = st.number_input("Steps per day", 0, 30000, 6000)
            alcohol = st.slider("Alcohol (g/day)", 0, 100, 10)
            smoker = st.radio("Do you smoke?", ["No", "Yes"])
            analyze = st.form_submit_button("Analyze My Health")

        if analyze:
            bmi = weight / ((height / 100) ** 2)
            body_fat = (1.2 * bmi) + (0.23 * age) - 10.8 * (1 if gender == "Male" else 0) - 5.4
            muscle_mass = 100 - body_fat - 15
            visceral_fat = round((weight / (height / 100)) + (1 if smoker == "Yes" else 0), 1)

            user_input = {
                "Age": age,
                "Gender": gender,
                "Gender_num": 0 if gender == "Male" else 1,
                "Height_cm": height,
                "Weight_kg": weight,
                "Calories_per_day": calories,
                "Steps_per_day": steps,
                "Alcohol_grams_per_day": alcohol,
                "Smoker": 1 if smoker == "Yes" else 0,
                "Body_Fat_Percent": round(body_fat, 1),
                "Muscle_Mass_Percent": round(muscle_mass, 1),
                "Visceral_Fat_Level": visceral_fat
            }

            user_df = pd.DataFrame([user_input])
            user_scaled = scaler.transform(user_df[features])
            risk_cluster = kmeans.predict(user_scaled)[0]
            risk = cluster_map[risk_cluster]

            st.success(f"Your Predicted Health Category: **{risk}**")

            # --- LLM-generated recommendations section (now with Groq) ---
            st.markdown("### Personalized Recommendations")
            with st.spinner("Generating personalized advice..."):
                prompt = (
                    f"You are a helpful health coach. Provide a detailed, encouraging, and easy-to-understand summary of a user's health status based on the following data, with a predicted health category of '{risk}'. "
                    f"Offer 3-5 actionable and friendly tips to help them improve their health based on their data. "
                    f"Do not use bullet points, use a conversational, paragraph-based format."
                    f"User Data: Age: {age}, Gender: {gender}, Weight: {weight}kg, Height: {height}cm, "
                    f"Daily Calories: {calories}kcal, Daily Steps: {steps}, "
                    f"Alcohol Intake: {alcohol}g, Smoking Status: {smoker}. "
                )
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "system", "content": "You are a helpful health coach."}, {"role": "user", "content": prompt}],
                        model=GROQ_MODEL,
                        max_tokens=1000,
                        temperature=0.7,
                    )
                    response = chat_completion.choices[0].message.content
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error generating LLM response: {e}")
                    st.info("Falling back to standard recommendations.")
                    recommendations = get_recommendations(user_input)
                    for tip in recommendations:
                        st.markdown(f"- {tip}")

            st.markdown("---")
            st.subheader("Your Body Composition")
            draw_body_chart(user_input)
            st.markdown("""
            **Definitions:**
            - **Body Fat %**: Percentage of your body that is fat, estimated using BMI, age, and gender.
            - **Muscle Mass %**: Estimated percentage of your muscle mass.
            - **Visceral Fat Level**: Fat surrounding your organs, estimated via weight-to-height ratio and smoking status.
            """)

            st.markdown("---")
            st.subheader("Daily Activity & Lifestyle Metrics")
            show_additional_charts(user_input)
            st.markdown("""
            **Comparisons:**
            - Calories/day: Recommended <= 2500 kcal
            - Steps/day: Recommended >= 7,000 steps
            - Alcohol intake: Recommended <= 20g/day
            """)

            st.markdown("---")
            st.subheader("Download Your Report")
            generate_pdf(user_input, risk, get_recommendations(user_input))

    elif page == "Calorie & Diet Planner":
        st.title("Calorie & Diet Planner")
        st.markdown("<p style='font-size: 1.1em;'>The Calorie & Diet Planner calculates your ideal daily calorie intake and macronutrient breakdown. It then generates a personalized meal plan with exact grams of carbohydrates, protein, and fats for breakfast, lunch, and dinner—making it easy to stay on track and reach your goals.</p>", unsafe_allow_html=True)
        st.info("You can also ask our Health Coach for a custom meal plan.")

        with st.form("diet_form"):
            age = st.slider("Age", 18, 100, 43)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            height = st.number_input("Height (cm)", 120, 250, 170)
            weight = st.number_input("Weight (kg)", 30, 200, 87)
            activity_level = st.selectbox("Activity Level", [
                "Sedentary (little to no exercise)",
                "Lightly active (light exercise 1-3 days in a week)",
                "Moderately active (moderate exercise 3-5 days in a week)",
                "Very active (hard exercise 6-7 days in a week)",
                "Super active (very hard exercise & physical job)"
            ], index=0)
            goal = st.selectbox("Goal", ["Lose Weight", "Maintain Weight", "Gain Weight"], index=0)
            
            carb_preference = st.selectbox("Diet Type", ["Low Fibre", "Low Carb", "Keto"], index=0)
            
            submit = st.form_submit_button("Calculate")

        if submit:
            bmr = calculate_bmr(weight, height, age, gender)
            tdee = calculate_tdee(bmr, activity_level)
            
            target_calories = tdee - 500 if goal == "Lose Weight" else tdee + 500 if goal == "Gain Weight" else tdee

            if carb_preference == "Keto":
                target_protein = (target_calories * 0.25) / 4
                target_fat = (target_calories * 0.70) / 9
                target_carbs = (target_calories * 0.05) / 4
            elif carb_preference == "Low Carb":
                target_protein = (target_calories * 0.40) / 4
                target_fat = (target_calories * 0.40) / 9
                target_carbs = (target_calories * 0.20) / 4
            else:
                target_protein = (target_calories * 0.30) / 4
                target_carbs = (target_calories * 0.40) / 4
                target_fat = (target_calories * 0.30) / 9
            
            st.markdown(f"### Your Estimated Calorie Needs: **{int(target_calories)} kcal/day**")
            st.markdown(f"**Macro Breakdown:**")
            st.write(f"- Carbohydrates: {int(target_carbs)} g/day")
            st.write(f"- Protein: {int(target_protein)} g/day")
            st.write(f"- Fat: {int(target_fat)} g/day")
            
            st.markdown("### **Feel free to choose any of the three diet plans generated below**")

            meal_plans = generate_custom_meal_plans(target_calories, target_protein, target_fat, target_carbs)
            c = 1 

            for plan_name, plan_data in meal_plans.items():
                st.markdown("---")
                st.markdown(f"### **Diet no. {c}: {plan_name}**")
                st.markdown(f"**Daily Totals:** {int(target_calories)} kcal | {int(target_protein)} g Protein | {int(target_fat)} g Fat | {int(target_carbs)} g Carbs")
                for meal_name, meal_info in plan_data.items():
                    st.markdown(f"#### **{meal_name}**")
                    st.markdown(f"- **Food:** {meal_info['Food']}")
                    
                    meal_p = 0
                    meal_f = 0
                    meal_c = 0
                    meal_cal = 0
                    
                    if isinstance(meal_info['Weights'], dict):
                        for item, weight_str in meal_info['Weights'].items():
                            food_key = item.lower().replace(' (cooked)', '').replace(' (90/10)', '').replace(' (93/7)', '').replace(' ', '_')
                            
                            if food_key == 'large_eggs':
                                num_eggs = int(weight_str.split()[0])
                                meal_p += num_eggs * food_db['large_egg']['protein']
                                meal_f += num_eggs * food_db['large_egg']['fat']
                                meal_c += num_eggs * food_db['large_egg']['carbs']
                                meal_cal += num_eggs * food_db['large_egg']['calories']
                            else:
                                if weight_str != "N/A":
                                    weight_g = int(weight_str.split()[0])
                                    if food_key in food_db:
                                        meal_p += (food_db[food_key]['protein'] / 100) * weight_g
                                        meal_f += (food_db[food_key]['fat'] / 100) * weight_g
                                        meal_c += (food_db[food_key]['carbs'] / 100) * weight_g
                                        meal_cal += (food_db[food_key]['calories'] / 100) * weight_g
                    else:
                        meal_p += food_db['protein_shake']['protein']
                        meal_f += food_db['protein_shake']['fat']
                        meal_c += food_db['protein_shake']['carbs']
                        meal_cal += food_db['protein_shake']['calories']

                    if isinstance(meal_info['Weights'], dict):
                        for item, weight in meal_info['Weights'].items():
                            st.markdown(f"   - **{item}:** {weight}")
                    else:
                        st.markdown(f"   - **Weights:** {meal_info['Weights']}")
                    
                    st.markdown(f"   - **Macros:** {int(meal_cal)} kcal, {int(meal_p)}g P, {int(meal_f)}g F, {int(meal_c)}g C")
                c = c + 1

    elif page == "Meal Search":
        st.title("Nutrition Meal Search")
        st.markdown("<p style='font-size: 1.1em;'>A powerful food search engine that lets you instantly find nutritional information for thousands of food items. Simply type the name of a food, and you’ll get detailed facts including carbohydrates, protein, fats, calories, and standard serving sizes. Whether you’re tracking macros, planning meals, or just curious about what’s in your food, this tool makes it quick and easy to get accurate, reliable nutrition data at your fingertips.😀</p>", unsafe_allow_html=True)

        search_query = st.text_input("Enter a food item", "")
        search_button = st.button("Search")

        if search_button and search_query.strip() != "":
            headers = {
                "x-app-id": NUTRITIONIX_APP_ID,
                "x-app-key": NUTRITIONIX_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {"query": search_query}

            try:
                response = requests.post(NUTRITIONIX_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                if "foods" in data and len(data["foods"]) > 0:
                    for food in data["foods"]:
                        st.subheader(food["food_name"].title())
                        st.write(f"Calories: {food.get('nf_calories', 'N/A')} kcal")
                        st.write(f"Protein: {food.get('nf_protein', 'N/A')} g")
                        st.write(f"Fat: {food.get('nf_total_fat', 'N/A')} g")
                        st.write(f"Carbohydrates: {food.get('nf_total_carbohydrate', 'N/A')} g")
                        st.write(f"Serving Size: {food.get('serving_qty', 'N/A')} {food.get('serving_unit', 'N/A')}")
                        st.image(food.get("photo", {}).get("thumb", ""), use_container_width=True)
                else:
                    st.warning("No results found. Please try a different food item.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while fetching data from the API: {e}")
                
    elif page == "Health Coach":
        st.title("Your Personal Health Coach")
        st.markdown("<p style='font-size: 1.1em;'>Interact with a personalized health coach to get answers to your fitness and nutrition questions. Ask about meal plans, exercise routines, or general health advice. Our coach is powered by a large language model and provides helpful, informed responses to guide you on your wellness journey.</p>", unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask your health coach..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                try:
                    # Get conversation history for the chat completion request
                    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    
                    # Make the Groq API call
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=GROQ_MODEL,
                        max_tokens=1000,
                        temperature=0.7,
                    )
                    
                    response_text = chat_completion.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.markdown(response_text)

                except Exception as e:
                    st.error(f"Error communicating with the health coach: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I'm having trouble connecting right now. Please try again later."})



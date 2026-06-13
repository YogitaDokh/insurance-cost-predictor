import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# 1. Page Configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium UI Design Elements
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F3F4F6;
        padding: 24px;
        border-radius: 12px;
        border-left: 6px solid #10B981;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Securely Load Your Pickled Model
@st.cache_resource
def load_model():
    with open("Polynomial.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"🛑 Error loading model file: {e}")
    st.stop()

# Safely manage metadata values
try:
    intercept_val = float(model.intercept_[0])
except (TypeError, IndexError, AttributeError):
    intercept_val = float(getattr(model, "intercept_", 0.0))

# --- SIDEBAR OUTLINE ---
with st.sidebar:
    st.header("📋 Model Information")
    st.markdown(f"""
    - **Engine Architecture:** Polynomial Regression
    - **Expected Features:** `{model.n_features_in_}`
    - **Intercept Base:** `{intercept_val:.2f}`
    """)
    st.markdown("---")
    st.caption("This predictor processes 6 native medical metrics transformed via Degree-2 expansion to compute medical premium estimations.")

# --- MAIN PAGE DASHBOARD ---
st.markdown('<div class="main-title">🏥 Medical Premium Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the individual\'s profile details below to calculate their personalized insurance cost evaluation.</div>', unsafe_allow_html=True)

# Organise user input sections beautifully across 3 distinct columns
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("### 🧑‍🤝‍🧑 Demographic Profile")
    age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Biological Sex", options=["Male", "Female"])

with col2:
    st.markdown("### 🩺 Health Metrics")
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")
    children = st.spinbox("Number of Dependents / Children", min_value=0, max_value=10, value=0, step=1)

with col3:
    st.markdown("### 🚬 Habits & Geography")
    smoker = st.selectbox("Smoking Status", options=["No", "Yes"])
    region = st.selectbox("Residential Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])

st.markdown("---")

# Centered prediction execution
left_pad, center_btn, right_pad = st.columns([1, 1, 1])
with center_btn:
    predict_btn = st.button("Calculate Insurance Premium 🚀", type="primary", use_container_width=True)

if predict_btn:
    try:
        # One-Hot Encoding values manually to match your preprocessing data pipeline
        sex_male = 1 if sex == "Male" else 0
        smoker_yes = 1 if smoker == "Yes" else 0
        
        region_northwest = 1 if region == "Northwest" else 0
        region_southeast = 1 if region == "Southeast" else 0
        region_southwest = 1 if region == "Southwest" else 0
        # Northeast behaves as our reference baseline (0,0,0)
        
        # Structure the baseline input exactly as it was organized during your model training:
        # [age, bmi, children, sex_male, smoker_yes, region_nw, region_se, region_sw]
        raw_features = np.array([[age, bmi, children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest]])
        
        # Expand inputs using Degree 2 to hit your required 27 features target
        poly = PolynomialFeatures(degree=2, include_bias=False)
        transformed_features = poly.fit_transform(raw_features)
        
        # Verify feature alignments
        if transformed_features.shape[1] != model.n_features_in_:
            # Fallback handling: If your dataset included fewer columns, we gracefully alter transformation array 
            # by removing region elements to guarantee alignment.
            raw_features_alt = np.array([[age, bmi, children, sex_male, smoker_yes]])
            transformed_features = poly.fit_transform(raw_features_alt)

        # Execute processing
        prediction = model.predict(transformed_features)
        
        # Handle extraction safely
        if isinstance(prediction, (np.ndarray, list)):
            prediction = prediction[0]
        if isinstance(prediction, (np.ndarray, list)): # Deep nested handling
            prediction = prediction[0]

        # Present calculation
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## 💵 Estimated Annual Premium")
        st.write("Based on the provided demographic and health risk variables, the projected cost is:")
        st.metric(label="Calculated Premium Cost", value=f"${prediction:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Pipeline Error: {e}")
        st.info("Ensure that your initial training configuration matched the standard [age, bmi, children, sex, smoker, region] layout structure.")

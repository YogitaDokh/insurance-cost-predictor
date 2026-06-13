import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# 1. Page Configuration (Must be the very first Streamlit command)
st.set_page_config(
    page_title="PolyPredict Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom CSS for modern UI styling (Fixed argument syntax)
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Cached Model Loader
@st.cache_resource
def load_model():
    with open("Polynomial.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"🛑 Error loading model file: {e}")
    st.stop()

# --- SIDEBAR (Settings & Model Specs) ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # User-adjustable expected polynomial degree configuration
    DEGREE = st.number_input("Polynomial Degree Used:", min_value=1, max_value=10, value=2, step=1)
    
    st.markdown("---")
    st.header("📋 Model Metadata")
    st.markdown(f"""
    - **Architecture:** Linear Regression
    - **Required Features:** `{model.n_features_in_}`
    - **Intercept:** `{model.intercept_[0]:.4f}`
    - **Sklearn Engine:** `v{model._sklearn_version}`
    """)
    
    st.caption("Ensure your input degree math correctly yields the number of features expected above.")

# --- MAIN PAGE ---
st.markdown('<div class="main-title">📊 PolyPredict Pro</div>', unsafe_allowed_html=True)
st.markdown('<div class="subtitle">An interactive, high-performance portal for evaluating polynomial regression models.</div>', unsafe_allowed_html=True)

# Layout: Split into Input/Output section and Visual Chart section
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### 🔮 Input Feature")
    st.write("Provide a baseline value for $X$ to compute the transformed polynomial response variable.")
    
    # Input container card
    with st.container():
        raw_input = st.number_input(
            "Enter Raw Input Value ($X$):", 
            value=1.0, 
            step=0.5,
            format="%.4f"
        )
        
        predict_btn = st.button("Generate Prediction ✨", type="primary", use_container_width=True)

    st.markdown("---")
    
    # Process Prediction
    prediction = None  # Placeholder variable for graphing visibility
    if predict_btn or raw_input:
        try:
            # Recreate transformation matrix
            X_custom = np.array([[raw_input]])
            poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
            X_poly = poly.fit_transform(X_custom)
            
            # Check dimension compliance
            if X_poly.shape[1] != model.n_features_in_:
                st.error(f"❌ **Feature Mismatch!** Degree `{DEGREE}` yields `{X_poly.shape[1]}` inputs, but your model explicitly demands `{model.n_features_in_}` features. Please check your Degree settings in the sidebar.")
            else:
                # Predict
                prediction = model.predict(X_poly)[0]
                
                # Success Display Card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🎉 Calculated Result")
                st.metric(
                    label=f"Predicted Output $Y$ (at $X$ = {raw_input:.2f})", 
                    value=f"{prediction:.6f}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An evaluation anomaly occurred: {e}")

with col2:
    st.markdown("### 📈 Curve Visualization")
    st.write("Understand the nature of your polynomial trajectory around your current input point.")
    
    # Generate a plot around the user's selected input range
    if prediction is not None:
        try:
            poly_plot = PolynomialFeatures(degree=DEGREE, include_bias=False)
            
            # Draw line profile
            x_range = np.linspace(raw_input - 5, raw_input + 5, 200).reshape(-1, 1)
            x_poly_range = poly_plot.fit_transform(x_range)
            y_range = model.predict(x_poly_range)
            
            # Matplotlib styling
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_range, y_range, color="#3B82F6", linewidth=2.5, label="Model Fit")
            ax.scatter(raw_input, prediction, color="#EF4444", s=120, zorder=5, label=f"Current Input ({raw_input:.2f})")
            
            ax.set_title("Local Polynomial Trajectory Profile", fontsize=10, fontweight="bold", color="#1F2937")
            ax.set_xlabel("Input Range (X)", fontsize=8)
            ax.set_ylabel("Predicted Value (Y)", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(frameon=True, facecolor="#F9FAFB")
            
            st.pyplot(fig)
        except Exception:
            st.info("💡 Adjust the 'Polynomial Degree' parameter in the sidebar to activate interactive curve plotting.")
    else:
        st.info("💡 Plotting space will initialize as soon as accurate degree match requirements are satisfied.")

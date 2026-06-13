# 🏥 Medical Premium Predictor (PolyPredict Pro)

An AI-powered Medical Insurance Cost Prediction system built using a trained **Polynomial Regression** model. The application leverages advanced feature engineering, dynamic polynomial transformations, and intelligent fallback mechanisms to deliver reliable premium cost estimations through an intuitive and responsive web interface.

---

## 🚀 Live Demo

Experience the application directly from your browser:

👉 **[Launch PolyPredict Pro](https://fmqofz32gnrljn9jaax7sr.streamlit.app/)**

---

## 🎯 Application Workflow

### Step 1: Demographic Information
- Select the beneficiary's age using the interactive slider.
- Choose the biological sex.

### Step 2: Health Metrics
- Enter the Body Mass Index (BMI).
- Specify the number of dependent children covered under the policy.

### Step 3: Lifestyle & Region
- Select smoking status.
- Choose the applicable geographical region.

### Step 4: Generate Prediction
- Click **Generate Cost Forecast ✨**
- The application automatically transforms the provided inputs into the required polynomial feature space and predicts the estimated medical insurance premium.

---

## ✨ Key Features

### 🔹 Intelligent Feature Recovery Engine
The system dynamically supports multiple feature structures and automatically aligns incoming data with the model's expected schema using:

- 6-column feature mapping
- 5-column fallback alignment
- 8-column raw structure handling

This ensures maximum compatibility with the serialized model metadata.

### 🔹 Interactive Premium Dashboard
- Modern responsive UI
- Multi-column layout design
- Enhanced typography using Google Inter Font
- Real-time prediction rendering

### 🔹 Model Metadata Explorer
The sidebar provides important model information such as:

- Model Type
- Expected Feature Count
- Input Structure
- Training Intercept
- Diagnostic Parameters

### 🔹 Robust Error Handling
Built-in exception management prevents application crashes and provides user-friendly guidance when invalid input structures are detected.

---

## 📐 Polynomial Feature Engineering

The trained model expects a feature vector containing **27 engineered features**.

### Raw Input Variables

```text
[Age, BMI, Children, Sex, Smoker, Region]
```

These six base features are transformed using:

```python
PolynomialFeatures(degree=2, include_bias=False)
```

The transformation generates:

### Base Features

```text
6 Features
```

### Squared Features

```text
6 Features
```

### Interaction Features

```text
15 Features
```

### Total Feature Count

```text
6 + 6 + 15 = 27 Features
```

This engineered feature space allows the regression model to capture nonlinear relationships and interaction effects among insurance variables.

---

## 🛠️ Project Structure

```text
insurance-cost-predictor/
│
├── app.py
├── Polynomial.pkl
├── requirements.txt
└── README.md
```

---

## 💻 Local Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/insurance-cost-predictor.git
cd insurance-cost-predictor
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🌐 Local Access

Once the server starts successfully, open:

```text
http://localhost:8501
```

in your browser.

---

## 📦 Dependencies

The project relies on the following libraries:

| Package | Purpose |
|----------|----------|
| Streamlit | Web Application Framework |
| Scikit-Learn | Polynomial Regression & Feature Transformation |
| NumPy | Numerical Computation |
| Pandas | Data Processing & Manipulation |

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🔍 Model Overview

**Algorithm:** Polynomial Regression

The model predicts medical insurance premiums using demographic, health, and lifestyle factors. Polynomial feature engineering enables the model to learn complex nonlinear relationships between variables and insurance costs.

---

## 📈 Prediction Inputs

| Feature | Description |
|----------|-------------|
| Age | Beneficiary Age |
| BMI | Body Mass Index |
| Children | Number of Dependents |
| Sex | Male / Female |
| Smoker | Yes / No |
| Region | Geographic Region |

---

## 📜 License

This project is intended for educational, academic, and portfolio demonstration purposes.

---

### ⭐ If you found this project useful, consider giving the repository a star!

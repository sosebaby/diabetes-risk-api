# 🩺 Diabetes Risk Prediction API

A machine learning API that predicts the risk of diabetes based on patient clinical data. Built with **FastAPI** and a **Random Forest Classifier** trained on the Pima Indians Diabetes Dataset.

---

## 👥 Users

| User Type | Description | Expected Usage |
|---|---|---|
| **Clinical Developer** | Hospital or clinic software teams integrating the API into patient management systems | Real-time predictions per patient consultation |
| **Healthcare Researcher** | Researchers running batch screening on patient cohorts | Occasional bulk use for population health studies |
| **System Administrator** | DevOps team managing the deployment | Monitoring uptime and model health |

**Expected daily request volume:** ~500–1,000 requests per day for a small clinic integration.

**User requirements:** Real-time responses (< 1 second), JSON input/output, no authentication required for this version.

---

## 🏗️ How Users Interact With This Service

```
                        ┌─────────────────────────────────┐
                        │     Diabetes Risk Prediction API │
                        │                                  │
 API Client  ──POST──►  │  1. Validate Input (Pydantic)   │
 (Hospital   ◄──JSON──  │  2. Build DataFrame              │
  Portal)               │  3. Run Random Forest Model      │
                        │  4. Return Risk Score & Category │
                        └─────────────────────────────────┘
```

**Input:** 8 patient clinical features as JSON  
**Output:** Diabetes prediction, probability score, and risk category (Low / Moderate / High)

---

## 📁 Project Structure

```
├── main.py               # FastAPI application and endpoints
├── train_model.py        # Model training script
├── diabetes.csv          # Pima Indians Diabetes Dataset
├── diabetes_model.pkl    # Trained Random Forest model
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
└── README.md             # This file
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the model

The trained model (`diabetes_model.pkl`) is already included. If you want to retrain it from scratch:

```bash
python3 train_model.py
```

### 4. Run the API

```bash
python3 -m uvicorn main:app --reload
```

The API will be running at: `http://localhost:8000`

### 5. Open the interactive docs

Go to: `http://localhost:8000/docs`

This is the auto-generated Swagger UI where you can test all endpoints directly in your browser.

---

## 🔌 API Endpoints

### `GET /`
Health check — confirms the API is running.

**Response:**
```json
{
  "message": "Diabetes Risk Prediction API is running"
}
```

---

### `POST /predict`
Submit patient clinical features and receive a diabetes risk prediction.

**Request Body:**
```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Response:**
```json
{
  "prediction": 1,
  "result": "Diabetic",
  "risk_category": "High",
  "probability": 0.89
}
```

| Field | Description |
|---|---|
| `prediction` | Binary outcome — `1` = Diabetic, `0` = Non-Diabetic |
| `result` | Human-readable label |
| `risk_category` | `Low` (< 0.33), `Moderate` (0.33–0.65), `High` (> 0.65) |
| `probability` | Model's predicted probability of diabetes (0.0 – 1.0) |

---

### Status Codes

| Code | Meaning |
|---|---|
| `200` | Prediction returned successfully |
| `422` | Validation error — check your input fields |
| `500` | Server error during model inference |

---

## 🐳 Running with Docker

### Build the image

```bash
docker build -t diabetes-risk-api .
```

### Run the container

```bash
docker run -p 8080:8080 diabetes-risk-api
```

The API will be available at: `http://localhost:8080`

---

## 🤖 Model Information

| Property | Detail |
|---|---|
| Algorithm | Random Forest Classifier |
| Dataset | Pima Indians Diabetes Dataset |
| Records | 768 patients |
| Features | 8 clinical features |
| Train / Test Split | 80% / 20% |
| Accuracy | 74% |
| Scaling | None required (Random Forest is scale-invariant) |

### Input Features

| Feature | Type | Description |
|---|---|---|
| `Pregnancies` | int | Number of pregnancies |
| `Glucose` | float | Plasma glucose concentration (mg/dL) |
| `BloodPressure` | float | Diastolic blood pressure (mmHg) |
| `SkinThickness` | float | Triceps skinfold thickness (mm) |
| `Insulin` | float | 2-hour serum insulin (µU/mL) |
| `BMI` | float | Body Mass Index (kg/m²) |
| `DiabetesPedigreeFunction` | float | Family history diabetes score |
| `Age` | int | Patient age in years |

---

## 📦 Dependencies

```
fastapi
uvicorn
pydantic
pandas
scikit-learn
joblib
```

---

## 👤 Author

**Euodia Ebalu— API Developer**  
Assignment 2 Part B — Systems Analysis and Design
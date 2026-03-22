# 🔐 Network Security - Phishing Website Detection System

## 🚀 Overview

**Network Security** is an industry-grade Machine Learning project designed to detect whether a website is **legitimate or phishing (malicious)** based on multiple engineered features extracted from URLs and web attributes.

This project implements a **complete ML lifecycle pipeline**, including:

* ETL Pipelines
* Data Validation & Transformation
* Model Training & Evaluation
* Deployment using FastAPI
* CI/CD with Docker, AWS, and GitHub Actions

---

## 🎯 Problem Statement

Phishing attacks are one of the most common cybersecurity threats. This system analyzes various URL and webpage features to classify a website as:

* ✅ Legitimate
* ❌ Phishing

---

## 📊 Dataset Features

The dataset contains multiple engineered features such as:

| Feature                     | Description                           |
| --------------------------- | ------------------------------------- |
| having_IP_Address           | Whether URL uses IP instead of domain |
| URL_Length                  | Length of the URL                     |
| Shortining_Service          | Use of URL shorteners                 |
| having_At_Symbol            | Presence of '@'                       |
| Prefix_Suffix               | Hyphen in domain                      |
| SSLfinal_State              | SSL certificate status                |
| Domain_registeration_length | Domain validity                       |
| HTTPS_token                 | Presence of HTTPS                     |
| web_traffic                 | Website traffic                       |
| Page_Rank                   | Ranking score                         |
| Google_Index                | Indexed by Google                     |
| Statistical_report          | External statistical flags            |

---

## 🧠 ML Pipeline Lifecycle

### 🔄 End-to-End Flow

```
Raw Data → ETL → MongoDB → Data Ingestion → Validation → Transformation → Model Training → Evaluation → Deployment
```

### ⚙️ Detailed Steps

1. **ETL Pipeline**

   * Extract raw data
   * Transform to required schema
   * Load into MongoDB Atlas

2. **Data Ingestion**

   * Fetch data from MongoDB

3. **Data Validation**

   * Validate schema using `schema.yaml`

4. **Data Transformation**

   * Feature engineering & preprocessing

5. **Model Training**

   * Train multiple models (XGBoost, etc.)

6. **Model Evaluation**

   * Select best model using metrics (F1, Precision, Recall)

7. **Model Pusher**

   * Save model & artifacts
   * Sync with S3 bucket

---

## 🌐 FastAPI Application

### Available Endpoints

| Endpoint   | Method | Description                     |
| ---------- | ------ | ------------------------------- |
| `/`        | GET    | Redirects to API docs           |
| `/train`   | GET    | Triggers full training pipeline |
| `/predict` | POST   | Upload CSV → Get predictions    |

---

## 🐳 Docker & Deployment

* Containerized using Docker
* Deployed on AWS EC2
* Images stored in AWS ECR

---

## 🔄 CI/CD Pipeline (GitHub Actions)

### Flow:

```
Push Code → CI → Build Docker Image → Push to ECR → Deploy on EC2
```

### Stages:

1. **Continuous Integration**

   * Code checkout
   * Linting & testing

2. **Continuous Delivery**

   * Docker build
   * Push to AWS ECR

3. **Continuous Deployment**

   * EC2 pulls latest image
   * Runs container automatically

---

## ☁️ Cloud & MLOps Integration

* **MongoDB Atlas** → Data storage
* **AWS S3** → Model/artifact storage
* **MLflow + DagsHub** → Experiment tracking
* **GitHub Actions** → CI/CD automation

---

## 🛠️ Tech Stack

### 💻 Programming & ML

* Python
* NumPy, Pandas, Matplotlib
* Scikit-learn, XGBoost

### ⚙️ Backend & APIs

* FastAPI
* Uvicorn

### 🧠 MLOps & Tracking

* MLflow
* DagsHub

### 🗄️ Database

* MongoDB (Atlas)

### ☁️ Cloud

* AWS EC2
* AWS ECR
* AWS S3

### 🔄 DevOps

* Docker
* GitHub Actions

---

## 📁 Project Structure

```
NetworkSecurity_Project/
│
├── .github/workflows/
│   └── main.yml                # CI/CD Pipeline
│
├── templates/
│   └── table.html             # HTML template for prediction output
│
├── valid_data/                # Sample test data
├── notebooks/                 # EDA & experimentation
├── data_schema/               # Schema validation config
│   └── schema.yaml
│
├── src/                       # Core project logic
│   ├── cloud/                 # S3 syncing
│   ├── pipelines/             # Training pipeline
│   ├── components/            # ML lifecycle components
│   ├── utils/                 # Helper functions
│   ├── constants/             # Config constants
│   ├── entity/                # Config classes
│   ├── exception/             # Custom exception handling
│   └── logging/               # Logging system
│
├── Data/                      # Training dataset
├── app.py                     # FastAPI entry point
├── main.py                    # Local training script
├── Dockerfile                 # Container configuration
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
├── push_data.py               # Upload data to MongoDB
├── test_mongo.py              # MongoDB testing
└── README.md
```

---

## ✨ Key Features

* ✅ End-to-End ML Pipeline
* ✅ ETL + Data Validation
* ✅ Custom Logging & Exception Handling
* ✅ FastAPI for real-time prediction
* ✅ Dockerized deployment
* ✅ CI/CD automation
* ✅ Cloud integration (AWS + MongoDB)
* ✅ Experiment tracking (MLflow + DagsHub)

---

## 🧪 How to Run Locally

```bash
# Clone repo
git clone <repo_url>

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

---

## 🌍 Demo

🔗 Demo Link: - http://15.207.51.82:8080/docs

---

## 🤝 Contribution

Feel free to fork, raise issues, or contribute to improve this project.

---

## 📌 Conclusion

This project demonstrates a **complete industry-level ML system**, combining:

* Data Engineering
* Machine Learning
* Backend APIs
* Cloud Deployment
* CI/CD Automation

---

💡 *Built to simulate real-world ML production systems with scalability and maintainability in mind.*

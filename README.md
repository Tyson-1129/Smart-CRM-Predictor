# Smart-CRM-Predictor

An automated, 24/7 machine learning pipeline that ingests incoming real estate inquiries, predicts buyer intent probability, and orchestrates an intelligent routing system to log CRM data and dispatch actionable alerts to brokers. 

This project demonstrates an end-to-end data science workflow: from training a Scikit-learn classification model to deploying a production-ready Flask API hosted in the cloud, fully integrated with a robust workflow automation architecture.

## System Architecture

The pipeline is entirely autonomous and highly resilient, utilizing error-handling directives (auto-retry and resume) to guarantee zero data loss during server timeouts.

1. **Data Ingestion:** Leads arrive via a webhook (simulating a front-end property portal).
2. **Inference API (Flask + PythonAnywhere):** The payload is routed to a custom Flask API hosted on PythonAnywhere. The application processes the data, handles missing values, and passes it to a pre-trained machine learning model.
3. **Machine Learning Model (Scikit-learn):** A Random Forest classifier evaluates features such as `budget_lakhs`, `property_type`, `time_on_page_sec`, and `site_visit_requested` to generate an `intent_score` (probability) and a binary classification (High/Low Intent). 
4. **Orchestration & Routing (Make.com):** Make.com captures the JSON response and routes the data based on the model's classification.
5. **CRM Logging (Google Sheets):** Every lead, regardless of intent, is logged into a Google Sheets CRM for historical tracking and broker outcome updates.
6. **Actionable Alerts (Gmail):** High-intent buyers trigger a dynamically generated, dark-mode HTML dashboard sent to the broker. The email includes the intent score, property requirements, extracted priority reasons, and a custom-generated call script.

## Repository Structure

* `app.py`: The production Flask application containing the web server logic, data extraction, validation, and JSON response formatting[cite: 25].
* `lead_scorer.py`: The core machine learning script detailing the data preprocessing, feature engineering, pipeline creation, and model training phases[cite: 25]. 
* `requirements.txt`: The dependency file for cloud deployment (includes Flask, Scikit-learn, pandas, and joblib)[cite: 25].

## Business Impact

* **Prioritization:** Dynamically filters low-intent window shoppers from high-priority buyers, allowing sales teams to allocate their time effectively.
* **Speed to Lead:** Reduces response time to seconds for hot leads, supplying brokers with an immediate, data-backed conversation opener.
* **Resilience:** Built-in error handling ensures the CRM remains up-to-date even if external APIs experience temporary latency.

## Setup & Deployment

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model by running `lead_scorer.py` to generate the `.pkl` pipeline file.
4. Deploy `app.py` and the exported `.pkl` file to a WSGI server (e.g., PythonAnywhere).
5. Point your Make.com or Zapier HTTP module to the deployed `/predict_lead` endpoint.

# Smart-CRM-Predictor (B2B SaaS)

An automated, 24/7 machine learning pipeline and SaaS platform designed for real estate agencies. It allows brokers to register for an AI assistant that seamlessly intercepts their incoming lead emails, predicts buyer intent probability, and dispatches actionable VIP alerts to prioritize high-value clients.

This project demonstrates an end-to-end data science workflow: from training a Scikit-learn classification model to deploying a production-ready Flask API hosted in the cloud, fully integrated with a dual-scenario Make.com automation architecture.

## System Architecture

The platform operates on two distinct, highly resilient workflows:

### Workflow A: Broker Onboarding (Frontend)
1. **SaaS Registration:** Brokers sign up via a modern, dark-mode web portal (`index.html`).
2. **Provisioning (Make.com Webhook):** The signup data is routed to a Make.com webhook. The system logs the new client into the master database and automatically emails them setup instructions, providing a unique "Mailhook" address to forward their leads to.

### Workflow B: The Core ML Pipeline
1. **Data Ingestion (Email Forwarding):** Leads arrive via the broker's auto-forwarded emails directly into the system's dedicated Mailhook.
2. **Inference API (Flask + PythonAnywhere):** The raw email text is routed to a custom Flask API. The application uses regex extraction to parse the lead details, handles missing values, and passes the data to the machine learning model.
3. **Machine Learning Model (Scikit-learn):** A Random Forest classifier evaluates features such as `budget_lakhs`, `property_type`, `time_on_page_sec`, and `site_visit_requested` to generate an `intent_score` (probability) and a binary classification (High/Low Intent). 
4. **CRM Logging (Google Sheets):** Every lead, regardless of intent, is logged into a Google Sheets CRM for historical tracking.
5. **Actionable Alerts (Gmail):** High-intent buyers trigger a dynamically generated HTML dashboard sent directly back to the broker. The email includes the intent score, property requirements, and a custom-generated call script.

## Repository Structure

* `index.html`: The SaaS registration front-end portal built with HTML and Tailwind CSS.
* `app.py`: The production Flask application containing the web server logic, regex data extraction, validation, and JSON response formatting.
* `lead_scorer.py`: The core machine learning script detailing the data preprocessing, feature engineering, pipeline creation, and model training phases. 
* `requirements.txt`: The dependency file for cloud deployment (includes Flask, Scikit-learn, pandas, and joblib).

## Business Impact

* **Frictionless Onboarding:** Brokers can connect their existing lead flow to the AI in 60 seconds using standard email forwarding, bypassing complex API integrations.
* **Prioritization:** Dynamically filters low-intent window shoppers from high-priority buyers, allowing sales teams to allocate their time effectively.
* **Speed to Lead:** Reduces response time to seconds for hot leads, supplying brokers with an immediate, data-backed conversation opener.

## Setup & Deployment

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model by running `lead_scorer.py` to generate the `.pkl` pipeline file.
4. Deploy `app.py` and the exported `.pkl` file to a WSGI server (e.g., PythonAnywhere).
5. Host `index.html` via GitHub Pages or Vercel for the client-facing registration portal.
6. Configure Make.com (or Zapier) to route the webhooks and mailhooks to your deployed `/predict_lead` endpoint.

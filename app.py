import re
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Load the trained pipeline once at server startup
pipeline = joblib.load('lead_scorer_pipeline.pkl')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "Lead Scorer API"}), 200

@app.route('/predict_lead', methods=['POST'])
def predict_lead():
    data = request.get_json(force=True)

    # 2. Check if the payload contains raw email text from Make.com Mailhook
    if 'email_text' in data:
        text = data['email_text']
        
        # Extract fields using Regular Expressions
        prop_match = re.search(r"Property:\s*(.*)", text, re.IGNORECASE)
        budget_match = re.search(r"Budget:\s*(\d+)", text, re.IGNORECASE)
        hour_match = re.search(r"Contact Hour:\s*(\d+)", text, re.IGNORECASE)
        time_match = re.search(r"Time on Page:\s*(\d+)", text, re.IGNORECASE)
        visit_match = re.search(r"Site Visit:\s*(\d+)", text, re.IGNORECASE)
        email_match = re.search(r"Email:\s*([^\s]+)", text, re.IGNORECASE)

        # DATA VALIDATION: Check if core fields are completely missing
        if not prop_match and not budget_match and not time_match:
            return jsonify({
                "status": "incomplete",
                "lead_status": "Insufficient Data",
                "message": "Lead assessment unavailable — data incomplete. Property type, budget, engagement metrics, and contact details could not be extracted reliably. Please review the original enquiry."
            }), 200  # Return 200 so Make.com doesn't throw a connection error

        # Assign extracted values or default fallbacks if partially missing
        property_type = prop_match.group(1).strip() if prop_match else "Unknown"
        budget_lakhs = float(budget_match.group(1)) if budget_match else 0.0
        contact_hour = int(hour_match.group(1)) if hour_match else 12
        time_on_page_sec = float(time_match.group(1)) if time_match else 0.0
        site_visit_requested = int(visit_match.group(1)) if visit_match else 0
        lead_email = email_match.group(1).strip() if email_match else "No Email Provided"

    # Fallback to discrete JSON validation
    else:
        required_fields = ['property_type', 'budget_lakhs', 'contact_hour', 'time_on_page_sec', 'site_visit_requested']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "incomplete",
                    "lead_status": "Insufficient Data",
                    "message": f"Lead assessment unavailable — data incomplete. Missing {field}."
                }), 200
        
        property_type = data['property_type']
        budget_lakhs = data['budget_lakhs']
        contact_hour = data['contact_hour']
        time_on_page_sec = data['time_on_page_sec']
        site_visit_requested = data['site_visit_requested']
        lead_email = data.get('lead_email', "No Email Provided")

    # 3. Convert extracted variables to a single-row DataFrame for the model
    lead_df = pd.DataFrame([{
        'property_type': str(property_type).strip(),
        'budget_lakhs': float(budget_lakhs),
        'contact_hour': int(contact_hour),
        'time_on_page_sec': float(time_on_page_sec),
        'site_visit_requested': int(site_visit_requested)
    }])

    # 4. Predict intent probability and binary status
    hot_probability = float(pipeline.predict_proba(lead_df)[0][1])
    
    # 5. Generate Dynamic Actionable Insights
    priority_reasons = []
    if site_visit_requested == 1:
        priority_reasons.append("Site visit request was detected.")
    if budget_lakhs > 0 and property_type != "Unknown":
        priority_reasons.append("Budget and property type were extracted successfully.")
    priority_reasons.append("Other signals should be confirmed from the original enquiry.")

    if hot_probability >= 0.65:
        lead_status = "High intent"
        suggested_action = "Call promptly"
    else:
        lead_status = "Low intent"
        suggested_action = "Review or nurture"

    suggested_opening = f"Hello, I'm calling about your enquiry for a {property_type} property. I wanted to understand your preferred location and arrange a suitable site visit."

    # Format budget dynamically (Crores, Lakhs, or Thousands)
    if budget_lakhs >= 100:
        val = budget_lakhs / 100
        formatted_budget = f"{int(val) if val.is_integer() else round(val, 2)} Crore"
    elif budget_lakhs >= 1:
        formatted_budget = f"{int(budget_lakhs) if budget_lakhs.is_integer() else round(budget_lakhs, 2)} Lakh"
    elif budget_lakhs > 0:
        val = budget_lakhs * 100
        formatted_budget = f"{int(val) if val.is_integer() else round(val, 2)} Thousand"
    else:
        formatted_budget = "Not specified"
    
    return jsonify({
        "status": "success",
        "lead_status": lead_status,
        "intent_score": round(hot_probability * 100, 1),
        "suggested_action": suggested_action,
        "priority_reasons": priority_reasons,
        "suggested_opening": suggested_opening,
        "property_type": property_type,
        "formatted_budget": formatted_budget, # <--- NEW VARIABLE
        "budget_lakhs": budget_lakhs,
        "time_on_page_sec": time_on_page_sec,
        "site_visit_requested": "Requested" if site_visit_requested == 1 else "Not requested",
        "lead_email": lead_email
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
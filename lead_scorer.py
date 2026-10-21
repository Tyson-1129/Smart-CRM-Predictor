import pandas as pd
import numpy as np

np.random.seed(42)
num_leads=500

# 1. Generate realistic lead inputs
property_types = np.random.choice(['1BHK', '2BHK', '3BHK', 'Villa', 'Commercial'], num_leads, p=[0.25, 0.35, 0.20, 0.10, 0.10])
contact_hours = np.random.choice(range(24), num_leads)

# Realistic budget assignment per property type (in Lakhs)
base_budgets = {
    '1BHK': np.random.normal(45, 10, num_leads),
    '2BHK': np.random.normal(85, 15, num_leads),
    '3BHK': np.random.normal(140, 25, num_leads),
    'Villa': np.random.normal(250, 40, num_leads),
    'Commercial': np.random.normal(180, 50, num_leads)
}

budgets = [round(max(15, base_budgets[pt][i])) for i, pt in enumerate(property_types)]

# 2. Realistic interaction signals
# Site visits requested: serious buyers ask for site visits
site_visit_requested = np.random.choice([1, 0], num_leads, p=[0.2, 0.8])
# Time spent on listing page (seconds)
time_on_page = np.random.exponential(scale=90, size=num_leads).round()

# 3. Ground-truth scoring logic (Hidden logic the model needs to discover)
# We calculate a log-odds score based on real buyer behavior
score = (
    (site_visit_requested * 2.5) +
    (np.where((contact_hours >= 9) & (contact_hours <= 19), 1.2, -0.8)) +
    (np.where(time_on_page > 120, 1.0, -0.5)) -
    2.5  # Base threshold to keep hot leads around 15-20%
)

# Convert score to probability via sigmoid function
probability = 1 / (1 + np.exp(-score))
status = np.where(probability > 0.5, 'Hot', 'Cold')

# 4. Construct DataFrame
df = pd.DataFrame({
    'property_type': property_types,
    'budget_lakhs': budgets,
    'contact_hour': contact_hours,
    'time_on_page_sec': time_on_page,
    'site_visit_requested': site_visit_requested,
    'status': status
})

print(df['status'].value_counts(normalize=True))
print("\nSample Data:")
print(df.head())

import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix

#Separate features(X) and target y
y=df['status'].map({'Cold':0,'Hot':1})
X=df.drop(columns=['status'])

#Train test split with stratification
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)

# 2. Build the ColumnTransformer
# handle_unknown='ignore' prevents crashes if a brand new property type appears later
# remainder='passthrough' keeps numerical columns (budget, time) untouched
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['property_type'])
    ],
    remainder='passthrough'
)

# 3. Build the Pipeline
# This chains the preprocessor and the Random Forest into one seamless workflow
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

# 4. Train the entire pipeline at once
pipeline.fit(X_train, y_train)

# 5. Evaluate the Pipeline
y_pred = pipeline.predict(X_test)
print("=== Production Pipeline Performance ===")
print(classification_report(y_test, y_pred, target_names=['Cold (0)', 'Hot (1)']))

# 6. Export the Model
joblib.dump(pipeline, 'lead_scorer_pipeline.pkl')
print("\nSuccess: Model saved as 'lead_scorer_pipeline.pkl'!")
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Expanded Dataset
# Columns: [Fever, Cough, Headache, Sore Throat, Fatigue, Chest Pain, Nausea, Skin Rash, Joint Pain, Chills, Disease]
COLUMNS = ['Fever', 'Cough', 'Headache', 'Sore_Throat', 'Fatigue', 'Chest_Pain', 'Nausea', 'Skin_Rash', 'Joint_Pain', 'Chills', 'Disease']
SYMPTOMS = COLUMNS[:-1]

# 1 = Yes, 0 = No
DATA = [
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 'Common Cold'],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 'Malaria'],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 'Migraine'],
    [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 'Flu'],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 'Allergies'],
    [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 'COVID-19'],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 'Anemia'],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 'Heart Disease'],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 'Dengue'],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 'Chickenpox'],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 'Arthritis'],
    [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 'Pneumonia'],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 'Food Poisoning'],
]

PRESCRIPTIONS = {
    'Common Cold': 'Rest, plenty of fluids, Vitamin C. Home remedy: Ginger tea and steam inhalation.',
    'Malaria': 'Antimalarial drugs. Seek immediate medical attention. Home remedy: Cinnamon and Basil tea.',
    'Migraine': 'Pain relievers, rest in dark room. Home remedy: Peppermint oil massage.',
    'Flu': 'Antivirals, rest, hydration. Home remedy: Honey, lemon, and warm water.',
    'Allergies': 'Antihistamines, identify triggers. Home remedy: Local honey and apple cider vinegar.',
    'COVID-19': 'Isolation, monitor oxygen levels. Home remedy: Turmeric milk and breathing exercises.',
    'Anemia': 'Iron supplements, green leafy vegetables. Home remedy: Beetroot and pomegranate juice.',
    'Heart Disease': 'Consult cardiologist IMMEDIATELY. Lifestyle changes required. Home remedy: Garlic and flaxseeds (supportive only).',
    'Dengue': 'Hydration, pain relievers (avoid aspirin). Home remedy: Papaya leaf juice.',
    'Chickenpox': 'Antiviral medication, calamine lotion. Home remedy: Neem leaf bath.',
    'Arthritis': 'Anti-inflammatory drugs, physiotherapy. Home remedy: Ginger and turmeric paste.',
    'Pneumonia': 'Antibiotics, rest. Hospitalization may be needed. Home remedy: Fenugreek tea.',
    'Food Poisoning': 'Rehydration salts, probiotics. Home remedy: Ginger and lemon water.',
}

class DiseasePredictor:
    def __init__(self):
        self.model = None
        self.symptoms = SYMPTOMS
        self.model_path = 'disease_model.pkl'

    def train_model(self):
        df = pd.DataFrame(DATA, columns=COLUMNS)
        X = df[self.symptoms]
        y = df['Disease']
        
        self.model = DecisionTreeClassifier()
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)
        print("Model trained and saved.")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.train_model()

    def predict(self, symptoms_dict):
        if not self.model:
            self.load_model()
        
        # Ensure input array matches the expanded symptom list order
        # UI sends symptoms with spaces (e.g. "Sore Throat"), model expects "Sore_Throat" logic or similar
        # Since get_symptoms_list() returns keys with spaces, app.py uses those keys.
        # self.symptoms has underscores. We need to match them.
        input_data = [symptoms_dict.get(sym.replace('_', ' '), 0) for sym in self.symptoms]
        prediction = self.model.predict([input_data])[0]
        prescription = PRESCRIPTIONS.get(prediction, "Consult a doctor for accurate diagnosis.")
        return prediction, prescription

    def get_symptoms_list(self):
        return [s.replace('_', ' ') for s in self.symptoms]

# Initialize
predictor = DiseasePredictor()
# Force retrain if columns changed (simple heuristic: just calling train if file exists is safer during dev)
predictor.train_model() 

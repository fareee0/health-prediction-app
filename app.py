from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
from models import db, User, Report
from ml_model import predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database_v2.db')
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    if current_user.is_authenticated:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your email and password.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter((User.email == email) | (User.username == username)).first()
        if user:
            flash('Email or Username already exists.')
        else:
            new_user = User(username=username, email=email, password=generate_password_hash(password, method='scrypt'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    symptoms_list = predictor.get_symptoms_list()
    prediction = None
    prescription = None
    
    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        input_data = {sym: 1 if sym in selected_symptoms else 0 for sym in symptoms_list}
        prediction, prescription = predictor.predict(input_data)
        
    return render_template('predict.html', symptoms=symptoms_list, prediction=prediction, prescription=prescription)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Simulate Generic AI Analysis
            import random
            analyses = [
                "Analysis Complete: Blood sugar levels appearing normal. Vitamin D deficiency detected. Recommendation: Increase sun exposure and Vitamin D supplements.",
                "Analysis Complete: All vitals within healthy range. Slight dehydration markers found. Recommendation: Increase water intake.",
                "Analysis Complete: Hemoglobin levels are low relative to baseline. Indications of possibility of anemia. Recommendation: Iron-rich diet advised.",
                "Analysis Complete: Cholesterol levels are elevated. Recommendation: Reduce saturated fat intake and increase cardio exercise.",
                "Analysis Complete: Normal report. No immediate health concerns detected."
            ]
            ai_analysis = random.choice(analyses)
            
            new_report = Report(
                user_id=current_user.id, 
                filename=filename, 
                file_path=file_path,
                status='Verified & Analyzed',
                analysis=ai_analysis
            )
            db.session.add(new_report)
            db.session.commit()
            flash('File uploaded and analyzed successfully!')
            return redirect(url_for('upload'))
            
    reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.upload_date.desc()).all()
    return render_template('upload.html', reports=reports)

if __name__ == '__main__':
    app.run(debug=True)

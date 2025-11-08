from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
import pickle, numpy as np, json, random, os

app = Flask(__name__)
app.secret_key = 'fintech-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# ------------------ Login Manager ------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ------------------ Load Model ------------------
model = pickle.load(open('../model/credit_model.pkl', 'rb'))
with open('../model/feature_importance.json', 'r') as f:
    feature_importance = json.load(f)
MODEL_ACCURACY = 96.67

# ------------------ Database Models ------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)
    role = db.Column(db.String(10), default='user')  # 'admin' or 'user'

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    score = db.Column(db.Integer)
    label = db.Column(db.String(20))
    features = db.Column(db.String(200))

with app.app_context():
    db.create_all()

# ------------------ Login Loader ------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ------------------ Routes ------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash("Username already exists", "error")
            return redirect(url_for('register'))
        new_user = User(username=username, password=password, role='user')
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for('predict'))
        else:
            flash("Invalid credentials", "error")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully", "info")
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('index.html', prediction_text=None, score=None, accuracy=MODEL_ACCURACY)

    try:
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)[0]

        if prediction == 1:
            label, score = "Good", random.randint(80, 100)
        elif prediction == 0:
            label, score = "Average", random.randint(50, 79)
        else:
            label, score = "Poor", random.randint(20, 49)

        # Save prediction to DB
        new_pred = Prediction(user_id=current_user.id, score=score, label=label, features=str(features))
        db.session.add(new_pred)
        db.session.commit()

        return render_template('index.html', prediction_text=label, score=score, accuracy=MODEL_ACCURACY)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", score=None, accuracy=MODEL_ACCURACY)

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role != 'admin':
        flash("Admin access required!", "error")
        return redirect(url_for('predict'))

    preds = Prediction.query.all()
    good = sum(1 for p in preds if p.label == 'Good')
    avg = sum(1 for p in preds if p.label == 'Average')
    poor = sum(1 for p in preds if p.label == 'Poor')

    return render_template('dashboard.html',
                           importance=feature_importance,
                           summary={'Good': good, 'Average': avg, 'Poor': poor},
                           total=len(preds))

if __name__ == "__main__":
    app.run(debug=True)

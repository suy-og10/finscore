from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
import numpy as np, json, random, os, sys

# Import model helpers. Use relative import when running as a package, fall back to a local import
try:
    from .model import load_pipeline, predict_one
except Exception:
    # When running `python app.py` from the app/ folder, the package context is missing.
    # Add the current directory to sys.path so `import model` succeeds.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import load_pipeline, predict_one

app = Flask(__name__)
app.secret_key = 'fintech-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# ------------------ Login Manager ------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ------------------ Load Model ------------------
try:
    pipeline = load_pipeline()
    fi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'feature_importance.json'))
    if os.path.exists(fi_path):
        with open(fi_path, 'r') as f:
            feature_importance = json.load(f)
        MODEL_ACCURACY = feature_importance.get('accuracy', 0)
    else:
        feature_importance = {}
        MODEL_ACCURACY = 0
except Exception as e:
    # If model isn't available, keep app running but predictions will error
    pipeline = None
    feature_importance = {}
    MODEL_ACCURACY = 0

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
        if pipeline is None:
            raise RuntimeError('Model pipeline not loaded')

        # Expected input fields (names) for the model
        expected = ['segment', 'elec_on_time_ratio', 'recharge_on_time_ratio', 'invoice_paid_on_time_ratio',
                    'supplier_on_time_ratio', 'business_days_open_ratio', 'monthly_upi_in_count',
                    'monthly_upi_in_amt', 'years_in_business', 'delivery_cancellations', 'avg_balance',
                    'min_balance_freq', 'monthly_revenue_variance']

        # Support JSON POST or form POST
        if request.is_json:
            data = request.get_json()
        else:
            data = {k: request.form.get(k) for k in expected}

        # Convert types where needed
        parsed = {}
        for k in expected:
            v = data.get(k)
            if v is None:
                raise ValueError(f'Missing feature: {k}')
            # segment stays as string; numeric fields convert to float/int appropriately
            if k in ['monthly_upi_in_count', 'delivery_cancellations']:
                parsed[k] = int(v)
            elif k in ['segment']:
                parsed[k] = str(v)
            else:
                parsed[k] = float(v)

        result = predict_one(pipeline, parsed)

        # Save prediction to DB (store legacy quality label to keep dashboard working)
        new_pred = Prediction(user_id=current_user.id, score=int(result['score']), label=result['quality'], features=str(parsed))
        db.session.add(new_pred)
        db.session.commit()

        # Show both quality and band
        prediction_text = f"{result['quality']} ({result['band']})"
        return render_template('index.html', prediction_text=prediction_text, score=result['score'], accuracy=MODEL_ACCURACY)
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

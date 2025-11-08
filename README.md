# 🏦 **FinScore — Alternative Credit Scoring for MSMEs**

FinScore is a **fintech web application** that predicts the **creditworthiness** of micro, small, and medium enterprises (MSMEs) using **non-traditional financial data** such as digital transactions, utility bills, and mobile recharge patterns.

The goal is to enable **fair credit access** for rural and small business owners who lack formal credit histories.

---

## 🚀 **Key Features**

✅ **Alternative Credit Scoring Model**
Predicts MSME creditworthiness (Good / Average / Poor) using behavioral data.

✅ **Credit Score Visualization**
Dynamic half-gauge chart shows credit score (0–100) with color-coded indicators.

✅ **User & Admin Login System**

* Regular users: Can predict scores.
* Admin: Access to analytics dashboard.

✅ **Interactive Dashboard**
Real-time feature importance and prediction distribution charts built with Chart.js.

✅ **SQLite Database Integration**
Stores user accounts and predictions securely using SQLAlchemy ORM.

✅ **Tailwind CSS UI**
Fully responsive and modern interface for a polished fintech look.

---

## 🧠 **Tech Stack**

| Layer                | Technology                            |
| -------------------- | ------------------------------------- |
| **Frontend**         | Tailwind CSS, Chart.js                |
| **Backend**          | Flask (Python)                        |
| **Database**         | SQLite with SQLAlchemy                |
| **Machine Learning** | RandomForestClassifier (Scikit-learn) |
| **Authentication**   | Flask-Login                           |
| **Visualization**    | Chart.js                              |

---

## 📁 **Project Structure**

```
FinScore/
│
├── model/
│   ├── train_model.py              # Generates credit_model.pkl and feature_importance.json
│   ├── credit_model.pkl            # Trained ML model
│   └── feature_importance.json     # Feature weights for dashboard
│
├── app/
│   ├── app.py                      # Flask application entry point
│   ├── database.db                 # SQLite database file
│   ├── static/
│   │   └── tailwind.css            # Custom styling
│   └── templates/
│       ├── base.html               # Global layout (Navbar, Footer)
│       ├── home.html               # Landing page
│       ├── login.html              # Login page
│       ├── register.html           # User registration page
│       ├── index.html              # Credit prediction page
│       └── dashboard.html          # Admin analytics dashboard
│
└── requirements.txt
```

---

## ⚙️ **Installation & Setup**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/FinScore.git
cd FinScore
```

### 2️⃣ Create and activate a virtual environment

#### Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the model (if not already done)

```bash
python model/train_model.py
```

### 5️⃣ Run the Flask app

```bash
cd app
python app.py
```

### 6️⃣ Open in your browser

```
http://127.0.0.1:5000/
```

---

## 🔑 **Creating an Admin Account**

To access the dashboard, you need one admin user.

Run this in Python shell:

```python
from app import app, db, User
with app.app_context():
    admin = User(username='admin', password='admin123', role='admin')
    db.session.add(admin)
    db.session.commit()
```

Now you can log in as:

```
Username: admin
Password: admin123
```

---

## 📊 **Using the App**

| Page         | Description                               |
| ------------ | ----------------------------------------- |
| `/`          | Landing page introducing FinScore         |
| `/register`  | Create a new user account                 |
| `/login`     | Log into your account                     |
| `/predict`   | Enter business data to get a credit score |
| `/dashboard` | Admin-only analytics (charts, stats)      |

---

## 📸 **Screenshots (add later)**

You can add screenshots after your app runs:

* 🏠 Home Page
* 🔐 Login Page
* 📈 Prediction Page (Gauge Chart)
* 🧾 Dashboard (Feature Importance + Summary)

---

## ⚡ **Sample Input for "Good" Credit Score**

| Field                      | Example Value |
| -------------------------- | ------------- |
| Monthly Electricity Bill   | `4200`        |
| Mobile Recharge Frequency  | `15`          |
| Digital Transaction Count  | `100`         |
| Average Transaction Amount | `6000`        |
| Payment Delay (Days)       | `2`           |

**Predicted Output:**
✅ *Creditworthiness: Good (Score 85–100)*

---

## 🤩 **Model Summary**

* Algorithm: **RandomForestClassifier**
* Data: **Synthetic behavioral dataset**
* Features:

  * Monthly electricity bill
  * Mobile recharge frequency
  * Digital transaction count
  * Average transaction amount
  * Payment delay days

Average model accuracy: **96.6%**

---

## 🚀 **Future Improvements**

* Integrate **real MSME datasets or APIs**
* Add **credit report PDF generation**
* Add **loan recommendation engine**
* Integrate **SHAP explainability** for transparency
* Deploy using **Render / Railway / Heroku**

---

## 👨‍💻 **Developed By**

**Project Name:** FinScore
**Developed by:** Suyog and Team
**Tech Domain:** Fintech | Machine Learning | Web Development

---

## 📜 **License**

This project is open-source under the [MIT License](https://opensource.org/licenses/MIT).

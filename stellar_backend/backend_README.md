# Stellar Analytics — Flask Backend

## Folder Structure

```
stellar_backend/
├── app.py                  ← Main Flask app
├── requirements.txt        ← Python dependencies
├── Procfile                ← For Render deployment
├── README.md
└── models/                 ← PUT YOUR .pkl FILES HERE
    ├── best_classifier.pkl
    ├── best_regressor.pkl
    ├── top_features_A.pkl
    ├── top_features_B.pkl
    ├── scaler_A.pkl
    ├── scaler_B.pkl
    ├── imputer_A.pkl
    └── imputer_B.pkl
```

## Setup (Local)

```bash
# 1. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your .pkl files to the models/ folder

# 4. Run the server
python app.py
```

Server runs at: http://localhost:5000

## API Endpoints

| Method | Endpoint            | Description                        |
|--------|--------------------|------------------------------------|
| GET    | /                  | API info                           |
| GET    | /health            | Health check                       |
| GET    | /fields            | Input field info (for frontend)    |
| POST   | /predict/classify  | Task A — Classification            |
| POST   | /predict/radius    | Task B — Radius prediction         |
| POST   | /predict/full      | Both tasks in one call (use this!) |
| GET    | /history           | Last 20 predictions                |
| DELETE | /history           | Clear history                      |

## Example API Call

```bash
curl -X POST http://localhost:5000/predict/full \
  -H "Content-Type: application/json" \
  -d '{
    "koi_period": 9.49,
    "koi_duration": 2.96,
    "koi_depth": 615.8,
    "koi_impact": 0.15,
    "koi_model_snr": 35.8,
    "koi_num_transits": 142.0,
    "koi_ror": 0.022,
    "teff": 5455.0,
    "logg": 4.47,
    "feh": 0.12
  }'
```

## Expected Response

```json
{
  "classification": {
    "prediction": "CONFIRMED",
    "label": 1,
    "confidence": 94.2,
    "probabilities": {
      "CONFIRMED": 94.2,
      "FALSE_POSITIVE": 5.8
    }
  },
  "regression": {
    "predicted_radius": 2.26,
    "unit": "Earth radii",
    "planet_category": "Mini-Neptune"
  },
  "input": { ... },
  "timestamp": "2026-02-26T10:00:00"
}
```

## Deploy to Render (Free)

1. Push this folder to a GitHub repo
2. Go to render.com → New Web Service
3. Connect your GitHub repo
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Deploy — get a public URL like `https://stellar-api.onrender.com`

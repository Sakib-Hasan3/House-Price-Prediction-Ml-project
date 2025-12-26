# House Price Dashboard (Bangladesh)

A simple, production-ready Flask web app that predicts apartment prices in major Bangladeshi cities. It provides an interactive dashboard and a JSON API. If the trained model files are missing, the app automatically falls back to a lightweight heuristic model so you can develop and demo without blocking.

## Overview
- Interactive form with dynamic city → area dropdowns
- Price prediction and per-sqft breakdown with Bengali currency formatting
- JSON API for programmatic access
- Safe fallback heuristic when `models/model.pkl` isn’t available

## Project Structure
```
app.py
README.md
requirements.txt
models/
static/
	css/
		style.css
	js/
		script.js
templates/
	index.html
	result.html
utils/
	predictor.py
```

## Requirements
- Python 3.10+ recommended
- Windows, macOS, or Linux
- Python packages in `requirements.txt` (chosen for Windows wheels):
	- Flask
	- numpy
	- pandas
	- scikit-learn
	- joblib

## Quick Start (Windows)
1. Create and activate a virtual environment:
	 ```powershell
	 python -m venv .venv
	 .venv\Scripts\activate
	 ```
2. Install dependencies:
	 ```powershell
	 pip install --upgrade pip
	 pip install -r requirements.txt
	 ```
3. Run the app:
	 ```powershell
	 python app.py
	 ```
4. Open http://localhost:5000 in your browser.

## Usage
### Web UI
1. Choose a city; the area list updates automatically.
2. Enter bedrooms, bathrooms, floor number, and floor area (sqft).
3. Submit to see total price and price per sqft.

### API
Endpoint: `POST /api/predict`

Request (JSON):
```json
{
	"bedrooms": 3,
	"bathrooms": 2,
	"floor_no": 6,
	"floor_area": 1200,
	"city": "Dhaka",
	"area": "Uttara"
}
```

Response (JSON):
```json
{
	"success": true,
	"predicted_price": 7450000.0,
	"predicted_price_formatted": "৳7,450,000",
	"price_per_sqft": 6208.3,
	"price_per_sqft_formatted": "৳6,208/sqft",
	"input_features": { /* echoed inputs */ }
}
```

### Form Submission Route
- `POST /predict` renders `templates/result.html` with a formatted result.

## Model Files
- `models/model.pkl`: Trained regression model
- `models/preprocessor.pkl`: Optional preprocessor to transform inputs

If these files are missing or fail to load, the app uses a heuristic fallback based on city and area adjustments, ensuring the UI and API remain functional in development.

## Configuration
- Runs in debug mode on `0.0.0.0:5000` by default. Change in `app.py` if needed.

## Troubleshooting
- If you hit build errors on Windows for scientific packages:
	- Upgrade pip: `python -m pip install --upgrade pip`
	- The pinned versions in `requirements.txt` are selected to use prebuilt wheels.
	- Use the provided virtual environment steps to isolate dependencies.

## Notes
- Prices are shown in Bangladeshi Taka with the `৳` symbol.
- Features used include: `Bedrooms`, `Bathrooms`, `Floor_no`, `Floor_area`, `City`, `Area_grouped`, plus engineered flags like `Has_high_floor`, `Has_many_bedrooms`, and `Luxury_bathroom_ratio`.


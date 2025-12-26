from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load ML model and preprocessor
MODEL_PATH = 'models/model.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'

try:
    model = joblib.load(MODEL_PATH)
    # Optional: load preprocessor if present
    preprocessor = joblib.load(PREPROCESSOR_PATH) if os.path.exists(PREPROCESSOR_PATH) else None
    print("✅ Model loaded successfully")
except BaseException as e:
    print(f"❌ Error loading model: {e}")
    # Fallback: lightweight heuristic model so UI works in dev
    class FallbackModel:
        def predict(self, df: pd.DataFrame):
            # Base rate per sqft by city
            city_multiplier = {
                'Dhaka': 6200,
                'Chattogram': 5000,
                'Cumilla': 3800,
                'Narayanganj City': 4200,
                'Gazipur': 3600
            }
            area_bonus = {
                'Gulshan 1': 1.30, 'Gulshan 2': 1.35, 'Banani': 1.20, 'Dhanmondi': 1.18,
                'Mirpur': 1.00, 'Uttara': 1.10, 'Bashundhara R-A': 1.22, 'Mohammadpur': 0.98,
                'Agrabad': 1.10, 'Khulshi': 1.15, 'GEC': 1.08, 'Panchlaish': 1.05,
                'Main Town': 1.05, 'Kandirpar': 1.08,
                'Bandar': 0.95, 'Fatullah': 0.97,
                'Tongi': 0.92, 'Konabari': 0.90,
                'Other': 1.00
            }
            prices = []
            for _, r in df.iterrows():
                base = city_multiplier.get(r.get('City', 'Dhaka'), 4000)
                bonus = area_bonus.get(r.get('Area_grouped', 'Other'), 1.0)
                sqft = float(r.get('Floor_area', 1000))
                bedrooms = int(r.get('Bedrooms', 2))
                bathrooms = int(r.get('Bathrooms', 1))
                floor_no = int(r.get('Floor_no', 1))
                # Simple heuristic
                price_per_sqft = base * bonus
                price_per_sqft *= 1.05 if r.get('Has_many_bedrooms', 0) else 1.0
                price_per_sqft *= 1.03 if r.get('Luxury_bathroom_ratio', 0) else 1.0
                price_per_sqft *= 0.98 if r.get('Has_high_floor', 0) else 1.0
                # minor adjustments
                price_per_sqft *= (1 + 0.01 * max(bedrooms - 2, 0))
                price_per_sqft *= (1 + 0.005 * max(bathrooms - 1, 0))
                price = price_per_sqft * sqft
                prices.append(np.log1p(price))
            return np.array(prices)

    model = FallbackModel()
    preprocessor = None
    print("ℹ️ Using fallback heuristic model (for development)")

# Available options for dropdowns
CITIES = ['Dhaka', 'Chattogram', 'Cumilla', 'Narayanganj City', 'Gazipur']
AREAS = {
    'Dhaka': ['Gulshan 1', 'Gulshan 2', 'Banani', 'Dhanmondi', 'Mirpur', 
              'Uttara', 'Bashundhara R-A', 'Mohammadpur', 'Other'],
    'Chattogram': ['Agrabad', 'Khulshi', 'GEC', 'Panchlaish', 'Other'],
    'Cumilla': ['Main Town', 'Kandirpar', 'Other'],
    'Narayanganj City': ['Bandar', 'Fatullah', 'Other'],
    'Gazipur': ['Tongi', 'Konabari', 'Other']
}

@app.route('/')
def home():
    """Render the main dashboard page"""
    return render_template('index.html', 
                         cities=CITIES, 
                         areas=AREAS['Dhaka'])

@app.route('/get_areas/<city>')
def get_areas(city):
    """Get areas for selected city (for dynamic dropdown)"""
    return jsonify({'areas': AREAS.get(city, ['Other'])})

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    # model is always set (real or fallback)
    
    try:
        # Get form data
        data = request.form
        
        # Prepare input features
        input_data = {
            'Bedrooms': int(data.get('bedrooms', 2)),
            'Bathrooms': int(data.get('bathrooms', 1)),
            'Floor_no': int(data.get('floor_no', 1)),
            'Floor_area': float(data.get('floor_area', 1000)),
            'City': data.get('city', 'Dhaka'),
            'Area_grouped': data.get('area', 'Other'),
            'Has_high_floor': 1 if int(data.get('floor_no', 1)) > 5 else 0,
            'Has_many_bedrooms': 1 if int(data.get('bedrooms', 2)) > 3 else 0,
            'Luxury_bathroom_ratio': 1 if int(data.get('bathrooms', 1)) / max(int(data.get('bedrooms', 2)), 1) > 0.8 else 0
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        log_price = model.predict(input_df)[0]
        price = np.expm1(log_price)
        price_per_sqft = price / input_data['Floor_area']
        
        # Prepare response
        result = {
            'success': True,
            'predicted_price': f"৳{price:,.0f}",
            'price_per_sqft': f"৳{price_per_sqft:,.0f}/sqft",
            'details': {
                'bedrooms': input_data['Bedrooms'],
                'bathrooms': input_data['Bathrooms'],
                'floor_area': f"{input_data['Floor_area']:,.0f} sqft",
                'city': input_data['City'],
                'area': input_data['Area_grouped']
            }
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON requests"""
    # model is always set (real or fallback)
    
    try:
        data = request.json
        
        # Prepare input features
        input_data = {
            'Bedrooms': int(data.get('bedrooms', 2)),
            'Bathrooms': int(data.get('bathrooms', 1)),
            'Floor_no': int(data.get('floor_no', 1)),
            'Floor_area': float(data.get('floor_area', 1000)),
            'City': data.get('city', 'Dhaka'),
            'Area_grouped': data.get('area', 'Other'),
            'Has_high_floor': 1 if int(data.get('floor_no', 1)) > 5 else 0,
            'Has_many_bedrooms': 1 if int(data.get('bedrooms', 2)) > 3 else 0,
            'Luxury_bathroom_ratio': 1 if int(data.get('bathrooms', 1)) / max(int(data.get('bedrooms', 2)), 1) > 0.8 else 0
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        log_price = model.predict(input_df)[0]
        price = np.expm1(log_price)
        price_per_sqft = price / input_data['Floor_area']
        
        return jsonify({
            'success': True,
            'predicted_price': float(price),
            'predicted_price_formatted': f"৳{price:,.0f}",
            'price_per_sqft': float(price_per_sqft),
            'price_per_sqft_formatted': f"৳{price_per_sqft:,.0f}/sqft",
            'input_features': input_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

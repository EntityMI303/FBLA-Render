from flask import Flask, render_template, request, redirect, url_for, session, send_file
import json, os, sys
from dotenv import load_dotenv
from transformers import pipeline
import torch

# Load environment variables from .env (optional)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "business25")

# Initialize local Hugging Face pipeline once at startup
# Using distilgpt2 for lightweight text generation; replace with a larger model if desired
local_generator = pipeline(
    "text-generation",
    model="distilgpt2",
    torch_dtype=torch.float32,
    device=-1  # -1 = CPU; use 0 if you have GPU
)

@app.route('/')
def home():
    return render_template('index.html', name="Sprite Blue Jet")

@app.route('/predict-future-sales', methods=['GET', 'POST'])
def sales():
    if request.method == 'POST':
        data = {
            'previous_sales': request.form.get('previous_sales'),
            'product': request.form.get('product'),
            'price': request.form.get('price'),
            'marketing_budget': request.form.get('marketing_budget'),
            'season': request.form.get('season'),
            'year': request.form.get('year'),
            'month': request.form.get('month'),
            'weeks': request.form.get('weeks'),
            'days': request.form.get('days')
        }
        session['sales_data'] = data
        file_path = os.path.join(os.path.dirname(__file__), 'sales_data.json')
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return '', 204

    return render_template('sales.html')

@app.route('/download-sales-data')
def download_sales():
    file_path = os.path.join(os.path.dirname(__file__), 'sales_data.json')
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "sales_data.json not found", 404

@app.route('/business-improvement-guide')
def improvement():
    data = session.get('sales_data')
    if not data:
        return redirect(url_for('sales'))

    previous_sales = float(data.get('previous_sales', 0))
    marketing_budget = float(data.get('marketing_budget', 0))

    # Simulated predicted and actual sales
    predicted_sales = round(previous_sales * (1.1 + marketing_budget * 0.001), 2)
    actual_sales = round(predicted_sales * (0.95 + 0.1), 2)

    data['predicted_sales'] = predicted_sales
    data['actual_sales'] = actual_sales

    # Local AI feedback using Transformers + Torch
    try:
        prompt = f"Sales data: {data}. Provide improvement suggestions in a professional tone."
        local_output = local_generator(prompt, max_length=150, num_return_sequences=1)
        ai_feedback = local_output[0]["generated_text"]
    except Exception as e:
        ai_feedback = f"Error generating local AI feedback: {e}"

    return render_template('improvement.html', data=data, feedback=ai_feedback)

@app.route('/update-sales-data', methods=['POST'])
def update_sales_data():
    updated = request.get_json()
    file_path = os.path.join(os.path.dirname(__file__), 'sales_data.json')
    with open(file_path, 'w') as f:
        json.dump(updated, f, indent=2)
    return '', 204

# ✅ New route: check Python version
@app.route('/version')
def version():
    return f"Python version: {sys.version}"

# ✅ New route: health check
@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    app.run(debug=True)
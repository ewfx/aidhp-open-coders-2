#├── recommendation_system/
#│   ├── app.py                     # Flask API (handles API requests)
#│   ├── train_model.py              # Model training script
#│   ├── utils.py                    # Helper functions (if needed)
#│   ├── models/
#│   │   ├── als_model.pkl           # Saved ALS model
#│   │   ├── customer_mapping.pkl    # Customer ID mapping
#│   │   ├── product_mapping.pkl     # Product ID mapping
#│   │   ├── customer_product_matrix.pkl  # User-item matrix
#│   ├── requirements.txt            # Dependencies (Flask, Surprise, etc.)
#│   ├── README.md                   # Documentation

# train_model.py - Train and Save ALS Model
import os

# Limit OpenBLAS & MKL threads to 1 for better performance
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sparse

# Sample data loading process (replace with actual data loading)
customer_product_matrix = sparse.csr_matrix([[0, 1], [1, 0]])
customer_mapping = {101: 0, 102: 1}
product_mapping = {201: 0, 202: 1}

# Train ALS model
model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
model.fit(customer_product_matrix)

# Save model and mappings
with open("models/als_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("models/customer_mapping.pkl", "wb") as file:
    pickle.dump(customer_mapping, file)

with open("models/product_mapping.pkl", "wb") as file:
    pickle.dump(product_mapping, file)

with open("models/customer_product_matrix.pkl", "wb") as file:
    pickle.dump(customer_product_matrix, file)

print("Model training complete and saved!")

# app.py - Flask API for recommendations
from flask import Flask, request, jsonify
import pickle
import scipy.sparse as sparse

app = Flask(__name__)
@app.route("/")  # <-- This defines the homepage
def home():
    return "Welcome to the Recommendation API!"

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/recommend/<cust_id>", methods=["GET"])
def recommend(customer_id):
    # Replace this with actual recommendation logic
    return jsonify({"cust_id": cust_id, "recommendations": ["Product1", "Product2", "Product3"]})

if __name__ == "__main__":
    app.run(debug=True)



# Load the trained ALS model
with open("models/als_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/customer_mapping.pkl", "rb") as file:
    customer_mapping = pickle.load(file)

with open("models/product_mapping.pkl", "rb") as file:
    product_mapping = pickle.load(file)

with open("models/customer_product_matrix.pkl", "rb") as file:
    customer_product_matrix = pickle.load(file)

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    cust_id = request.args.get("cust_id", type=int)

    if cust_id not in customer_mapping:
        return jsonify({"error": "Customer ID not found"}), 400

    customer_index = customer_mapping[cust_id]
    recommended_indices, scores = model.recommend(customer_index, customer_product_matrix[customer_index], N=5)
    recommended_product_ids = [list(product_mapping.keys())[i] for i in recommended_indices]

    return jsonify({"cust_id": cust_id, "recommended_products": recommended_product_ids})

if __name__ == "__main__":
    app.run(debug=True)

print("Flask API is running at http://127.0.0.1:5000/")

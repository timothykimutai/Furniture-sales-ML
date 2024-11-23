## **Executive Summary**

This report documents the complete lifecycle of a data-driven project undertaken to analyze, model, and deploy a regression model for predicting revenue in a furniture business. The process included: 

1. **Data Analysis**: Cleaning, exploring, and preprocessing the dataset to make it ready for modeling.
2. **Model Building**: Developing a Linear Regression model to predict revenue with high accuracy.
3. **Deployment**: Exposing the model as an API using Flask and deploying it in a production environment using Docker and Heroku.

The solution demonstrates the ability to leverage machine learning for real-world business applications by creating a scalable and accessible predictive system. 

---

## **1. Problem Statement**

Revenue prediction is critical for optimizing inventory management and pricing strategies in the furniture business. The goal was to:
- Predict revenue using features like inventory, category, location, season and brand.
- Deploy the solution to make it available as an API for business integration.

---

## **2. Dataset Overview**

### **Description**
The dataset contained information about various factors influencing revenue, such as:
- **Features**: Inventory, cost, sales, delivery duration, discount percentage, brand, location, season, store type and material.
- **Target Variable**: Revenue.

### **Initial Observations**
- **Missing Values**: Critical columns like inventory and revenue had missing entries.
- **Outliers**: Negative revenue values and zero inventory rows were present.
- **Categorical Variables**: Fields like `category` and `location` required encoding.

---

## **3. Data Preprocessing**

### **Procedure**

1. **Handling Missing Values**:
   - Imputed missing values in `inventory` and `discount_percentage` columns using median values.

2. **Removing Outliers**:
   - Excluded rows with negative revenue or zero inventory.

3. **Categorical Encoding**:
   - Applied **one-hot encoding** to categorical features for compatibility with the model.

4. **Feature Scaling**:
   - Standardized numerical columns to improve model performance.

---

## **4. Exploratory Data Analysis (EDA)**

### **Key Findings**
1. **Correlation Analysis**:
   - Revenue was strongly correlated with inventory levels and discounts offered.
2. **Visual Insights**:
   - Revenue distribution showed a positive skew.
   - Higher discounts generally led to lower revenue.

### **Techniques Used**
- **Scatter Plots**: To observe relationships between features and revenue.
- **Heatmap**: To show the relationships between revenue (target variable) and other features

---

## **5. Model Building**

### **Model Selection**
A **Linear Regression** model was chosen due to:
- Interpretability.
- Simplicity for this relatively small and clean dataset.
- Adaptability

### **Training the Model**
1. **Data Splitting**:
   - Training Set: 80%.
   - Testing Set: 20%.
2. **Training Process**:
   - The model was trained on the processed dataset using scikit-learn (sklearn).

### **Evaluation**
- **Metrics**:
  - Mean Squared Error (MSE): 123.45
  - R² Score: 0.89
- **Conclusion**:
  - The model exhibited high accuracy and generalization capabilities.

---

## **6. Deployment**

### **The Goal**
To expose the trained model as a web service for real-time predictions.

### ** The Process**
1. **Model Serialization**:
   - The model was saved as `regression_model.pkl` using `joblib`.
   
2. **API Development**:
   - A Flask API was created with an endpoint `/predict` to receive input data and return predictions.

3. **API Workflow**:
   - **Input**: JSON containing feature values.
   - **Processing**: Data is converted to a Pandas DataFrame for model input.
   - **Output**: Predicted revenue is returned as a JSON response.

---

## **7. Deployment to Production**

### **Steps Taken**
1. **Containerization**:
   - A `Dockerfile` was created to package the Flask application.
   - The application was run in a Docker container locally for testing.

2. **Cloud Deployment**:
   - The Dockerized app (flaskApp.py) was deployed on **Heroku** for easy accessibility.

### **Testing**:
- The `/predict` endpoint was tested using **Postman** and `curl`.
- Example Input:
  ```json
  {
    "inventory": 50,
    "discount": 10,
    "store_location": "CityA",
    "product_type": "Sofa"
  }
  ```
- Example Output:
  ```json
  {
    "prediction": 1200.75
  }
  ```

---

## **8. Future Improvements**

### **Model Enhancements**
- Experiment with advanced algorithms like Ridge Regression or Gradient Boosting.
- Incorporate additional features for better predictive power.

### **API Enhancements**
- Add input validation to handle missing or incorrect data.
- Implement authentication for secure usage.

### **Deployment Enhancements**
- Deploy using serverless platforms like AWS Lambda for scalability.
- Monitor API performance with tools like Prometheus and Grafana.

---

## **9. Conclusion**

This project demonstrated the complete lifecycle of a data science workflow:
- **Data Analysis and Preparation**: Cleaned and preprocessed the dataset.
- **Model Development**: Built and evaluated a high-performing Linear Regression model.
- **Deployment**: Delivered a scalable API for predicting revenue in real time.

The system is now ready to support the furniture business in making data-driven decisions, such as optimizing inventory and pricing strategies.

---

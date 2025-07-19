# Predictive Maintenance for Scania Trucks using Machine Learning

*An end-to-end project to predict Air Pressure System (APS) failures in Scania trucks, featuring an XGBoost model, SHAP explanations, and a deployed interactive web application with Streamlit.*

---

### **Live Application Screenshot**

![Streamlit App Screenshot](https://github.com/Sourabh1710/predictive-maintenance-truck-fleet/blob/main/Live%20Application%20Screenshot.png)

---

### Table of Contents
* [Problem Statement](#problem-statement)
* [Key Features](#key-features)
* [Tech Stack](#tech-stack)
* [Key Results & Business Impact](#key-results--business-impact)
* [How to Run This Project](#how-to-run-this-project)
* [Project Structure](#project-structure)

---

### Problem Statement

In the trucking industry, unexpected vehicle failures lead to significant operational disruptions and high costs. A failure in the Air Pressure System (APS), which is critical for braking and gear changing, can result in expensive roadside repairs, towing fees, and delivery delays. The maintenance schedule is often reactive, meaning repairs are only performed after a failure has occurred.

The objective of this project is to develop a predictive maintenance system that can identify trucks at a high risk of an APS failure. By shifting from a reactive to a proactive maintenance strategy, the company can schedule repairs in advance, reducing costs and improving overall fleet reliability and safety. The core challenge lies in building a model that can accurately detect the rare failure events from a vast amount of noisy sensor data.

---

### Key Features
*   **Advanced Feature Engineering:** Created rolling-window features (mean, standard deviation, max) to capture the dynamic behavior and instability of sensors leading up to a failure.
*   **Imbalanced Classification:** Trained a powerful **XGBoost Classifier**, using the `scale_pos_weight` parameter to effectively handle the severe class imbalance between failure and non-failure cases.
*   **Model Interpretability (XAI):** Implemented **SHAP (SHapley Additive exPlanations)** to explain the model's predictions, providing actionable insights for maintenance teams by identifying which sensors are the primary drivers of a failure prediction.
*   **Quantifiable Business Impact:** Performed a detailed cost-benefit analysis to translate the model's performance (True Positives, False Positives, False Negatives) into a tangible dollar-value savings.
*   **Interactive Deployment:** Developed and deployed a user-friendly web application using **Streamlit**, allowing users to upload new sensor data and receive real-time failure predictions.

---

### Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-2077B4?style=for-the-badge&logo=shap&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)


---

### Key Results & Business Impact

The model's performance on the test set demonstrates its value. By proactively identifying failures, it generates significant savings.

> **Cost-Benefit Analysis Results:**
> *   **Cost without Model (Reactive):** `$1,875,000`
> *   **Cost with Model (Proactive):** `$250,400`
> *   **Total Estimated Savings on Test Set:** **`$1,624,600`**

*(Note: These figures are based on the test set and assumed costs. The actual savings will scale with the size of the fleet.)*

#### Model Interpretability with SHAP

![SHAP Force Plot](https://github.com/Sourabh1710/predictive-maintenance-truck-fleet/blob/main/shap_explanation.png)

This plot shows exactly which sensor readings (in red) pushed the model to predict a failure for a specific truck, providing a clear starting point for mechanics.

---

### How to Run This Project

Follow these steps to set up and run the project on your local machine.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    All the required libraries are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser. You can now upload a CSV file (like `test_failure_case.csv` generated from the notebook) to see the model in action.

---

### Project Structure
```
predictive-maintenance-truck-fleet/
|
├── app.py                      # The Streamlit web application script
├── src/
│   ├── final_model.pkl         # Saved trained XGBoost model       
|   ├── imputer.pkl             # Saved Scikit-learn imputer
|   ├── model_columns.pkl       # Saved list of columns for the model
|   ├── original_columns.pkl    # Saved list of original columns before feature engineering
├── requirements.txt            # List of Python dependencies
├── README.md                   # This file
|
├── data/
│   ├── aps_failure_training_set.csv
│   └── aps_failure_test_set.csv
├── test/                       # csv files to test streamlit app
│   ├── test_failure_case.csv
│   └── test_normal_case.csv
└── notebooks/
    └── 01-EDA-and-Feature-Engineering.ipynb   # Jupyter Notebook with the full analysis
```

---

## 👤 Author
**Sourabh Sonker**                                                                                                                 
**Aspiring Data Scientist**

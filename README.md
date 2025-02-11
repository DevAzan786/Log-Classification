# Log Classification With Hybrid Classification Framework

This project implements a hybrid log classification system, integrating multiple approaches to handle varying levels of complexity in log patterns. By combining different classification methods, the system ensures flexibility and effectiveness in processing predictable, complex, and poorly labeled data patterns.

## Classification Approaches

### 1. Regular Expression (Regex):
- Handles simple and predictable patterns.
- Useful for logs that can be captured using predefined rules.

### 2. Sentence Transformer + Logistic Regression:
- Manages complex patterns when sufficient training data is available.
- Utilizes embeddings generated by Sentence Transformers and applies Logistic Regression as the classification layer.

### 3. Large Language Models (LLM):
- Used for handling complex patterns when sufficient labeled training data is unavailable.
- Provides a fallback or complementary approach to the other methods.

---

## Folder Structure

### `training/`
- Contains the code for training models using Sentence Transformer and Logistic Regression.
- Includes the code for regex-based classification.

### `models/`
- Stores the saved models, including Sentence Transformer embeddings and the Logistic Regression model.

### `resources/`
- This folder contains resource files such as test CSV files, output files, images, etc.

### Root Directory
- Contains the FastAPI server code (`server.py`).
- Contains the Streamlit app for user interaction with the FastAPI server.

---

## Setup Instructions

### Install Dependencies
Ensure you have Python installed on your system. Install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

### Run the FastAPI Server
To start the server, use the following command:

```bash
uvicorn server:app --reload
```

Once the server is running, you can access the API at:

- **Main endpoint:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative API Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## How to Run This Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/DevAzan786/Log-Classification.git
   cd Log-Classification
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI Server:**
   ```bash
   uvicorn server:app --reload
   ```

4. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

5. **Access API and Streamlit Interface:**
   - Upload a CSV file containing logs to the FastAPI endpoint for classification.
   - Ensure the file has the following columns:
     - `source`
     - `log_message`
   - The output will be a CSV file with an additional column `target_label`, representing the classified label for each log entry.
   - Use the Streamlit app for a user-friendly interface to interact with the FastAPI server.


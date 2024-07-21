# Tourist Revisitation Prediction

## Project Overview 

This repository contains the code and resources for predicting the probability of a tourist revisiting a destination using machine learning techniques. The primary model employed is a Random Forest classifier, which has been implemented within a Streamlit application to facilitate user-friendly interaction and accessibility.


## Installation

### Prequisites

 - Python 3.8 or above
 - Jupyter Notebook
 - Required libraries (listed in 'requirements.txt')

### Steps

1. Clone the repository:
   
    ```bash
    git clone https://github.com/yyohyy/Tourist_Revisitation_Prediction.git
    ```
    
2. Navigate to the Project directory:
   
    ```bash
    cd Tourist_Revisitation_Prediction
    ```
    
3. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    ```
    
     On Windows:
             
       venv\Scripts\activate
      
     On macOS/Linux:
   
       source venv/bin/activate
       
    
4. Install the required dependencies:
   
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
 
    ```bash
    streamlit run predict.py
    ```
    
2. Open your browser and navigate to `http://localhost:8501` to access the app.
3. Observe prediction based on input.
   

## Train model

1. Open Jupyter notebook

   ```bash
   jupyter notebook
   ```
2. Navigate to notebook 'Tourist_Revisitation_Predictive_Modeling.ipynb' in the main project folder.   
3. Modify the dataset path if using custom dataset.
4. Run all the cells in the notebook.
5. The trained model will be saved in the 'model' folder.   




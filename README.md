# Thyroid-Disease-Prediction 🩺
<strong>Thyroid Disease Prediction ML project</strong>

## Data Description ℹ️
    From Garavan Institute
    Documentation: as given by Ross Quinlan
    6 databases from the Garavan Institute in Sydney, Australia
    Approximately the following for each database:

        ** 2800 training (data) instances and 972 test instances
        ** Plenty of missing data
        ** 29 or so attributes, either Boolean or continuously-valued 

    2 additional databases, also from Ross Quinlan, are also here

        ** Hypothyroid.data and sick-euthyroid.data
        ** Quinlan believes that these databases have been corrupted
        ** Their format is highly similar to the other databases 




## Workflows 🔧	
    1. Update config.yaml
    2. Update schema.yaml
    3. Update params.yaml
    4. Update entity
    5. Update configuration manager (configuration.py) in src config
    6. Update components
    7. Update pipeline
    8. Update the main.py
    9. Update the app.py

<!-- ## MLflow experiments 🔱 -->


<!-- MLFLOW_TRACKING_URI=https://dagshub.com/tejas05in/Thyroid-Disease-Prediction.mlflow \
MLFLOW_TRACKING_USERNAME=tejas05in \
MLFLOW_TRACKING_PASSWORD=9efcb5c7b79d0e949378459b922b1462a80fa413  -->





## How to run the project
![Static Badge](https://img.shields.io/badge/Conda%20-%20Project%20Run%20Details%20-%20css?style=flat&logo=pypi&logoColor=rgb)

1. Clone the project
```bash
git clone https://github.com/tejas05in/Thyroid-Disease-Prediction.git
```
2. Change into the project directory
```bash
cd /Thyroid-Disease-Prediction
```
3. Create a conda environment 


```bash
conda create -p env python==3.11.4 -y
```
4. Activate the conda environment
```bash
conda activate env/
```
5. Install the requirements
```bash
pip install -r requirements.txt
```
6. Start the streamlit app
```bash
streamlit run app.py
# This will redirect you to the end point in your browser
```
\
![Static Badge](https://img.shields.io/badge/python%20-%20model%20training%20-%20rgb?style=flat&logo=pypi&logoColor=rgb)
#### Initialization of training pipeline 
```bash
python main.py
```
#### MLflow experiments 🔱
#### Optional: Run this to export the environemnt variables which will log the results in mlflow at dagshub:
![Static Badge](https://img.shields.io/badge/mlflow%20-%20v2.10.0%20-%20rgb?style=flat&logo=pypi&logoColor=rgb)

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/tejas05in/Thyroid-Disease-Prediction.mlflow \
export MLFLOW_TRACKING_USERNAME=tejas05in \
export MLFLOW_TRACKING_PASSWORD=9efcb5c7b79d0e949378459b922b1462a80fa413
```
If the above variables are not exported then mlfow will store the results locally
and it can be accessed by passing
```bash
mlflow ui --port 5000
```


## ![alt text](image.png)
```bash
docker pull tejas05in/tdpapp
docker run -p 5000:5000 tdpapp:latest
```

## Model Drift Reports and Tests
- [Drift Report](drift_reports/report.html) : Contains information about the dataset and model drift parameters 
- [Tests](drift_reports/test.html) : Contains information about the various tests performed on the dataset , model and features

## Dagshub Repository :  
[Repo link](https://dagshub.com/tejas05in/Thyroid-Disease-Prediction) : Directs you to the Dagshub repository

## Pipline Version Control : 
![Static Badge](https://img.shields.io/badge/DVC%20-%20v3.43.1%20-%20css?style=flat&logo=pypi&logoColor=rgb)



## **1. Clustering Antarctic Penguin Species**
![Penguin Image](Clustering_Antarctic_Penguin_Species/iter_pemguins.jpg)
This project aims to support researchers in identifying different species of penguins in Antarctica using K-means clustering. The dataset, derived from Dr. Kristen Gorman and the Palmer Station LTER, contains measurements such as culmen length, culmen depth, flipper length, body mass, and penguin sex. The project involves preprocessing the data by scaling it using `StandardScaler`, transforming categorical variables into dummy variables, and then applying the K-means algorithm. The optimal number of clusters is determined using the Elbow Method, and the clusters are visualized to reveal patterns in the dataset. The project was implemented in Python, with libraries like `pandas`, `matplotlib`, and `scikit-learn`.

## **2. Modeling Car Insurance Claim Outcomes**
![Car Image](Modeling_Car_Insurance_Claim_Outcomes/car.jpg)

This project involves building a logistic regression model to predict whether a customer will make a claim on their car insurance during the policy period. Using a dataset provided by an insurance company, which includes variables such as age, gender, driving experience, credit score, and past accidents, the project focuses on identifying the single feature that results in the best performing model in terms of accuracy. The analysis is performed using Python with libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `statsmodels`. The best feature identified for prediction was the client's driving experience, resulting in a model accuracy of 77.71%.

## **3. Predicting Movie Rental Durations**
![DVD Image](Predicting_Movie_Rental_Durations/dvd_image.jpg)

This project focuses on building regression models to predict the duration for which customers will rent DVDs, helping a rental company optimize their inventory planning. Using features such as rental rates, movie length, and replacement costs from the provided dataset, various models were tested, including Lasso regression and Random Forest. After hyperparameter tuning, the Random Forest model achieved the best performance with a Mean Squared Error (MSE) below the target threshold of 3. The project was implemented using Python, leveraging libraries such as `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.

## **4. Predictive Modeling for Agriculture**
![DVD Image](Predictive_Modeling_for_Agriculture/farmer_in_a_field.jpg)

In this project, a multi-class classification model was developed to assist farmers in selecting the optimal crop based on essential soil metrics such as nitrogen, phosphorus, potassium levels, and pH value. Using a dataset that captures these soil measurements, the model predicts the most suitable crop for maximizing yield. After evaluating different features, potassium (K) was identified as the most critical predictor for crop selection. The model was implemented using Python, with key libraries including `pandas`, `scikit-learn`, and `metrics` for model training and performance evaluation.

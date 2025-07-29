&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&													&
&													&
&   README FILE: FOR INTRODUCTION TO MACHINE LEARING, DVAD91, STEVEN KRAAMWINKEL, KAUID: stevkraa100	&
&													&
&													&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


This readme file is made for educational purposes to guide the reader through the Jupyter notebooks that have been developed by me. It contains the following notebooks: 
		- Heart_Disease_Prediction: with supervised classification algorithms.
		- Loan_Prediction: with supervised regression algorithms. 

The Python version that was used is: version 3.13
The libraries in the notebooks include: numpy, pandas, seaborn, matplotlib, and sklearn

The notebooks contains relatively low code, and high analysis and transparancy. This approach makes data preparation and machine learning modelling more efficient, explainable, and understandable for people with very few AI and data knowledge. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


PROBLEM 1: Heart Disease prediction using classifation algorithms

In this problem a dataset named 'cleaned merged heart dataset.csv' was used as the primary data source. 

Before the classifacation algorithms were developed the dataset was explored first and some data analysis was applied. The steps during the exploration and data analysis phase included: 

- step 1: Checking the variables datatypes. Goal was to make sure that all variables are from the int or float datatype. This is caused by the fact that calculations and data manipulations in Python with both numerical and object datatypes result oftentimes in errors, makes machine learning (ML) model building, and even proper data analysis impossible. In this situation the dataset contained only integer and float type variables, so no datatype conversions were necessary. 
- step 2: Provide a short summary of the dataset. This is done by the Python .describe() function
- step 3: Checking the data for missing values. If the dataset has missing values addiotional data cleaning techniques were needed to apply. However, the dataset contained no missing values. 
- step 4: Checking for correlation between the predictive variables and the target variable, and visualizing the correlations  in a heatmap. This was done by the Python .corr() function with the statistical Spearman method, a method that calculates the correlation between variables and a target variable. 
- step 5: Checking if the data is complete, makes sense, and is consistent. The .to_string() function was used for this purpose in the notebook. 
step 6: Reshaping the dataframe that will be used for the ML prediction phase. Columns with poor correlation with the target variable were dropped, and additional features with slightly better correlation were engineered. 

Machine learning model prediction phase: 
In this phase five different machine learning models were developed. Prior to each model prediction, the five features that contained the highest correlation with the 'target' feature were used as the predictive columns. Next, the dataset was split using the train_test_split method from the sklearn library. The model was trained using the MLmodel.fit method, and predictions were made with MLmodel.predict. Predicted was whether an instance belonged to one of the binary [0,1] classes. Main metric that was used is the classifation accuracy. This means how well the model is able to predict each example it sees to one of the classes correctly. Additionally a confusion matrix was given. Also precision, recall, and f1-score metrics were calculated. 

The following five models were used in the ML modelling predictive phase: 
- Logistic Regression
- KNN (k-nearest-neighbour)
- Naive Bayes
- Random Forest
- Support Vector Machine

The performances of each of the five model on the performance metrics are listed below: 

				Accuracy	Precision*	Recall*		F1-score*

Logistic Regression		0.75		0.75		0.75		0.75
KNN (k-nearest-neighbour)	0.87		0.88		0.87		0.87
Naive Bayes			0.75		0.75		0.75		0.74
Random Forest			0.90		0.90		0.90		0.90
Support Vector Machine		0.84		0.84		0.84		0.84

* the precision, recall, and f1-scores are based on their weighted averages over both binary classes.

Conclusion: 
Out of the five machine learning models, the Random Forest performed the best with classification results of approx 0.90, meaning that the algorithm was able to predict 90% of all examples correctly. Second best was the KNN algorithm with scores around 0.87, third best was the Support Vector Machine. The Logistic Regression and Naive Bayes algorithms still performed acceptable, with metric scores of approx 0.75, but were less accurate than the other three models. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


PROBLEM 2 Loan prediction using regression models

In this problem loan data from the 'loan_data.csv' dataset was used as the dataset. The objective of this problem was to make exploratory analysis from the dataset, and the model regression models. 

The first step was to explore the dataset by checking for missing values, and for consistency between the datatypes of the variables. The dataset contained no missing values. However, the datatypes of the variables contained both object datatypes and float and integer datatypes. Since calculation and data manipulations with both object and numerical data types is very error prone and unfeasable, the object datatypes were transformed to integer data types. This was done with the Python built in .map function and using dictionaries for each categorical value. 

After this pre step. The real exploratory data analysis (EDA) phase began. The steps during the EDA phase included: 

-step 1: Summary of statistics. This shows the mean, standard deviation, min, max, and the three inner quartile values of each of the variables. 
-step 2: Visualizing the continuous variables with boxplot visualization. This was done with the matplotlib plt.boxplot method. Goal was to identify outliers for each continuous variable.
-step 3: Removal of outliers. Outliers were removed based on a threshold z-score of 3. This means that datapoints from continuous variables that were more than three standard deviations higher than the mean were removed. Only the approx top 1% data values were removed, with the goal not to lose to many data examples. The numpy library was used for calculations. 
-step 4: Visualisation of the continuous variables. This was done both by histograms and distribution plots, using the seaborn libaray with .hist for histograms and .displot for the distribution plots. 
-step 5: The categorical variables were visualized using frequency bar plots. The x-axis of the bar plots contain the different classes and the y-axis the frequency of each of the classes. 

Machine learning model phase: 
In the modelling prediction phase different regression models were trained on the loan dataset where example with one of more outliers were removed. This resulted in a slight decrease of examples from 45000 to 42866, of almost 5% of all the data. 

For predictive goals four different regression models were trained. First a simple linear regression model was trained. This type of model is usefull to predict one variable based on one other variable. Therefore correlation between all variables with the target variable 'credit score' were calculated, and one variable with some correlation to the target variable with more than two categorical class values was selected. A linear regression model was trained with the LinearRegression() model. Training and test set was determined by the train_test_split function, with 25% test data. The training set was trained with the .fit function, and predictions were made with the .predict function. The final prediction resulted in a array containing predicted credit scores for each test example. As the performance metrics the root mean squared error and mean absolute error were computed. 

For the other three models: the lasso regression, decision tree, and random forest algorithm the whole dataset minus the outliers was used. The target y variable is the 'credit score' variable, and the predictive variables were the other variables from the loan dataset. Training and prediction was used with similar methods as were used with the linear regression model. As performance metrics also the root mean squared error and mean absolute error were computed. 

The performance results of all the applied ML-regression models are listed below: 


				RMSE		MAE

Linear regression		49.19`		39.38
Lasso regression		47.05		37.67
Decision tree			47.68		38.18
Random forest			47.61		38.13


Conclusion:
Whether the models performed well or poor is debate for discussion. With a standard deviation of 50 is the credit scores, it is questionable if any of the models made accurate predictions. All four models have RMSE (root mean squared error) and MAE (mean average error) values that are very similar. No one model outperformed any other model significantly. This can be caused because of a lack of correlation between the predictive variables and the target variables. For model improvement additional feature engineering, and/or hyperparameter tuning can be applied, to achieve better RMSE and MAE scores. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

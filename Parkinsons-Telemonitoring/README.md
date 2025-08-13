Predicting Parkinson's Disease Severity from Voice Features
This project aims to predict the severity of Parkinson's disease using voice recordings and clinical data from a dataset of 5,875 records from 42 patients. The dataset includes acoustic features (e.g., jitter, shimmer, noise ratios, RPDE, DFA, PPE) and demographic/clinical information (e.g., age, sex, test_time) to predict motor_UPDRS scores, which measure motor impairment severity. The project involves comprehensive data preprocessing, exploratory data analysis, feature engineering, training multiple machine learning and deep learning models, evaluating their performance, and visualizing results to assess the relationship between voice characteristics and disease severity.
Below, we outline the steps performed in the project, focusing on the actions taken without explaining the internal workings of the models.
Project Overview in Steps

Data Loading:

Loaded the Parkinson's disease dataset from a CSV file (parkinsons_updrs.csv) with 22 columns: subject#, age, sex, test_time, motor_UPDRS, total_UPDRS, Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP, Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA, NHR, HNR, RPDE, DFA, and PPE.
Displayed the first few rows to inspect the data structure.


Exploratory Data Analysis (EDA):

Summarized the dataset by printing column names, dimensions (2D), size (129,250 elements), shape (5,875 rows, 22 columns), and unique values for age, subject#, sex, and test_time.
Generated descriptive statistics (e.g., mean, min, max) for all features using df.describe().
Checked data types and memory usage with df.info().
Computed a correlation matrix to analyze relationships between features and targets (motor_UPDRS, total_UPDRS).
Verified no duplicate rows exist using df.duplicated().sum().
Confirmed no missing values with df.isna().sum().


Data Visualization:

Created a heatmap to confirm the absence of null values using sns.heatmap(df.isnull()).
Generated pairplots with sns.pairplot(df) to visualize pairwise relationships between features.
Plotted histograms for motor_UPDRS and total_UPDRS to examine their distributions, using 20 bins and distinct colors (#cce639 for motor_UPDRS, #e639b2 for total_UPDRS).
Visualized the correlation matrix as a heatmap with sns.heatmap(df.corr()) to identify feature correlations.
Created a pie chart to show the gender distribution (sex_counts) with labels for Male and Female, including percentage labels.
Plotted a pie chart for the age distribution (age_counts) with percentage labels.
Generated a violin plot to compare motor_UPDRS across sex using sns.violinplot.
Created a box plot to visualize motor_UPDRS across age using sns.boxplot.


Data Preprocessing:

Defined the target variable as motor_UPDRS and dropped irrelevant columns (subject#, motor_UPDRS, total_UPDRS) to create the feature set X.
Assigned y as motor_UPDRS.
Split the dataset into training (80%) and testing (20%) sets using train_test_split with random_state=42.
Standardized numerical features using StandardScaler, fitting on X_train and transforming both X_train and X_test to create X_train_scaled and X_test_scaled.


Model Training and Evaluation:

Linear Regression:
Trained a LinearRegression model on X_train_scaled and y_train.
Generated predictions for training (y_train_pred) and test (y_test_pred) sets.
Evaluated performance using Mean Squared Error (MSE), R-squared (R2), and Mean Absolute Error (MAE) for both sets.
Printed MSE, R2, and MAE for training and test sets.
Visualized results with scatter plots for each feature in X_test against y_test (color #16c7bb) and y_test_pred (color #c7161f), arranged in a 7x3 grid (15x30 inches), with best-fit lines, titles, labels, and legends.


Support Vector Regressor (SVR):
Performed hyperparameter tuning using GridSearchCV with parameters: kernel=['rbf'], gamma=[0.3, 0.4], C=[5.0, 10.0, 25], and 5-fold cross-validation, optimizing for negative MSE.
Trained the best SVR model (best_estimator) on X_train and y_train.
Generated predictions for training and test sets.
Evaluated performance with MSE, R2, and MAE for both sets.
Printed best parameters, MSE, R2, and MAE.
Visualized results with scatter plots for each feature in X_test against y_test (color green) and y_test_pred (color red), in a 7x3 grid.


Decision Tree Regressor:
Conducted hyperparameter tuning with GridSearchCV using parameters: max_depth=[3, 5, 7, 10, 15], min_samples_split=[2, 5, 10, 15, 20], min_samples_leaf=[1, 2, 4, 6, 8], and 5-fold cross-validation, optimizing for negative MSE.
Trained the best DecisionTreeRegressor model on X_train_scaled and y_train.
Generated predictions for training and test sets.
Evaluated performance with MSE, R2, and MAE.
Printed best parameters, MSE, R2, and MAE.
Visualized results with scatter plots for each feature in X_test against y_test (color yellow) and y_test_pred (color purple), in a 7x3 grid.


Gradient Boosting Regressor:
Trained a GradientBoostingRegressor with fixed parameters: n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42 on X_train_scaled and y_train.
Generated predictions for training and test sets.
Evaluated performance with MSE, R2, and MAE.
Printed MSE, R2, and MAE.
Visualized results with scatter plots for each feature in X_test against y_test (color blue) and y_test_pred (color red), in a 7x3 grid.


AdaBoost Regressor:
Trained an AdaBoostRegressor with a DecisionTreeRegressor base estimator (max_depth=10), n_estimators=100, and random_state=42 on X_train_scaled and y_train.
Generated predictions for training and test sets.
Evaluated performance with MSE, R2, and MAE.
Printed MSE, R2, and MAE.
Visualized results with scatter plots for each feature in X_test against y_test (color pink) and y_test_pred (color brown), in a 7x3 grid.


TensorFlow Neural Network:
Re-prepared data by dropping motor_UPDRS and total_UPDRS from X, re-splitting into training and test sets, and re-applying StandardScaler.
Built a sequential neural network with 5 layers: input layer (20 features), 4 hidden layers (500, 250, 100, 50 units, ReLU activation), and output layer (1 unit, no activation).
Compiled the model with SGD optimizer (learning_rate=0.001) and MSE loss.
Trained the model on X_train_scaled and y_train for 150 epochs with a batch size of 32 and 20% validation split.
Evaluated the model on X_test_scaled and y_test, computing test loss.
Generated predictions for training and test sets.
Calculated MSE, R2, and MAE for both sets.
Printed MSE, R2, and MAE.
Plotted the training loss over epochs using plt.plot(history.history['loss']).
Visualized results with scatter plots for each feature in X_test against y_test (color #16c1c7) and y_test_pred (color #c71677), in a 7x3 grid.





Dependencies

Python 3.x
Pandas for data loading and manipulation
NumPy for numerical operations
Scikit-learn for preprocessing, splitting, model training, and evaluation
Matplotlib and Seaborn for visualizations
TensorFlow for neural network modeling

Dataset
Parkinson's disease telemonitoring dataset: Contains 5,875 records from 42 patients, with voice-derived acoustic features and UPDRS scores collected over time.

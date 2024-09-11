{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Key Takeaways from the model :\n",
    "#### -Age and BMI are strong predictors of insurance charges, with older individuals and those with higher BMIs expected to have higher charges.\n",
    "#### -Number of children has a smaller positive effect.\n",
    "#### -Gender (being male) doesn't seem to affect the charges significantly in this model.\n",
    "#### -Smoking has the most substantial impact, with smokers incurring significantly higher charges."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# -----------------------------------------------------------------\n",
    "\n",
    "##### In this project we will create a ridge regression model to gain an understanding of the relationships between the features of our datasets and insurance charges\n",
    "\n",
    "##### We will start off importing the dataset into a pandas dataframe :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   age     sex     bmi  children smoker      charges\n",
      "0   19  female  27.900         0    yes  16884.92400\n",
      "1   18    male  33.770         1     no   1725.55230\n",
      "2   28    male  33.000         3     no   4449.46200\n",
      "3   33    male  22.705         0     no  21984.47061\n",
      "4   32    male  28.880         0     no   3866.85520\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Specify the file path\n",
    "file_path = '/Users/harshpatel/Desktop/codingproject/regression/expenses.csv'\n",
    "\n",
    "# Load the CSV into a DataFrame\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "# Display the first few rows of the DataFrame\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Now I will to see if there are any missing values in our dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age         0\n",
       "sex         0\n",
       "bmi         0\n",
       "children    0\n",
       "smoker      0\n",
       "charges     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check to see if there are any missing values in our dataset. No missing values exist. \n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Since I do not need to clean missing values, I will proceed. The next steps will be to import the necessary libraries we will use and get dummies for our categorical variables (sex and smoker). "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import necessary libraries\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.preprocessing import StandardScaler\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# There are two categorical variables in our dataset: sex and smoker which are binary columns. We will convert this to a dummy variable. \n",
    "df = pd.get_dummies(df, columns=['sex', 'smoker'], drop_first=True)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Now that the libraries are imported and dummy variables are processed, the next thing to check for is multicollinearity by checking VIF factors : "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      Feature       VIF\n",
      "0         age  1.021030\n",
      "1         bmi  1.014626\n",
      "2    children  1.004466\n",
      "3    sex_male  1.005621\n",
      "4  smoker_yes  1.007995\n"
     ]
    }
   ],
   "source": [
    "# Define features (X) and target (y) variables\n",
    "X = df.drop('charges', axis=1)  # Features (all columns except 'charges')\n",
    "y = df['charges']  # Target variable (insurance charges)\n",
    "\n",
    "\n",
    "#Split the dataset into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "#Standardize the feature variables\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "#Check VIF factors for multi-collinearity amongst features. A high VIF factor indicates a strong correlation amongst variables which reduces the reliability of the model. \n",
    "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
    "import pandas as pd\n",
    "\n",
    "#Use X scaled for preprocessed data\n",
    "vif = pd.DataFrame()\n",
    "vif['Feature'] = X_train.columns\n",
    "vif['VIF'] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]\n",
    "\n",
    "# Display the VIF values\n",
    "print(vif)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Now the model is ready to be trained, fitted and tested : "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      age     bmi  children  sex_male  smoker_yes       Actual     Predicted  \\\n",
      "764    45  25.175         2         0           0   9095.06825   8555.003063   \n",
      "887    36  30.020         0         0           0   5272.17580   6973.844831   \n",
      "890    64  26.885         0         0           1  29330.98315  36797.399178   \n",
      "1293   46  25.745         3         1           0   9301.89355   9418.107438   \n",
      "259    19  31.920         0         1           1  33750.29180  26871.066378   \n",
      "\n",
      "      Percent Difference  \n",
      "764            -5.938000  \n",
      "887            32.276409  \n",
      "890            25.455731  \n",
      "1293            1.249357  \n",
      "259           -20.382714  \n"
     ]
    }
   ],
   "source": [
    "#We will be using a Ridge model. This helps with overfitting and, although we do not have high multicollinearity, helps to maintain it. It also helps explain variability rather than overfitting our model during training. \n",
    "ridge = Ridge(alpha=.05) #alpha set to .05. Tested other alphas. \n",
    "ridge.fit(X_train_scaled, y_train)\n",
    "\n",
    "#Creating predictions using the model. \n",
    "y_pred = ridge.predict(X_test_scaled)\n",
    "\n",
    "\n",
    "\n",
    "#Storing predicted values into dataframe\n",
    "results_df = X_test.copy()  # Copy the test features\n",
    "results_df['Actual'] = y_test  # Add the actual target values\n",
    "results_df['Predicted'] = y_pred  # Add the predicted target values\n",
    "\n",
    "#Add a column for to see percent difference from actual\n",
    "results_df['Percent Difference'] = ((results_df['Predicted'] - results_df['Actual']) / results_df['Actual']) * 100\n",
    "\n",
    "\n",
    "\n",
    "# Print the first few rows to see a sample of the output\n",
    "print(results_df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Let's see the model performance. I want to be careful of overfitting the model as it will not be as reliable. An R^2 of 0.75-0.90 is good to have for our purposes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error: 33979563.00870515\n",
      "R²: 0.7811282405840095\n"
     ]
    }
   ],
   "source": [
    "#Checking model performance. Our model explains 78% of variance is explainable by age, bmi, children, sex and smoker predictors. \n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "print(f'Mean Squared Error: {mse}')\n",
    "\n",
    "\n",
    "from sklearn.metrics import r2_score\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "print(f'R²: {r2}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Now lets take a look at the coefficients in our model. This will help to understand relationships between the predictors and our target variable which is insurance charges : "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      Feature  Coefficient\n",
      "0         age  3616.103798\n",
      "1         bmi  1978.413305\n",
      "2    children   519.283904\n",
      "3    sex_male    -3.942320\n",
      "4  smoker_yes  9559.144012\n"
     ]
    }
   ],
   "source": [
    "coefficients = ridge.coef_\n",
    "\n",
    "coeff_df = pd.DataFrame({\n",
    "    'Feature': X_train.columns,\n",
    "    'Coefficient': coefficients\n",
    "})\n",
    "\n",
    "# Display the coefficients\n",
    "print(coeff_df)\n",
    "\n",
    "#See key takeaways in the first cell for interpretation of coefficients"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

Ridge Regression Model for Insurance Charges

Key Takeaways from the Model:

-Age and BMI are strong predictors of insurance charges, with older individuals and those with higher BMIs expected to have higher charges.

-Number of children has a smaller positive effect.

-Gender (being male) doesn't seem to affect the charges significantly in this model.

-Smoking has the most substantial impact, with smokers incurring significantly higher charges.


Dataset
The dataset includes the following features:

age: Age of the individual

sex: Gender of the individual (male/female)

bmi: Body Mass Index

children: Number of children

smoker: Smoking status (yes/no)

charges: Insurance charges (target variable)




1. Load the Dataset

We start by importing the dataset into a pandas DataFrame.

```python
import pandas as pd

# Load the CSV into a DataFrame
file_path = '/Users/harshpatel/Desktop/codingproject/regression/expenses.csv'
df = pd.read_csv(file_path)
```
```python
# Display the first few rows of the DataFrame
print(df.head())
```

```yaml
Output:


   age     sex     bmi  children smoker      charges
0   19  female  27.900         0    yes  16884.92400
1   18    male  33.770         1     no   1725.55230
2   28    male  33.000         3     no   4449.46200
3   33    male  22.705         0     no  21984.47061
4   32    male  28.880         0     no   3866.85520

```
2. Check for Missing Values

Next, we check whether there are any missing values in the dataset.


```python
# Check for missing values
df.isnull().sum()
```

```yaml
Output:

age         0
sex         0
bmi         0
children    0
smoker      0
charges     0
```

3. Convert Categorical Variables to Dummy Variables

We convert categorical variables like sex and smoker into dummy variables for use in the model.

```python

# Convert categorical variables into dummy variables
df = pd.get_dummies(df, columns=['sex', 'smoker'], drop_first=True)
```

4. Check for Multicollinearity

We check for multicollinearity between the features using Variance Inflation Factor (VIF).

```python

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X = df.drop('charges', axis=1)
X_train_scaled = scaler.fit_transform(X)

# Calculate VIF
vif = pd.DataFrame()
vif['Feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]

# Display VIF values
print(vif)
```

```yaml
Output:

markdown
Copy code
      Feature       VIF
0         age  1.021030
1         bmi  1.014626
2    children  1.004466
3    sex_male  1.005621
4  smoker_yes  1.007995

```

5. Train the Ridge Regression Model


We use the Ridge regression model to train and predict insurance charges.

```python

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['charges'], test_size=0.2, random_state=42)

# Standardize the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Ridge regression model
ridge = Ridge(alpha=0.05)
ridge.fit(X_train_scaled, y_train)

# Predict insurance charges
y_pred = ridge.predict(X_test_scaled)

```

6. Model Performance

```python

from sklearn.metrics import mean_squared_error, r2_score

# Calculate and print Mean Squared Error and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R²: {r2}')
```

```yaml
Output:

Mean Squared Error: 33979563.00870515
R²: 0.7811282405840095
```

7. Coefficients of the Model

Finally, we examine the coefficients of the Ridge model to understand the relationship between each feature and insurance charges.

```python
coefficients = ridge.coef_
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
})

# Display the coefficients
print(coeff_df)

```
```yaml
Output:

      Feature  Coefficient
0         age  3616.103798
1         bmi  1978.413305
2    children   519.283904
3    sex_male    -3.942320
4  smoker_yes  9559.144012
```

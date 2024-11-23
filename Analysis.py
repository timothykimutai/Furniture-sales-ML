# %%
import pandas as pd

# %%
data= pd.read_csv('Furniture.csv')
df = pd.DataFrame(data=data)


# %%
df.head(10)

# %%
negative_revenue = df[df['revenue']< 0]
negative_revenue

# %% [markdown]
# Zero Inventory

# %%
zero_inventory = df[df['inventory'] == 0]
zero_inventory

# %%
"Negative revenue rows:\n", negative_revenue

# %%
"\nZero inventory rows:\n", zero_inventory

# %%
df_cleaned = df[(df['revenue']>=0) & (df['inventory']>0)]

# %%
df_cleaned

# %% [markdown]
# # categorical to numerical

# %%
categorical_coumns =['category', 'material', 'color', 'location', 'season', 'store_type', 'brand']
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_coumns, drop_first=True)

# %% [markdown]
# # Feature Scaling


# %%
import sklearn
from sklearn.preprocessing import StandardScaler

# %%
# Standardize numerical features using StandardScaler
# Numerica columns
numerical_columns = ['price', 'cost', 'sales', 'profit_margin', 'inventory', 'discount_percentage', 'delivery_days', 'revenue']
scaler = StandardScaler()
df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

# %% [markdown]
# # Correlational Analysis

# %%
# The correlation between revenue(target) ad other variables
corr_matrix = df_encoded.corr()

# %%
# Focus on correlation with revenue
revenue_corr = corr_matrix['revenue'].sort_values(ascending=False)

# %%
revenue_corr

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# # Define Features and Target Variable

# %%
X = df_encoded.drop(columns=['revenue'])
y = df_encoded['revenue']

# %%
# Split sets into training and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# %% [markdown]
# Train a linear regression model

# %%
model = LinearRegression()
model.fit(X_train, y_train)

# %%
# Predict
y_predict= model.predict(X_test)
y_predict

# %% [markdown]
# Model Evaluation

# %%
mse = mean_squared_error(y_test, y_predict)
r2 =r2_score(y_test, y_predict)


# %%
"Mean Squared Error:", mse

# %%
"R^2 Score:", r2

# %% [markdown]
# # Visualization

# %%
# Import liraries
import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# Heatmap of correlation matrix

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# %% [markdown]
# Scatter plot of price vs revenue

# %%
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_cleaned, x = 'price', y='revenue', hue='category')
plt.title("Price vs Revenue")
plt.show()

# %%
import joblib

# %%
# Save the model
joblib.dump(model, 'regression-model.pkl')



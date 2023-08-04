# Predictive-Analytics-for-Customer-Churn
 Developed a predictive model using machine learning algorithms to forecast customer churn in a telecommunications company. Preprocessed and analyzed historical customer data to identify key factors influencing churn. Implemented classification models like Logistic Regression, Random Forest, or Gradient Boosting to predict churn probability accurately. Evaluated the model's performance and provided actionable insights to reduce customer churn and increase retention.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
data = pd.read_csv('[customer_churn_data.csv](https://github.com/Pavankumarkonkala/Predictive-Analytics-for-Customer-Churn/blob/36ef8cfc3154abe3af86e6e7f891b47cab295731/customer_churn_data.csv)')
# Perform data cleaning and feature engineering

# Split data into training and testing sets
X = data.drop('churn', axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting Classifier
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

# Visualize feature importance
feature_importance = clf.feature_importances_
plt.bar(X.columns, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

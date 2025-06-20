import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

#https://drive.google.com/file/d/1O-Xpj0OAbv_huRxNZFJRjIMr88w_UZw8/view?usp=drive_link
df= pd.read_csv("/content/drive/MyDrive/Data/emails.csv")
print(df.head())
print(df['spam'].value_counts(),'\n')

sns.countplot(data=df,x=df['spam'])
plt.title("Spam (1) vs Not Spam (0)")
plt.show()

vectorizer = TfidfVectorizer(stop_words = 'english', max_features=5000)
x= vectorizer.fit_transform(df['text'])
y= df['spam']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42,test_size=0.2,stratify=y)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))

probs = model.predict_proba(x_test)[:,1]
fpr,tpr,_=roc_curve(y_test,probs)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label=f"Logistic Regression (AUC = {roc_auc: .2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

import pandas as pd
df=pd.read_csv("Iris.csv")
df.drop(columns=["Id"],inplace=True)
from sklearn.model_selection import train_test_split
X=df.drop(columns=["Species"])
y=df["Species"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler() 
sc.fit(X_train) 
X_train_std=sc.transform(X_train) 
X_test_std=sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn=Perceptron(eta0=0.1,random_state=1)
ppn.fit(X_train_std,y_train)
y_pred=ppn.predict(X_test_std)

from sklearn.metrics import accuracy_score
print("Accuracy Perceptron: %.3f" %accuracy_score(y_test,y_pred))

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=5)
dt.fit(X_train_std,y_train)
y_pred_dt=dt.predict(X_test_std)
print("Accuracy Decision Tree: %.3f"%accuracy_score(y_pred_dt,y_test))

from sklearn.ensemble import RandomForestClassifier
rft=RandomForestClassifier(max_depth=5)
rft.fit(X_train_std,y_train)
y_pred_rft=rft.predict(X_test_std)
print("Accuracy Score RFC: %.3f"%accuracy_score(y_pred_rft,y_test))

SepalLength=float(input("Enter the Sepal Length of the Flower in cms: "))
SepalWidth=float(input("Enter the Sepal Width of the Flower in cms: "))
PetalLength=float(input("Enter the Petal Length of the Flower in cms: "))
PetalWidth=float(input("Enter the Petal Width of the Flower in cms: "))

X_df=[SepalLength,SepalWidth,PetalLength,PetalWidth]
df1=pd.DataFrame([X_df])
df1.columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
X_df1=sc.transform(df1)
ans=dt.predict(X_df1)
print("Predicted Flower Type: ",ans[0])


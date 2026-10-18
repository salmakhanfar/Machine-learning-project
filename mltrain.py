import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

data=pd.read_excel(r"c:\Users\user\Downloads\disease_prediction.xlsx")
label = LabelEncoder()
X=data[['Age','SugerLevel','BMI']]
data['Disease']=label.fit_transform(data['Disease'])
y=data['Disease']

tree=DecisionTreeClassifier()
tree_score=cross_val_score(tree,X,y,cv=5).mean()
rf=RandomForestClassifier(n_estimators=100)
rf_score=cross_val_score(rf,X,y,cv=5).mean()
knn=KNeighborsClassifier(n_neighbors=5)
knn_score=cross_val_score(knn,X,y,cv=5).mean()

result=pd.DataFrame({'Model':['DecisionTree','RandomForest','KNN'],'Accuracy':[tree_score,rf_score,knn_score]})
print(result)

best=max(result['Accuracy'])
worst=min(result['Accuracy'])
print("The best accuracy = ",best)
print("The worst accuracy = ",worst)

model=rf
model.fit(X,y)
prediction=model.predict([[45,180,30]])
result=label.inverse_transform(prediction)
print('prediction = ',result[0])

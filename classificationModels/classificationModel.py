from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
class classificationModel:
    def __init__(self):
        self.supportedModels="Random Forest"  
        self.dataframe=None
        self.datasetName=None
        self.targetFeature=None
        self.test_size=None
    def getConfusionMatrix(self):
        return None
    
class RandomForest(classificationModel):
    def __init__(self,ds,targetFeature,test_size=0.2):
        classificationModel.__init__(self)
        self.supportedModels="Random Forest"  
        self.dataframe=ds
        self.targetFeature=targetFeature
        self.test_size=test_size
    def getMetrics(self):
        features_cols=self.dataframe.columns.tolist()
        features_cols.remove(self.targetFeature)
        X=self.dataframe[features_cols]  # Features
        y=self.dataframe[self.targetFeature].astype('category').cat.codes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        clf=RandomForestClassifier(n_estimators=1000)
        rf = GridSearchCV(RandomForestClassifier(random_state=42), 
                              {'n_estimators': [50,30,20,10], 'max_depth': [3, 5, None]},
                              cv=10)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        acc_score=accuracy_score(y_test, preds)
        cm=confusion_matrix(y_test, preds,labels=y.unique().tolist())
        f1_Score=f1_score(y_test, preds, average='macro')
        return cm,acc_score,f1_Score
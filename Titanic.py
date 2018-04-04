import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, average_precision_score, f1_score, precision_score
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def wrangle(data):
    #data = data.drop(['Name'], axi s=1)

    data['Ticket'] = data['Ticket'].replace(np.NaN,'X')
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Sex'] = data['Sex'].replace(np.NaN,'WAS_BLANK')

    #calculate how frequently data values occur in 'Tickets' column
    frequency = data['Ticket'].value_counts()
    #create a list as temporary storage of results
    temp = []
    #'row' refers to the value in the 'Tickets' column
    for row in data['Ticket']:
        temp.append(frequency[row])

    data['Tckt_Share'] = temp
    data = data.drop(['Ticket'],axis=1)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    #seperate leading letter and numbers, add column for multiple cabins??
    data['Cabin'] = data['Cabin'].replace(np.NaN,'None')
    
    #calculate how frequently data values occur in 'Cabin' column
    frequency2 = data['Cabin'].value_counts()
    #create a list as temporary storage of results
    temp2 = []
    #'row' refers to the value in the 'Tickets' column
    for row in data['Cabin']:
        if frequency2[row] > 10:
            temp2.append(10)
        else:
            temp2.append(frequency2[row])

    data['Cabin_Share'] = temp2

    data['Embarked'] = data['Embarked'].replace(np.NaN,'XX')

    
    return(data)

def OneHot(X):
    X_CAT = X.drop(CAT_DROP,axis=1)
    X_CONT = X.drop(CONT_DROP,axis=1)
 
    label_encoder = LabelEncoder()
    
    
    #labelencoder is usually only designed to encode a single column
    X_CAT = X_CAT.apply(LabelEncoder().fit_transform)
    #print(X_CAT.head())
    #print(X_CAT.dtypes) 
    
    #OneHotLabels:
    enc = preprocessing.OneHotEncoder()
    enc.fit(X_CAT)
    onehotlabels = enc.transform(X_CAT).toarray()   
    
    X_TOTAL_NP_Col = np.concatenate([onehotlabels, X_CONT],axis=1)
    #print(type(X_TOTAL_NP_Col))
    #X_TOTAL_PD_Col =  pd.concat([X_CAT, X_CONT], axis = 1)
    #test which features make what impact:
    #X_TOTAL_PD_Col = X_TOTAL_PD_Col.drop(['Cabin','Embarked','Age','SibSp','Parch','Tckt_Share','Cabin_Share'],axis=1)

    return(X_TOTAL_NP_Col)
  
def make_prediction():
        test_data_url = 'test.csv'
        test_data = pd.read_csv(test_data_url)
        ##create frame with PassengerId
        result = test_data.filter(['PassengerId'],axis=1)
        test_data = wrangle(test_data)
        test_data = test_data.drop(['Name','PassengerId'],axis=1)
        test_data = OneHot(test_data)
        

        test_predict = clf.predict(test_data)
        #print(type(test_predict))
        #add results to DF with Passengger ID
        result['Survived'] = test_predict
        print(result)
        #make the PassengerID the index of the DF
        result.set_index('PassengerId',inplace=True)
        result.to_csv('submission.csv', sep=',', encoding ='utf-8')
    
def Correlation(data):
    #investigate which features are most closlely correlated - convert categorical features into numeric (only "Sex" is interesting here:)
    new = pd.get_dummies(data['Sex'])
    data = pd.concat([data,new],axis=1)
    data=data.drop('Sex',axis=1)
    print(data.corr())
    
if __name__ == "__main__":
    dataset_url = 'train.csv'
    data = pd.read_csv(dataset_url)
    data = wrangle(data)
    #Correlation(data)
    y = data.Survived
    X = data.drop(['Survived','Name','PassengerId'],axis=1)
    CAT_DROP = ['Parch','Age','SibSp','Fare','Tckt_Share','Cabin_Share','Embarked','Cabin']
    CONT_DROP = ['Pclass','Embarked','Cabin','Sex','Parch','Age','SibSp','Tckt_Share']
    
    #CAT_DROP = ['Parch','Age','SibSp','Fare','Tckt_Share','Cabin_Share']
    #CONT_DROP = ['Pclass','Embarked','Cabin','Sex']
    
    X = OneHot(X)
    
    #X = data.drop(['Pclass'],axis=1)
    #print(X.head())
    #create list for models
    models = []
    #models.append(('QDA', QuadraticDiscriminantAnalysis()))
    #models.append(('GNB', GaussianNB()))
    #models.append(('SVC', SVC()))
    #models.append(('KNN', KNeighborsClassifier()))
    #models.append(('RFC', RandomForestClassifier(n_estimators=100)))
    #models.append(('DTC', DecisionTreeClassifier()))
    #models.append(('MLP', MLPClassifier(max_iter=1200)))
    #models.append(('ADB', AdaBoostClassifier()))
    models.append(('GPC', GaussianProcessClassifier()))

    for name, model in models:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
        
        print(name)
        #create pipeline - normalize data and select method
        pipeline = make_pipeline(preprocessing.StandardScaler(), model)
        #display available hyperparameters:
        #print(sorted(pipeline.get_params().keys()))
        #display available hyperparameters
        #print(pipeline.get_params())
        #define hyperparamaters to tune
        #hyperparameters1 = {'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],
        #              'randomforestclassifier__max_depth': [None, 10, 5, 3, 1] }
        #blank hyperparameters to use when testing across all algorithms
        hyperparameters = {}
        #range of hyperparameters to use in hte best algorithm (KNN is this case)
        #hyperparameters = {
         #               'kneighborsclassifier__n_neighbors':[5,6,7,8,9,10],
          #              'kneighborsclassifier__leaf_size':[1,2,3,5],
           #             'kneighborsclassifier__weights':['uniform', 'distance'],
            #            'kneighborsclassifier__algorithm':['auto', 'ball_tree','kd_tree','brute'],
             #           'kneighborsclassifier__n_jobs':[-1],
              #          'kneighborsclassifier__metric':['euclidean','cityblock', 'manhattan'],
               #         }
        #best parameters as discovered above (for KNN)
        #best_hyperparameters = {
         #               'kneighborsclassifier__n_neighbors':[9],
          #              'kneighborsclassifier__leaf_size':[1],
           #             'kneighborsclassifier__weights':['uniform'],
            #            'kneighborsclassifier__algorithm':['auto'],
             #           'kneighborsclassifier__n_jobs':[-1],
              #          'kneighborsclassifier__metric':['euclidean'],
               #         }
        #carry out cross validation pipeline (tests training data against all hyperparameter permutations)
        clf = GridSearchCV(pipeline, hyperparameters, cv=25)
        # Fit and tune model
        clf.fit(X_train, y_train)
        #print('best hyperparameters are\n',clf.best_para
        
        #predict target against test data
        y_pred = clf.predict(X_test)
        #test prediction against actual test data
        print(accuracy_score(y_test, y_pred))
        #print(average_precision_score(y_test, y_pred))
        #print(f1_score(y_test, y_pred))

make_prediction()
        


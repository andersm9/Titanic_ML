# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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

def Conversion(combine):
    #convert 'Sex" feature to numberic:
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    #Convert age into agebands:
    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16),  'Age'] = 1

        
    #Convert 'Embarked' into ordinals:  
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, 'U':3} ).astype(int)
    
    #Create Banding for Tckt_Share

    for dataset in combine:
        dataset.loc[(dataset['Tckt_Share'] > 0) & (dataset['Tckt_Share'] <=4) , 'Tckt_Share'] = 0
        dataset.loc[(dataset['Tckt_Share'] > 4), 'Tckt_Share'] = 1
        dataset['Tckt_Share'] = dataset['Tckt_Share'].astype(int)    

    #Create a banding for fare
    
    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
    
    return(combine)

def Drop(combine):
    combine = combine.drop(['Name','Ticket','Cabin','PassengerId','SibSp','Parch'],axis=1)
    return(combine)

def Fill(df):
    df['Embarked'] = df['Embarked'].replace(np.NaN,'U')
    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
    df['Age'].fillna(df['Age'].dropna().median(), inplace=True)
    return(df)


    
def make_prediction(test):
        ##create frame with PassengerId

        result = test.filter(['PassengerId'],axis=1)
        test = Drop(test)
        test_predict = clf.predict(test)
        #print(type(test_predict))
        #add results to DF with Passengger ID
        result['Survived'] = test_predict
        #make the PassengerID the index of the DF
        result.set_index('PassengerId',inplace=True)
        result.to_csv('submission.csv', sep=',', encoding ='utf-8')

def T_Share(data):
    #calculate how frequently data values occur in 'Tickets' column
    frequency = data['Ticket'].value_counts()
    #create a list as temporary storage of results
    temp = []
    #'row' refers to the value in the 'Tickets' column
    for row in data['Ticket']:
        temp.append(frequency[row])

    data['Tckt_Share'] = temp

        
    return(data)

if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    train_df = Fill(train_df)
    test_df = Fill(test_df)
    train_df = T_Share(train_df)
    test_df = T_Share(test_df)

    combine = [train_df, test_df]
    combine = Conversion(combine)
    
    
    #use the following to compare the connection between "survived" and values in "age":
    #print(train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=False))
    
    
    
    train_df = Drop(train_df)
    
    y = train_df.Survived
    X = train_df.drop(['Survived'],axis=1)

    models = []
    #models.append(('QDA', QuadraticDiscriminantAnalysis()))
    #models.append(('GNB', GaussianNB()))
    #models.append(('SVC', SVC()))
    #models.append(('KNN', KNeighborsClassifier()))
    #models.append(('RFC', RandomForestClassifier(n_estimators=100)))
    models.append(('DTC', DecisionTreeClassifier()))
    #models.append(('MLP', MLPClassifier(max_iter=1200)))
    #models.append(('ADB', AdaBoostClassifier()))
    #models.append(('GPC', GaussianProcessClassifier()))

    for name, model in models:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123, stratify=y)
        
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
        clf = GridSearchCV(pipeline, hyperparameters, cv=10)
        # Fit and tune model
        clf.fit(X_train, y_train)
        #print('best hyperparameters are\n',clf.best_para
        
        #predict target against test data
        y_pred = clf.predict(X_test)
        #test prediction against actual test data
        print(accuracy_score(y_test, y_pred))
        #print(average_precision_score(y_test, y_pred))
        #print(f1_score(y_test, y_pred)

make_prediction(test_df)

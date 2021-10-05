from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

model = None
df = None
Result = ""
CroosValide = None
Percentage = None
result = None
X_train = None
X_test = None
Y_train = None
Y_test = None
TestSpam = ""
Clean = ""
def Clean(request):
    global df
    global Clean
    df['label'].fillna("ham", inplace=True)
    df['message'].fillna("Hi", inplace=True)
    Clean = "Your data is clean now !!"
    context = {'Clean':Clean}
    return render(request,'Detection/cleandata.html',context)


def Test(request):
    global TestSpam
    if(request.method == "POST"):
        TEXT = request.POST['w3review']
        Model = joblib.load(os.path.dirname(__file__) + "\\MyModel.pkl")
        finalAns = Model.predict([TEXT])[0]
        if(finalAns == "ham"):
            TestSpam = "This message is not a SPAM"
        else :
            TestSpam = "This message is a SPAM "

    context = {'TestSpam':TestSpam}
    return render(request,'Detection/result.html',context)

def home(request):
    return render(request,'Detection/home.html')

def parameter(request):
    global result
    if request.method == "POST":
            result = request.POST['algorithm']
            if (result == '1'):
                context = {'result': result}
                return render(request, 'Detection/parameter_1.html', context)
            elif (result =='2'):
                    context = {'result': result}
                    return render(request, 'Detection/parameter_2.html', context)
            elif (result =='3'):
                context = {'result': result}
                return render(request, 'Detection/parameter_3.html', context)
            elif (result =='4'):
                context = {'result': result}
                return render(request, 'Detection/parameter_4.html', context)
            elif (result =='5'):
                context = {'result': result}
                return render(request, 'Detection/parameter_5.html', context)
            elif (result =='6'):
                context = {'result': result}
                return render(request, 'Detection/parameter_6.html', context)

def CreatModel(request):

    global model
    global X_train, X_test, Y_train, Y_test , x , y
    global Result
    SavedModel = model
    Acc = 0
    if(CroosValide == None):
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        print(confusion_matrix(Y_test, Y_pred))
        print(classification_report(Y_test, Y_pred))
        print(accuracy_score(Y_test, Y_pred))

        Result = str(classification_report(Y_test, Y_pred))
        Result +="\nAccuracy Score : "

        Result += str(accuracy_score(Y_test, Y_pred))
        Result += "\n"

        Result +="\nConfusion Matrix : \n"

        Result += str(confusion_matrix(Y_test, Y_pred))
        context = { 'Result' : Result }
        joblib.dump(model, os.path.dirname(__file__) + "\\MyModel.pkl")

    else :
        skf = StratifiedKFold(n_splits = CroosValide)
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x.loc[train_index], x.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            print("Accuracy: " + accuracy.__str__())


            if(Acc < accuracy):
                SavedModel = model
                Acc = accuracy
                Result = str(classification_report(y_test, predictions))
                Result +="\nAccuracy Score : "

                Result += str(accuracy_score(y_test, predictions))
                Result += "\n"

                Result +="\nConfusion Matrix : \n"

                Result += str(confusion_matrix(y_test, predictions))
                context = { 'Result' : Result }

        joblib.dump(SavedModel, os.path.dirname(__file__) + "\\MyModel.pkl")



    return render(request,'Detection/startmodel.html',context)

def startmodel(request):
    global model
    global X_train, X_test, Y_train, Y_test , x , y
    if request.method == "POST":

        hamDf = df[df['label'] == "ham"]
        spamDf = df[df['label'] == "spam"]

        hamDf = hamDf.sample(spamDf.shape[0])

        finalDf = hamDf.append(spamDf ,ignore_index = True)

        X_train, X_test, Y_train, Y_test = train_test_split(finalDf['message'], finalDf['label'], test_size = Percentage, random_state = 0, shuffle = True, stratify = finalDf['label'])

        if (result == '1'):
            N_neighbors = request.POST["parameter1_neighbors"]
            model = Pipeline([('tfidf',TfidfVectorizer()),('model' ,  KNeighborsClassifier(n_neighbors = int(N_neighbors)))])
        elif (result =='2'):
            c = request.POST["parameter2_c"]
            g = request.POST["parameter2_gamma"]
            model = Pipeline([('tfidf',TfidfVectorizer()),('model' , SVC(C = int(c), gamma = g))])

        elif (result =='3'):
            model = Pipeline([('tfidf',TfidfVectorizer()),('model' , DecisionTreeClassifier())])

        elif (result =='4'):
            Job = request.POST["parameter4_estimator"]
            Est = request.POST["parameter4_jobs"]
            model = Pipeline([('tfidf',TfidfVectorizer()),('model' , RandomForestClassifier(n_estimators = Est, n_jobs = Job))])

        elif (result =='5'):
            S = request.POST["parameter5_solver"]
            P = request.POST["parameter5_penalty"]

            model = Pipeline([('tfidf',TfidfVectorizer()),('model' , LogisticRegression(solver='liblinear', penalty='l1'))])


        elif (result =='6'):
            model = Pipeline([('tfidf',TfidfVectorizer()),('model' , MultinomialNB())])
        if(CroosValide != None):
            x = finalDf['message']
            y = finalDf['label']

    return render(request,'Detection/startmodel.html')

def cleandata(request):
    global df
    global Clean
    if request.method == "POST":
        File = request.FILES['file']
        df=pd.read_csv(File, sep="\t")
        print(df)
    context = {'Clean':Clean}
    return render(request,'Detection/cleandata.html',context)

def classify(request):
    global CroosValide
    global Percentage
    if request.method == "POST":
        result = request.POST['parameter2_gamma']
        if(result == "1"):
            Value = request.POST['test_1']
            CroosValide = int(Value)
            Percentage = 0.2
        if(result == "2"):
            Value = request.POST['test_2']
            CroosValide = None
            Percentage = int(Value)/100
        if(result =="3"):
            Percentage = 0.2

    print(Percentage)


    return render(request,'Detection/classify.html',)

def testoptions(request):

    return render(request,'Detection/test_options.html',)

def result(request):

    return render(request,'Detection/result.html',)

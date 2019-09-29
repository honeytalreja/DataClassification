import sklearn
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("pima-indians-diabetes.data.csv")
print("number of data and number of features respectively : "+str(data.shape))

def main(train, test, percent):
    best_x = ['kNN', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVC']
    best_y = []

    train_X = train[train.columns[:8]]
    test_X = test[test.columns[:8]]
    train_Y = train['Outcome']
    test_Y = test['Outcome']
    listy=[]
    listx=[]
    knn_max, k_max = 0, 0

    for i in range(5,35):

        kn=KNeighborsClassifier(n_neighbors=i, algorithm='auto')
        kn.fit(train_X,train_Y)
        prediction4=kn.predict(test_X)
        # print("K-Nearest Neighbour Classifier : ")
        answer=metrics.accuracy_score(test_Y,prediction4)
        # print(str(answer*100)+"% accuracy for k=", i)
            # listx.append(i)
        answer *= 100
        listx.append(i)
        listy.append(answer)

        if answer > knn_max:
            knn_max, k_max = answer, i
        # print(confusion_matrix(prediction4,test_Y))

    plt.title('Accuracy of KNN for various values of k')
    plt.annotate('neighbors='+str(k_max), (k_max, knn_max))
    plt.plot(listx,listy)
    plt.ylabel("Accuracy")
    plt.xlabel("K-Value")
    plt.show()

    print('KNN prediction for highest accuracy:')
    print('Maximum accuracy is', (knn_max), 'for k =', k_max, '\n')
    best_y.append(knn_max)


    model = LogisticRegression(C=1.8)
    model.fit(train_X,train_Y)
    prediction = model.predict(test_X)
    answer = metrics.accuracy_score(test_Y, prediction)
    answer *= 100
    print("Logistic Regression Prediction :", answer)
    print(confusion_matrix(prediction,test_Y))
    best_y.append(answer)

    dt_x = []
    dt_y = []
    dt_max = 0
    dt_max_depth = 0
    for i in range (2,20):
        dt=DecisionTreeClassifier(max_depth=i,random_state=0)
        dt.fit(train_X,train_Y)
        prediction3 = dt.predict(test_X)
        answer = metrics.accuracy_score(test_Y,prediction3)
        answer *= 100
        if answer > dt_max:
            dt_max = answer
            dt_max_depth = i
        dt_x.append(i)
        dt_y.append(answer)

    plt.plot(dt_x, dt_y)
    plt.annotate('depth='+str(dt_max_depth), (dt_max_depth, dt_max))
    plt.title('Decision tree variation of max depth vs accuracy')
    plt.show()
    print('Maximum accuracy for decision tree is at max_depth =', dt_max_depth)
    print('Maximum accuracy is', dt_max)
    best_y.append(dt_max)
    print(confusion_matrix(prediction3,test_Y))



    rf_estimators = 0
    rf_max = 0
    rf_x, rf_y = [], []
    for i in range(20, 100, 5):
        rfc=sklearn.ensemble.RandomForestClassifier(n_estimators=i,random_state=0)
        rfc.fit(train_X,train_Y)
        prediction1=rfc.predict(test_X)
        answer = metrics.accuracy_score(test_Y,prediction1)
        answer *= 100
        if answer > rf_max:
            rf_max = answer
            rf_estimators = i
        rf_x.append(i)
        rf_y.append(answer)
        # print("Random Forest Classifier : ")
    # print(confusion_matrix(prediction1,test_Y))

    plt.plot(rf_x, rf_y)
    plt.annotate('estimators='+str(rf_estimators), (rf_estimators, rf_max))
    plt.title('RF variation of estimators vs accuracy')
    plt.show()
    print('Best accuracy for random forest is found at estimators =', rf_estimators)
    print('best accuracy is = ', rf_max)
    best_y.append(rf_max)

    svc = SVC(gamma='auto')
    svc.fit(train_X, train_Y)
    y_pred = svc.predict(test_X)
    accuracy = accuracy_score(y_pred, test_Y)
    accuracy *= 100
    print('Accuracy of SVC:', accuracy)
    best_y.append(accuracy)

    plt.scatter(best_x, best_y)
    plt.title('Models and their accuracy')

    for i in range(len(best_x)):
        plt.annotate(percent,(best_x[i],best_y[i]))

    plt.show()


print('\n\nFor test size: 20%\n')
train, test = train_test_split(data, test_size=0.20, random_state=0, stratify=data['Outcome'])
main(train, test,"20%")

print('\n\nFor test size: 30%\n')
train, test = train_test_split(data, test_size=0.30, random_state=0, stratify=data['Outcome'])
main(train, test,"30%")

print('\n\nFor test size: 40%\n')
train, test = train_test_split(data, test_size=0.40, random_state=0, stratify=data['Outcome'])
main(train, test,"40%")

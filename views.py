
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime

import re
import string

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,search_ratio_model,drowsiness_details_model,drowsiness_classification_model,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "SProvider" and password =="SProvider":
            drowsiness_details_model.objects.all().delete()
            search_ratio_model.objects.all().delete()
            drowsiness_classification_model.objects.all().delete()

            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')


def viewtreandingquestions(request,chart_type):
    dd = {}
    pos,neu,neg =0,0,0
    poss=None
    topic = drowsiness_details_model.objects.values('ratings').annotate(dcount=Count('ratings')).order_by('-dcount')
    for t in topic:
        topics=t['ratings']
        pos_count=drowsiness_details_model.objects.filter(topics=topics).values('names').annotate(topiccount=Count('ratings'))
        poss=pos_count
        for pp in pos_count:
            senti= pp['names']
            if senti == 'positive':
                pos= pp['topiccount']
            elif senti == 'negative':
                neg = pp['topiccount']
            elif senti == 'nutral':
                neu = pp['topiccount']
        dd[topics]=[pos,neg,neu]
    return render(request,'SProvider/viewtreandingquestions.html',{'object':topic,'dd':dd,'chart_type':chart_type})

def Test_Driver_Drowsiness_DataSet_Details(request): # k-NN Test Based Results
        search_ratio_model.objects.all().delete()
        ratio=""
        kword = 'Head Movement'
        print(kword)
        obj = drowsiness_classification_model.objects.all().filter(Q(classification=kword))
        obj1 = drowsiness_classification_model.objects.all()
        count =obj.count();
        count1 = obj1.count();
        ratio=(count/count1)*100
        if ratio != 0:
            search_ratio_model.objects.create(names=kword, ratio=ratio)

        ratio1 = ""
        kword1 = 'Eye Blinking'
        print(kword1)
        obj1 = drowsiness_classification_model.objects.all().filter(Q(classification=kword1))
        obj11 = drowsiness_classification_model.objects.all()
        count1 = obj1.count();
        count11 = obj11.count();
        ratio1 = (count1 / count11) * 100
        if ratio1 != 0:
            search_ratio_model.objects.create(names=kword1, ratio=ratio1)

        ratio12 = ""
        kword12 = 'Depression'
        print(kword12)
        obj12 = drowsiness_classification_model.objects.all().filter(Q(classification=kword12))
        obj112 = drowsiness_classification_model.objects.all()
        count12 = obj12.count();
        count112 = obj112.count();
        ratio12 = (count12 / count112) * 100
        if ratio12 != 0:
            search_ratio_model.objects.create(names=kword12, ratio=ratio12)

        obj = search_ratio_model.objects.all()
        return render(request, 'SProvider/Test_Driver_Drowsiness_DataSet_Details.html', {'list_objects': obj,'ratio': ratio})


def Train_Driver_Drowsiness_DataSetDetails(request):
    detection_accuracy.objects.all().delete()
    df = pd.read_csv('Driver_DataSets.csv')
    df
    df.columns
    df.rename(columns={'driver_drowsiness_status': 'dstatus', 'causeOf_drowsiness': 'cdrow'}, inplace=True)

    def apply_results(results):
        if (results == 'No Accident'):
            return 0  # No Accident
        elif (results == 'About to Accident'):
            return 1  # About to Accident
        elif (results == 'Accident'):
            return 2  # Accident

    df['results'] = df['cdrow'].apply(apply_results)

    X = df['dstatus']
    y = df['results']

    cv = CountVectorizer()

    x = cv.fit_transform(X)

    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    X_train.shape, X_test.shape, y_train.shape

    print("Naive Bayes")

    from sklearn.naive_bayes import MultinomialNB

    NB = MultinomialNB()
    NB.fit(X_train, y_train)
    predict_nb = NB.predict(X_test)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    print("ACCURACY")
    print(naivebayes)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_nb))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_nb))
    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)

    # SVM Model
    print("SVM")
    from sklearn import svm

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print("ACCURACY")
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

    print("Logistic Regression")

    from sklearn.linear_model import LogisticRegression

    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, y_pred) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    print("Decision Tree Classifier")
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtcpredict = dtc.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, dtcpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, dtcpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, dtcpredict))
    detection_accuracy.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, dtcpredict) * 100)

    print("SGD Classifier")
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=0)
    sgd_clf.fit(X_train, y_train)
    sgdpredict = sgd_clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, sgdpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, sgdpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, sgdpredict))
    detection_accuracy.objects.create(names="SGD Classifier", ratio=accuracy_score(y_test, sgdpredict) * 100)

    print("KNeighborsClassifier")
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier()
    kn.fit(X_train, y_train)
    knpredict = kn.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, knpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, knpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, knpredict))
    detection_accuracy.objects.create(names="KNeighborsClassifier", ratio=accuracy_score(y_test, knpredict) * 100)

    labeled = 'labeled_data.csv'
    df.to_csv(labeled, index=False)
    df.to_markdown

    se = ''
    # K-Nearest Neighbour
    obj1 = drowsiness_details_model.objects.values('names',
                                                    'reg_State',
                                                    'driver_dl_no',
                                                    'vehicle_status',
                                                    'driver_steering_behavior',
                                                    'driver_drowsiness_status',
                                                    'causeOf_drowsiness',
                                                    'detection_speed',
                                                    'detection_place',
                                                    'detection_city',
                                                    'detection_time',
                                                    'detection_date')

    drowsiness_classification_model.objects.all().delete()
    for t in obj1:
        names = t['names']
        reg_State= t['reg_State']
        driver_dl_no= t['driver_dl_no']
        vehicle_status= t['vehicle_status']
        driver_steering_behavior= t['driver_steering_behavior']
        driver_drowsiness_status= t['driver_drowsiness_status']
        causeOf_drowsiness= t['causeOf_drowsiness']
        detection_speed= t['detection_speed']
        detection_place= t['detection_place']
        detection_city= t['detection_city']
        detection_time= t['detection_time']
        detection_date= t['detection_date']

        for f in driver_drowsiness_status.split():
            if f in ('head','headshake'):
                se = 'Head Movement'
            elif f in ('eyes','Nap','sleepy'):
                se = 'Eye Blinking'
            elif f in ('Depression','deprivation','tired','restless'):
                se = 'Depression'

        drowsiness_classification_model.objects.create(names=names,
            reg_State=reg_State,
            driver_dl_no=driver_dl_no,
            vehicle_status=vehicle_status,
            driver_steering_behavior=driver_steering_behavior,
            driver_drowsiness_status=driver_drowsiness_status,
            causeOf_drowsiness=causeOf_drowsiness,
            detection_speed=detection_speed,
            detection_place=detection_place,
            detection_city=detection_city,
            detection_time=detection_time,
            detection_date=detection_date,
            classification=se)

    obj = drowsiness_classification_model.objects.all()

    return render(request, 'SProvider/Train_Driver_Drowsiness_DataSetDetails.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = drowsiness_details_model.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def negativechart(request,chart_type):
    dd = {}
    pos, neu, neg = 0, 0, 0
    poss = None
    topic =drowsiness_details_model.objects.values('ratings').annotate(dcount=Count('ratings')).order_by('-dcount')
    for t in topic:
        topics = t['ratings']
        pos_count = drowsiness_details_model.objects.filter(topics=topics).values('names').annotate(topiccount=Count('ratings'))
        poss = pos_count
        for pp in pos_count:
            senti = pp['names']
            if senti == 'positive':
                pos = pp['topiccount']
            elif senti == 'negative':
                neg = pp['topiccount']
            elif senti == 'nutral':
                neu = pp['topiccount']
        dd[topics] = [pos, neg, neu]
    return render(request,'SProvider/negativechart.html',{'object':topic,'dd':dd,'chart_type':chart_type})


def charts(request,chart_type):
    chart1 = search_ratio_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Driver_DataSet_Details(request):
    obj =drowsiness_details_model.objects.all()
    return render(request, 'SProvider/View_Driver_DataSet_Details.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =drowsiness_details_model.objects.values('names').annotate(dcount=Avg('Development'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})

def View_Driver_Drowsiness_Classification_Details(request):

    obj = drowsiness_classification_model.objects.all()
    return render(request, 'SProvider/View_Driver_Drowsiness_Classification_Details.html', {'list_objects': obj})

def View_Test_Results(request):
    obj = search_ratio_model.objects.all()
    return render(request, 'SProvider/View_Test_Results.html', {'list_objects': obj})








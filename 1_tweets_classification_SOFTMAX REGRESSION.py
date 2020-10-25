# -*- coding: utf-8 -*-
"""
Created on Thursday Oct 15 14:37:48 2020

@author: Ivan Martino

This is a Quick and Dirty algorithm for tweets classifications
using a Softmax Regression algorithm from mlxtend package

There are few important variables to set:
PATH, FILE_USERS, FILE_INFO, FOLDER, FILE_DATA_BASE.

PATH="..." contains the directory location, where to read and store the file.

FILE_USERS="..." is a text file where to store the screen_names of the Twitter users.
For example, the content of FILE_USERS="users.txt" could be

users.txt
BOF
@realDonaldTrump
@JoeBiden
@KamalaHarris
@Mike_Pence
EOF

FILE_INFO="..." this should contain the name of the file where to store the account information,
i.e. FILE_INFO = 'info'. Depending of the needs, this could be a txt-file or a pny-file.

FOLDER="..." is the name of the folder to place all information we get from TWITTER

FILE_DATA_BASE="..." is the root of the file names for the data we are creating
"""

#for the right folder and to save and load list
import os
import json
import sys

#to save sparce matrix
import pickle

#needed first to writes the tweets
import numpy as np
import random

#to get today
from datetime import date

#nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer

#to split the dataset
from sklearn.model_selection import train_test_split

#to standardize the inputs
from sklearn.preprocessing import StandardScaler

#pacchetti utili per la parte tre
from mlxtend.classifier import SoftmaxRegression

#to print output
import pandas

#VARIABLES
##string
PATH=""
FILE_USERS="users.txt" #you may want to change this
LANGUAGES = ['en'] #this is the lenguage 'it'=Italian, 'en'=English
FOLDER="" #folder where to save the files of the tweets
FILE_DATA_BASE = '' #this is the root of the file names for the data we are creating
FILE_INFO = '' #summary file



"""
We set the right path
"""
def setpath():
    os.chdir(PATH)
    return True


def load_data():
    with open(FOLDER+'/all_tweets.txt', 'r') as g:
        return json.loads(g.read())

"""
The function starndardize the data with mean 0 and variance 1
"""
def standardize_data(X_train, X_dev, X_test):
    scale = StandardScaler(with_mean=0, with_std=1)
    scale.fit(X_train)
    new_X_train = scale.transform(X_train)
    new_X_dev = scale.transform(X_dev)
    new_X_test = scale.transform(X_test)
    return new_X_train, new_X_dev, new_X_test


"""
We read the user screen names from FILE.
"""
def get_users(FILE):
    if os.path.isfile(FILE)==True:
        g=open(FILE, 'r' ,encoding='utf-8')
        NAMES = [l.strip('\n\r') for l in g.readlines()]
        g.close()
    else:
        print("Hej, there is no users to access API-TWITTER.\n Sorry we stop here!")
        sys.exit()

    a=-1;
    while a != False:
        a=clean(NAMES)
        if a!=False:
            NAMES.remove(a)

    print("We will analyze", len(NAMES), "twitter users.")
    return NAMES

"""
This function cleans a list of twitter users from double names.
"""
def clean(List):
    ''' Check if given list contains any duplicates '''
    S = set()
    for e in List:
        if e in S:
            return e
        else:
            S.add(e)
    return False

"""
The function returns the np.array
["@users", #ofTweetsGotten, #lasttweetsid]
"""
def get_info():
    return np.load(FOLDER+"/"+FILE_INFO +".npy")


def subset(Vectotized_tweets, Y_output, List):
    TT=[]
    if len(Vectotized_tweets)!=len(Y_output):
        print("Inputs have wrong domension!")
        return False
    for i in range(len(Vectotized_tweets)):
        if Y_output[i] in List:
            TT.append([Vectotized_tweets[i], Y_output[i]])

    return TT

"""
Provide the argmax of an array.
"""
def to_classlabel(z):
    return z.argmax(axis=1)


"""
The function provides accuracies and precisions for a learning model.

Input
        -model
        -d, if d=0, we call y_pred = to_classlabel(model.predict(X_train))
            if d=1, we call y_pred = model.predict(X_train)
            id d=2, we call y_pred = to_classlabel(model.predict(X_train.toarray()))
        -screen_names, the list of users
        and also
        -X_train,
        -X_dev,
        -Y_train,
        -Y_dev

Does
        -print score matrix
        -print false positive matrix and accuracies
        -print precision matrix
        for both training set and development set


Output
        -it provides pandas tables for
        -print score matrix
        -print false positive matrix and accuracies
        -print precision matrix
        of development set (only)
"""
def compute_accuracies(model, d, screen_names, X_train, X_dev, Y_train, Y_dev):
    if d==0:
        y_pred = to_classlabel(model.predict(X_train))
    elif d==1:
        y_pred = model.predict(X_train)
    elif d==2:
        y_pred = to_classlabel(model.predict(X_train.toarray()))

    U=len(screen_names)

    #compute accuracies on the training set
    score=np.zeros((U,U))
    acc=np.zeros((U,U))
    pre=np.zeros((U,U))
    tot=[0]*U
    est=[1]*U#Laplace smothing

    for i in range(len(y_pred)):
        tot[Y_train[i]]=tot[Y_train[i]]+1
        est[y_pred[i]]=est[y_pred[i]]+1
        score[y_pred[i],Y_train[i]]=score[y_pred[i],Y_train[i]]+1

    print("Score matrix on training set")
    df_score = pandas.DataFrame(score, columns=screen_names, index=screen_names)
    print(df_score)
    print("\n")

    for i in range(U):
        for j in range(U):
            acc[i,j]=score[i,j]/tot[j]
    print("False positive matrix on the training set\n -- the diagonal shows accuracy")
    print("--i.e. the score matrix divided by the total number of existent outcomes with that label")
    df_fp = pandas.DataFrame(acc, columns=screen_names, index=screen_names)
    print(df_fp)
    print("\n")


    for i in range(U):
        for j in range(U):
            pre[i,j]=score[i,j]/est[i]
    print("False positive (estimate) matrix on the training set\n -- the diagonal shows precision: correct estimate/number of estimate")
    print("--i.e. the score matrix divided by the total number of estimate with that label")
    df_pre = pandas.DataFrame(pre, columns=screen_names, index=screen_names)
    print(df_pre)
    print("\n")

    if d==0:
        y_pred = to_classlabel(model.predict(X_dev))
    elif d==1:
        y_pred = model.predict(X_dev)
    elif d==2:
        y_pred = to_classlabel(model.predict(X_dev.toarray()))


    #compute accuracies on the training set
    score=np.zeros((U,U))
    acc=np.zeros((U,U))
    pre=np.zeros((U,U))
    tot=[0]*U
    est=[1]*U#Laplace smothing

    for i in range(len(y_pred)):
        tot[Y_dev[i]]=tot[Y_dev[i]]+1
        est[y_pred[i]]=est[y_pred[i]]+1
        score[y_pred[i],Y_dev[i]]=score[y_pred[i],Y_dev[i]]+1

    print("=======================")
    print("Score matrix on development set")
    df_score = pandas.DataFrame(score, columns=screen_names, index=screen_names)
    print(df_score)
    print("\n")

    for i in range(U):
        for j in range(U):
            acc[i,j]=score[i,j]/tot[j]
    print("False positive matrix on the development set\n -- the diagonal shows accuracy")
    print("--i.e. the score matrix divided by the total number of existent outcomes with that label")
    df_fp = pandas.DataFrame(acc, columns=screen_names, index=screen_names)
    print(df_fp)
    print("\n")


    for i in range(U):
        for j in range(U):
            pre[i,j]=score[i,j]/est[i]
    print("False positive (estimate) matrix on the development set\n -- the diagonal shows precision: correct estimate/number of estimate")
    print("--i.e. the score matrix divided by the total number of estimate with that label")
    df_pre = pandas.DataFrame(pre, columns=screen_names, index=screen_names)
    print(df_pre)
    print("\n")

    return df_score, df_fp, df_pre



"""
This is the main :)

We classify the users in FILE_USERS using the data in all_tweets.txt via Softmax Regression algorithm.
To create all_tweets.txt have a look at the file create_data.py

There are a few parameters to set, all in the beginning of the main:
*train
*dev
*test
*n_features
*min_df
*DO_STANDARDIZE_DATA
*regularization_lambda
*ETA
*EPOCHS
These the meaning of this parameters is commented below.
"""
def main():
    #SETUP!!!
    train=0.9 #percetage of data for training
    dev=0.05 #percetage of data for development
    test=0.05 #percetage of data for test

    n_features=1500 #this could be adjusted later by the algorithm

    #this is setting the  CountVectorizer from sklearn.feature_extraction.text
    vectorizer = CountVectorizer(min_df=20, #you may want to adjust this
                                 max_features=n_features,
                                 lowercase=False)

    DO_STANDARDIZE_DATA=1 #1 yes, 0 no

    regularization_lambda=0.1
    ETA=0.00005
    EPOCHS=50
    model_sm = SoftmaxRegression(eta=ETA,
                           epochs=EPOCHS,
                           l2=regularization_lambda,
                           #n_classes=U,
                           minibatches=1,
                           random_seed=1,
                           print_progress=3)


    print("-----------------------------")
    print("METHOD - SOFTMAX REGRESSION")
    print("-----------------------------")


    print("Hello,\nwe will use Softmax Regression to classify twitter users\n")
    setpath()

    #get the users
    screen_names=get_users(FILE_USERS);
    info_data=get_info()
    U=len(screen_names)#number of users
    for i in range(U):
        print("For", screen_names[i], " one has ", info_data[i,1], "tweets")

    if os.path.isfile(FOLDER+"/update_SM"+str(U)+".txt")==True:
        with open(FOLDER+"/update_SM"+str(U)+".txt", "r") as h:
            update=h.read()
            h.close()

        print("We load the dataset.")
        file=FOLDER+"/X_train_politic"+update+".npy"
        with open(file,'rb') as f:
            X_train=pickle.load(f)

        file=FOLDER+"/Y_train_politic"+update+".npy"
        with open(file,'rb') as f:
            Y_train=pickle.load(f)

        file=FOLDER+"/X_dev_politic"+update+".npy"
        with open(file,'rb') as f:
            X_dev=pickle.load(f)

        file=FOLDER+"/Y_dev_politic"+update+".npy"
        with open(file,'rb') as f:
            Y_dev=pickle.load(f)

        file=FOLDER+"/X_test_politic"+update+".npy"
        with open(file,'rb') as f:
            X_test=pickle.load(f)

        file=FOLDER+"/Y_test_politic"+update+".npy"
        with open(file,'rb') as f:
            Y_test=pickle.load(f)
    else:
        all_tweets=load_data()
        random.shuffle(all_tweets)
        random.shuffle(all_tweets)#Always shuffle your opponent cards when you play :)

        tweets=[];
        YY=[]
        for i in range(len(all_tweets)):
            tweets.append(all_tweets[i][2])
            YY.append(all_tweets[i][0])

        if len(tweets)==len(all_tweets):
            print("We load the data and we create the data set!")

        Y=np.array(YY)#this is the output label vector


        print("-----------------------------")
        m=len(tweets)
        X_train_1, x_appoggio, Y_train, y_appoggio = train_test_split(tweets, Y, test_size=(dev+test))
        X_dev_1, X_test_1, Y_dev, Y_test = train_test_split(x_appoggio, y_appoggio, test_size=(test/(dev+test)))
        print("We will train with the", train*100," % of the data;")
        print(dev*100, "% of the data is reserve for the method development;")
        print(test*100, "% of the data is for the test.")

        vectorizer.fit(X_train_1)
        X_train=vectorizer.transform(X_train_1)
        X_dev=vectorizer.transform(X_dev_1)
        X_test=vectorizer.transform(X_test_1)

        if DO_STANDARDIZE_DATA==0:
            print("We don't standardize data")
        else:
            print("We will provide to the model with standardize data, mean zero and variance 1")
            X_train, X_dev, X_test=standardize_data(X_train, X_dev, X_test)

        del(all_tweets)
        del(X_train_1,X_dev_1,X_test_1, x_appoggio,y_appoggio)


        today = date.today()
        today_string=today.strftime("%y_%b_%d")
        #we save the data we have prepared
        with open(FOLDER+"/X_train_politic"+today_string+"_SM"+str(U)+".npy",'wb') as f:
            pickle.dump(X_train,f)


        file=FOLDER+"/X_train_politic"+today_string+"_SM"+str(U)+".npy"
        with open(file, "wb") as f:
            pickle.dump(X_train,f)

        file=FOLDER+"/Y_train_politic"+today_string+"_SM"+str(U)+".npy"
        with open(file, "wb") as f:
            pickle.dump(Y_train,f)

        file=FOLDER+"/X_dev_politic"+today_string+"_SM"+str(U)+".npy"
        with open(file, "wb") as f:
            pickle.dump(X_dev,f)

        file=FOLDER+"/Y_dev_politic"+today_string+"_SM"+str(U)+".npy"
        with open(file, "wb") as f:
            pickle.dump(Y_dev,f)

        file=FOLDER+"/X_test_politic"+today_string+"_SM"+str(U)+".npy"
        with open(file, "wb") as f:
            pickle.dump(X_test,f)

        file=FOLDER+"/Y_test_politic"+today_string+"_SM"+str(U)+".npy"
        with open(file, "wb") as f:
            pickle.dump(Y_test,f)

        with open(FOLDER+"/update_SM"+str(U)+".txt", "w") as h:
            h.write(today_string+"_SM"+str(U))
            h.close()


    D=X_test.toarray().shape[1] #this is the lengh of the input vector

    print("\n")
    if n_features>D:
        n_features=D
    print("The # of features is", n_features)
    print("The regularization parameter is", regularization_lambda)
    print("The learning step is", ETA)
    print("The # of cycle is", EPOCHS)
    print("\n")


    #WE START TRAINING THE MODEL
    model_sm.fit(X_train.toarray(), Y_train)

    acc=model_sm.score(X_train.toarray(), Y_train)
    acc_dev=model_sm.score(X_dev.toarray(), Y_dev)
    print("\n")
    print("Accuracy on the training set", acc)
    print("Accuracy on the development set", acc_dev)

    #print some statistics about the model
    df_score, df_fp, df_pre=compute_accuracies(model_sm,1,screen_names, X_train, X_dev, Y_train, Y_dev)



if __name__ == '__main__':
    main()

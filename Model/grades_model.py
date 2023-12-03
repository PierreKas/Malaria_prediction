# import pandas as pds
# from sklearn.tree import DecisionTreeClassifier
# GradeData= pds.read_excel('Data.xlsx')
# X=GradeData.drop(columns=['SNAMES ','Total Marks','Marks /20','Grading '])
# y=GradeData['Grading ']
# model= DecisionTreeClassifier()
# model.fit(X.values,y)
# predictions= model.predict([[12,1,30,40]])
# print(predictions)
#################################################################
#####ACCURACY########
# # IMPORTS ALL LIBRARIES
# import pandas as pds
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# #IMPORT DATASET
# GradeData=pds.read_excel('Data.xlsx')

# #SPLIT(DIVISER) DATASET INTO TRAINING SET AND TEST SET
# X=GradeData.drop(columns=['SNAMES ','Total Marks','Marks /20','Grading '])
# y=GradeData['Grading ']
# X_for_train,x_for_test,y_for_train,y_for_test=train_test_split(X,y,test_size=0.2)

# #Create a Decision Tree, Logistic Regression, Support Vector Machine and Random Forest Classifiers
# Decision_Tree_Model=DecisionTreeClassifier()
# Logistic_Regression_Model=LogisticRegression()
# Support_Vector_Machine_Model=svm.SVC(kernel='linear')
# Random_Forest_Model=RandomForestClassifier(n_estimators=100)

# #TRAIN THE MODEL USING THE TRAINING SETS
# Decision_Tree_Model.fit(X_for_train,y_for_train)
# Logistic_Regression_Model.fit(X_for_train,y_for_train)
# Support_Vector_Machine_Model.fit(X_for_train,y_for_train)
# Random_Forest_Model.fit(X_for_train,y_for_train)

# #PREDICT THE MODEL
# DT_prediction=Decision_Tree_Model.predict(x_for_test)
# LR_prediction=Logistic_Regression_Model.predict(x_for_test)
# SVM_prediction=Support_Vector_Machine_Model.predict(x_for_test)
# RF_prediction=Random_Forest_Model.predict(x_for_test)

# #CALCULATION OF MODEL ACUURACY
# DT_score=accuracy_score(y_for_test,DT_prediction)
# LR_score=accuracy_score(y_for_test,LR_prediction)
# SVM_score=accuracy_score(y_for_test,SVM_prediction)
# RF_score=accuracy_score(y_for_test,RF_prediction)

# #DISPLAY ACCURACY
# print ("Decistion Tree accuracy =", DT_score*100,"%")
# print ("Logistic Regression accuracy =", LR_score*100,"%")
# print ("Suport Vector Machine accuracy =", SVM_score*100,"%")
# print ("Random Forest accuracy =", RF_score*100,"%")

###########################################################################################################
#######PERSISTING MODEL#######
# import pandas as pds
# from sklearn import svm
# import joblib

# GradeData=pds.read_excel('Data.xlsx')

# X=GradeData.drop(columns=['SNAMES ','Total Marks','Marks /20','Grading '])
# y=GradeData['Grading ']

# model=svm.SVC(kernel='linear')

# model.fit(X.values,y)

# # create a persisting model#
# joblib.dump(model,'Grade_model.joblib')

###########################################################################################################
####PERSISTING MODEL USING USER INPUT####
# import pandas as pds
# from sklearn import svm
# import joblib

# #User input
# Quiz=int (input ("Enter Quiz Marks :"))
# Assignment= int(input ("Enter Assignment Marks: "))
# Mid=int (input ("Enter Mid Exam Marks Marks :"))
# Final= int(input ("Enter Final Exam Marks: "))

# #Prediction from the created model
# model=joblib.load('Grade_model.joblib')
# prediction=model.predict([[Quiz,Assignment,Mid,Final]])
# print(prediction)

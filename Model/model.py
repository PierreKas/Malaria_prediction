###ACCURACY########
# IMPORTS ALL LIBRARIES
import pandas as pds
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer


#IMPORT DATASET
MalariaData=pds.read_csv('malaria_clinical_data.csv')
imputer = SimpleImputer(strategy='median')

X=MalariaData.drop(columns=['SampleID','consent_given','location','Enrollment_Year','bednet','fever_symptom','Suspected_Organism','Suspected_infection','RDT','Blood_culture','Urine_culture','Taq_man_PCR','parasite_density','Microscopy','Laboratory_Results','Clinical_Diagnosis'])
y=MalariaData['Clinical_Diagnosis']
X_imputed = imputer.fit_transform(X)
# columns_with_missing_values = ['temperature','wbc_count','rbc_count','hb_level','hematocrit','mean_cell_volume','mean_corp_hb', 'mean_cell_hb_conc', 'platelet_count', 'platelet_distr_width', 'mean_platelet_vl', 'neutrophils_count','lymphocytes_percent' ,'lymphocytes_count','mixed_cells_percent', 'mixed_cells_count', 'RBC_dist_width_Percent']

# # Fill missing values with the mode of respective columns
# for column in columns_with_missing_values:
#     New_MalariaData = MalariaData[column].mode()[0]
#     MalariaData[column].fillna(New_MalariaData, inplace=True)
    
# print(MalariaData.info())
#SPLIT(DIVISER) DATASET INTO TRAINING SET AND TEST SET

X_for_train,x_for_test,y_for_train,y_for_test=train_test_split(X_imputed,y,test_size=0.2)


#Create a Decision Tree, Logistic Regression, Support Vector Machine and Random Forest Classifiers
Decision_Tree_Model=DecisionTreeClassifier()
Logistic_Regression_Model=LogisticRegression()
# Support_Vector_Machine_Model=svm.SVC(kernel='linear')
Random_Forest_Model=RandomForestClassifier(n_estimators=100)

#TRAIN THE MODEL USING THE TRAINING SETS
Decision_Tree_Model.fit(X_for_train,y_for_train)
Logistic_Regression_Model.fit(X_for_train,y_for_train)
# Support_Vector_Machine_Model.fit(X_for_train,y_for_train)
Random_Forest_Model.fit(X_for_train,y_for_train)

#PREDICT THE MODEL
DT_prediction=Decision_Tree_Model.predict(x_for_test)
LR_prediction=Logistic_Regression_Model.predict(x_for_test)
# SVM_prediction=Support_Vector_Machine_Model.predict(x_for_test)
RF_prediction=Random_Forest_Model.predict(x_for_test)

#CALCULATION OF MODEL ACUURACY
DT_score=accuracy_score(y_for_test,DT_prediction)
LR_score=accuracy_score(y_for_test,LR_prediction)
# SVM_score=accuracy_score(y_for_test,SVM_prediction)
RF_score=accuracy_score(y_for_test,RF_prediction)

#DISPLAY ACCURACY
print ("Decistion Tree accuracy =", DT_score*100,"%")
print ("Logistic Regression accuracy =", LR_score*100,"%")
# print ("Suport Vector Machine accuracy =", SVM_score*100,"%")
print ("Random Forest accuracy =", RF_score*100,"%")
##After accuracy testing (except SVM) the most accurate is Random Forest Model
#######################################

# import pandas as pds
# from sklearn.ensemble import RandomForestClassifier
# import joblib
# from sklearn.impute import SimpleImputer

# MalariaData=pds.read_csv('malaria_clinical_data.csv')
# imputer = SimpleImputer(strategy='most_frequent')  ##Replace missing values with the mode of the column

# X=MalariaData.drop(columns=['SampleID','consent_given','location','Enrollment_Year','bednet','fever_symptom','Suspected_Organism','Suspected_infection','RDT','Blood_culture','Urine_culture','Taq_man_PCR','parasite_density','Microscopy','Laboratory_Results','Clinical_Diagnosis'])
# y=MalariaData['Clinical_Diagnosis']
# X_imputed = imputer.fit_transform(X)

# model=RandomForestClassifier(n_estimators=100)

# model.fit(X_imputed,y)

# # TO SAVE THE TRAINED MODEL#
# joblib.dump(model,'malaria_model.joblib')

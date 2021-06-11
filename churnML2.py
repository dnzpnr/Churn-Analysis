
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt

'''warningleri gostermesin istiyorsan asagidaki kodu yaz'''
from warnings import filterwarnings
filterwarnings('ignore')

from churnFirst import df_
df_.head()
df = df_.copy()

y = df['Exited']
x = df.drop(['Exited'], axis= 1)


y = df['Exited'].copy()
x = df.drop('Exited', axis = 1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

'''degiskenlerimize normalizasyon uygulayalim'''

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
x_train_normed = mms.fit_transform(x_train)
x_test_normed= mms.fit_transform(x_test)

'''standardizasyon'''
from sklearn import preprocessing
stdandard_scale = preprocessing.StandardScaler()
x_train_std = stdandard_scale.fit_transform(x_train)
x_test_std = stdandard_scale.fit_transform(x_test)

# 1A- LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
loj = LogisticRegression()
loj_model = loj.fit(x_train,y_train)
y_pred = loj_model.predict(x)
acc1A = accuracy_score(y, y_pred)


'''
y_probs = loj_model.predict_proba(x)
y_probs = y_probs[:,1]
y_probs[0:10]
y_pred = [1 if i > 0.5 else 0 for i in y_probs]
y_pred[0:10]
confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
loj_model.predict_proba(x)[:,1][0:5]


logit_roc_auc = roc_auc_score(y, loj_model.predict(x))
logit_roc_auc

fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(x)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()
'''

# 1B - LOGISTIC REGRESSION - NORM

loj_norm = LogisticRegression().fit(x_train_normed,y_train).predict(x_test_normed)
acc1B = accuracy_score(y_test,loj_norm)

# 1C - LOGISTIC REGRESSION- CV

loj = LogisticRegression()
loj_model = loj.fit(x_train,y_train)

acc1C = cross_val_score(loj_model, x_test, y_test, cv = 10).mean()

# 1D - LOGISTIC REGRESSION - CV- NORM

loj_cv_norm = loj.fit(x_train_normed,y_train)

acc1D = cross_val_score(loj_cv_norm, x_test_normed, y_test, cv = 10).mean()


# 2A- NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_ = nb.fit(x_train, y_train).predict(x_test)
acc2A = accuracy_score(y_test, nb_)

# 2B- NAIVE BAYES - NORM

nb_norm = nb.fit(x_train_normed,y_train).predict(x_test_normed)
acc2B = accuracy_score(y_test,nb_norm)


# 2C- NAIVE BAYES- CV

acc2C = cross_val_score(nb, x_test, y_test, cv = 10).mean()


# 2D- NAIVE BAYES -CV -NORM

nb_cv_norm = nb.fit(x_train_normed,y_train)
acc2D = cross_val_score(nb_cv_norm, x_test_normed, y_test, cv = 10).mean()

# 2E- NAIVE BAYES - CV- STD


nb_cv_std = nb.fit(x_train_std,y_train)
acc2E = cross_val_score(nb_cv_std, x_test_std, y_test, cv = 10).mean()


# 3A -  KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn_ = knn.fit(x_train,y_train).predict(x_test)
acc3A = accuracy_score(y_test,knn_)

# 3B- KNN - NORM

knn_norm = knn.fit(x_train_normed,y_train).predict(x_test_normed)
acc3B = accuracy_score(y_test,knn_norm)


# 3C- KNN - CV

knn_params = {'n_neighbors': np.arange(1,20)}

knn_cv = GridSearchCV(knn, knn_params, cv = 10).fit(x_train,y_train)
knn_cv.best_params_

knn_cv_ = KNeighborsClassifier(n_neighbors=18).fit(x_train,y_train).predict(x_test)
acc3C = accuracy_score(y_test,knn_cv_)

# 3D - KNN - CV- NORM

knn_cv_norm = GridSearchCV(knn, knn_params, cv = 10).fit(x_train_normed,y_train)
knn_cv.best_params_

knn_cv_norm_ = KNeighborsClassifier(n_neighbors=18).fit(x_train_normed,y_train).predict(x_test_normed)
acc3D = accuracy_score(y_test,knn_cv_norm_)


# 4A- SVC

from sklearn.svm import SVC
svc = SVC()
svc_ = svc.fit(x_train,y_train).predict(x_test)
acc4A = accuracy_score(y_test,svc_)

# 4B - SVC - NORM

svc_norm = svc.fit(x_train_normed,y_train).predict(x_test_normed)
acc4B = accuracy_score(y_test,svc_norm)

# 4C- SVC - CV

svc_params = {'C': np.arange(1,10)}
svc_cv = GridSearchCV(svc,svc_params,cv= 10).fit(x_train,y_train)
svc_cv.best_params_

svc_cv_ = SVC(C=1).fit(x_train,y_train).predict(x_test)
acc4C = accuracy_score(y_test,svc_cv_)

# 4D - SVC- CV- NORM

svc_cv_norm = GridSearchCV(svc,svc_params,cv = 10).fit(x_train_normed,y_train)
svc_cv_norm.best_params_

svc_cv_norm_ = SVC(C = 7).fit(x_train_normed,y_train).predict(x_test_normed)
acc4D = accuracy_score(y_test,svc_cv_norm_)

# 5A- MLP

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp_ = mlp.fit(x_train,y_train).predict(x_test)
acc5A = accuracy_score(y_test,mlp_)

# 5B - MLP - NORM

mlp_norm = mlp.fit(x_train_normed,y_train).predict(x_test_normed)
acc5B = accuracy_score(y_test,mlp_norm)


# 5C- MLP - CV

mlp_params = {'alpha':[0.1,2],
              'solver':['lbfgs','adam','sgd'],
              'activation': ['relu','logistic']}

mlp_cv = GridSearchCV(mlp, mlp_params,cv = 10).fit(x_train, y_train)
mlp_cv.best_params_

mlp_cv_ = MLPClassifier(alpha=2,activation='relu',solver='lbfgs').fit(x_train,y_train).predict(x_test)
acc5C = accuracy_score(y_test,mlp_cv_)

# 5D - MLP - CV - NORM

mlp_cv_norm = GridSearchCV(mlp, mlp_params,cv = 10).fit(x_train_normed,y_train)
mlp_cv_norm.best_params_

mlp_cv_norm_ = MLPClassifier(alpha=2,activation='relu',solver='lbfgs').fit(x_train_normed,y_train).predict(x_test_normed)
acc5D = accuracy_score(y_test,mlp_cv_norm_)

# 5E - MLP - CV -STD


mlp_cv_std = MLPClassifier(alpha=2,activation='relu',solver='lbfgs').fit(x_train_std,y_train).predict(x_test_std)
acc5E = accuracy_score(y_test,mlp_cv_std)


# 6A- CART

from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier()
cart_ = cart.fit(x_train,y_train).predict(x_test)
acc6A = accuracy_score(y_test,cart_)

'''
Train test ayrimi olmadan deneme yapalim
'''
from skompiler import skompile
crt = DecisionTreeClassifier()
X =  df['Tenure']
X = pd.DataFrame(X)
crt_ = crt.fit(X,y)
print(skompile(crt_.predict).to('python/code'))
# CALISMADI

# 6B - CART- NORM

cart_norm = cart.fit(x_train_normed,y_train).predict(x_test_normed)
acc6B = accuracy_score(y_test,cart_norm)

# 6C- CART -CV

cart_params = {'min_samples_split': range(1,20)}

cart_cv = GridSearchCV(cart, cart_params,cv = 10).fit(x_train,y_train)
cart_cv.best_params_

crt_cv = DecisionTreeClassifier(min_samples_split=19).fit(x_train,y_train).predict(x_test)
acc6C = accuracy_score(y_test,crt_cv)

# 6D- CART - CV- NORM

cart_cv_norm = GridSearchCV(cart,cart_params, cv = 10).fit(x_train_normed,y_train)
cart_cv_norm.best_params_

cart_cv_norm_ = DecisionTreeClassifier(min_samples_split=19).fit(x_train_normed,y_train).predict(x_test_normed)
acc6D = accuracy_score(y_test,cart_cv_norm_)


# 7A - RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_ = rf.fit(x_train,y_train).predict(x_test)
acc7A = accuracy_score(y_test,rf_)

# 7B- RANDOM FOREST - NORM

rf_norm = rf.fit(x_train_normed,y_train).predict(x_test_normed)
acc7B = accuracy_score(y_test,rf_norm)


# 7C - RANDOM FOREST- CV

rf_params = {'n_estimators': [200,500],
             'min_samples_split':range(1,5),
             'max_features': [4,8]}

rf_cv = GridSearchCV(rf, rf_params,cv=10).fit(x_train,y_train)
rf_cv.best_params_

rf_cv_ = RandomForestClassifier(n_estimators=500, max_features=4,min_samples_split=3).fit(x_train,y_train).predict(x_test)
acc7C = accuracy_score(y_test,rf_cv_)

# 7D- RANDOM FOREST - CV - NORM

rf_cv_norm = GridSearchCV(rf,rf_params,cv = 10).fit(x_train_normed,y_train)
rf_cv_norm.best_params_

rf_cv_norm_ = RandomForestClassifier(n_estimators=500, max_features=4,min_samples_split=4).fit(x_train_normed,y_train).predict(x_test_normed)
acc7D = accuracy_score(y_test,rf_cv_norm_)

# 8A - GBM

from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier()
gbm_ = gbm.fit(x_train,y_train).predict(x_test)
acc8A = accuracy_score(y_test, gbm_)


# 8B - GBM -NORM

gbm_norm = gbm.fit(x_train_normed,y_train).predict(x_test_normed)
acc8B = accuracy_score(y_test,gbm_norm)

# 8C - GBM - CV

gbm_params = {'n_estimators': [200,500],
              'learning_rate': [0.01,0.2],
              'min_samples_split':[2,4]}

gbm_cv = GridSearchCV(gbm,gbm_params,cv = 10).fit(x_train,y_train)
gbm_cv.best_params_

gbm_cv_ = GradientBoostingClassifier(n_estimators=500,learning_rate=0.01,min_samples_split=2).fit(x_train,y_train).predict(x_test)
acc8C = accuracy_score(y_test,gbm_cv_)


# 8D- GBM - CV - NORM

gbm_cv_norm = GridSearchCV(gbm,gbm_params,cv = 10).fit(x_train_normed,y_train)
gbm_cv_norm.best_params_


gbm_cv_norm_ = GradientBoostingClassifier(n_estimators=200,learning_rate=0.2,min_samples_split=2).fit(x_train_normed,y_train).predict(x_test_normed)
acc8D = accuracy_score(y_test,gbm_cv_norm_)

# 8E- GBM - CV - STD

gbm_cv_std = GradientBoostingClassifier(n_estimators=200,learning_rate=0.2,min_samples_split=2).fit(x_train_std,y_train).predict(x_test_std)
acc8E = accuracy_score(y_test,gbm_cv_std)


# 9A- XGBOOST

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb_ = xgb.fit(x_train,y_train).predict(x_test)
acc9A = accuracy_score(y_test,xgb_)

# 9B - XGBOOST - NORM

xgb_norm = xgb.fit(x_train_normed,y_train).predict(x_test_normed)
acc9B = accuracy_score(y_test,xgb_norm)


# 9C - XGBOOST- CV

xgb_cv = GridSearchCV(xgb,gbm_params, cv = 10).fit(x_train,y_train)
xgb_cv.best_params_

xgb_cv_ = XGBClassifier(n_estimators=500,learning_rate=0.01,min_samples_split=2).fit(x_train,y_train).predict(x_test)
acc9C = accuracy_score(y_test,xgb_cv_)


# 9D- XGBOOST -CV -NORM

xgb_cv_norm = GridSearchCV(xgb,gbm_params,cv = 10).fit(x_train_normed,y_train)
xgb_cv_norm.best_params_


xgb_cv_norm_ = XGBClassifier(n_estimators=500,learning_rate=0.01,min_samples_split=2).fit(x_train_normed,y_train).predict(x_test_normed)
acc9D = accuracy_score(y_test,xgb_cv_norm_)


# 9E - XGBOOST - CV - STD

xgb_cv_std = XGBClassifier(n_estimators=500,learning_rate=0.01,min_samples_split=2).fit(x_train_std,y_train).predict(x_test_std)
acc9E = accuracy_score(y_test,xgb_cv_std)

# 10A- LIGHTGBM

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
lgbm_ = lgbm.fit(x_train,y_train).predict(x_test)
acc10A = accuracy_score(y_test,lgbm_)


# 10B - LIGHTGBM - NORM

lgbm_norm = lgbm.fit(x_train_normed,y_train).predict(x_test_normed)
acc10B = accuracy_score(y_test,lgbm_norm)


# 10C- LIGHTGBM - CV

lgbm_cv = GridSearchCV(lgbm, gbm_params,cv = 10).fit(x_train,y_train)
lgbm_cv.best_params_

lgbm_cv_ = LGBMClassifier(n_estimators=500,learning_rate=0.01,min_samples_split=2).fit(x_train,y_train).predict(x_test)
acc10C = accuracy_score(y_test,lgbm_cv_)

# 10D - LIGHTGBM - CV- NORM

lgbm_cv_norm = GridSearchCV(lgbm,gbm_params,cv = 10).fit(x_train_normed,y_train)
lgbm_cv_norm.best_params_

lgbm_cv_norm_ = LGBMClassifier(n_estimators=500,learning_rate=0.01,min_samples_split=2).fit(x_train_normed,y_train).predict(x_test_normed)
acc10D = accuracy_score(y_test,lgbm_cv_norm_)

# 10E - LIGHTGBM - CV - STD

lgbm_cv_std = LGBMClassifier(n_estimators=500,learning_rate=0.01,min_samples_split=2).fit(x_train_std,y_train).predict(x_test_std)
acc10E = accuracy_score(y_test,lgbm_cv_std)



# 11A - CATBOOST

from catboost import CatBoostClassifier
catbst = CatBoostClassifier()
catbst_ = catbst.fit(x_train,y_train).predict(x_test)
acc11A = accuracy_score(y_test,catbst_)

# 11B - CATBOOST- NORM

catbst_norm = catbst.fit(x_train_normed,y_train).predict(x_test_normed)
acc11B = accuracy_score(y_test,catbst_norm)




# 11C - CATBOOST - CV


ctbst_params = {'iterations': [200,500],
              'learning_rate': [0.01,0.2],
              'depth':[2,5,8]}

catbst_cv = GridSearchCV(catbst,ctbst_params,cv = 10).fit(x_train,y_train)
catbst_cv.best_params_

catbst_cv_ = CatBoostClassifier(iterations=500, depth=5,learning_rate=0.01).fit(x_train,y_train).predict(x_test)
acc11C = accuracy_score(y_test,catbst_cv_)


# 11D - CATBOOST - CV -NORM

catbst_cv_norm = GridSearchCV(catbst,ctbst_params, cv = 10).fit(x_train_normed,y_train)
catbst_cv_norm.best_params_

catbst_cv_ = CatBoostClassifier(iterations=500, depth=8,learning_rate=0.01).fit(x_train_normed,y_train).predict(x_test_normed)
acc11D = accuracy_score(y_test,catbst_cv_)


# 11E - CATBOOST -CV -STD

catbst_cv_std = CatBoostClassifier(iterations=500, depth=8,learning_rate=0.01).fit(x_train_std,y_train).predict(x_test_std)
acc11E = accuracy_score(y_test,catbst_cv_std)

'''
LOGISTIC REGRESSION
NAIVE BAYES
KNN
SVC
MLP
CART
RANDOM FOREST
GBM
XGBOOST
LIGHTGBM
CATBOOST
'''

labels = ['LREG', 'NAIVE', 'KNN', 'SVC', 'MLP', 'CART', 'RF', 'GBM', 'XGB', 'LGBM', 'CATB']
model_A = [acc1A, acc2A, acc3A, acc4A,acc5A,acc6A,acc7A,acc8A,acc9A,acc10A,acc11A]
model_B = [acc1B, acc2B, acc3B, acc4B,acc5B,acc6B,acc7B,acc8B,acc9B,acc10B,acc11B]
model_C = [acc1C, acc2C, acc3C, acc4C,acc5C,acc6C,acc7C,acc8C,acc9C,acc10C,acc11C]
model_D = [acc1D, acc2D, acc3D, acc4D,acc5D,acc6D,acc7D,acc8D,acc9D,acc10D,acc11D]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize = (18,10))

rects1 = ax.bar(x- 1.5*width, model_A, width, label='None')
rects2 = ax.bar(x- width/2, model_B, width, label='NORM')
rects3 = ax.bar(x + width/2, model_C, width, label='CV')
rects4 = ax.bar(x + 1.5*width, model_D, width, label='CV-NORM')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by different ML Models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()



'''eger indexi sifirlamak istiyorsan

df.reset_index(drop=True)
'''


''''
butun satirlari gormek istiyorsan;

pd.options.display.max_rows = 999
pd.set_option('display.float_format',lambda x: '%.2f' % x)
'''

# Importing libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns % matplotlib
inline
matplotlib.style.use('fivethirtyeight')

# data
x = pd.DataFrame({
    # Distribution with lower outliers
    'x1': np.concatenate([np.random.normal(20, 2, 1000), np.random.normal(1, 2, 25)]),
    # Distribution with higher outliers
    'x2': np.concatenate([np.random.normal(30, 2, 1000), np.random.normal(50, 2, 25)]),
})
np.random.normal

scaler = preprocessing.RobustScaler()
robust_df = scaler.fit_transform(x)
robust_df = pd.DataFrame(robust_df, columns=['x1', 'x2'])

scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(x)
standard_df = pd.DataFrame(standard_df, columns=['x1', 'x2'])

scaler = preprocessing.MinMaxScaler()
minmax_df = scaler.fit_transform(x)
minmax_df = pd.DataFrame(minmax_df, columns=['x1', 'x2'])

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 5))
ax1.set_title('Before Scaling')

sns.kdeplot(x['x1'], ax=ax1, color='r')
sns.kdeplot(x['x2'], ax=ax1, color='b')
ax2.set_title('After Robust Scaling')

sns.kdeplot(robust_scaled_df['x1'], ax=ax2, color='red')
sns.kdeplot(robust_scaled_df['x2'], ax=ax2, color='blue')
ax3.set_title('After Standard Scaling')

sns.kdeplot(standard_df['x1'], ax=ax3, color='black')
sns.kdeplot(standard_df['x2'], ax=ax3, color='g')
ax4.set_title('After Min-Max Scaling')

sns.kdeplot(minmax_df['x1'], ax=ax4, color='black')
sns.kdeplot(minmax_df['x2'], ax=ax4, color='g')
plt.show()

import pandas as pd
import numpy as np

'''
ML modellerimi kurarken;
A- Normalizasyon ve cross validation yapilmadan,
B- Normalizasyon yapilip cross validation uygulanmadan,
C- Cross validation uygulayip normalizasyon yapmadan,
D- Normalizasyon ve cross validation uygulanarak,
 olacak sekilde 4 sekilde inceleme yapacagim.
 Boylece yeni baslayanlar neden normalizasyon ve CV(cross validation)
 uygulandigini daha iyi gozlemleyebileceklerdir'''

from churnFirst import df_
df_.head()
df = df_.copy()



y = df['Exited'].copy()
x = df.drop('Exited', axis = 1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state = 42)

'''
Yukaridaki komutla verisetini %80'i train, %20'si de
test verisi olacak sekilde parcaladik. Ayni kod baska bir yerde calistirildiginda 
ayni sonucu versin diye de random_state parametresini 42 olarak belirttik.
'''


'''degiskenlerimize normalizasyon uygulayalim'''

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
x_train_normed = mms.fit_transform(x_train)
x_test_normed= mms.transform(x_test)
'''
Once basit dogrusal regresyon uygulayarak baslayalim. 
Aslinda gercek hayatta karsilacagimiz problemlerin bircogu dogrusal olmayan 
turden problemlerdir. Fakat ben yeni baslayanlarin gormesi icin,
ornegin Balance ile Exited yani churn olma arasinda bir model kurarak
sonuclari gozlemlemek istiyorum.
'''
# 1A- BASIT DOGRUSAL REGRESYON

import statsmodels.api as sm

x = df[['Balance']]

'''
Arka plandaki matris islemlerinin duzgun yapilabilmesi icin bir sutun 
daha eklememiz gerekiyor. O yuzden bir sutun daha ekliyorum.
'''
x = sm.add_constant(x)

y = df['Exited']
bdr = sm.OLS(y,x).fit()
bdr.summary()

'''
R-squared degeri 1'e ne kadar yakin olursa model o kadar dogru aciklanmis demektir.
Bizim durumumuz icin R-squared degeri oldukca dusuk cikti. Bunun sebebi aslinda
dogrusal olmayan bir dagilima sahip olmamizdi fakat biz yine de dusuk cikacagini 
gozlemlemek istedik.
'''

'''
Modelin anlamliligini ifade eden p-value degeri;
'''

bdr.f_pvalue

'''
Virgulden sonraki 4 basamagi gormek istersek;
'''
print('f_pvalue: ', '%.4f' % bdr.f_pvalue)

'''
Modelin MSE(mean squared error) degerine erismek istersek;
'''
bdr.mse_model

'''Modelimizin tahmin ettigi Exited degerleri;'''
bdr.fittedvalues

'''Gercekte olan Exited degerleri;'''

y[0:10]

'''Goruldugu uzere modeli statsmodels kutuphanesini kullanarak olusturdum.
Ayni model sklearn kutuphanesiyle de olusturulabilirdi. Bundan sonra kuracagim
modelleri sklearn kutuphanesini kullanarak kuracagim icin statsmodels'i de
sizlere kisaca tanitmak istedim. '''


# 2A- COKLU DOGRUSAL REGRESYON

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error,r2_score

from sklearn.linear_model import LinearRegression
cdr = LinearRegression().fit(x_train, y_train)
cdr_ = cdr.predict(x_test)
cdr_rmse = np.sqrt(mean_squared_error(y_test,cdr_))
cdr_rmse

# 2B - COKLU DOGRUSAL REGRESYON - NORM

cdr_norm = LinearRegression().fit(x_train_normed,y_train).predict(x_test_normed)



# 2C- COKLU DOGRUSAL REGRESYON - CV
'''
Oncelikle cv neden yapilir? Yukarida verisetini train ve test olarak ayirirken
bu ayirma islemi rastgele yapiliyor. Verisetinden bu sekilde rastgele secim 
yapildiginda birbirinden cok farkli sonuclar almak kacinilmaz olacaktir.
Iste bu yuzden cv ile bu ayirma islemi birden fazla kez yapilir ve sonuc her birinin
ortalamasini yansitacagi icin gercege daha yakin bir sonuc elde etmis oluruz.
'''

np.sqrt(-cross_val_score(cdr, x_test,y_test, cv = 15, scoring='neg_mean_squared_error')).mean()


# 3A- RIDGE REGRESSION

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1).fit(x_train,y_train)
y_pred_ridge = ridge.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred_ridge))

# 3C- RIDGE REGRESSION - CV

from sklearn.linear_model import RidgeCV

lamdas = [[0.1], [0.2], [0.5],[1],[5]]

ridge_cv = RidgeCV(alphas= lamdas, scoring='neg_mean_squared_error',
                   normalize= False)
ridge_cv.fit(x_train,y_train)
ridge_cv.alpha_[0]

'''
Alpha parametresini 5 olarak secti
'''
ridge_ = Ridge(alpha=5).fit(x_train,y_train)
y_pred_ridge_ = ridge_.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred_ridge_))

# 3D- RIDGE REGRESSION - CV - NORM.

lamdas_ = [[5], [6], [10],[15],[50]]

ridge_cv_ = RidgeCV(alphas= lamdas_, scoring='neg_mean_squared_error',
                   normalize= True)
ridge_cv_.fit(x_train,y_train)
ridge_cv.alpha_[0]

'''
Alpha parametresini tekrar 5 olarak secti
'''

ridge_2 = Ridge(alpha=5).fit(x_train,y_train)
y_pred_ridge_2 = ridge_2.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred_ridge_2))

# 4A- LASSO REGRESSION

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.1).fit(x_train,y_train)
lasso_ = lasso.predict(x_test)
np.sqrt(mean_squared_error(y_test,lasso_))

# 4C- LASSO REGRESIION- CV

from sklearn.linear_model import LassoCV
#lambdalar = [[0.1],[0.3],[5],[10]]
lasso_cv = LassoCV(alphas=None, cv = 15, normalize=False)\
    .fit(x_train,y_train)
lasso_cv.alpha_

'''
Alphayi biz vermedik, kendisi secti
'''
lasso = Lasso(alpha = lasso_cv.alpha_).fit(x_train,y_train)
lasso_ = lasso.predict(x_test)
np.sqrt(mean_squared_error(y_test,lasso_))

# 4D- LASSO REGRESSION - CV - NORM.

lasso_cv_ = LassoCV(alphas=None, cv = 15, normalize=True)\
    .fit(x_train,y_train)
lasso_cv_.alpha_

lasso_cv_norm = Lasso(alpha = lasso_cv.alpha_).fit(x_train,y_train)
lasso_cv_norm_ = lasso_cv_norm.predict(x_test)
np.sqrt(mean_squared_error(y_test,lasso_cv_norm_))

# 5A- ELASTICNET

from sklearn.linear_model import ElasticNet

elnet = ElasticNet().fit(x_train,y_train).predict(x_test)
np.sqrt(mean_squared_error(y_test,elnet))



# 5C- ELASTICNET- CV

from sklearn.linear_model import ElasticNetCV

elnet_cv = ElasticNetCV(cv = 15, normalize=False,random_state=42).fit(x_train,y_train).predict(x_test)
np.sqrt(mean_squared_error(y_test,elnet_cv))

# 5D- ELASTICNET- CV- NORM

elnet_cv_ = ElasticNetCV(cv = 15, normalize=True,random_state=42).fit(x_train,y_train).predict(x_test)
np.sqrt(mean_squared_error(y_test,elnet_cv_))

# DOGRUSAL OLMAYAN MODELLER

# 6A- KNN

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor().fit(x_train,y_train).predict(x_test)
np.sqrt(mean_squared_error(y_test,knn))










# 6C- KNN - CV

'''
Gridsearch yontemiyle olasi parametreleri deneyip en iyi sonucu veren parametreyi 
secerek cv uygulamis olacagiz.
'''
from sklearn.model_selection import GridSearchCV
'''
Ornegin 1'den 20'ye kadar olan sayilardan en uygun parametreyi secelim
'''
k_degerleri = {'n_neighbors': np.arange(1,20,1)}
knn_ = KNeighborsRegressor()
knn_cv = GridSearchCV(knn_, k_degerleri, cv = 10)
knn_cv.fit(x_train,y_train)
knn_cv.best_params_['n_neighbors']
'''
Goruldugu uzere en iyi parametreyi 19 olarak secti
'''
knn_cv1 = KNeighborsRegressor(n_neighbors= knn_cv.best_params_['n_neighbors'])
knn_cv1_= knn_cv1.fit(x_train,y_train).predict(x_test)
np.sqrt(mean_squared_error(y_test,knn_cv1_))

# 7A- YAPAY SINIR AGLARI
'''
YSA'da standardizasyon cok onemli oldugu icin once veriyi
uygun hale donusturelim. Ancak normalizasyon ile standardizasyon birbiriyle 
karistirilmamali;
'''
from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit(x_train)
x_std_ = x_std.transform(x_train)
x_test_ = x_std.transform(x_test)

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor().fit(x_std_, y_train)
mlp_ = mlp.predict(x_test_)
np.sqrt(mean_squared_error(y_test,mlp_))

# 7C- YAPAY SINIR AGLARI - CV

parametreler = {'alpha': [1,2,5,15],
                'hidden_layer_sizes':[(20,20),(20,30),(20,50)],
                'activation':['relu','logistic']}
mlp_cv = GridSearchCV(mlp,parametreler,cv =15).fit(x_std_,y_train)
mlp_cv.best_params_

mlp_cv1 = MLPRegressor(alpha= 1, hidden_layer_sizes = (20,30)).fit(x_std_, y_train)
y_pred_mlp_cv = mlp_cv1.predict(x_test_)
np.sqrt(mean_squared_error(y_test,y_pred_mlp_cv))


# 8A- DECISION TREE

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor().fit(x_train,y_train)
dtr_ = dtr.predict(x_test)
np.sqrt(mean_squared_error(y_test,dtr_))

# 8C- DECISION TREE- CV

params = {'min_samples_split':range(2,15),
          'max_leaf_nodes':range(2,10)}
dtr_cv = GridSearchCV(dtr,params,cv = 10)
dtr_cv.fit(x_train,y_train)
dtr_cv.best_params_

dtr_cv_ = DecisionTreeRegressor(max_leaf_nodes=9,min_samples_split=2).\
    fit(x_train,y_train)
y_pred_dtr_cv = dtr_cv_.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred_dtr_cv))

# 9A- BAGGING REGRESSION

from sklearn.ensemble import BaggingRegressor
bagging = BaggingRegressor().fit(x_train,y_train)
bagging_ = bagging.predict(x_test)
np.sqrt(mean_squared_error(y_test,bagging_))
bagging.n_estimators
'''
n_estimators kac tane agac yapisi olusturulacagidir.
'''

# 9C- BAGGING REGRESSION- CV

prmtrs = {'n_estimators': range(2,15)}
bagging_cv = GridSearchCV(bagging, prmtrs, cv = 12)
bagging_cv.fit(x_train,y_train)
bagging_cv.best_params_

br = BaggingRegressor(n_estimators=14).fit(x_train,y_train)
br_ = br.predict(x_test)
np.sqrt(mean_squared_error(y_test,br_))

# 10A- RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor().fit(x_train,y_train)
rf_ = rf.predict(x_test)
np.sqrt(mean_squared_error(y_test,rf_))

# 10C- RANDOM FOREST- CV

paramtrs = {'max_features':[2,4,6,8,10],
            'n_estimators':[100,200,500,1000]}
rf_cv = GridSearchCV(rf, paramtrs, cv = 12)
rf_cv.fit(x_train,y_train)
rf_cv.best_params_

rf_cv_ = RandomForestRegressor(max_features=4, n_estimators= 1000).\
    fit(x_train,y_train)
rf_cv1 = rf_cv_.predict(x_test)
np.sqrt(mean_squared_error(y_test,rf_cv1))

# 11A- GBM

from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor()
gbm_ = gbm.fit(x_train,y_train).predict(x_test)
np.sqrt(mean_squared_error(y_test,gbm_))

# 11C- GBM - CV

params_ = {'n_estimators':[500,1000]}

gbm_cv = GridSearchCV(gbm,params_,cv = 12)
gbm_cv.fit(x_train,y_train)
gbm_cv.best_params_

gbm_cv_= GradientBoostingRegressor(n_estimators=500).fit(x_train,y_train)
y_ = gbm_cv_.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_))


# 12A- XGBOOST

#pip install xgboost

import xgboost as xgb
from xgboost import XGBRegressor
xgb_ = XGBRegressor()
xgb_model= xgb_.fit(x_train,y_train).predict(x_test)
np.sqrt(mean_squared_error(y_test,xgb_model))

'''
Bir de xgboost'un kendi veri yapisini kullanarak deneme yapalim
'''

trainXGB = xgb.DMatrix(data=x_train, label=y_train)
testXGB = xgb.DMatrix(data=x_test, label=y_test)

param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
bst = xgb.train(param, trainXGB,num_boost_round=10)
y_pred_xgb = bst.predict(testXGB)
np.sqrt(mean_squared_error(y_test,y_pred_xgb))

'''
Gercekten de soylendigi gibi daha iyi bir sonuc verdi
'''

# 12C- XGBOOST - CV

xgb_params  = {'n_estimators':[500,1000],
               'max_depth':[2,5],
               'learning_rate':[0.1,2]}

xgb_model_cv = GridSearchCV(xgb_,xgb_params,cv=12)
xgb_model_cv.fit(x_train,y_train)
xgb_model_cv.best_params_

xgb_cv = XGBRegressor(learning_rate=0.1,max_depth=2,n_estimators=500).fit(x_train,y_train)
y_xgb = xgb_cv.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_xgb))


# 13A -LIGHTGBM

#pip install lightgbm

from lightgbm import LGBMRegressor

lgb = LGBMRegressor()
lgb_ = lgb.fit(x_train,y_train).predict(x_test)
np.sqrt(mean_squared_error(y_test,lgb_))

# 13C- LIGHTGBM- CV

lgb_cv  = GridSearchCV(lgb, xgb_params, cv = 12)
lgb_cv.fit(x_train,y_train)
lgb_cv.best_params_
lgb_cv_ = LGBMRegressor(learning_rate=0.1,max_depth=2,n_estimators=500).fit(x_train,y_train)
y_lgb = lgb_cv_.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_lgb))

# 14A - CATBOOST

from catboost import CatBoostRegressor
catb = CatBoostRegressor()
catb_ = catb.fit(x_train,y_train).predict(x_test)
np.sqrt(mean_squared_error(y_test,catb_))


# 14C - CATBOSST- CV

catb_parms  = {'iterations':[500,1000],
               'depth':[2,5],
               'learning_rate':[0.1,2]}

catb_cv = GridSearchCV(catb,catb_parms, cv = 12).fit(x_train,y_train)
catb_cv.best_params_

catb_model = CatBoostRegressor(iterations=500,depth=2,learning_rate=0.1).fit(x_train,y_train)
catb_model_ = catb_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,catb_model_))

#SINIFLANDIRMA MODELLERI

# 15A- LIGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
df.head()

y = df['Exited']
x = df.drop(['Exited'], axis= 1)
logstc = sm.Logit(y,x).fit()
logstc.summary()

log = LogisticRegression(solver='liblinear')
log_ = log.fit(x,y)
log_


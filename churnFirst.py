import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Churn_Predictions.csv')
df.head()

df.info()

'''ayni kisiler tekrar yazilmis olabilir, kontrol etmeliyim!!!'''
df[df.duplicated()]

'''ayni satir hic tekrar etmemis, bu iyi. Ama ayni kisi tekrar etmis olabilir!!!
 tekrarlayan kisileri bulmak icin customerid'lere bakacagim!'''
df[df['CustomerId'].duplicated()]

'''simdi Gender icin one-hot donusumu yapalim'''
df_one_hot = pd.get_dummies(df, columns = ['Gender'], prefix = ['Gender'])
df_one_hot.head()

'''yalniz bu donusum sonrasinda olusan dummy degisken tuzagindan korunmaliyiz!
o yuzden Gender_Female'i siliyorum'''
del df_one_hot['Gender_Female']
df_one_hot.head()

'''simdi Geography icin one-hot donusumu yapalim'''
df_one_hot = pd.get_dummies(df_one_hot, columns = ['Geography'], prefix = ['Geography'])
df_one_hot.head()

'''
 yine dummy degisken tuzagindan kacinmak icin bir tanesini silmemiz gerekiyor. 
Fransayi siliyorum
'''
del df_one_hot['Geography_France']
df_one_hot.head()

'''
simdi numerik degiskenler arasindaki iliskiyi inceleyelim
'''
num_columns = df_one_hot.select_dtypes(include = [np.number])
correlations = num_columns.corr()
print(correlations['Exited'].sort_values(ascending = False))

'''
bu iliskileri bir de heatmap'te gosterelim
'''
corr_ = df_one_hot.corr()
top_corr_features = corr_.index
plt.figure(figsize=(12,12))
g = sns.heatmap(df_one_hot[top_corr_features].corr(),annot=True)
plt.show()

'''
simdi akla gelen bazi sorulari veriyi gorsellestirerek incelemeye calisalim
'''
# SORU-1: Kredi karti olmayan kisilerin yuzde kaci churn etmis?
x = df_one_hot[df_one_hot['HasCrCard'] == 0]['Exited'].value_counts()
kkolmyn_yuzde = (x[1]/x.sum())*100
print('% ' + str(kkolmyn_yuzde))

# SORU-2: Kredi karti olan kisilerin yuzde kaci churn etmis?
y = df_one_hot[df_one_hot['HasCrCard'] == 1]['Exited'].value_counts()
kkolan_yuzde = (y[1]/y.sum())*100
print('% ' + str(kkolan_yuzde))

# SORU-3: Hangi ulkeden ne kadar veri kaydi yapilmis?
df['Geography'].value_counts().plot.barh()
plt.show()

# SORU-4: Urun sayilarina gore kayitli kisi sayisi nedir?
sns.barplot(x = "NumOfProducts", y = df_one_hot.NumOfProducts.value_counts(),
            data= df_one_hot);
plt.show()

# SORU-5: Kadin ve erkeklerin aktif uyelik durumu nedir?
df.groupby('Gender')['IsActiveMember'].value_counts().plot.barh()
plt.show()

# SORU-6: Kadin ve erkeklerin kacar tane urunleri var?
df.groupby(['Gender'])['NumOfProducts'].value_counts().plot.barh()
plt.show()

# SORU-7: Age'deki aykiri degerler?
sns.boxplot(df_one_hot['Age'])
plt.show()

# SORU-8: CreditScore'deki aykiri degerler?
sns.boxplot(df_one_hot['CreditScore'])
plt.show()

# SORU-9: Balance'deki aykiri degerler?
sns.boxplot(df_one_hot['Balance'])
plt.show()
'''
Yukarida boxplot grafigine bakarsak normale gore buyuk oldugunu goruyoruz. 
Peki bunu nasil yorumlamayiz? Soyle, aslinda boxplot grafiginde bir problem yok,
fakat balance(bakiye) degerleri cok daginik. 
Bakiyenin cok kucuk oldugu degerleri de olmus, cok buyuk degerleri de. 
Bu nedenle boxplot grafigi siskin cikmis.'''

# SORU-10: EstimatedSalary'deki aykiri degerler?
sns.boxplot(df_one_hot['EstimatedSalary'])
plt.show()

'''
Simdi degiskenlerdeki aykiri gozlemlerin yuzdesini veren bir dongu yazalim,
hangi degiskenin yuzde kac oraninda aykiri gozlem verdigini tespit etsin
'''
for i in num_columns.columns:
    Q1 = df_one_hot[i].quantile(0.25)
    Q3 = df_one_hot[i].quantile(0.75)
    IQR = Q3 - Q1
    ust_sinir = Q3 + 1.5*IQR
    alt_sinir = Q1 - 1.5*IQR
    toplam_aykiri_deger_sayisi = (df_one_hot[df_one_hot[i] < alt_sinir].value_counts().sum()+
                                  df_one_hot[df_one_hot[i] > ust_sinir].value_counts().sum())
    toplam_deger_sayisi = df_one_hot[i].value_counts().sum()
    print(i + '--> % ' + str((toplam_aykiri_deger_sayisi/toplam_deger_sayisi)*100))

'''
Aslinda verisetini dikkatli bir sekilde incelersek bircok degiskenin 
sayisal gorunumlu olmasina ragmen kategorik oldugunu anlariz. Biz normalde bu
donguyu sadece kategorik olmayan sayisal degiskenler icin uygulamaliydik.
Ornegin 'Exited' degiskeni 0 ve 1'lerden olusmasina(kategorik olmasina) ragmen
%20 oraninda aykiri deger tespit edilmis. Bu sonuc bizi yaniltmaktadir.
O yuzden sadece yas(Age),bakiye(Balance),tahmini maas(EstimatedSalary)
degiskenleri icin bu degerler gecerlidir.'''

# SORU-11: Ulkelere gore erkek ve kadinlarin bakiyeleri?
sns.boxplot(x="Geography", y="Balance",
            hue="Gender", palette=["m", "g"],data=df)
plt.show()
'''
Bu grafige bakarak Almanlarin bakiye ortalamalarinin digerlerine gore
daha fazla oldugunu ve standart sapmalarinin da daha az oldugunu goruyoruz.
'''

# SORU-12: Ulkelere gore erkek ve kadinlarin maas tahminleri?
sns.boxplot(x="Geography", y="EstimatedSalary",
            hue="Gender", palette=["m", "g"],data=df)
plt.show()

# SORU-13: Kullanilan vade miktarlarinin yas ve cinsiyet ile iliskisi?
sns.boxplot(x="Tenure", y="Age",
            hue="Gender", palette=["m", "g"],data=df)
plt.show()

# SORU-14: Kredi karti sahiplik durumunun kredi skoru ve cinsiyet ile iliskisi?
sns.boxplot(x="HasCrCard", y="CreditScore",
            hue="Gender", palette=["m", "g"],data=df)
plt.show()

# SORU-15: Aktif uyelerin churn etme durumlari ve bunun yas ile iliskisi?
sns.boxplot(x="Exited", y="Age",
            hue="IsActiveMember", palette=["m", "g"],data=df)
plt.show()

# SORU-16: Urun sayisina gore aktif uyeligin yas ile iliskisi?
sns.boxplot(x="NumOfProducts", y="Age",
            hue="IsActiveMember", palette=["m", "g"],data=df)
plt.show()

# SORU-17: Urun sayisina gore kredi skoru ve aktif uyelik iliskisi?
sns.boxplot(x="NumOfProducts", y="CreditScore",
            hue="IsActiveMember", palette=["m", "g"],data=df)
plt.show()

'''
Soru sayisi ne kadar fazla olursa veri o kadar iyi taninmis olur. Bu sorulari
ornek olmasi icin koydum, verisetini nasil anlariz, ilk asamada ne yapmaliyiz
gibi sorulariniza umarim yanit olmustur. Simdi gereksiz bilgileri 
silerek ML modelleri icin verisetimizi hazirlayalim.'''

df_one_hot.head(15)
'''
Verisetinde cok fazla degisken varsa ve kullandiginiz programda bazi 
degiskenleri goremiyorsaniz asagidaki komutlari kullanabilirsiniz.
'''
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

'''
Verisetinde RowNumber, CustomerId, Surname gibi verileri silebiliriz, 
aksi halde ML modelimizde gurultuye sebep olabilir bu veriler.
'''

del df_one_hot['RowNumber']
del df_one_hot['CustomerId']
del df_one_hot['Surname']
df_one_hot.head()

df_ = df_one_hot.copy()
df_.head()

'''
Veri on isleme ve gorsellestirme adimlarimizi tamamladik.
Simdi ML modellerimizi kurmaya baslayabiliriz
'''

df_.head()

df[df['Age'] == 42]
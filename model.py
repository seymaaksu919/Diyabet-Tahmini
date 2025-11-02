import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



#Veri seti yüklenir ve ilk inceleme yapılır.
df = pd.read_csv("data/diabetes.csv")
print(df.head())
print(df.describe().T)
print(df.isna().sum())


#Sütunların grafikleri 
fig = plt.figure(figsize=(12, 8))
df.hist(color="green", bins=20, edgecolor="black", grid=False, figsize=(12, 8))
plt.tight_layout()
plt.show()

#Korelasyon matrisi oluşturma
numerical_df = df.select_dtypes(include=["float64", "int64"])
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.8, linecolor='gray', cbar_kws={"shrink": .8})

plt.title("Korelasyon Matrisi", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



#Yaş grubu analizi yapılır.
df['Age_group'] = pd.cut(df['Age'], bins=[20,30,40,50,60,70,80],
                         labels=['21-30','31-40','41-50','51-60','61-70','71-80'])
sns.countplot(data=df, x='Outcome', hue='Age_group')
plt.title("Diyabet Hastalığı ile Yaş Arasındaki İlişki")
plt.xlabel("Diyabet Durumu (0=Yok, 1=Var)")
plt.ylabel("Hasta Sayısı")
plt.legend(title='Yaş Grubu')
plt.show()


#Glikoz grubu analizi
df['Glucose_group'] = pd.cut(df['Glucose'], bins=[0, 100, 125, 200],
                             labels=['Normal','Prediyabet','Yüksek'])
sns.countplot(data=df, x='Outcome', hue='Glucose_group')
plt.title("Diyabet Hastalığı ile Glikoz Seviyesi Arasındaki İlişki")
plt.xlabel("Diyabet Durumu (0=Yok, 1=Var)")
plt.ylabel("Hasta Sayısı")
plt.legend(title='Glikoz Seviyesi')
plt.show()


#Feature Engineering yapılır ve modelin daha doğru tahmin yapması sağlanır
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Age_Insulin'] = df['Age'] * df['Insulin']
df['Log_Glucose'] = np.log1p(df['Glucose'])


#Feature Scaling(Outcome yani hedef değişken hariç tüm sayısal değişkenler ölçeklendirilir.)
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if "Outcome" in numerical_columns:
    numerical_columns.remove("Outcome")

scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


#Bağımlı ve bağımsız değişkenler ayrılır.
X = df.drop(["Outcome", "Age_group", "Glucose_group"], axis=1)
y = df["Outcome"]

#Hedef değişkenin sadece 0 ve 1 değerlerinden oluştuğunu doğrularız.
print("y unique values:", sorted(y.unique()))
if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Hedef sütun 'Outcome' 0/1 formatında değil!")



#Eğitim ve test verisi olarak ayırma işlemi yapılır.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Eğitim veri boyutu:", X_train.shape)
print("Test veri boyutu:", X_test.shape)



from xgboost import XGBClassifier

#Model için hiperparametrelerini sistematik olarak dener. Deneyerek elde eder en iyi parametreleri.
from sklearn.model_selection import GridSearchCV

#MOdel için kullanılacak metrikler
from sklearn.metrics import make_scorer, f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay


#Modelin oluşturulması ve hiperparametre optimizasyonu
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum()  #Veri dengesizse örnek 0 sınıfı daha fazla ise bu  oranlama ile sınıf dengesizliğini önler.
                                                                
)

#Hiperparametre ızgarası
param_grid = {
    'max_depth': [3, 4],             #Ağaç derinliği
    'learning_rate': [0.05],        #Küçük değer → daha stabil  Her adımdaki öğrenme oranı
    'n_estimators': [100, 150, 200],    #Ağaç sayısı (Oluşacak)
    'subsample': [0.7, 0.8],            #Rastgele seçilecek örnek oranı
    'colsample_bytree': [0.8, 0.9]      #Özelliklerin rastgele seçimi
}

#GridSearchCV Nesnesi oluşturma
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=make_scorer(f1_score),  #Pozitif ve negatif sınıfları dengeler
    cv=5,
    verbose=2,
    n_jobs=-1
)


#Grid Search ile eğit tüm hiperparametre kombinasyonlarını dener.
#Her eğitimde farklı  veri sütunları ile test eder.
#F1 skoruna göre en iyi modeli seçer.
grid_search.fit(X_train, y_train)



#En iyi model seçimi en iyi sonucu veren model döndürülür.
best_model = grid_search.best_estimator_
print("En iyi parametreler:", grid_search.best_params_)  #Ve en iyi modeli veren parametreler yazdırılır.


# Test verisi ile tahmin
y_predict = best_model.predict(X_test)
y_predict_proba = best_model.predict_proba(X_test)[:,1]  #Her örnek için 0 ve 1 olasılıklarını döndürür.



#Sonuçlar
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Classification Report:\n", classification_report(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("ROC AUC:", roc_auc_score(y_test, y_predict_proba))

#ROC eğrisi çizdirilir.
#Duyarlılık ve özgüllük arasındaki dengeyi gösterir.
#Eğri ne kadar sol üst köşeye yakınsa, model o kadar iyidir.
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

# Confusion matrix hesapla
cm = confusion_matrix(y_test, y_predict)

# Görselleştir
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap="Greens", values_format='d')
plt.title("Karışıklık Matrisi", fontsize=14)
plt.show()



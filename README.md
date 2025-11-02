# Diyabet Tahmin Modeli – XGBoost ile Veri Analizi ve Tahmin

Bu proje, Pima Indians Diabetes veri seti kullanılarak kişilerin diyabet riskini tahmin etmeyi amaçlamaktadır. Veri analizi, feature engineering ve makine öğrenmesi adımları Python ekosistemi kullanılarak uygulanmıştır.


## 🚀 Proje Hedefi

* Kişilerin demografik ve biyometrik özelliklerine göre diyabet riskini tahmin etmek.
* Veri analizi ve görselleştirme ile değişkenler arasındaki ilişkileri incelemek.
* XGBoost Classifier ile yüksek performanslı bir sınıflandırma modeli geliştirmek.



## 📂 Veri Seti

* **Kaynak:** [Kaggle – Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
* **Özellikler:**

  * Pregnancies: Gebelik sayısı
  * Glucose: Glikoz seviyesi
  * BloodPressure: Kan basıncı
  * SkinThickness: Cilt kalınlığı
  * Insulin: İnsülin seviyesi
  * BMI: Vücut kitle indeksi
  * DiabetesPedigreeFunction: Genetik faktör skoru
  * Age: Yaş
  * Outcome: 0 (Diyabet yok) / 1 (Diyabet var)



## 🧾 Veri Analizi

1. **İlk İnceleme**

   * Veri seti yüklendi, ilk 5 satır ve istatistiksel özet çıkarıldı.
   * Eksik değerler kontrol edildi.

2. **Veri Görselleştirme**

   * Histogramlar ve dağılım grafikleri ile sütunların dağılımları incelendi.
   * Korelasyon matrisi ile değişkenler arası ilişkiler analiz edildi.
   * Yaş ve glikoz seviyelerine göre diyabet durumunun dağılımı görselleştirildi.



## 🛠 Feature Engineering

Model performansını artırmak için bazı yeni özellikler oluşturuldu:

* **Glucose_BMI:** Glikoz seviyesi × BMI
* **Age_Insulin:** Yaş × İnsülin seviyesi
* **Log_Glucose:** Glikoz seviyesinin logaritmik dönüşümü

Bu sayede model, değişkenler arasındaki etkileşimleri daha iyi öğrenebiliyor.



## ⚖️ Feature Scaling

* Tüm sayısal değişkenler `StandardScaler` ile standartlaştırıldı.
* Outcome değişkeni (0/1) hariç tüm değişkenler ölçeklendirildi.



## 📊 Modelleme

* **Model:** XGBoost Classifier
* **Hiperparametre Optimizasyonu:** GridSearchCV ile yapıldı.
* **Veri Dengesizliği:** `scale_pos_weight` kullanılarak pozitif/negatif sınıf dengesi sağlandı.

**Hiperparametre Örneği:**

* max_depth: 3-4
* learning_rate: 0.05
* n_estimators: 100-200
* subsample: 0.7-0.8
* colsample_bytree: 0.8-0.9



## 📈 Model Performansı

**Test Seti Sonuçları:**

| Metric       | Value |
| ------------ | ----- |
| Accuracy     | 0.77  |
| F1-Score (1) | 0.69  |
| ROC AUC      | 0.84  |

**Confusion Matrix:**

```
[[77 23]
 [13 41]]
```

**ROC Eğrisi:**
ROC eğrisi, modelin sınıfları ayırma yeteneğini gösterir. Eğri ne kadar sol üst köşeye yakınsa, model o kadar iyi çalışır.



## 🖼 Görselleştirmeler

* Histogramlar ve dağılım grafikleri
* Korelasyon Matrisi
* Yaş / Glikoz gruplarına göre dağılım
* Confusion Matrix
* ROC Eğrisi


## 📦 Kullanılan Kütüphaneler

* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn
* xgboost



## 💡 Sonuç ve Öneriler

* Model, diyabet tahmini için iyi bir başlangıç noktası sağlar (ROC AUC: 0.84).
* Daha fazla veri, farklı feature engineering ve model tuning ile performans artırılabilir.
* Görselleştirmeler ve metrikler, modelin güvenilirliğini ve açıklanabilirliğini artırır.


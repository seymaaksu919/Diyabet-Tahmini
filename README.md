Tabii, tüm README dosyasını baştan sona, performans sonuçlarını da içinde barındıracak şekilde aşağıda hazırladım. İstersen direkt kullanabilirsin:

---

# Diyabet Tahmini için Karar Ağacı Modeli

## Proje Hakkında

Bu proje, Pima Indians Diabetes veri seti kullanılarak diyabet hastalığının tahmin edilmesini amaçlamaktadır. Karar ağacı algoritması ile oluşturulan model, hastaların çeşitli medikal verilerine dayanarak diyabet olup olmadığını sınıflandırır.

## Kullanılan Veri Seti

* **Pima Indians Diabetes Dataset**: UCI Machine Learning Repository’den alınmıştır.
* İçeriği: Yaş, gebelik sayısı, glukoz seviyesi, kan basıncı, cilt kalınlığı, insülin, BMI, genetik faktörler gibi 8 özellik.
* Hedef değişken: Diyabet durumu (pozitif/negatif).

## Yöntem

* Veri ön işleme ve temizleme yapıldı.
* Karar ağacı sınıflandırma algoritması kullanıldı.
* Model eğitildi ve test verisi üzerinde doğruluk ve diğer metriklerle değerlendirildi.
* Modelin performansı analiz edildi.

## Model Performans Sonuçları

Model, diyabet tahmininde aşağıdaki performans değerlerini elde etmiştir:

| Metrik    | Değer  |
| --------- | ------ |
| Accuracy  | %75.97 |
| Precision | %68.00 |
| Recall    | %61.82 |
| F1 Score  | %64.76 |

### Detaylı Sınıflandırma Raporu

| Sınıf (0 = Diyabet Yok) | Precision | Recall | F1-Score | Destek (Örnek Sayısı) |
| ----------------------- | --------- | ------ | -------- | --------------------- |
| 0                       | 0.80      | 0.84   | 0.82     | 99                    |
| 1 (Diyabet Var)         | 0.68      | 0.62   | 0.65     | 55                    |

### Yorum

Model, diyabet olmayanları daha iyi sınıflandırırken (precision ve recall değerleri yüksek), diyabetli hastalar için doğru tespit oranı (recall) biraz daha düşük kalmıştır.
Bu, bazı diyabet hastalarının model tarafından gözden kaçırıldığı anlamına gelir.
Bu sebepten ilerleyen aşamalarda performansın artırılması için veri seti dengelenebilir, farklı algoritmalar denenebilir veya parametre optimizasyonları yapılabilir.



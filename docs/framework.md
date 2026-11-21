# CS415 Group 21 · Araştırma çerçevesi ve iş bölümü

**Sürüm 2 · 18 Kasım 2026.** Hocanın proposal feedback'ine göre güncellendi; gerekçeler 04_Feedback_Uyum.md (Can'ın planlama klasöründe) §3–4'te.
**Durum:** Grup onayı bekliyor. Progress report teslimi **22 Kasım** olduğu için uygulama hemen başlıyor (Can'ın kararı, 2026-11-18).

**Dayanak belgeler:**
- 01_Oneri_Denetimi.md (Can'ın planlama klasöründe): onaylı önerinin kaynak denetimi.
- 02_Cerceve_Kararlari.md (Can'ın planlama klasöründe): kararların gerekçeleri, kaynakları ve hesapları (Tur 1–9).
- 04_Feedback_Uyum.md (Can'ın planlama klasöründe): hoca feedback'i, seçenekler, LLM council kararı.

**Kullanım notu.** Bu bir planlama belgesidir. Progress report ve final raporu grup kendi sözleriyle yazar; ders üretken yapay zekâ için açık izin vermiyor.

---

## 1. Amaç ve neden

**Görev.** DEAM'deki 45 saniyelik kliplerden statik valence ve arousal tahmin etmek. Konu ders tarafından Group 21'e atandı.

**Neden yeniden çerçeveledik.** Önerideki H2, σ'nın hem kalibre olduğunu hem de dinleyici anlaşmazlığını yansıttığını iddia ediyordu.
- Watcharasupat ve ark. 2025 aynı veride σ ile dinleyici anlaşmazlığı arasındaki ilişkiyi ≈ 0 bulmuş.
- Hoca da feedback'te aynı noktaya bağımsız olarak dikkat çekti: ortalama puana karşı NLL ile öğrenilen σ, dinleyiciler arası varyans değildir.

**Birincil iddialar (hocanın önceliği, F-g):**
1. Önceden eğitilmiş temsiller, aynı başlıkla kullanıldığında el yapımı özelliklerden daha iyi tahmin verir.
2. Tahmin edilen σ'nın tam olarak neyi temsil ettiği belirlenir.

Bunun dışındaki her şey ikincil ya da rapordan sonraya bırakılıyor.

**σ'nın tanımı (raporda ve kodda bu cümle aynen kullanılır):**
> "σ, ortalama puanı tahmin eden modelin öngörü hatasının tahminidir. İçinde modelin yaklaşım hatası ve ortalama almadan kalan gürültü (≈ s²/n) bulunur. Bireysel dinleyicilerin yayılımı değildir. Tek bir ağın σ'sı epistemik belirsizliği kapsamaz; onun için seed ensemble'ının yayılımına bakılır."

**Özgün katkı (rapordan sonra):** Watcharasupat'ın açık bıraktığı soru: "Donuk omurga anlaşmazlık bilgisini kaybetti mi?" (H2b). Kendimizi açıkça "kısmi replikasyon ve uzatma" olarak konumluyoruz.

**Kapsam dışı:** klip içi dinamik tahmin, 2015 seti. Bireysel dinleyici dağılımının modellenmesi (O4) için bkz. §3.

**Başarısızlık sayılacak durumlar (anti-hedefler):**
- Test setine bakarak eşik ya da model değiştirmek.
- σ'yı dinleyici anlaşmazlığıyla eşit saymak.
- Sondanın gösterdiği "bilgi bulunuyor" sonucunu "model bu bilgiyi kullanıyor" diye sunmak.
- Progress report'ta H2b sonucu ima etmek. H2b'nin boş modeli (rastgele MERT) henüz çalıştırılmadı.

## 2. Hipotezler

| Hipotez | İddia | Ölçütler | Ne zaman |
|---|---|---|---|
| **H1** | Önceden eğitilmiş temsil (MERT), aynı başlıkla openSMILE'ı geçer. MERT, PANNs'ı geçer (ikincil). | C2 (birincil), C2b, C3 (ikincil), C1 (sağlamlık) | C2 progress report'ta; C2b ve C3 sonra |
| **H2a** | Heteroskedastik başlığın σ'sı kalibredir, gerçek hatayla sıralanır ve sabit σ'dan iyidir. Bu, σ'nın modelin kendi hatasının tahmini olarak işlediği anlamına gelir. | C5, C7 (birincil), C4 (ikincil) ve §4.3'teki betimsel analizler | İlk sonuçlar progress report'ta |
| **H2b** | Donuk temsilde, μ'nün açıkladığından fazla ve doğrusal yolla çıkarılabilir anlaşmazlık (s) bilgisi vardır. | C6a, C6b | Rapordan sonra |

## 3. Veri

- **Kullanılan:** DEAM 2013–2014 setinden 1.744 adet 45 sn klip.
  - Statik etiketler 1–9 ölçeğinde μ ve s olarak verilmiş.
  - Ölçekleme: μ → (μ − 5)/4, s → s/4.
  - Lisans CC BY-NC.
- **Çıkarılan:** 2015 setinin 58 tam şarkısı (S9).
- **Kılavuzdan doğrulananlar (2026-11-18):**
  - Dinamik etiketler için dinleyici başına ham eğriler var.
  - Metadata başlık, tür ve sanatçı içeriyor.
  - Statik bireysel puanlardan **söz edilmiyor**.
- **Veri denetimi** (betikle; raporda bölüm olarak; Q3):
  - Klip sayısı ve μ, s aralıkları; (x − 5)/4 dönüşümünün tutarlılığı; eksik değerler.
  - Klip başına dinleyici sayısı n. s/√n hesabı için gerekli.
  - Ses, `features.zip` ve csv arasında kimlik eşleşmesi.
  - Sanatçı adlarının normalizasyonu ve sanatçı başına şarkı sayısı.
  - 2013 ve 2014 grupları ayrı bir değişken olarak işaretlenir.
  - Statik bireysel puan dosyası var mı (evet/hayır)?
- **Bireysel puanlar (Q9):**
  - **Varsa:** rapordan sonra split-half güvenilirliğiyle gürültü tavanı hesaplanır. Dağılım modellemesi (O4) ek hedef olur.
  - **Yoksa:** analitik tavan R²_max ≈ 1 − ort(s²/n) / Var(μ) verilir. Raporda (μ, s) üzerinde Gauss–Gauss KL ya da W2 kayıplı bir dağılım başlığı, çalıştırılmamış bir seçenek olarak anılır.
  - Dinamik ham eğrilerden türetilen dinleyici ortalamaları yalnızca betimsel kullanılır. Bunun sınırı, eğrilerin yalnızca 15–45 sn aralığını kapsaması.
- **Sanatçı alanı yoksa ya da kullanılamazsa:** şarkı kimliğine göre gruplanır ve bu, sapma olarak raporlanır.

## 4. Değerlendirme protokolü

### 4.1 Bölme (S8 revizyonu, Q2)

- **Sanatçıya göre gruplanmış, V–A çeyreğine göre tabakalı 10 katlı bölme** (StratifiedGroupKFold).
  - Her katta test %10.
  - Doğrulama, eğitim kısmından aynı gruplama kuralıyla ayrılır (≈ 80/10/10).
  - Hiperparametreler yalnızca o katın doğrulama setinde seçilir.
- **Neden değişti:** Tekrarlı rastgele 70/15/15 bölmede test setleri örtüşüyor. Bölmeler arası bootstrap ve P(A>B) bu yüzden fazla iyimser oluyor. Katlı bölmede her klip tam bir kez test ediliyor; böylece σ analizleri 1.744 klibin birleşik katdışı (OOF) tahminleriyle yapılabiliyor (262 yerine).
- **Seed:** Her model her katta 3 seed ile eğitilir.
- Bölme dosyası sabit bir seed ile üretilir, depoya işlenir ve bir testle doğrulanır: katlar arasında sanatçı ya da şarkı örtüşmesi sıfır olmalı.
- **Rastgele (sanatçı ayrımsız) 10 katlı bölme** baseline ve en iyi MERT başlığıyla koşulur. Sanatçı etkisini ve önceden eğitilmiş temsilin kazancının korunup korunmadığını betimsel olarak ölçer. Rapordan sonra (Q10).

### 4.2 Karşılaştırma

- Katlar üzerinden eşleştirilmiş farklar.
- Ana karar: P(A>B) ≥ 0,75 (Bouthillier 2021).
- CI'lar sanatçıya göre gruplanmış bootstrap ile hesaplanır.
- Bir hipotez ailesinde birden fazla test varsa Holm düzeltmesi uygulanır.
- **Aynı başlık kuralı:** C2'de temsil etkisi başlık etkisinden ayrılır. Birincil karşılaştırma ridge(openSMILE) ile ridge(MERT) arasında. SVR baseline ayrıca raporlanır.

### 4.3 Ölçümler (tanımlar sabit)

| Ölçüm | Tanım ve not |
|---|---|
| R² | Test setinin kendi ortalamasına göre (sklearn tanımı) |
| RMSE | Hem [−1, 1] hem 1–9 ölçeğinde |
| CCC | Sapmayı cezalandırır; ridge'in büzülmesi CCC'yi düşürür |
| Pearson r | CCC ile birlikte raporlanır; düşük CCC'nin düşük sıralama başarısı sanılmaması için |
| PICP | 68 ve 95 düzeylerinde ve tam kalibrasyon eğrisi |
| NLL | Düz Gauss NLL; β-NLL'in kayıp değeri değil |
| z-skorları | z = (y − μ̂)/σ̂ için ortalama, varyans ve kuyruk histogramı |
| Güvenilirlik diyagramı ve ENCE | Levi ve ark. 2022. Birleşik OOF üzerinde, eşit sayılı kutular ve bootstrap CI. Mükemmel kalibre simülasyonundan bir referans değer verilir. |
| ρ(σ, \|e\|) | Spearman. Tavanı simülasyonla verilir: e ~ N(0, σ̂²). Ayrıca μ̂'nün merkeze uzaklığı kontrol edilerek kısmi ρ. |
| ρ(σ, s) | Betimsel. Ham değer ve esnek biçimde μ ile μ̂ kontrol edilmiş kısmi Spearman. Ayrıca ρ(\|e\|, s), kısmi ρ(σ, s \| \|e\|) ve σ ile s/√n ilişkisi. |
| σ ayrıştırması | σ̂² üzerine betimsel regresyon: s²/n (etiket gürültüsü), seed ensemble varyansı (epistemik) ve \|μ̂\| (merkezden uzaklık). Hocanın "σ tam olarak neyi temsil ediyor" sorusuna doğrudan cevap. |
| AURC | C7 için |

**Ortak tahmin dosyası:**
`fold_id, split_type, seed, model, song_id, mu_V, mu_A, sigma_V, sigma_A`
Nokta tahmin eden modellerde σ boş bırakılır (NaN).

### 4.4 Ölçütler

| Kod | Statü | Koşul |
|---|---|---|
| C1 | **Sağlamlık kontrolü** (geç/kal değil) | openSMILE baseline katların medyanında R²V ve R²A, CI ile raporlanır. Referans olarak Cheuk'un değerleri (0,32 / 0,64, sanatçı ayrımı belirtilmemiş) ve Castellon'un sanatçıya göre tabakalı MFCC değerleri (0,27 / 0,48) verilir. Altında kalması H1'i geçersiz kılmaz. |
| C2 | Birincil | Aynı başlıkla MERT, openSMILE'ı her boyutta P(A>B) ≥ 0,75 ile geçer. |
| C2b | İkincil, rapordan sonra | MERT, PANNs'ı P(A>B) ≥ 0,75 ile geçer. |
| C3 | İkincil, betimsel | ΔR²V > ΔR²A. İki boyutun tavanları ve test varyansları farklı olduğu için hipotez testi değil. |
| C4 | İkincil | β-NLL ile MSE eşi arasındaki ΔR² için %90 CI ±0,02 içinde (TOST); CI daha genişse "belirsiz". |
| C5 | Birincil | Medyan PICP95 ∈ [0,90; 0,98] ve PICP68 ∈ [0,60; 0,76]. NLL, sabit σ referansından iyi (P ≥ 0,75). Sabit σ doğrulama artıklarından ve aynı μ̂ ile hesaplanır. |
| C6a, C6b | Rapordan sonra | Değişmedi (02 Tur 8). |
| C7 | Birincil | σ'ya göre sıralamanın AURC'si, μ̂'nün merkeze uzaklığına göre sıralamanınkinden küçük (P ≥ 0,75). |

**Önceden yazılmış yorumlar:**
- **C5'in kalması beklenen bir olasılık.** Yaklaşık 1.400 eğitim klibi ve yüksek boyutlu bir embedding ile eğitim artıkları test artıklarından küçük olur, bu da σ'yı küçük tahmin ettirebilir. Bu durumda doğrulama setinde tek bir ölçek katsayısıyla yeniden kalibrasyon **ikincil** olarak raporlanır. Sıralama ölçüleri bu katsayıdan etkilenmez.
- **C1 kalırsa** bu, sanatçı ayrımının beklenen etkisi olarak yorumlanır (Castellon, yön kanıtı).
- **C6a geçip C6b kalırsa:** "Donuk temsilde μ'nün ötesinde çıkarılabilir anlaşmazlık bilgisi yok."

## 5. Yöntem

### 5.1 Temsiller (bir kez çıkarılır ve saklanır)

| Temsil | Ayar | Ne zaman |
|---|---|---|
| openSMILE | DEAM'in 500 ms kare özelliklerinden klip başına ortalama ve std (`features.zip`; ses gerekmez) | Hemen |
| MERT-v1-95M | 24 kHz, 5 sn pencereler, **13 katmanın hepsi tek geçişte** (Q1) | Hemen |
| Rastgele ağırlıklı MERT | Aynı mimari, eğitilmemiş ağırlıklar | Rapordan sonra |
| PANNs CNN14 | 32 kHz, pencere ortalaması | Rapordan sonra |

### 5.2 Başlıklar

- **Baseline:** openSMILE üzerinde ridge ve SVR (RBF).
- **H1:** MERT üzerinde ridge (başlıktaki sonuç) ve tek gizli katmanlı MLP (512, MERT protokolü). Katman doğrulama setinde seçilir. openSMILE üzerinde de aynı başlıklar.
- **H2a:** μ ve σ çıkaran bir MLP. β = 0,5 birincil, β = 0 karşılaştırma koşulu; ikisinin kalibrasyonu yan yana raporlanır. Referanslar: doğrulama artıklarından hesaplanan sabit σ ve seed ensemble'ının yayılımı.

### 5.3 Varyansın sayısal kararlılığı (F-e, Q5)

| Önlem | Ayar | Kaynak |
|---|---|---|
| Parametreleme | Yumuşak sınır: σ = σ_min + (σ_max − σ_min)·sigmoid(z). Sert kırpma gradyanı sıfırladığı için kullanılmaz. | Lakshminarayanan 2017 dn. 2 (softplus + taban); Seitzer 2022 Ek D (sınırlar) |
| σ_min | 0,01 (ölçekli birim). Ortalama alma gürültüsü s/√n ≈ 0,03–0,1 bunun üstünde. | Analitik |
| σ_max | Eğitim setindeki hedefin std'sinin 2 katı. Bu std, ortalamayı tahmin eden boş modelin hatasıdır; daha büyük bir σ bilgi taşımaz. Yalnızca eğitim verisinden hesaplanır. | Analitik (tasarım tercihi) |
| Isınma | İlk epoch'larda μ yalnızca MSE ile eğitilir, sonra β-NLL'e geçilir. | Skafte 2019 (ikincil aktarım, doğrulanmadı) |
| Aşırı uyum | Weight decay ve doğrulama setinde düz NLL ile erken durdurma | Seitzer 2022 §4.1 |
| İzleme | σ yüzdelikleri (1, 5, 50, 95, 99), sınırdaki σ oranı, düz NLL ve RMSE eğrileri | Seitzer 2022 §4.1 |
| Seçim | Model seçimi doğrulama setinde düz NLL ve RMSE ile yapılır; β-NLL'in kayıp değeriyle değil. | Seitzer 2022 §4.1 |

- **Uyarı kuralı (doğrulama setinde, teste bakmadan):** σ'ların %5'inden fazlası sınırlardan birine %1 yakınlıktaysa, bu bir kararlılık sorunu olarak raporlanır.
- Stirn 2023'teki "faithful" kayıp, rapordan sonra sabit bir ikincil koşul olarak denenir. Otomatik geçiş kuralı kaldırıldı, çünkü sonuçlara göre dallanmaya yol açıyordu.

### 5.4 H2b sondası (rapordan sonra)

02'deki S6–S7 kararları değişmedi. Özetle:
- Hedefler: s_V ve s_A.
- μ-referansı ve ridge sondası.
- Rastgele MERT ve μ pozitif kontrolü.
- 13 katmanın eğrisi betimsel; hipotez testi tek bir katman ölçüsüyle.
- Pencere ortalaması; ön analizden geçerse std de eklenir.

Birincil test MERT'te yapılır; PANNs ve openSMILE betimsel.

### 5.5 Ek hedef: LP-FT

Takvim izin verirse tek katta yapılır. Yapılmazsa sınırlılık olarak raporlanır.

## 6. İş bölümü

| Üye | 22 Kasım'a kadar | Rapordan sonra |
|---|---|---|
| **Can** | DEAM indirme, veri denetiminin etiket ve ses kısmı, MERT önbelleği (13 katman) | Rastgele MERT ve PANNs önbelleği, H2b |
| **Efe** | openSMILE özetleme, ridge ve SVR baseline (sesi beklemeden), sapma tablosu, ilgili çalışmaların güncellenmesi | C2b ve rastgele bölme baseline'ı |
| **Cem** | Ridge ve MLP başlıkları. Heteroskedastik başlık, MERT önbelleği hazırlanırken önce openSMILE özellikleri üzerinde kurulur ve sınanır; sonra MERT'e geçer. | Stirn koşulu, LP-FT |
| **Emir** | Sanatçı normalizasyonu, 10 katlı bölme dosyası ve bütünlük testi. Ölçüm betiğinin ilk sürümü: R², RMSE, CCC, Pearson, PICP, NLL, Spearman. | Gruplanmış bootstrap, P(A>B), TOST, Holm, ENCE, AURC, σ ayrıştırması, C1–C7 raporu |
| **Rapor sahibi** | Grup belirler. Yazım günü 21 Kasım. | — |

## 7. Takvim (22 Kasım progress report)

| Gün | İşler |
|---|---|
| **18 Kasım** | Grup bu belgeyi onaylar. Repo açılır. Emir bölme iskeletini kurar. Can zip'leri denetler: sanatçı alanı, statik bireysel puan, n. Efe openSMILE özetlemesine başlar. Cem başlık kodunu kurar. |
| **19 Kasım** | Bölme dosyası ve bütünlük testi işlenir. Veri denetimi biter. MERT önbelleği çıkarılır. openSMILE baseline koşulur. |
| **20 Kasım** | MERT ridge/MLP, openSMILE ile aynı başlıkla karşılaştırılır. Ölçüm betiğinin ilk sürümü hazır. Heteroskedastik başlığın ilk koşusu yapılır (β = 0 ve 0,5, kararlılık logları). |
| **21 Kasım** | Kalibrasyon, σ–hata ve σ–s analizleri; sabit σ karşılaştırması. Rapor ve sapma tablosu yazılır. |
| **22 Kasım** | Teslim. Yeni deney yapılmaz. Hangi ölçütlerin çalıştırılmadığı açıkça yazılır. |

**Kritik yol:** zip ve sanatçı alanı → bölme dosyası → MERT önbelleği → MERT başlıkları → heteroskedastik başlık.

**Darboğaz önlemleri:**
- Efe ve Cem önbelleği beklemez: Efe `features.zip` ile, Cem openSMILE özellikleriyle çalışır.
- Ölçüm betiği sonuçları beklemez; sahte bir tahmin dosyasıyla sınanır.

**Asgari teslim (heteroskedastik kısım bitmezse):** veri denetimi, baseline, MERT karşılaştırması ve heteroskedastik başlık "devam ediyor" notuyla.

## 8. Uygulayıcıya bırakılanlar

- MLP genişliği, öğrenme hızı, epoch sayısı, weight decay ve ısınma süresi doğrulama setinde seçilir.
- Spline düğümleri doğrulama setinde seçilir.
- Seed'ler sabitlenip raporlanır.
- Repo yapısı uygulama planında belirlenir.

## 9. Açık bağımlılıklar

- **D1:** Grubun bu belgeyi onaylaması, S3 (boş sonuç kabulü) ve rapor sahibinin belirlenmesi.
- **D2:** Zip denetimi. Sanatçı alanı ve statik bireysel puan olup olmadığı. 19 Kasım'a kadar bitmeli.
- **D3:** Progress report'un tarihi biliniyor (22 Kasım). Biçim ve sayfa sınırı hâlâ bilinmiyor.
- **D4:** GPU erişimi. MERT çıkarımı süresinin tahmini (2–8 dk) GPU varsayıyor; CPU'da çok daha uzun sürer.
- **D5:** Ortak repo.

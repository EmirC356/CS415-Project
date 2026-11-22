# openSMILE baseline ve iki veri sorunu (Efe, 22 Kasım)

## Herkesi etkileyen iki bulgu

### 1. 2013 statik etiketlerinde valence ve arousal yer değiştirmiş

2013 grubundaki 744 klipte statik `valence_mean` sütunu aslında arousal'ı, `arousal_mean` sütunu valence'ı taşıyor. 2014 grubu tutarlı. Kanıtlar `data/audit/va_swap_report.md` içinde, betik `data/audit/check_va_swap.py`.

| grup | r(statik V, dinamik V) | r(statik V, dinamik A) | r(statik A, dinamik A) | r(statik A, dinamik V) |
|---|---|---|---|---|
| 2013 | 0.575 | **0.956** | 0.570 | **0.949** |
| 2014 | **0.912** | 0.591 | **0.942** | 0.569 |

- Statik etiketler aynı kliplerin dinamik (2 Hz) anotasyonlarının zaman ortalamasıyla karşılaştırıldı.
- Ses enerjisi (openSMILE RMS) de aynı yönü gösteriyor: 2014'te enerji arousal ile daha güçlü ilişkili (ρ 0.42'ye karşı 0.32). 2013'teki ham etiketlerde ise ilişki valence ile daha güçlü (0.54'e karşı 0.32), ve etiketler yer değiştirilince örüntü 2014 ile aynı hale geliyor.
- DEAM kılavuzunda ve web'de bununla ilgili bir erratum bulamadım.
- Kişi başına statik puan dosyası da averaged dosyayla tutarlı, yani aynı yer değişimi orada da var.

**Ne yapmalı:**
- Bütün modeller `data/interim/labels_fixed.csv` dosyasını kullanmalı. Bu dosya audit'in `labels.csv` çıktısından şu komutla üretiliyor:
  `python data/audit/check_va_swap.py --raw data/raw --fix`
  Komut 2013 grubunda V ve A sütunlarını (μ, s, raw, file, adj) geri çeviriyor ve `va_swap_fixed` bayrağını ekliyor.
- **Emir:** `folds_artist.csv` dosyasındaki `quadrant` sütunu ters etiketlerle hesaplandı. 2013 kliplerinde çeyrek 2 ile 4 yer değiştiriyor. Sanatçı gruplaması geçerli; sadece tabakalama 2013 kısmında hafif kaymış. Dosyayı yeniden üretmeyi ya da raporda bunu not etmeyi sen seç.
- **Cem:** Heteroskedastik başlığın prototip sonuçları ters etiketlerle üretildi; düzeltilmiş etiketlerle yeniden koşulması lazım.
- Repo'daki prototip sonuçların (`outputs/`) hepsi bu hatayı içeriyor.

### 2. 2013 openSMILE dosyaları tam şarkıyı kapsıyor

- 2013 kliplerinin `features/<id>.csv` dosyaları 45 saniyelik kesiti değil, tam şarkıyı kapsıyor (en uzunu 585 s). Ses dosyası ve etiketler ise 45 saniyelik kesite ait.
- `features/aggregate_opensmile.py` artık `metadata_2013.csv` içindeki segment başlangıcına göre kesiti alıyor. "min.sec" biçiminde: `1.29` = 89 s, `0.2` = 20 s.
- Kesit konumu 21 klipte doğrulandı: mp3 kesitinin enerji eğrisi tam şarkının openSMILE enerjisiyle eşleştirildi, hepsi 0,5 saniye içinde tuttu.
- **Can:** MERT zaten 45 saniyelik mp3'ten çalıştığı için bu sorun onu etkilemiyor.

## Baseline sonuçları (düzeltilmiş etiketler, kırpılmış feature'lar)

Ayarlar:
- 520 boyutlu openSMILE (260 tanımlayıcının kesit üzerinden ortalaması ve std'si).
- Hiperparametreler her katta ve her boyutta doğrulama RMSE'sine göre seçildi, model 8 eğitim katında yeniden eğitildi.
- Tahminler `results/predictions/` altında sözleşme biçiminde.

10 test katı üzerinden ortalama ± SD:

| Model | split | R²_V | R²_A | CCC_V | CCC_A | RMSE_V | RMSE_A |
|---|---|---|---|---|---|---|---|
| Ridge | artist | 0.372 ± 0.079 | 0.671 ± 0.061 | 0.563 | 0.806 | 0.216 | 0.189 |
| SVR (RBF) | artist | 0.379 ± 0.075 | 0.693 ± 0.045 | 0.565 | 0.822 | 0.215 | 0.183 |
| Ridge | random | 0.395 ± 0.046 | 0.677 ± 0.036 | 0.579 | 0.810 | 0.215 | 0.189 |
| SVR (RBF) | random | 0.414 ± 0.047 | 0.698 ± 0.031 | 0.599 | 0.823 | 0.211 | 0.183 |

- **C1 (sağlamlık kontrolü):** Literatürle (Cheuk 0.32 / 0.64) tutarlı ve proposal eşiğinin (0.30 / 0.60) üstünde.
- **SVR ile ridge:** Arousal'da SVR anlamlı derecede daha iyi (8/10 kat, ΔR² = +0.022, Wilcoxon p = 0.049). Valence'ta fark yok (5/10 kat).
- **Sanatçı etkisi:** Sanatçı-gruplu split rastgele split'ten biraz zor. SVR valence'ta ΔR² ≈ 0.035, arousal'da ≈ 0.005. Hocanın "artist-disjoint protokolde eşikler iyimser" uyarısı valence için geçerli, arousal için değil.
- **Gruplara göre (SVR, artist):** 2013: R² 0.37 / 0.65. 2014: 0.43 / 0.74. Düzeltmeden önce 2013 arousal'ı 0.29'du.
- **Sabit σ referansı (`*_homo`, σ = doğrulama RMSE'si):** PICP68 0.67–0.68, PICP95 0.95. Ortalama düzeyde zaten kalibre. Heteroskedastik başlığın bunu NLL ile ve σ–hata sıralamasıyla geçmesi gerekiyor.

Düzeltmelerin etkisi (ridge, birleşik OOF R², V / A):

| koşul | V | A |
|---|---|---|
| veri olduğu gibi | 0.46 | 0.50 |
| kesit kırpma | 0.51 | 0.51 |
| kırpma ve etiket düzeltmesi | 0.38 | 0.68 |

Düzeltmeden önceki yüksek valence, 2013'te "valence" adı altında aslında arousal'ın tahmin edilmesinden kaynaklanıyordu.

## Dosyalar

| Dosya | Ne yapıyor |
|---|---|
| `features/aggregate_opensmile.py` | openSMILE önbelleği (kırpmalı) |
| `data/audit/check_va_swap.py`, `va_swap_report.md` | V/A yer değişimi kontrolü ve düzeltmesi |
| `models/baseline_opensmile.py` | ridge, SVR ve `*_homo` referansı |
| `eval/evaluate_v1.py` | Metrikleri hesaplamak için yazdım. **Emir, bu senin alanın**: istersen temel al, istersen kendi betiğinle değiştir. |
| `splits/folds_random.csv` | Rastgele 10 kat (StratifiedKFold, çeyreğe göre, seed 415). Emir'in alanı; sözleşme biçiminde, gözden geçir. |
| `results/predictions/opensmile_*` | Baseline tahminleri (artist ve random) |
| `results/ablation_*`, `results/label_swap_check/` | Ablasyonlar ve düzeltme öncesi sonuçlar |
| `report/` | Rapor taslakları: related work, sapma tablosu, refs.bib, figürler, tablolar |

# Sonuçlardaki sayısal değerlerin açıklaması (TR)

Bu doküman, `results/` altındaki tablolarda ve README'de geçen — ilk bakışta
anlaşılması zor olabilen — sayısal metrikleri Türkçe açıklar. Amaç: sunumda
"bu sayı ne demek?" sorusuna hızlı cevap verebilmek.

## Temel kavramlar

- **Embedding (512-d vektör):** ArcFace modeli her yüzü 512 sayıdan oluşan bir
  vektöre çevirir. Aynı kişinin farklı fotoğrafları benzer vektörlere, farklı
  kişiler uzak vektörlere düşer.
- **Cosine similarity (kosinüs benzerliği):** İki embedding arasındaki açının
  kosinüsü. `+1` = birebir aynı yön (çok benzer), `0` = alakasız, negatif = ters
  yön. Bizde iki fotoğrafın "aynı kişi mi" skoru budur.
- **Eşik / threshold (0.44):** Karar sınırı. Kosinüs benzerliği `>= 0.44` ise
  "aynı kişi", altındaysa "farklı kişi" deriz. Bu değer **sadece validation
  (doğrulama) kümesinde** seçildi, test kümesine dokunulmadan sabitlendi.

## Doğrulama (verification) metrikleri — `verification_metrics.csv`

- **AUC (Area Under ROC Curve):** 0–1 arası. Rastgele bir "aynı kişi" çiftinin
  skorunun, rastgele bir "farklı kişi" çiftinden yüksek olma olasılığı.
  `1.000` = kusursuz ayrım; `0.5` = rastgele tahmin. Bizde her poz kutusunda
  `1.000` (bu yüzden README tablosundan bu sabit sütun çıkarıldı).
- **EER (Equal Error Rate):** Yanlış kabul oranı ile yanlış ret oranının
  eşitlendiği noktadaki hata. Düşük = iyi. Bizde `0.000` (her kutuda sabit).
- **FAR (False Accept Rate):** Farklı kişileri yanlışlıkla "aynı" sayma oranı.
  `1e-3 = %0.1`, `1e-2 = %1`. Güvenlik uygulamalarında düşük FAR istenir.
- **TAR@FAR (True Accept Rate @ belirli FAR):** FAR'ı sabit bir seviyeye
  (örn. %0.1) ayarlayan eşikte, aynı kişileri doğru kabul etme oranı.
  Örn. `TAR@1e-3 = 0.996` → "yanlış kabul oranını %0.1'de tutarken, aynı kişi
  çiftlerinin %99.6'sını doğru eşledik". Tek düşüş burada: **frontal/profile**
  kutusunda 1.000 yerine 0.996.
- **accuracy@threshold (acc@1e-2):** Sabitlenen eşikte doğru sınıflanan
  (aynı+farklı) çiftlerin toplam orana. Bizde ~1.000.
- **mean pos. cosine (ortalama pozitif benzerlik = "marj"):** Aynı kişi
  çiftlerinin ortalama kosinüs benzerliği. **Asıl bilgi burada:** poz açısı
  arttıkça bu değer düşüyor — frontal↔frontal **0.867**, frontal↔profile
  **0.702** (~%19 göreli düşüş). AUC/EER "1.000/0.000" olarak sabit kalsa bile,
  modelin kimlik marjı profilde daralıyor. Yani poz etkisi kararı değil, marjı
  vuruyor.

### Neden her şey 1.000/0.000 çıktı?
FEI temiz, stüdyo koşullarında, sadece 20 test kimliği olan küçük bir veri
seti. "Farklı kişi" çiftleri çok kolay ayrıldığı için metrikler tavan yapıyor
(ceiling effect). Gerçek dünya (kötü ışık, çözünürlük, kalabalık galeri,
benzer yüzler) bu kadar kolay olmaz — bkz. README > Limitations.

## Bootstrap %95 güven aralığı (CI) — `[1.000, 1.000]`

profile/profile kutusunda sadece **21 pozitif çift** var. Bu az örnekte
belirsizliği ölçmek için 1000 kez "yerine koyarak örnekleme" (bootstrap) yapıp
her seferinde AUC hesapladık. 21 çiftin hepsi kolayca doğru olduğu için her
tekrarda AUC yine 1.000 çıktı → aralık `[1.000, 1.000]`. Bu **dar bir aralık iyi
tahmin** anlamına gelmez; sadece "örnek çok küçük ve hepsi kolay" demektir.

## Embedding görselleştirme — Silhouette skoru

- **Silhouette (siluet) skoru:** Bir noktanın kendi kümesine, en yakın komşu
  kümeye göre ne kadar iyi oturduğu. `+1`'e yakın = kümeler net ayrık,
  `0` civarı = ayrım yok, negatif = karışık.
- **by identity = 0.777:** Kimliğe göre kümeler çok net ayrışıyor.
- **by pose_bin = -0.006:** Poza göre neredeyse hiç kümelenme yok.
- **Yorum:** Embedding kimliği güçlü, pozu neredeyse hiç kodlamıyor. Bu, poz-
  bağımsız doğrulama/retrieval sonuçlarının nedenidir.

## Retrieval (getirme) metrikleri — `retrieval_metrics.csv`, `retrieval_crosspose_metrics.csv`

- **top-1 accuracy:** Sorgu yüzüne en yakın 1 galeri görüntüsü aynı kişiyse
  doğru. `1.000` = her sorguda en yakın komşu doğru kişi.
- **top-5 accuracy:** En yakın 5 komşudan en az biri aynı kişiyse doğru
  (top-1'den her zaman ≥). İkisi de 1.000 olduğu için üst üste bilgi vermez.
- **same-pose pool:** Galeri = diğer tüm test görüntüleri (her pozdan).
- **cross-pose (daha zor):** Galeri = **sadece frontal** görüntüler; sorgu =
  yan/profil görüntüler. Yani "profilden çekilmiş yüzü, kişinin cepheden
  fotoğrafıyla eşleştir". Bizde bu da 1.000 → gerçek poz-bağımsızlık kanıtı.

## Poz kutuları ve yaw eşikleri

- **yaw:** Başın sağa/sola dönme açısı (derece). 0° = tam cephe.
- **frontal:** `|yaw| < 20°` · **half-profile:** `20°–60°` · **profile:** `>60°`.
- Kestirilen yaw ~**±65–68°**'de doyuma ulaşıyor (±90°'a çıkamıyor); bu yüzden
  en uç profiller `profile` kutusunda az temsil ediliyor olabilir.

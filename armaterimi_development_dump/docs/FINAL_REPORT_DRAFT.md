# ArmaTerimi — Trendyol × TEKNOFEST 2026 Stage-2 Teknik Raporu

> Aksi belirtilmedikçe tüm macro-F1 sayıları public setin DEV yarısında, query-disjoint
> out-of-fold (OOF) protokolüyle ölçülmüştür.

---

## Summary of the pipeline flow

Görev: sorgu–ürün ikilisini `relevant / semi-relevant / irrelevant` olarak sınıflamak;
metrik üç sınıflı macro-F1, değerlendirme 90.000 satırlık private set üzerinde ve süre
puanı container başlangıcından çıktı dosyasına kadar sayılıyor. Çözümümüz dört bloktan
oluşuyor:

1. **Metin normalizasyonu (iki katman).** `train.csv` iki farklı katalog export'undan
   birleştirilmiş ve satırın hangi export'tan geldiği etiketi ele veriyor (bkz. EDA).
   Deterministik bir normalizasyon katmanı (NFC + Türkçe noktalı-i düzeltmesi, sızıntılı
   attribute key'lerinin düşürülmesi, `unisex → unknown`) bu sızıntıyı kapatıyor. Üstüne
   ikinci bir attribute temizliği (60 key + bilgi taşımayan kalıp değerler) girdiyi
   %15–17 kısaltıp DEV macro-F1'i +0,0155 artırıyor. Aynı transform'lar inference'ta
   byte-identical çalışıyor.
2. **Eksik `irrelevant` sınıfının üretilmesi.** Train'de hiç irrelevant örneği yok; test
   dengeli üçte-birlik. Sınıfı, organizatörün gerçek irrelevant profiline göre tasarlanmış
   10 reçeteli bir kural madencisi + tf-idf tabanlı bir "getirme yakın-ıskası" reçetesiyle
   ürettik; seçim, dağılım kapılarından (adversarial ayırt-edilebilirlik, zorluk profili)
   geçmeden hiçbir satır etiketlenmedi. Üretilen adayların **%100'ü** doğrulanmış bir
   LLM-judge tarafından etiketlendi (rubrik karşısında macro-F1 0,924); kör 3 okuyuculu el
   denetimi nihai kümede **0,987 precision** ölçtü. Sonuç: 145.848 satırlık donmuş
   irrelevant kümesi + yargıçtan dönen semi/relevant "ekstralar".
3. **Model.** `dbmdz/bert-base-turkish-128k-cased` (BERTurk-128k, 12 katman, 184.347.651
   parametre) cross-encoder olarak, 3 sınıflı head ile 6 epoch fine-tune edildi. Karar kuralı
   ordinal: olasılıklardan tek skaler (`x = p_semi + 2·p_rel`) türetilir, iki eşikle üç sınıfa
   bölünür. Eşikler **yalnızca DEV (3.750 satır) üzerinde** fit edilir; HOLDOUT hiç okunmadan
   kalır ve sadece son ölçümde bir kez test eder. DEV OOF iki-sınırlı macro-F1 **0,8546**;
   teslim edilen imajın organizatörün kendi `evaluate.py`'siyle public 7.500 satırda ölçülen
   skoru **0.857943456149** (DEV yarısı 0,8557 / HOLDOUT yarısı **0,8602**). HOLDOUT'un DEV'in
   *üzerinde* çıkması, eşiklerin DEV'e aşırı uydurulmadığının doğrudan kanıtıdır.

   *Model seçimi neden BERTurk?* Ölçülen cephe (hepsi aynı veriyle, kapsayıcı içi hız):
   TY-ecomm 12L 0,8643 @ 2.680 · TY-ecomm 8L 0,8581 @ 3.322 · **BERTurk 12L 0,8546 @ ~4.500** ·
   turkish-medium 8L 0,8337 @ 5.427. BERTurk, en iyi F1'in 0,0097 altında kalarak **%68 daha
   yüksek** verim veriyor ve rakip arketiplerine karşı **en iyi kötü-durum** puanını üretiyor
   (bkz. §Model seçimi). Kalite ucunu (TY-ecomm 12L) değil, Pareto dengesini seçtik.

4. **Serving.** Tek Docker imajı, iki mod (tahmin + explainability). Tahmin yolu için
   `transformers`'a başvurmayan/lazy-import bir servis hattı, süreç-paralel veri okuma ve
   uzunluk-kovali batching; ölçülen uçtan uca hız: gerçek 90.000 satırlık girdide konteyner içi **~20,0 s = ~4.500 satır/s** (jüri bayraklarıyla, tek koşu; ardışık ısınmış koşularda 17,8 s / ~5.030 satır/s'e iner). İmaj organizatör
   sözleşmesine birebir uyar (digest-pinli base, read-only dosya sistemi, ağ yok).

Strateji bilinçli: puan formülünde hız ekseni `(SPS/best_SPS) × 10` bir MaxNorm oranı ve
+500 SPS hızlı bir lidere karşı ~2 puan oynatırken, +0,01 macro-F1 ~0,35 puan oynatıyor.
Bu yüzden hedefi "≥0,85 macro-F1'i ≥3.000 satır/s hızında teslim etmek" olarak koyduk; teslim
edilen model bu hedefin iki eksende de üzerindedir ve bu Pareto noktasının altında kalan her
hız/kalite takasını ölçüp reddettik.

**Uçtan uca hat (8 adım).**

1. **Girdi.** `train.csv` 500.000 satır — 250k *relevant* + 250k *semi-relevant*; **`irrelevant`
   sınıfı organizatör tarafından hiç verilmemiştir**. `public_test.csv` (7.500 etiketli satır)
   yalnızca ÖLÇÜM içindir ve hiçbir aşamada eğitime girmez.
2. **Metin normalizasyonu.** İki-snapshot sızıntısının temizliği (T1 NFC + U+0307, T2/T3 sızdıran
   nitelik anahtarları, T4 `gender` alanındaki `unisex` **ve `çocuk`** değerlerinin
   nötrlenmesi) + v2 nitelik temizliği (60 anahtar + kalıp değer filtresi).
3. **Eksik sınıfın üretimi.** 10 kural reçetesi + tf-idf getirme yakın-ıskası → ~170k aday çift.
4. **Dağılım kapıları — harcamadan ÖNCE.** Adversarial AUC 0,7185, organizatörün kendi
   train-test kaymasının (0,748) altında; ek olarak Asure zorluk bandı.
5. **Yargıç etiketleme.** 182k çift; kanarya kontrolleri + kör 3-okuyucu denetim, kesinlik 0,987.
6. **Eğitim kümesi.** 813.269 satır: organizatör R/S + yargılanan I + judge-S/R ekstralar.
7. **Model.** BERTurk-128k tabanlı 12 katmanlı, 184.347.651 parametreli çapraz-kodlayıcı;
   6 epoch, sınıf ağırlıklı çapraz entropi, 3 sınıflı baş.
8. **Karar ve servis.** Ordinal skaler `x = p_kısmen + 2·p_ilgili`, iki global eşikle dilimlenir
   (eşikler yalnızca DEV üzerinde fit edilir; HOLDOUT sadece test eder). Servis
   `transformers`'sız yalın BERT portu, süreç-paralel veri yolu ve işçi süreçlerinde
   tokenizasyon ile 90k satırı kapsayıcı içinde **~20 saniyede** üretir.

Açıklanabilirlik arayüzü **aynı checkpoint** üzerinden çalışır: karar skalerine göre Integrated
Gradients, kelime düzeyinde dikkat hizalaması, karşı-olgusal yeniden puanlama, sadakat (silme
eğrisi) ve Türkçe sağlamlık kontrolleri; hiçbir aşamada LLM kullanılmaz.


## EDA and key findings regarding the data (if any)

**Bulgu 1 — iki-snapshot label leak (kritik).** `train.csv` (500k satır; 250k relevant +
250k semi-relevant, hiç irrelevant yok) iki farklı katalog export'undan birleştirilmiş ve
satırın hangi export'tan geldiği etiketin kendisi. Dört biçim markörü bunu kanıtlıyor
(500.000 satırın tamamında ölçüldü):

| Markör | Train n | P(relevant) | Public-test n |
|---|---:|---:|---:|
| `attributes` içinde `color detail:` | 196.763 | 1.000000 | **0** |
| `gender == unisex` | 22.518 | 0,999911 | **0** |
| `gender == çocuk` | 16.634 | 0,000000 (%100 semi) | 664 (testte %43 irrelevant / %30 relevant) |
| Title'da noktalı-i (U+0307) | 32.822 | 0,000000 (%100 semi) | 1.110 |
| Boş `attributes` | 2.247 | 0,000000 (%100 semi) | 85 |

Satırların **%46.9'u** en az bir markör taşıyor; iki kurallı bir oracle raw train'in
**%60.8'ini %100.000 doğrulukla** sınıflandırabiliyor. Kök neden: relevant satırlar eski
bir export'tan (snapshot A), semi satırlar ve **testin üç sınıfının tamamı** daha yeni bir
export'tan (snapshot B) geliyor; iki export `İ` harfini farklı küçültüyor ve şema revizyonu
geçirmiş (`renk` 347 değerden 31'e inmiş, bazı key'ler yeniden adlandırılmış — ör.
`color detail` → `web color`). Encoding farkı taksonomiyi bile bölmüş: U+0307 temizlenince
3.173 kategori string'inden 252'si birleşiyor. Test tarafında markörler ya hiç yok ya da
sınıftan bağımsız (noktalı-i DEV'de χ² p=0,19; boş-attributes anti-korele). Yani raw
train'e fit edilen her model, test'te ya hiç ateşlemeyen ya da gürültü olarak ateşleyen
mükemmel bir shortcut öğreniyor. Normalizasyon yapılmadan hesaplanan her train istatistiği
de (kategori bazlı oranlar dahil) bu sızıntıyla kirli. Çözüm: Data Cleaning bölümü.

**Bulgu 2 — Stage-2 `relevant` sınıfı Stage-1 pozitif seti ile aynı.** 250.000 relevant
satırın 249.984'ü Stage-1 eğitim çiftleriyle eşleşiyor. Bu, R sınırının görece kolay
olduğunun erken işaretiydi ve Stage-1 birikimini referans olarak kullanılabilir kıldı.

**Bulgu 3 — train→test dağılım kayması.** Sorgu token'larının ürün metnine düşme oranı
relevant sınıfta train'de 0,850, public test'te 0,688; `giyim` train sorgularının %12.5'i,
public'in %31.8'i. Kaymanın bir kısmı snapshot şema farkı, bir kısmı gerçek üretim farkı.

**Bulgu 4 — problem ordinal ve asıl fark semi/irrelevant sınırında.** 12B'lik bir LLM'in
zero-shot scalar'ı üzerinde ölçtük: tek scalar + iki eşik, modelin kendi üç sınıflı
generation'ından +0,018 iyi (0,7259 vs 0,7083). Oracle ayrıştırması önceliği kesinleştirdi:
mükemmel bir R-kapısı +0,050 getirirken **mükemmel bir S/I-kapısı +0,110** getiriyor — S/I
sınırı R sınırının 2.2 katı headroom taşıyor ve dönüşüm oranı ölçülü:
Δmacro-F1 ≈ 0,87 × ΔAUC(S|I). Tüm mimari ve veri kararları bu iki sayıya hizalandı.

**Bulgu 5 — organizatörün irrelevant'ları kolay negatif değil.** Gerçek irrelevant profili:
Asure-12B margin medyanı −1.0 (model onları "alakasız ama yakın" buluyor), %63'ünde sorgu
ile title arasında sıfır kelime kesişimi, %62'sinde gender alanı dolu. Tipik desen: sorgunun
head noun'u üründe yok, modifier'ları var. Sözcük örtüşmesine güvenen model tam burada
çöküyor: sorgu kapsama oranı %75–100 olan dilimde irrelevant F1'i yalnızca **0,406**
(satırların %21'i). Bu profil negative mining tasarımını doğrudan belirledi.

**Bulgu 6 — semi bandı dikeyler arasında farklı genişlikte.** Uyumluluk dikeylerinde
(oto, elektronik) S bandı giyime göre 3.1× geniş; erken probe'larda per-vertical eşik
ölçekleme +0,056 getiriyordu. Nihai modelde bu bilgiyi eşik ayarı yerine öğrenilmiş
olasılıklara bıraktık (bkz. What didn't work — batch/dikey bazlı post-processing'i bilinçli
kullanmadık).

**Bulgu 7 — "boş attributes" yazımı bile snapshot işaretiydi.** Train'de "attribute yok"
iki farklı yazımla geziyor: `""` (2.247 satır, %100 semi) ve `unknown` (4.097 satır, %100
relevant); public'te yalnızca `""` var. Attribute temizliği v2 ikisini tek sentinel'e
(`unknown`) katlıyor.

## Negative sampling process (if any)

Train'de `irrelevant` örneği yok; sınıfı organizatör profilinde üretmek zorundaydık. Nihai
üretim hattına üç başarısız denemeden sonra ulaştık — üçü de ölçümle elendi:

- **Cosine-band mining (başarısız).** Bi-encoder benzerlik komşuluğundan aday çekmek
  relevant/semi komşuluğuna düşüyor; yargıca göre irrelevant verimi yalnızca %6–21.
- **"Hard band" seçimi (başarısız).** Adayları 12B modelin en çok karıştırdığı banttan
  seçmek, organizatör profilinin tam tersini üretti (onların irrelevant'ı model için
  *kolay*). Ürettiğimiz kümeyi organizatör irrelevant'ından ayıran adversarial probe
  **AUC 0,9101**'e çıktı; öğrenci modelin gerçek-irrelevant isabeti 0,374'te kaldı.
  Dağılım eşleştirmesiyle (importance matching) onarım AUC'yi 0,6264'e, isabeti 0,605'e
  çekti — ama bu bir yama olduğu için nihai hatta taşınmadı; ders alındı: **seçim baştan
  doğru dağılımda yapılmalı.**
- **Yargıçsız kural etiketi (başarısız).** Bir ara sürümde 61.8k satır yalnız kural
  güveniyle `irrelevant` damgalanmıştı (~%11 beklenen kontaminasyon). Bu keşfedildiğinde
  küme reddedildi ve kural konuldu: **yargıçtan geçmeyen hiçbir satır eğitim etiketi
  alamaz.**

**Nihai üretim hattı (donmuş):**

1. **Üretim — 10 reçete.** Organizatör profilinden türetilen kural reçeteleri (marka
   değişimi; aynı bölümde tür değişimi; uzak renk; yanlış cihaz modeli; yanlış sayı/beden;
   yanlış aksesuar; yanlış varlık; tür kelimesi yalnız attribute'ta olan "lexical tuzak";
   uzak rastgele; niteleyici-soyma) + organizatörün kendi "getir-sonra-yargıla" mekanizmasını
   taklit eden **tf-idf getirme yakın-ıskası** (title+category+brand üzerinde top-80, sorgunun
   kendi R/S satırları ve yasak yaprak-kategorileri düşülür). Her reçete sorgu başına
   sıralı liste üretir; train-çifti/public-sorgu düşürme ve reçeteler arası dedup uygulanır.
2. **Seçim — profil ve kotalarla.** Sorgu başına kota, ürün başına küresel tekrar sınırı
   (3), bölüm profili DEV-irrelevant köklerinden, moda payı gerçek profile göre (nihai
   kümede %33). Bu sabitler DEV-irrelevant'tan **bir kez** okundu, hiç yeniden fit edilmedi;
   HOLDOUT hiçbir aşamada okunmadı.
3. **Kapılar — yargıçtan önce, zorunlu.** (a) *Adversarial kapı:* ürettiğimiz irrelevant'ı
   organizatör irrelevant'ından ayırmaya çalışan bir probe'un AUC'si, organizatörün **kendi**
   train→test kaymasının tabanını (train-R'yi DEV-R'den ayırma AUC'si) en fazla +0,02
   aşabilir — mutlak bir bar değil, çünkü train'den üretilen hiçbir şey domain-shift
   tabanının altına inemez (bunu da ölçtük: eski 0,65 barı fiziksel olarak ulaşılmazdı).
   Seçim tarihçesi: yalnız kural reçeteleri 0,798 → +getirme reçetesi 0,756 → +moda
   kotaları 0,728 (o günkü taban 0,738'in altı). (b) *Zorluk kapısı:* Asure-12B margin
   profili gerçek irrelevant bandında olmalı — frac(margin>0) ölçüldü 0,186 (bant
   0,18–0,28; DEV-I 0,234), medyan −1.5 (DEV-I −1.0). (c) Hijyen ve kapsam kontrolleri.
   **Kapılar geçilmeden tek bir yargıç çağrısı yapılmadı.**
4. **Etiketleme — %100 yargıç.** Önce 20k'lık tabakalı canary (kill kuralı: tutma ≥0,85),
   sonra kalan. Toplam **182.000 negatif aday yargıçlandı**: 152.425 irrelevant, 17.590
   semi, 12.778 relevant çıktı. Kural reçetelerinde tutma oranı 0,89–1.00; bilerek geniş ağ
   atan iki reçetede (niteleyici-soyma, getirme yakın-ıskası) 0,62–0,66 — onların işi zaten
   yargıca sınırda aday taşımaktı, saflık yargıcın işi.
5. **Denetim ve donduma.** Nihai küme üzerinde kör el denetimi: 300 satır, 3 bağımsız
   okuyucu, yalnız rubrikle — **precision 0,987**, relevant'a dönen 0 satır, %98.3 oybirliği,
   her reçete ≥0,955. Yargıç-sonrası kısıtlı yeniden-seçimle donmuş dosyalar:
   `irrelevant` **145.848** satır (+6.577 satır doz-ablasyonu rezervi), yargıcın semi/relevant
   dediği **29.551** "ekstra" (ayrı dosyada, önce karantina — sonra ablasyonla kanıtlanıp
   eğitime alındı), 902 cihaz-model kontrast satırı. Yargıç-sonrası adversarial kapı: **0,7185**
   (organizatör tabanı 0,748 + 0,02 barının altında). Tümü manifest + sha256 ile donduruldu.

Etiket öncesi kontrol deneyi: aynı seçim yalnız kural etiketleriyle eğitilirse DEV 0,752 ve
gerçek-irrelevant recall 0,72'de kalıyor — yargıç etiketlemesinin katkısı ölçülü.

## All processes that can be considered as data generation

Yarışma verisi dışında ürettiğimiz her veri ve üretim süreci:

| Üretilen veri | Süreç | Ölçek |
|---|---|---:|
| Normalize train / public test | deterministik snapshot normalizasyonu (etiket kullanmaz) | 500k + 7.5k satır |
| Temiz attribute metni (v2) | 60 key + kalıp-değer filtresi, sentinel birleştirme | tüm satırlar |
| Irrelevant aday havuzları | 10 reçeteli kural madenciliği + tf-idf getirme, seed'li | yüz binlerce aday |
| Aday zorluk skorları | Asure-12B zero-shot binary margin (yalnız kapı/profil için; asla etiket olarak değil) | havuzun tamamı |
| LLM-judge etiketleri (nihai üretim) | Gemini 3.7 Flash, rubrik system prompt, canary + kill kuralları | 182.000 satır |
| LLM-judge etiket deposu (önceki turlar dahil) | aynı yargıç, önbellekli; her satır tek kez etiketlenir | ~323.000 çift |
| Kontrast satırları | şablonlu sorgular için cihaz-model eşleşmeleri | 902 satır |
| Kör el denetimi kayıtları | 300 satır × 3 bağımsız okuyucu, yalnız rubrik | 900 karar |
| DEV/HOLDOUT bölme dosyası | query-disjoint, seed'li, sha256 ile sabit | 3.750 / 3.750 |

Yargıç asla submission tahmini üretmedi; yalnızca eğitim verisi etiketledi (build-time,
internetin serbest olduğu dönemde). Tüm üretimler seed'li ve manifest dosyalı; development
dump'ta veri + kod birlikte teslim ediliyor.

## Model Selection

Seçim ölçüme dayandı: aynı donmuş veri, aynı eğitim/değerlendirme düzeneği, ön-kayıtlı
">0,012 fark" kuralı (gürültü tabanının altındaki fark "kazanan" ilan edilmez).

**Tur 1 — Türkçe encoder bake-off (madencilik verisinin erken sürümüyle):**
BERTurk-128k 0,7358 > tybert 0,7265 ≈ BERTurk-cased 0,7249 (gürültü içinde) ≫ XLM-R-base
0,6885. İlk çalışma atı BERTurk-128k oldu.

**Tur 2 — donmuş veri + temiz metin üzerinde geniş karşılaştırma (3 epoch, cinsiyet
düzeltmesi öncesi; yalnızca backbone sıralaması için):**

| Backbone (cross-encoder, 3 sınıflı head) | DEV OOF macro-F1 | Not |
|---|---:|---|
| TY-ecomm-embed-multilingual-base (306M) | 0,8529 | en yüksek kalite |
| BERTurk-128k cased (184M) | 0,8205 | fark 0,032 > 0,012 tabanı |
| mdeberta-v3-base | ~0,812–0,819 | GPU çekişmeli ölçüm; yine de açıkça altta |
| multilingual-e5-large | ~0,806–0,810 | GPU çekişmeli ölçüm |
| Qwen3-0.6B metin kulesi | 0,7660 | 4× yavaş, en zayıf S/I sınırı |

**Tur 3 — nihai cephe (hepsi aynı sızıntısız veriyle; 6 epoch BERTurk ve turkish-medium, 3 epoch
TY-ecomm; kapsayıcı içi hız, jüri saati):**

| Aday | DEV OOF | public 7.500 | kapsayıcı içi SPS |
|---|---:|---:|---:|
| TY-ecomm 12L | 0,8643 | 0,8653 | 2.680 |
| TY-ecomm 8L | 0,8581 | 0,8626 | 3.322 |
| **BERTurk-128k cased 12L — SEÇİLEN** | **0,8546** | **0,8579** | **~4.500** |
| turkish-medium 8L/512 | 0,8337 | 0,8409 | 5.427 |

Aynı 6-epoch reçetede BERTurk-128k *uncased* da denendi (0,8511, −2,7 sd) — cased omurga
küçük harfli veride bile daha iyi tokenize ediyor (aynı 90k'da %2 daha az token). Teslim
reçetesinin tohum yayılımı n=3 için sd 0,0013'tür; tablodaki farklar bunun katbekat üstündedir.

Trendyol'un kendi e-ticaret embedding modeli (GTE mimarisi) kalitede öndedir: 306M
parametrenin ~%63'ü embedding tablosunda oturduğundan hesap katmanları göründüğünden küçüktür.
Veri katkıları da backbone'dan bağımsız çıktı (ekstraların +0,018'i BERTurk'te de görüldü),
yani model seçimi veri hattından bağımsız, temiz bir karşılaştırmadır.

**Buna rağmen teslim edilen model BERTurk-128k'dır.** Sebep kalite değil, puan formülüdür.
Kapsayıcı içi ölçümle TY-ecomm 12L 2.680 SPS, BERTurk 12L **~4.500 SPS** üretiyor; F1 farkı
ise 0,0074 (0,8653 vs 0,8579, public 7.500 satır). Hız ekseni `(SPS/best_SPS) × 10` bir
MaxNorm oranıdır ve **sıra çarpanı taşımaz**; dahası bizim hızımız *herkesin* paydasını
belirler — yavaş bir kalite liderinin hız puanını da biz bastırırız. Rakip arketipleri
üzerinde taranan senaryolarda BERTurk en iyi **kötü-durum** puanını veriyor. Yani kalite
ucunu değil, Pareto dengesini seçtik; bu bilinçli bir puan-optimizasyonu kararıdır.

**Ölçümle elenenler:** ModernBERT (İngilizce-merkezli BPE Türkçe'de 3.327 token/kelime vs
BERTurk 2.053; üstüne aynı uzunlukta daha yavaş forward ⇒ doğal Türkçe girdide ~3.6× yavaş);
12B LLM'i teslim etmek (iki eşikli kuralla DEV 0,8373'e ulaşıyordu ama ~60 satır/s hız,
puan matematiğinde hız MaxNorm'u yüzünden her senaryoda kaybettiriyor — 12B, üzerine
eğittiğimiz S/I fine-tune ile yalnızca **referans çıtası** olarak tutuldu: DEV scalar OOF
0,8724); CORAL/CORN gibi ordinal loss head'leri (mevcut kararımız zaten ordinal bir head;
tek fark eşiklerin loss yerine doğrudan hedef metriğe fit edilmesi — literatürde CORAL düz
cross-entropy'ye kaybediyor, beklenen kazanç ≈ 0); iki modeli kademelendirme ve derinlik
budama (bkz. What didn't work — kademelendirme reddedildi; derinlik budama ilk turda reddedilmiş, düzeltilmiş reçeteyle yeniden ölçülünce TY-ecomm 8L olarak canlı aday hâline gelmiştir).

## Model Architecture

**Backbone:** `dbmdz/bert-base-turkish-128k-cased` (BERTurk-128k; HF revizyonu `fea3225…`,
MIT lisansı) — 12 katman, 768 gizli boyut, 12 baş, **128.000 kelime-parçalık** WordPiece
sözlük; toplam 184.347.651 parametre (bunun ~98M'si sıfır FLOP'luk gömme tablosudur).
Genel Türkçe derlemler üzerinde ön-eğitilmiş, yalnızca Türkçe bir BERT. Büyük sözlük burada
işe yarar: aynı 90k satırı 32k sözlüklü Türkçe BERT'lerden %9–18 daha az token'a böler, yani
daha az hesap. Cross-encoder olarak
kullanılır: girdi `(sorgu, ürün metni)` çifti; ürün metni alan-etiketli kompakt şablonla
kurulur (`title | brand | category | gender age_group | attributes`; bilinmeyen alan
atlanır), 256 token'da truncate edilir (temiz metinle satırların yalnızca ~%0.1–0.2'si
kesiliyor).

**Head:** sıfırdan eğitilen 3 sınıflı softmax head → satır başına `(p_irr, p_semi, p_rel)`.

**Karar kuralı (ordinal, satır-bazlı):** olasılıklardan tek scalar türetilir —
`x = p_semi + 2·p_rel` — ve iki eşikle üç sınıfa bölünür. Eşikler out-of-fold fit edilir;
taşınım maliyeti ölçüldü: eşikleri taze bir query-disjoint yarıda yeniden fit etmek yalnızca
0,0031 ± 0,0024 kaybettiriyor, yani eşik fitting overfit kaynağı değil. Aynı olasılıklar
üzerinde iki-sınırlı varyant da ölçülür (`relevant if p_rel > eşik_R; değilse irrelevant if
(1−p_irr) ≤ eşik_I; değilse semi`).
PROTOKOL: eşikler **yalnızca DEV (3.750 satır)** üzerinde fit edilir ve HOLDOUT yalnızca test
eder — 7.500 satırın tamamında yeniden fit ETMİYORUZ; bu, tek temiz test kümemizi tüketirdi.
Nihai teslim kuralı: **ordinal** — skaler x = p_semi + 2·p_rel, iki global eşik
**[0,9518, 1.0619]**, DEV'de fit edilmiş, `decision.json` içinde donmuş. Kategori-rejimli
varyant da ölçüldü; bu omurgada out-of-fold kestirim global eşikleri tercih etti
(0,8533 vs 0,8506), rejim tablosu checkpoint'te pasif dokümantasyon olarak durur. Önemli bir tasarım kısıtı: **her etiket yalnız kendi
satırından üretilir** — private 90k tek batch olarak işlenmeyeceği için batch-istatistiği
kullanan hiçbir post-processing (ör. kantil/rank tabanlı etiketleme) teslimde yok.

## İki-snapshot sızıntısı: `gender = çocuk`

Bu, çalışmanın tek en yüksek getirili teknik bulgusudur ve organizatörün verisindeki bir
*veri* kusurudur, bir hiperparametre değil.

`train.csv` iki ayrı katalog dışa-aktarımından birleştirilmiştir: `relevant` satırlar bir
aktarımdan, `semi-relevant` satırlar diğerinden gelir. Bu yüzden iki aktarım arasında farklı
olan her şey, eğitimde **kusursuz ama anlamsız** bir yordayıcıya dönüşür. `gender` alanında iki
böyle değer bulduk:

| değer | eğitimde relevant | eğitimde semi | P(semi) | test'te görülme |
|---|---:|---:|---:|---:|
| `unisex` | 22.516 | 2 | 0,0001 | **%0** — testte hiç yok |
| **`çocuk`** | **0** | **16.634** | **1,0000** | **%8,9** (664/7.500) |

16.634 satırın istisnasız hepsinin `semi-relevant` olması rastlantı olamaz (250k/250k dengeli
bir kümede olasılığı 2⁻¹⁶⁶³⁴'tür). Model bunu bir kesinlik olarak öğrenir. Oysa **testte** aynı
satırlar %43 `irrelevant` / %30 `relevant` / %27 `semi`'dir (public'in tamamı, 664 satır). Ölçtük: model bu satırlarda
**%69,0**, diğer her yerde **%88,1** doğruluk veriyordu — test kümesinin onda birinde 19 puanlık
bir delik.

`unisex` zaten T4 adımında nötrleniyordu; **alanın yalnızca bir değerini temizlemiş, geri
kalanını hiç denetlememiştik.** `çocuk` da `unknown`'a eşlenince:

| | yama yok | çıkarımda yama | **sızıntısız yeniden eğitim** |
|---|---:|---:|---:|
| TY-ecomm 12L | 0,8587 | +0,0069 | **0,8643** (+0,0056) |
| TY-ecomm 8L | 0,8474 | — | **0,8581** (+0,0107) |
| BERTurk 12L 6ep | 0,8428 | +0,0120 | **0,8546** (+0,0118) |
| turkish-medium 8L | 0,8221 | — | **0,8337** (+0,0116) |

Kazanç **her mimaride** ortaya çıkıyor ve tüm servis optimizasyonlarımızın toplamından
büyüktür. Ardından `gender` ve `age_group`'un **her** değerini taradık: başka bozuk değer yok
(`age_group` en kötü kaymada 0,10; `gender=çocuk` 0,51). `brand = 'genel markalar'` de aynı
imzayı taşıyor (11.243 satır, 0 relevant) ancak marka alanında nötr bir "sink" değer
bulunmadığından çıkarımda yeniden yazmak yapay bir dağıtım-dışı token üretiyordu. Doğru yol,
eğitimde rastgele marka-dropout ile bir "sink" yaratıp GM'yi oraya katlamaktı; bunu %5 ve %10
dropout ile iki kez eğittik — sonuçlar (0,8505 ve 0,8554) teslim reçetesinin tohum yayılımı
içinde kaldı, dropout'un gerçek marka sinyalini vergilendirmesi kazancı yedi. Artefakt
gerçektir, düzeltmesi ödemiyor; **bilinçli olarak düzeltilmedi** ve README'de böyle
belgelenmiştir.

## Data Cleaning & Input Preprocessing

İki deterministik, etiket gerektirmeyen katman; ikisi de train'e ve **inference'ta private
CSV'ye birebir aynı** uygulanıyor (serving koduna gömülü kopya, 7.500 public satırın
tamamında repo transformuyla byte-identical doğrulandı):

**Katman 1 — snapshot normalizasyonu:**
- **T1:** tüm metin alanlarında NFC + U+0307 (noktalı-i) temizliği; tek başına 252 mükerrer
  kategori string'ini birleştiriyor.
- **T2:** varlığı train etiketini ≥0,93 saflıkla tahmin eden 18 attribute key'inin
  düşürülmesi (donmuş liste). Sızıntının gerçek imzası key *varlığı* olduğu için filtre
  saflık üzerinden.
- **T4:** `gender: unisex → unknown` (train'in %4.5'i, test'te %0).

Ölçülen etki: iki kurallı leak-oracle kapsamı train'in %46.89'undan (0,999991 doğruluk)
**%0,45'ine** düştü; metadata-only adversarial probe'un train→DEV genelleme farkı 0,444'ten
0,176'ya indi. Kalan farkın ayrıştırması yapıldı: attribute zenginliği ve satır şekli her
iki sette de aynı yönde gerçek sinyal; key-varlığı kaynaklı ~0,13'lük rezidü hiçbir attribute
cerrahisiyle sıfırlanamıyor (iki export tüm attribute sözlüğünde farklı) — kabul edildi ve
probe her veri değişikliğinde standart kapı olarak yeniden koşuldu.

Bu filtrenin **üç yanlış denemesi** kayıt altında (tekrarlanmasın diye): key'i düşürmek
yerine yeniden adlandırmak (196.763 satırlık markör meşru bir ismin altında hayatta kaldı);
tek yönlü filtre (testte train'den *daha sık* olan snapshot-B key'lerini kaçırdı);
prevalence-oranı proxy'si (kategori kaymasıyla karıştı; aksesuar akıl yürütmesinin yükünü
taşıyan `uyumlu marka/cihaz` key'lerini yanlışlıkla düşürdü). Doğru sinyal: etiket saflığı.

**Katman 2 — attribute temizliği (v2):**
- Toplam **60 attribute key** düşülür: 18 sızıntı key'i + sorgu kelimelerinin hiç düşmediği
  42 kalıp key (bakım/yıkama talimatları, menşei, garanti, sürdürülebilirlik, paketleme,
  yasal metinler, üretici kimliği...).
- Bilgi taşımayan **değerler** düşülür (`belirtilmemiş`, `mevcut değil`, bakım cümleleri;
  uyumluluk key'lerinde `belirtilmemiş` bilgi olduğu için tutulur; `yok`/`var` asla düşülmez
  — `kapüşon: yok` bilgidir).
- "Attribute kalmadı" durumu tek sentinel'e (`unknown`) katlanır — EDA Bulgu 7'deki
  `""`/`unknown` snapshot işareti böylece kapanır.

Ölçülen etki: attribute karakterlerinin %34'ü, model token'larının %15–17'si gidiyor (train
ve public'te aynı oranda — asimetri yaratmıyor; en büyük tek kalem her iki tarafta da
satırların %81–88'inde bulunan `menşei`); 256-token truncation oranı ihmal edilebilire
düşüyor. Eşleştirilmiş eğitim: **0,7994 → 0,8149 (+0,0155)**, R-vs-S AUC 0,872 → 0,909.
Kural: v2 metniyle eğitilen model yalnız v2 metniyle servis edilir — train/serve kayması
otomatik parity testiyle kilitli.

## Training Strategy

**Yaklaşım: düz (hard) etiketlerle tek aşamalı fine-tuning — distillation yok.** Erken
planımız 12B'den distillation'dı; iki bulgu vazgeçirdi: (1) kalite sorununun asıl kaynağı
kapasite değil **etiket kalitesiydi** — yargıçlanmamış kural etiketleri atılıp her satır
yargıçlanınca öğrenci sıçradı; (2) yargıçlı veriyle düz etiketli öğrenci, 12B referans
çıtasının (DEV scalar OOF 0,8724) yakınına tek başına geldi ve hız kazancı (~30×) aradaki
farkı puan matematiğinde ezici biçimde geri ödüyor.

**Nihai eğitim tablosu (dondurulan koşu):** 813.269 satır — organizatör 498.374 (250k semi +
248.374 relevant; public sorgularla çakışan 1.626 satır düşülmüş), yargıç tarafından
`irrelevant` denen 263.932 satır (v5 madenciliğinden 152k + önceki etiket deposundan 112k)
ve 50.963 `contrast` satırı (yargıç-semi/relevant ekstralar + cihaz-modeli karşıtlık satırları).
Sınıf dağılımı 284.136 semi / 265.119 relevant / 264.014 irrelevant — private kümenin
30k/30k/30k dengesine yakın. `gender` alanındaki `çocuk`/`unisex` değerleri bu tabloda
`unknown`'a katlanmıştır (T4; bkz. iki-snapshot sızıntısı bölümü).
**Public test satırları hiçbir eğitim koşusuna girmemiştir**; builder her kaynakta public
sorguları düşürür (`drop_public`) ve D10 kontrolü sıfır sorgu çakışması doğrular.

**Hiperparametreler (teslim edilen BERTurk koşusu):** 6 epoch, batch 64, LR 3e-5, AdamW
(wd 0,01), %5 linear warmup + cosine, grad clip 1.0, bf16, maxlen 256, seed 42;
**class-weighted cross-entropy** (sınıf ağırlıkları frekansın tersinden) + label smoothing
0,05; uzunluk-gruplu batching. Epoch sayısı omurgaya özgüdür: TY-ecomm 306M'de 3 epoch
optimum (4. epoch overfit ediyor), BERTurk 184M'de 3→6 epoch +0,0056 getiriyor, 9 epoch ise
−0,0057 ile geri düşüyor; LR 2e-5 de 3e-5'in altında kaldı (−0,0031).

Tohum duyarlılığı (teslim reçetesi, n=3): 0,8546 / 0,8533 / 0,8559 → ortalama 0,8546,
sd 0,0013. Teslim edilen checkpoint (seed 42) kendi reçete ortalamasının tam üzerindedir —
şanslı bir çekiliş değil, reçetenin beklenen değeridir. Tek-tohumluk her karşılaştırma bu
±0,0013 yayılıma karşı okunmalıdır.

**Her veri kalemi ablasyonla kanıtlandı** (aynı backbone, aynı metin, tek değişken):

| Adım | DEV OOF macro-F1 |
|---|---:|
| Organizatör + yargıçlı irrelevant (taban) | 0,8217 |
| + yargıç ekstraları (29.5k semi/relevant) | **0,8441** (+0,022) |
| + 3. epoch + backbone kararı sonrası (şampiyon veri) | 0,8529 |
| + önceki turların yargıçlı deposu (136.489 çift) | **0,8553** (scalar 0,8426 → 0,8543, AUC(S∣I) 0,8964 → 0,9037) |

Son satırın iki-sınırlı farkı tek başına gürültü içinde; ama scalar +0,012 ve AUC +0,007
aynı yönde — depo zarar vermiyor, üç okumadan ikisi anlamlı iyileşiyor, eklendi. (Bu veri
ablasyonu TY-ecomm omurgasında yürütüldü; veri katkılarının omurgadan bağımsız olduğu ayrıca
gösterildi.)

Public-test etiketlerinin eğitimde kullanımı organizatörce serbest bırakıldı; biz model
ağırlıklarını public'e hiç göstermedik (250k+250k organizatör verisi + kendi ürettiğimiz
yargıçlı veri yetti); public yalnızca karar eşiklerinin fit'inde ve doğrulamada kullanıldı.

## Validation Strategy

Bu bölüm bilinçli olarak uzun. Elimizdeki tek gerçek-dağılım etiketli veri 7.500 satırlık
public set; private 90.000 satır. Public'e bakılarak yapılan her seçim 7.500 satıra overfit
etme şansıdır ve proje baştan sona şu disiplinle yürüdü:

1. **DEV / HOLDOUT ayrımı — query-disjoint, bir kez, geri dönüşsüz.** Public'in 7.500
   satırı sorgu bazında ikiye bölündü (seed'li; sorgu asla iki yarıya bölünmez; satır ve
   sınıf sayıları dengeli: 3.750 / 3.750). **DEV**: tüm fitting, tüm varyant karşılaştırması,
   tüm eşik seçimi. **HOLDOUT**: karantinada. Bölme dosyası sha256 ile sabit; karantina
   kontrolü otomatik.
2. **HOLDOUT protokolü: milestone başına tek okuma, kayıtlı.** Kural: günde en fazla bir
   okuma, yalnız DEV'in seçtiği şampiyonda, HOLDOUT asla model seçmez. Okumalar:
   - Okuma 1: DEV 0,8217 → HOLDOUT 0,8235 (DEV'den yüksek — overfit yok).
   - Okuma 2: DEV 0,8441 → HOLDOUT 0,8385 (fark −0,006, 0,012 tabanının içinde — DEV dürüst).
   - Okuma 3 (2026-08-25, o günkü şampiyon TY-ecomm konteyneri üzerinden): HOLDOUT **0,8646**, DEV 0,8633, tamamı 0,8639.
   - Okuma 4 (2026-08-27/28, **teslim edilen BERTurk imajı üzerinden**): HOLDOUT **0,8602**, DEV 0,8557, tamamı **0.857943456149** — HOLDOUT yine DEV'in üzerinde.
3. **OOF fitting her yerde.** Eşikler DEV içinde query-disjoint fold'larla out-of-fold fit
   edilir; rapor edilen sayı her zaman OOF sayısıdır. Eşik yeniden-fit taşınımı ölçüldü:
   0,0031 ± 0,0024 — eşik fitting stabil; asıl risk **çok adaydan seçim yapmak**.
4. **Gürültü tabanı ve ön-kayıtlı kapılar.** 3.750 satırda macro-F1 bootstrap sd'si 0,0062
   (%95 CI ±0,012) ⇒ ön-kayıtlı kural: iki varyant arasında **>0,012** fark yoksa kazanan
   ilan edilmez ve basit olan kalır. N adayın en iyisini seçmek ~0,006·√(2·ln N) saf gürültü
   şişmesi alır — bu yüzden karşılaştırma sayısı bilinçli az tutuldu ve her merdiven
   (veri katkıları, backbone, karar kuralı) önceden yazılmış tek karar kuralıyla koşuldu.
   Kill kuralları deney koşulmadan önce yazıldı (ör. yargıç canary tutma ≥0,85, el denetimi
   precision ≥0,85, prompt-probe kapısı +0,010 AUC).
5. **Metrik doğrulaması.** Kendi macro-F1 fonksiyonumuz organizatörün `evaluate.py`'ı ile
   rastgele denemeler üzerinde bit-identical doğrulandı. Manşet sayıların yanında sorgu
   bazlı (satırlar bağımsız değil) bootstrap CI raporlanır.
6. **Yargıç da aynı disiplinle seçildi ve denetlendi.** 8 aday LLM-judge, rubrik system
   prompt ile, aynı tabakalı 1.200 DEV satırında **paired** karşılaştırıldı: kazanan
   Gemini 3.7 Flash macro-F1 **0,9238** (CI 0,9073–0,9377); ikinci 0,8795. Yargıç çıktısı
   ayrıca kör el denetiminden geçti (nihai kümede 300 satır × 3 okuyucu, precision 0,987).
7. **Adversarial probe'lar standart kapı.** Snapshot-leak rezidüsü her eğitim verisi
   değişiminde, üretilmiş-irrelevant dağılım uyumu her madencilik değişiminde yeniden
   ölçüldü (Negative sampling'deki eşikli kapılar).
8. **Karar kuralı ayrıca strese sokuldu** ("eşikler 7.5k'ya overfit olmasın" endişesi):
   parametresiz alternatifler ölçüldü — argmax 0,8322, train-prior düzeltmesi 0,8354,
   prior-düzeltmesi ve sıcaklık birlikte 0,8359; fit'li eşikler 0,8529 (+0,021 katkı, taşınım iyimserliği yalnız
   0,003). Batch-kantil tabanlı parametresiz bir kural HOLDOUT'ta fit'li eşikleri geçiyordu
   (0,8461 vs 0,8385) ama satır-bağımsızlık kısıtına aykırı olduğu için **bilinçli olarak
   teslim edilmedi**; offline teşhis aracı olarak tutuldu. Donmuş eşikler **yalnızca DEV'de**
   fit edilir; 7.500'ün tamamına yeniden fit etmek tek temiz ölçüm kümemizi tüketirdi.
   Teslim modelinde aynı stres testi: argmax 0,8495 → DEV'de fit edilmiş eşikler 0,8579
   (+0,0084). Public'e hiç erişilmeseydi ve eşikler yalnızca kendi eğitim verimizin bir
   diliminde fit edilseydi 0,8501 alınırdı — yani public etiketlerinin toplam katkısı
   kalibrasyondaki bu +0,008'dir, modelde değil. Eşiklerin sağlamlığı ayrıca 12'şer kez
   yeniden bölünerek ölçüldü: rastgele / sorgu-ayrık / ürün-ayrık / yaprak-ayrık bölmelerde
   fit yarısı diğer yarıyı yalnız ~0,003 abartıyor; en sert kategori-kökü-ayrık bölmede
   aktarım 0,850 ± 0,014. Private için dürüst beklentimiz bu aralıktır.

PUBLIC-F1 (teslim edilen BERTurk imajından, organizatörün `evaluate.py`'si ile):
**0.857943456149** — DEV yarısı 0,8557 / HOLDOUT yarısı 0,8602; çıktı sha256
`a6890ceed4a557bc05a376e295e2005d4240abb1c656aa1b62f77ce65f5a160c`. Komut: README §10.

## Inference Strategy for Solution Routine

Hız puanı `(SPS/best_SPS) × 10` ve runtime container başlangıcından çıktı dosyasının
tamamlanmasına kadar — cold start dahil. Servis hattı buna göre tasarlandı ve her optimizasyon
**parity kapısıyla** (çıktı byte-identical değilse ret) alındı:

- **Lazy import / `transformers`'sız yol.** Servis betiği işe koyulma süresi 5.9 s → 0,16 s;
  model kurulumu arka plan thread'inde, sınıf doğrudan yüklenir (−1.2 s). BERTurk yolunda
  el yazımı forward'ın HF'e karşı parity'si max|Δlogit| 1.05e-5, argmax uyumu 1.0000; GTE
  yolunda graph parity 4.8e-06. LEAN-YOL: **transformers'sız port imaja girdi** — teslim edilen model BERTurk olduğu için
  imajda çalışan yol `lean_infer.py`'dir (düz BERT); `lean_gte.py` GTE ("New") mimarisinin torch+safetensors ile yeniden yazımıdır (RoPE yarı-döndürme + sabit-NTK taban düzeltmesi, füzyonlu qkv, kapılı MLP, tanh-pooler baş); transformers'a karşı parite **max|Δlogit| = 0.0** ve tam public kümede bayt-bayt aynı CSV. Kazanım: model hazır olma 6,37 s → 3,44 s. Servis betiği seçimi checkpoint'in kendi
  `config.json`'ından yapılır (remote-code `auto_map` varsa GTE portu, yoksa düz BERT yolu) —
  teslim imajında fiilen `lean_infer.py` koşar; `serve_hf.py` geri düşüş olarak durur.
- **Süreç-paralel veri yolu.** 90k ölçeğinde çalışmanın ~%46'sı modelde değil tek iş
  parçacıklı Python veri yolundaydı (okuma + normalizasyon + tokenizasyon). Fork worker'lı
  paralel yol bunu 90k projeksiyonunda ~17.9 s → ~11.9 s'ye indirdi; veri yolu çıktısı
  (id'ler + token'lar) tek-süreçli yola karşı byte-identical.
- **Uzunluk-kovali batching.** Token uzunluğuna göre stable sort; satır + token çifte
  kapaklı batch'ler; batch başına pad; logit'ler orijinal sıraya geri dağıtılır.
- **Standardizasyon.** Kolonlar isimle seçilir, donmuş normalizasyon + v2 attribute
  temizliği serving koduna gömülü (7.500 satırda byte-identical doğrulandı), çıktı
  `row_id,real_label` atomik yazılır (temp + rename), şema çıkışta doğrulanır.

**Ölçülen uçtan uca hız.** SPS **duvar saatidir**: `run_solution.sh` başlangıcından teslim
dosyası yazılana kadar geçen sürenin tamamı sayılır. Aşağıdaki tablo bu nedenle *kapsayıcı içi*
(jüri bayraklarıyla, gerçek 90.000 satır, aday başına 2 tekrar, en iyisi) ölçülmüştür; çıplak
(kapsayıcısız) sayılar kıyas için ayrı sütunda ve **karşılaştırma amacıyla kullanılamaz**.

Tüm adaylar **aynı, sızıntısı giderilmiş veriyle** eğitildi; tek değişken modeldir.
DEV OOF eşik uydurmayı kendi katlarının içinde yapar (karamsar, aday sıralaması için);
DEV/HOLDOUT sütunları teslim edilen imajdan, jürinin kendi `run_solution.sh`'i ile ölçüldü.

| Aday | DEV OOF | kapsayıcı içi 90k | DEV | HOLDOUT | public 7.500 |
|---|---:|---:|---:|---:|---:|
| TY-ecomm 12L | 0,8643 | 2.680 | 0,8677 | 0,8629 | 0,8653 |
| TY-ecomm 8L | 0,8581 | 3.322 | 0,8608 | 0,8643 | 0,8626 |
| **BERTurk-128k 12L — TESLİM EDİLEN** | **0,8546** | **~4.500** | **0,8557** | **0,8602** | **0,8579** |
| turkish-medium 8L/512 | 0,8337 | 5.427 | 0,8434 | 0,8383 | 0,8409 |

Not: teslim edilen modelde **HOLDOUT (0,8602), DEV'in (0,8557) üzerindedir.** Eşikler yalnızca
DEV üzerinde fit edildiğinden bu, eşiklerin fit yarısına aşırı uydurulmadığının doğrudan
kanıtıdır — TY-ecomm 12L ve turkish-medium'da fark ters yöndedir.

**Duvar saatinin dağılımı** (teslim edilen BERTurk imajı, 20,0 s): `3,9 s docker --gpus all`
(bunun ~3,6 s'si NVIDIA hook'unun sürücüyü enjekte etmesi — `--entrypoint /bin/true` ile izole
edildi: GPU'suz 0,69 s, GPU'lu 4,33 s; bizim kodumuz değil) + `0,1 s önyükleme` + `14,8 s sunucu`
(veri yolu 4,8 s ile model hazırlığı 5,2 s eşzamanlı; 9,5 s çıkarım; 0,1 s yazma) + `~1,2 s kuyruk`.
Modelden bağımsız kısım ~10,5 s'dir (%53) ve bunun 3,9 s'si yapısaldır. Bundan çıkan **fiziksel
tavan: her GPU kapsayıcısı için ~90000/6,3 s ≈ 14.000 SPS** — hız paydası kaçamaz, bu da hızlı
adayın kayıp riskini sınırlar.

**Hız kazanımları** (hepsi çıktı-korumalı; her adayda 90.000 satırda **0 etiket değişimi**
doğrulandı): `_is_boiler` sabitlerinin döngü dışına alınması (90k'da 8,6M `_fold()` çağrısı →
~1,4M; seri normalizasyon 11,15 s → 3,22 s), havuz IPC'sinin int32/int8'e daraltılması,
**rayon aşırı-aboneliğinin giderilmesi** (her işçi kendi rayon havuzunu açtığı için 12 çekirdekte
~120 iş parçacığı oluşuyor ve model yükleme iş parçacığı açlığa giriyordu: `model_ready` 5,93 s,
tek başına maliyeti 2,0 s — `OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1` ile modele göre +%5…+%17;
işçi sayısı kapsayıcı içinde ayrıca taranıp 10'a sabitlendi — jüri bayrakları altında `nproc`
1 döndüğü için otomatik tespit tek işçiye çökerdi), cuInit yoklamasının kritik yoldan
çıkarılması (1,394 s → 0,019 s), `predict.py` içindeki gereksiz 90k CSV yeniden-okumasının
kaldırılması ve çıktı fsync'lendikten sonra yorumlayıcı kapanışının atlanması (`os._exit`).

**Docker teslimi.** Tek imaj (`armaterimi_solution`), organizatörün donmuş
`run_solution.sh`/`run_explainability.sh` betikleri byte-identical (sha256 her build'de
doğrulanır). Base imaj digest-pinli (`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`);
tüm pip'ler build sırasında sürüm-assert'li; ağırlıklar ve tüm env imajın içinde (runtime'da
internet yok, hiçbir env enjeksiyonu gerekmez). Jüri bayraklarının tamamı altında test
edildi: `--network none --read-only --cap-drop ALL --user <uid>` + tmpfs — cache dizinleri
boş tmpfs'te kurulur, CUDA yoksa fail-fast. Gizli bilgi taraması imaj katmanlarında yapıldı
(temiz). CPU smoke testi geçti; GPU jüri provası ve nihai zamanlama GPU-PROVA: jüri bayraklarıyla (`--read-only`, `--cap-drop ALL`, `--network none`, tmpfs `/tmp`, `--user 1000:1001`, `--gpus all`) tam koşu **başarılı**; duvar saati 13 s / 7.500 satır, çıktı şeması ve izinler doğrulandı, public macro-F1 0,863942 (25 Ağustos, o günkü TY-ecomm imajı). CUDA 12.8 kullanıcı alanı sürücü altında sorunsuz çalıştı. Konteyner ek yükü ayrıştırıldı: nvidia hook ~3,9 s + create + sarmalayıcı/mount (yapısal, tüm takımlar için aynı). NİHAİ PROVA (28 Ağustos, teslim edilen BERTurk imajı, zip'ten kurulan bağlamla): 90.000 satır 20,7 s, çıktı bayt-bayt önceki koşularla aynı; public macro-F1 **0.857943456149**; imaj 8,69 GB.

**Ölçülüp reddedilen hız fikirleri:** torch.compile (+%6 inference, ama shape başına
16–69 s derleme *skorlanan sürenin içinde* ⇒ net negatif); iki-model kademesi ve derinlik
budama (bkz. What didn't work).

## Tekrarlanabilirlik (aynı girdi → aynı çıktı)

Organizatörün şartı: 90.000 satırlık test kümesi üzerinde **iki koşunun tahmin ettiği
etiketlerin %99'dan fazlası** uyuşmalıdır. Ölçtük — teslim edilen imajla, jürinin kendi
`run_solution.sh` betiğiyle, üç bağımsız koşu:

| karşılaştırma | etiket uyuşması | farklı satır |
|---|---:|---:|
| koşu 1 ↔ 2 | **%100,0000** | 0 |
| koşu 1 ↔ 3 | **%100,0000** | 0 |
| koşu 2 ↔ 3 | **%100,0000** | 0 |

Üç çıktı dosyasının sha256'sı da aynıdır: `cbacea4b68a5b99b…`. Yani hat "tolerans içinde"
değil, **bit düzeyinde belirlenimcidir**. Bunu mümkün kılan tasarım kararları: örnekleme yok
(argmax/eşik), uzunluğa göre kararlı sıralama (`np.argsort(..., kind="stable")`), sabit
gruplama, tek atomik yazım (temp + `os.replace`), ve akış (STREAM) modunun kapalı olması —
akış, grup kompozisyonunu değiştirdiği için dolgu kaynaklı bf16 gürültüsüyle etiketlerin
%0,1'ini oynatıyordu; ölçüp reddettik.

## Inference Strategy for Explainability Routine

**Kullanılan model için explainability nedir?** Ekrandaki her şey, **tahminleri üreten
checkpoint'in kendi ileri geçişinin** fonksiyonudur — aynı ağırlıklar, aynı donmuş
preprocessing, aynı karar kuralı. İkinci bir model ya da anlatıcı LLM yoktur; organizatörün
2026-08-25 açıklamasındaki "girdi+çıktıyı yorumlayan ayrı açıklayıcı" sınıfına giren hiçbir
bileşen yoktur. Açıklamanın hedefi, sınıf logit'i değil, **kararı fiilen veren büyüklüktür**:
ordinal skaler `x = p_semi + 2·p_rel`. Bu ayrım önemlidir — eşikli kuralda argmax ile servis
edilen etiket satırların ~%3'ünde ayrışır ve tam o satırlarda bir sınıf-logit atfı yanlış
büyüklüğü açıklardı. Arayüzün verdiği etiketin konteynerin verdiği etiketle birebir aynı olduğu
otomatik bir parite testiyle korunur (`verify_ui_parity.py`: aynı logit'ler iki karar
vericiden geçirilir; 300/300 aynı).

**Bilgi nasıl üretiliyor / çıkarılıyor?** Her panel modelin kendi sinyalidir:

1. **Karar sayı doğrusu.** Satırın skaleri, iki eşik (`[0.9518, 1.0619]`, yalnızca DEV'de fit)
   ve üç sınıf bölgesi; eşiğe uzaklık kararın ne kadar kıl payı olduğunu gösterir. Eşiğe
   yakın satırlar "sınırda" olarak işaretlenir.
2. **Integrated Gradients (birincil açıklama).** Kelime-gömme katmanında, hedef = karar
   skaleri (softmax·rank ile türevlenebilir), taban = tokenizer'ın kendi `[UNK]` gömmesi.
   Bu checkpoint'te integrand plato→uçurum→plato şeklindedir; eş aralıklı düşük adımlı IG
   uçurumu ıskalayıp tamlığı bozuyordu, bu yüzden **uyarlanabilir kuadratür** kullanılır
   (17 noktalı ucuz ön-tarama, gradyan bütçesi |Δg|'ye göre dağıtılır, tamlık toleransı
   aşılırsa bütçe en fazla iki kez ikiye katlanır ve en iyi geçiş tutulur). **Tamlık
   kontrolü** (Σ atıf = f(girdi) − f(taban)) her istekte hesaplanıp ekranda basılır, asla
   yeniden normalize edilmez. Token katkıları karakter ofsetleriyle **kelimelere** toplanır
   (IG toplamsal olduğundan kesin), sonra alan bazında (başlık / marka / kategori / cinsiyet-yaş
   / nitelikler) toplanır.
3. **Karşı-olgusal yeniden puanlama (tıkama).** Her ürün alanı, her `anahtar: değer` niteliği
   ve her sorgu kelimesi tek tek çıkarılıp **model yeniden koşturulur** (tek gruplanmış ileri
   geçiş); etiketi çeviren çıkarmalar listelenir. Hiçbiri çevirmiyorsa karar dağıtılmıştır —
   bu da bir bulgudur.
4. **Sadakat eğrisi.** IG'nin en yüksek atıflı k kelimesi sırayla silinip skaler yeniden
   ölçülür ve rastgele k kelime silinmesine karşı çizilir: atıf sıralaması gerçekse skor
   çok daha hızlı çöker. Tamlık aksiyomatik bir kontroldür; bu, deneysel olanıdır.
5. **Yöntem uyumu.** IG (gradyan tabanlı) ile tıkama (yeniden puanlama tabanlı) aynı alan
   sıralamasını mı veriyor? İki bağımsız yöntemin Spearman uyumu tek sayı olarak gösterilir.
6. **Dikkat hizalaması (destekleyici).** Son katmanların sorgu→ürün dikkati + rollout,
   **kelime düzeyinde** toplanmış, satır başına %100'e normalize edilmiş ısı haritası; sütun
   üstünde alan renk şeridi. "Attention is not explanation" tartışması (Jain & Wallace '19 /
   Wiegreffe & Pinter '19) gereği bilinçli olarak *destekleyici* etiketlidir; birincil açıklama
   IG'dir. Cross-encoder içlerinin okunabilirliği için literatür dayanağı: "Pathway to
   Relevance" (arXiv 2502.04645).
7. **Türkçe sağlamlık şeridi.** Sorgunun diyakritiksiz, boşluksuz, tek-harf-hatalı ve büyük
   harfli varyantları yeniden puanlanır; kararın hangisinde değiştiği gösterilir. Model
   boşluksuz bileşiklerde (`çelikbileklik`) ve tek karakter silinmesinde kırılgandır — bu
   bilinçli olarak gösterilir, gizlenmez (bkz. sınırlılıklar).
8. **Ön-işleme farkı.** Model ham katalog satırını değil normalize edilmiş halini görür; her
   alan için ham → modele giden değer ve değişikliğin gerekçesi listelenir (ör. `gender: çocuk → unknown`, T4 — public satırların %8,9'unda görünür).
9. **Türkçe özet.** Sabit şablonlarla, ölçülen sayılardan üretilir (etiket, olasılık, skalerin
   eşiğe konumu, yön-duyarlı sürücü alanlar ve kelimeler, etiketi çeviren karşı-olgu).

**GUI tasarımı.** Tek statik HTML (vanilla JS + inline SVG/canvas, sıfır dış istek — imaj
çevrimdışı çalışır), Türkçe. Üstte organizatörün zorunlu bloğu: aranabilir `row_id` seçici,
ürünün görünür alanları (düzenlenebilir), seçilen satırın sorgusuyla başlayan serbest sorgu
kutusu, **Açıkla** düğmesi. Altında sırasıyla: karar kartı (etiket, olasılık çubukları, sayı
doğrusu, Türkçe özet, gecikme), kelime katkıları (mavi = "ilgili" yönüne, kırmızı = "ilgisiz"
yönüne; alt-parça görünümü isteğe bağlı), alan katkıları, karşı-olgusal kartlar (ürün tarafı /
sorgu tarafı), dikkat ısı haritası (isteğe bağlı alan-özeti çubukları), sadakat eğrisi,
sağlamlık şeridi, ön-işleme farkı ve tamlık satırı. Her kartın altında "nasıl okunur" metni
vardır; teknik dipnot yöntemi ve sınırlarını açıkça yazar.

**Ortalama kullanım rutini.** Jüri bir satır arar ve seçer → Açıkla (GPU'da ~0,3 s; tünel
üzerinden ölçülen uçtan uca 0,9 s) → etiketi, skalerin eşiklere göre konumunu ve karara en
çok katkı veren kelime/alanları okur → karşı-olgu kartlarından etiketi neyin çevirdiğini görür
→ sorguyu ya da herhangi bir ürün alanını düzenleyip yeniden Açıkla'ya basar; aynı model
yeniden puanlar ve yeniden açıklar (örneğin `celikbileklik` sorgusuna bir boşluk eklemek
etiketi "kısmen ilgili"den "ilgili"ye taşır ve sayı doğrusundaki nokta eşiği geçer).

**Servis.** Aynı imajın ikinci modu (`run_explainability.sh`, organizatörün donmuş betiği).
stdlib `http.server`; sunucu model yüklenmeden **önce** bind olur (`/healthz` ~1 s'de 200,
model arka planda yüklenir, `/readyz` hazır olunca 200) → imaja vendored ngrok v3.39.11
(sha256 pinli) tüneli açar, public URL loga basılır. `NGROK_AUTHTOKEN` yoksa fail-closed
(exit 2); token asla yazdırılmaz ve imajın hiçbir katmanında bulunmaz (bayt taramasıyla
kanıtlandı). Konteyner kapanınca UI ve tünel birlikte kapanır. Son tur teyidi teslim imajıyla
2026-08-28'de yapıldı: organizatörün betiği + gerçek ngrok tüneliyle URL 60 sn'nin altında
canlı (sınır 10 dk), dış ağdan sayfa/API 200, uçtan uca Explain 0,9 sn, konteyner
öldürüldüğünde tünel temiz kapandı (404). Geliştirme sırasında kullanılan gerçek-etiket
katmanı yalnızca bir CLI bayrağıyla açılır; organizatörün betiği bu bayrağı geçmediği için
teslimde görünmez.

## What worked & What didn't work?

**Çalışanlar**

- **Snapshot normalizasyonu:** leak-oracle kapsamı %46.9 → %0,45; adversarial fark
  0,444 → 0,176. Projenin en değerli tek adımı.
- **"Her satır yargıçtan geçer" kuralı + kör el denetimi:** nihai irrelevant kümesinde
  precision 0,987; kural-etiketli hiçbir satır eğitime girmedi.
- **Dağılım kapıları yargıç harcamasından önce:** adversarial kapı (0,7185, organizatör
  tabanı 0,748+0,02'nin altında) + zorluk-profili kapısı; kapı geçilmeden etiket parası
  harcanmadı.
- **Getirme yakın-ıskası reçetesi:** organizatörün "getir-sonra-yargıla" mekanizmasını
  taklit etmek, yalnız-kural seçiminin geçemediği adversarial kapıyı geçirdi (0,798 → 0,728).
- **Yargıç ekstraları ve eski etiket deposunun geri dönüşümü:** +0,022 ve +0,012 (scalar);
  her ikisi ablasyonla kanıtlanıp alındı. Etiket deposu sayesinde hiçbir yargıç kararı çöpe
  gitmedi — başarısız seçim turlarının etiketleri bile sonunda katkıya döndü.
- **E-ticaret ön-eğitimli backbone (TY-ecomm):** kalitede BERTurk'e +0,032; teslimde ise puan
  matematiği hızı ödüllendirdiği için BERTurk seçildi (bkz. Model Selection) — kalite ucu
  ölçüldü ve bilinçli olarak bırakıldı.
- **Attribute temizliği v2:** −%15–17 token ve +0,0155 F1 aynı anda — hem hız hem kalite.
- **Ordinal scalar + iki eşik:** zero-shot'ta bile 3-way generation'ı +0,018 geçti; ucuz,
  hızlı, tek boyutlu olduğu için açıklanabilir.
- **DEV/HOLDOUT + OOF + ön-kayıtlı eşikler:** dört HOLDOUT okumasının üçü DEV'in üzerinde
  çıktı (+0,002, −0,006, +0,001, +0,005); hiçbir karar tek public sayısıyla verilmedi.
- **Parity-kapılı hız mühendisliği:** her optimizasyon byte-identical çıktı şartıyla
  alındı; veri yolu paralelleştirmesi modelden bağımsız ~%33 uçtan-uca kazanç verdi.

**Çalışmayanlar (ölçümle reddedildi)**

- **Cosine-band negative mining:** adaylar relevant/semi komşuluğuna düştü; irrelevant
  verimi %6–21.
- **"Hard band" aday seçimi:** modelin karıştırdığı bandı seçmek organizatör profilinin
  tam tersini üretti (adversarial AUC 0,9101); importance-matching onarımı işledi
  (0,6264) ama yama olduğu için hat yeniden tasarlandı.
- **Yargıçsız kural etiketleri:** 61.8k'lık "temiz" küme hiç yargıçlanmamış çıktı (~%11
  beklenen kontaminasyon); tespit edilip reddedildi — bu keşif "distillation yerine etiket
  kalitesi" dönüşünün tetiği oldu.
- **Distillation hattı:** 12B teacher + soft-target öğrenci planı, yargıçlı veriyle düz
  etiketli eğitimin tek başına yetiştiği görülünce kapatıldı; 12B yalnız referans çıtası
  (0,8724) olarak kaldı. Basit olan kazandı.
- **Prompt/verbalizer probe'ları (12B):** en iyi varyant +0,008 AUC(S|I); ön-kayıtlı kapı
  ≥ +0,010 → reddedildi.
- **CORAL/ordinal loss head'leri:** mevcut karar zaten ordinal; literatürde CORAL düz
  CE'ye kaybediyor → hiç denenmedi (bilinçli).
- **Yardımcı S/I loss'u:** etkisiz (0,8444, AUC(S|I) değişmedi) → merdivenin devamı iptal.
- **ModernBERT:** Türkçe fertility 3.327 vs 2.053 + yavaş forward ⇒ ~3.6× yavaş; elendi.
- **İki-model kademesi (hızlı model emin olduğu satırları, büyük model kalanı):** %0
  yönlendirme 0,827 @0,69×, %50 0,847 @0,93×, %100 0,850 @1.00× — her satırda hızlı modeli
  de koşturmak kazancı yiyor, tatlı nokta yok. Model seçimi ikili.
- **Derinlik budama (12 → 8 katman) — İLK RED GERİ ALINDI.** İlk ölçüm 0,8393 @1.938 SPS idi ve
  reddedilmişti; ancak o koşu **eski `g2_extras` veri reçetesiyle** ve **hızlandırma öncesi
  servis yoluyla** yapılmıştı, yani iki değişken aynı anda oynuyordu. Şampiyon reçetesiyle ve
  ayarlı servis yoluyla yeniden eğitildiğinde TY-ecomm 8L **0,8474** (3 tohum: 0,8530/0,8436/
  0,8455, sd 0,0041) ve **3.351 SPS kapsayıcı içi** veriyor; 12L ise 0,8587 @ 2.652.
  Yani gerçek takas **+%26 hız karşılığında 0,0113 F1** — reddedilen "+%10 için 0,6–1,6 puan"
  değil. 8L artık canlı bir Pareto adayıdır. Ders: bir seçeneği reddederken tek değişken
  oynatılmalı; aksi hâlde red, seçeneğin değil o günkü reçetenin reddi olur.
- **Derinlik budama, daha ileri (8 → 6 katman):** TY-ecomm 6L **0,7991** — hız puanının
  sıfırlandığı 0,80 eşiğinin *altında*. Bu mimaride diz 8 katmandadır ve keskindir. Reddedildi.
- **torch.compile:** +%6 inference, shape başına 16–69 s derleme skorlanan sürede ⇒ net
  negatif.
- **Fazla epoch:** TY-ecomm'da 4. epoch overfit ediyor (~0,845 → ~0,808); BERTurk'te 6 optimum,
  9 epoch −0,0057 geri düşüyor. Epoch sayısı omurgaya özgü.
- **Global recalibration (prior/sıcaklık) ve batch-kantil kuralı:** model semi'yi biraz fazla
  çağırıyor (teslim modelinde %37 vs %33 — DEV'de macro-F1'i en büyükleyen eşiklerin seçtiği
  denge) ama satır-bazlılık kısıtı ve ölçülen küçük katkılar nedeniyle hiçbiri teslimde değil.
- **Per-vertical eşik ölçekleme:** erken probe'larda +0,056 vaat etti; nihai modelin
  öğrenilmiş olasılıkları üzerinde ayrıca eşik cerrahisine girilmedi (sadelik ve overfit
  riski; tek global eşik çifti kaldı).
- **Normalizasyonda üç yanlış filtre denemesi:** yeniden adlandırma, tek yönlü filtre,
  prevalence proxy'si (Data Cleaning'de anlatıldı) — kayıt altında, tekrarlanmadı.
- **Operasyonel ders:** bir gece kuyruğu yanlışlıkla iki kez başlatılıp aynı GPU'da
  çakıştı; etkilenen tüm koşular "kirli" işaretlenip karantinaya alındı (hiçbiri şampiyon
  değildi), "tek seferde tek eğitim + süreç kilidi" kuralı kondu. Kirli sayı rapora girmedi.

## Project development environment specifications (Hardware & Software)

**Geliştirme/test makinesi (organizatör runtime'ı ile aynı spec):** Debian 12, x86_64,
1× NVIDIA A100-SXM4 40 GB (driver 580.126.20, CUDA 13.0), 12 vCPU, 83 GB RAM. Stage-1
geliştirmesi ayrıca tek RTX 4080 SUPER 16 GB üzerinde yapılmıştı.

**Yazılım:** Python 3.11 (geliştirme) / 3.12 (imaj); PyTorch 2.10.0+cu128; transformers 4.57.6 (pinli); tokenizers;
safetensors; numpy/pandas; scikit-learn (yalnız analiz). Veri üretim döneminde 12B referans
modeli vLLM 0.26.0 ile servis edildi (yalnız build-time). Container base:
`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`, digest-pinli; tüm sürümler build sırasında
assert'li.

**Harici servisler (yalnızca build-time, iznin açık olduğu dönemde):** LLM-judge etiketleme
OpenRouter API üzerinden (Gemini 3.7 Flash). Harcama kayıtlı: ilk turlar ~44 USD (29.47 +
14.7), nihai üretim ~60 USD (canary 7.36 + ek 3.71 + kalan ~49) — toplam **~105 USD** (API kullanım sayacıyla mutabık).
Runtime'da internet yok; tüm ağırlıklar imajın içinde.

IMAJ: nihai build **8,69 GB** (açıklanabilirlik uygulaması + vendored ngrok dâhil); model
ağırlığı `model.safetensors` **369 MB** bf16 (184.347.651 parametre, BERTurk-128k tabanı,
revizyon `fea3225…`, sha256 `2a0d4c78…`); servis bf16. Peak VRAM < 4 GB, konteyner RAM'i
70 GB sınırının çok altında; tüm sayılar teslim README'siyle (§9.3) birebir aynıdır.

## Ablation Study (How effective was each stage of the solution on its own?) (if any)

Tüm sayılar query-disjoint OOF, public setin DEV yarısı (3.750 satır), aksi belirtilmedikçe
üç sınıflı macro-F1. Her satır tek değişkenli, eşleştirilmiş bir karşılaştırmadır:

**A. Karar kuralı ve problem yapısı (12B zero-shot referanslar):**

| Aşama | macro-F1 | Katkı |
|---|---:|---|
| Yüzey özellikleri (20 lexical/structural özellik, gradient boosting) | 0,6392 | taban |
| 12B zero-shot, 3-class generation | 0,7083 | +0,069 model gücü |
| Aynı model, tek scalar + iki eşik | 0,7259 | +0,018 ordinal karar |
| İki-sınırlı 12B kuralı (Stage-1 birikimi dahil) | 0,8373 | referans/floor |
| 12B + S/I fine-tune (referans çıtası, scalar OOF) | 0,8724 | teslim edilmedi (hız) |

**B. Veri katkıları (geliştirme sırasında, TY-ecomm/BERTurk öğrencileri üzerinde; satır başına tek değişken):**

| Adım | macro-F1 | Δ |
|---|---:|---:|
| Yargıçlanmamış kural-etiketli irrelevant seti (eski, BERTurk) | 0,7766 | — |
| → tamamı yargıçlı donmuş set (BERTurk, aynı düzenek) | 0,7920 | +0,015 |
| Attribute temizliği v2 (TY-ecomm, eşleştirilmiş) | 0,7994 → 0,8149 | +0,0155 |
| + yargıç ekstraları (29.5k) | 0,8217 → 0,8441 | +0,022 |
| Backbone: BERTurk → TY-ecomm (şampiyon veri, 3 ep) | 0,8205 → 0,8529 | +0,032 |
| + eski yargıçlı etiket deposu (136.489 çift) | 0,8529 → 0,8553 | +0,002 (scalar +0,012) |

**C. Karar kuralı ablasyonu (aynı olasılık dosyaları):** argmax 0,8322 → train-prior
düzeltmesi 0,8354 → +sıcaklık 0,8359 → fit'li iki eşik **0,8529** (+0,021; eşik taşınım
iyimserliği 0,003).

**D. Madencilik zincirinin izole etkisi:** aday seçim dağılımı düzeldikçe hem kapı metriği
hem gerçek transfer birlikte hareket etti — adversarial AUC 0,9101 → 0,6264 (onarım) →
yeniden tasarımla 0,7185 (organizatör tabanının altında); öğrencinin gerçek-irrelevant
isabeti 0,374 → 0,466 → 0,605 → nihai modelde S/I AUC 0,9037. Yargıç etiketinin katkısı:
aynı seçim kural etiketiyle DEV 0,752 / gerçek-I recall 0,72'de kalıyor.

**E. Hız/kalite takasları:** 8-katman budama ilk ölçümde reddedildi, düzeltilmiş reçetede
TY-ecomm 8L canlı aday oldu (public 0,8626 @ 3.322 SPS) ama nihai cephede BERTurk'e kaybetti;
kademe: tatlı nokta yok (ret); veri yolu paralelleştirme + iş parçacığı sabitleme: kaliteye
dokunmadan uçtan uca ~%33 + %5–17 (kabul, 0 etiket değişimi).

**F. Teslim edilen BERTurk üzerinde doğrulanan katkılar (aynı reçete, tek değişken):**

| Değişiklik | DEV OOF | Δ |
|---|---:|---:|
| 3 → 6 epoch (cinsiyet düzeltmesi öncesi) | 0,8372 → 0,8428 | +0,0056 |
| `gender: çocuk → unknown` ile sızıntısız yeniden eğitim | 0,8428 → 0,8546 | **+0,0118** |
| İki-eşikli ordinal kural vs argmax (public 7.500) | 0,8495 → 0,8579 | +0,0084 |
| Tohum yayılımı (n=3) | 0,8533–0,8559 | sd 0,0013 |
| Denenip reddedilenler: 9 epoch −0,0057 · LR 2e-5 −0,0031 · uncased omurga −0,0035 · marka-sink %5 −0,0041 · sorgu yapıştırma/typo artırımı ±0,002 (gürültü) | | |

Normalizasyonun izole etkisi bilerek macro-F1 olarak raporlanmıyor: raw-train'e fit edilmiş
bir karşılaştırma modeli hiç eğitilmedi (kirli sayı üretmemek için); etkisi leak
metrikleriyle ölçülü (oracle %46.9 → %0,45; adversarial fark 0,444 → 0,176).

**Bileşen-çıkarma ablasyonu (ölçülmüş, aynı omurga ve aynı OOF protokolü):**

| yapılandırma | DEV OOF | fark |
|---|---:|---:|
| yargılanan I + v1 metin, 2 epoch | 0,8149 | — |
| + öznitelik temizliği v2 | 0,8217 | +0,0068 |
| + yargıç-semi/relevant ekstralar | 0,8441 | **+0,0224** |
| + önceki etiket deposu (136k) | 0,8529 | +0,0088 |
| + 3. epoch (nihai reçete) | 0,8553 | +0,0024 |
| + iki-eşikli ordinal karar kuralı (o omurgada rejim bazlı) | **0,8587** | +0,0034 |
| *karşılaştırma:* argmax kararı | 0,8434 | −0,0119 |

Her satır tek bir bileşeni ekler; en büyük tek kazanç yargıç tarafından semi/relevant'a çevrilen
"ekstra" satırlardır (+0,0224), yani madenciliğin ıskaladığı sınırı yargıcın düzeltmesi.
Bu tablo TY-ecomm omurgasında ölçülmüştür; teslim edilen BERTurk omurgasında bileşen
katkılarının aynı YÖNDE olduğu doğrulandı (örn. sızıntısız yeniden eğitim +0,0118, iki-eşikli
karar kuralı argmax'a karşı +0,0073 DEV OOF / +0,0084 public), tablo bazındaki tam tekrar yapılmadı ve yapılmayan şey
rapora yazılmamıştır.


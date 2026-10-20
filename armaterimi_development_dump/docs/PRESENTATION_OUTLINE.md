# Final Sunumu — Slayt Planı (5–6 Eylül, Trendyol kampüs)

> Kaynak doküman: `docs/FINAL_REPORT_DRAFT.md` (sayıların tek kaynağı).
> `⟦FREEZE: …⟧` işaretli her yer, model/imaj dondurulduğunda raporla **aynı anda, aynı sayıyla** doldurulur.
> Dil: Türkçe; teknik terimler İngilizce bırakılır (fine-tuning, cross-encoder, macro-F1) —
> organizatörün "gereksiz çeviri okunabilirliği öldürüyor" uyarısına birebir uyum.
> Yapmadığımız hiçbir şey söylenmez; her sayı raporda ve development dump'ta izlenebilir.

---

## 0. Transkriptten format bulguları (sunumu bağlayan maddeler)

- **Şimdilik serbest format.** Organizatör webinarda süre/şablon vermedi: "free format gitmenizi
  tavsiye ediyoruz… beklentilerimizi 5–6 Eylül'de kampüste detaylandıracağız. Ters köşe yok:
  yaptığınız işleri anlattığınız bir sunum hazırlayın; belki birkaç eksik kalem söyleriz."
  → Plan: **10 dk konuşma + ~3 dk canlı demo** varsayımı; her slayt tek mesajlık ve modüler,
  yerinde gelecek kalem listesine yarım saatte uyarlanabilir.
- **Canlı demo zorunlu ve puanlanıyor.** "Final günü sunumunuzu yaparken jürinin önünde bu
  [explainability arayüzü] üzerinden interaktif olarak sonuçlarınızı gösteriyor olacaksınız…
  kalitesini, intuitive olup olmamasını final günündeki sunumda jürimiz değerlendirecek."
  Review döneminde yalnız fonksiyon testi (ngrok linki 10 dk içinde kalkıyor mu, farklı
  row_id + elle değiştirilmiş query çalışıyor mu); **kalite puanı sunum günü veriliyor.**
- **Açıklama, tahmini üreten modelden gelmek zorunda.** "Küçük model tahmin edip büyük modele
  açıklatma — bunu kabul etmiyoruz." → Demo anlatımında bu cümleye açık gönderme yapıyoruz
  (bizim tasarımımız zaten böyle; bunu avantaja çeviriyoruz).
- **Geçen yılın dersi (organizatör ağzından):** demo tarafındaki özgün işler "takımların puan
  olarak ayrışmasını sağladı"; verdikleri template "10–15 dakikada yazılmış POC — belirgin
  şekilde üstüne çıkın." → Demoya sunum içinde geniş yer.
- **Dil:** İngilizce sunuma "şimdilik yapılabilir" dendi ama resmî onay verilmedi → Türkçe hazırla.
- **Rapor için söylenen okunabilirlik uyarısı slaytlar için de geçerli:** tek okumada anlaşılır,
  kaş kaldırtmayan metin; jüri mühendis — pazarlama dili yok, ölçüm dili var.
- Puanlama teyidi (webinar): final-set F1 = MaxNorm × sıralama çarpanı × 20; hız = SPS MaxNorm × 10
  (sıralama çarpanı yok); iki sert eşik: **≥0.85 macro-F1 VE 90k ≤ 150 dk**, yoksa hız puanı 0.
  Kronometre `run_solution.sh` start → çıktı CSV; solution modunda ağ kapalı.

---

## 1. Zamanlama planı (hedef: 10:00 konuşma + demo)

| # | Slayt | Süre | Kümülatif |
|---|---|---|---|
| S1 | Kapak | 0:20 | 0:20 |
| S2 | Problem tek resimde | 0:35 | 0:55 |
| S3 | Çözüm tek bakışta | 0:40 | 1:35 |
| S4 | Veri tuzağı: iki snapshot | 1:10 | 2:45 |
| S5 | Eksik sınıfı üretmek ve yargılamak | 1:10 | 3:55 |
| S6 | Model seçimi: 306M alan-içi encoder | 0:40 | 4:35 |
| S7 | Asıl savaş: semi/irrelevant sınırı | 1:00 | 5:35 |
| S8 | Hız: resmî kronometre | 0:55 | 6:30 |
| S9 | Validasyon disiplini | 0:50 | 7:20 |
| S10 | Çalışmayanlar | 1:00 | 8:20 |
| S11 | Sonuçlar (donmuş) | 0:40 | 9:00 |
| S12 | Demo geçişi | 0:15 | 9:15 |
| — | **CANLI DEMO** | 2:30–3:00 | ~12:00 |
| S13 | Ekip, maliyet, kaynak | 0:25 | ~12:30 |
| S14 | Kapanış | 0:20 | ~12:50 |

**Esneme planı:** toplam süre 10 dk'ya sabitlenirse S3, S2 içine katlanır; S6+S7 tek slayta iner;
S13 30 saniyeye düşer. **Asla kısaltılmayacaklar:** S4 (kanca), S5, S9, S10 ve demo.
Süre 15 dk verilirse demo 4–5 dk'ya uzar (jüriye klavye teklif edilir), slaytlar uzamaz.

---

## 2. Slaytlar

### S1 — Kapak: "90.000 satırda doğru VE hızlı"
- Takım adı, yarışma/aşama, tarih.
- Görev: sorgu–ürün → `relevant / semi-relevant / irrelevant` (üç sınıflı macro-F1).
- Tek cümle iddia: macro-F1 ⟦FREEZE: DEV=…, HOLDOUT=…⟧, 90k'yı 150 dk limitine karşı
  ~1 dakika mertebesinde işleyen tek Docker imajı ⟦FREEZE: SPS=…⟧.
- **SÖYLE:** "Bu sunum iki sorunun cevabı: doğru sınırı nasıl öğrendik ve resmî kronometrede
  nasıl hızlı kaldık — ikisinde de tek yöntemimiz ölçmekti."
- **GÖRSEL:** Sade kapak; ortada üç bölgeli ordinal sayı doğrusu motifi
  (irrelevant | semi | relevant, iki eşik çizgisi) — sunum boyunca dönecek görsel imza.

### S2 — Problem tek resimde
- Girdi: sorgu + ürün metadatası; çıktı: üç sınıftan biri; değerlendirme: **görünmeyen 90.000
  satırlık private set**.
- Puan: F1 (MaxNorm × sıralama çarpanı × 20) + hız (SPS MaxNorm × 10) + rapor 10 + sunum ve
  açıklanabilirlik ~20.
- İki sert eşik: ≥0.85 macro-F1 VE ≤150 dk — biri bile tutmazsa hız puanı 0.
- Kronometre container start'ından çıktı dosyasına; cold start dahil, ağ kapalı.
- Bizim okumamız: bu bir Pareto problemi → hedefi baştan "≥0.85 F1'i ≥3.000 satır/s'de teslim et"
  diye yazdık; altında kalan her takası ölçüp reddettik.
- **SÖYLE:** "Puan formülünü ilk gün ezberledik; bundan sonraki her mühendislik kararı bu
  formüle karşı verildi."
- **GÖRSEL:** Tek resim: "apple telefon" sorgusu altında üç ürün kartı — iPhone 17 Pro (relevant),
  telefon kılıfı (semi), çorap (irrelevant); altta puan formülü tek şerit halinde.
  (Örnek bilinçli olarak organizatörün webinardaki kendi örneği.)

### S3 — Çözüm tek bakışta: dört blok
- **1) Metin normalizasyonu:** deterministik snapshot düzeltmesi + attribute temizliği
  (girdi %15–17 kısalır, DEV +0.0155).
- **2) Eksik sınıf üretimi:** 10 reçete + tf-idf getirme → dağılım kapıları → %100 LLM-judge
  etiketi → kör denetim 0.987.
- **3) Model:** 306M e-ticaret ön-eğitimli encoder, cross-encoder; ordinal karar
  (tek scalar + iki eşik).
- **4) Serving:** tek Docker imajı, iki mod (tahmin + explainability); lazy-import hızlı yol,
  süreç-paralel veri okuma.
- **SÖYLE:** "Dört blok; her birinin katkısı ablation'la ayrı ayrı ölçüldü — sunumun geri kalanı
  bu dört bloğun kanıtıdır."
- **GÖRSEL:** ⟦FREEZE: ŞEMA⟧ Soldan sağa dört bloklu pipeline şeması; her bloğun altında tek
  ölçüm rozeti (+0.0155 / precision 0.987 / DEV ⟦FREEZE⟧ / ⟦FREEZE: SPS⟧). Raporla aynı şema.

### S4 — Veri tuzağı: iki snapshot, tek etiket sızıntısı  *(kanca slaydı)*
- `train.csv` iki farklı katalog export'unun birleşimi — ve satırın hangi export'tan geldiği,
  etiketin kendisini ele veriyor.
- Satırların **%46.9'u** en az bir biçim markörü taşıyor; iki kurallı bir oracle, raw train'in
  **%60.8'ini %100.000 doğrulukla** sınıflıyor.
- Örnek markörler: `color detail` attribute key'i → 196.763 satır, hepsi relevant, public'te 0;
  title'da noktalı-i encoding'i → %100 semi; `unisex` → %99.99 relevant, public'te 0.
- Test tarafında bu markörler ya yok ya sınıftan bağımsız → raw train'e fit edilen her model,
  test'te hiç ateşlemeyen mükemmel bir shortcut öğreniyor.
- Çözüm: etiket kullanmayan deterministik normalizasyon (NFC + noktalı-i, 18 sızıntı key'i,
  sentinel birleştirme) → oracle kapsamı **%46.9 → %0.45**; inference'ta byte-identical aynı
  transform.
- **SÖYLE:** "Modelin eline mükemmel bir kopya kâğıdı tutuşturulmuştu ve o kâğıt sınavda
  geçersizdi — ilk işimiz kâğıdı yok etmek oldu; bu, projenin en değerli tek adımı."
- **GÖRSEL:** İki snapshot şeması (Export A → 250k relevant; Export B → 250k semi + testin üç
  sınıfının tamamı); yanında 3 satırlık markör mini-tablosu; büyük rakam geçişi "%46.9 → %0.45".

### S5 — Eksik sınıf: irrelevant'ı üretmek ve yargılamak
- Train'de 0 irrelevant; test dengeli üçte-bir → sınıfı **organizatörün gerçek profili**nde
  üretmek zorundaydık (kolay negatif işe yaramıyor: gerçek irrelevant'ların %63'ü sorgu ile
  sıfır kelime kesişimli ama "yakın" görünüyor).
- Üretim: 10 kural reçetesi + organizatörün "getir-sonra-yargıla" mekanizmasını taklit eden
  tf-idf getirme yakın-ıskası.
- **Kapılar yargıçtan önce, zorunlu:** adversarial ayırt-edilebilirlik 0.7185 ≤ organizatörün
  kendi train→test tabanı (0.748) + 0.02; zorluk profili gerçek bantta (12B margin medyanı
  −1.5 vs gerçek −1.0). Kapı geçilmeden tek etiket harcanmadı.
- Etiketleme: **182.000 adayın %100'ü** doğrulanmış LLM-judge'dan (8 aday yargıç aynı 1.200
  DEV satırında paired yarıştı; kazanan macro-F1 0.924). Yargıç yalnız eğitim verisi etiketledi,
  asla tahmin üretmedi.
- Kör el denetimi: 300 satır × 3 bağımsız okuyucu, yalnız rubrikle → **precision 0.987**,
  relevant'a dönen 0 satır, %98.3 oybirliği. Donmuş küme: 145.848 irrelevant + sha256 manifest.
- **SÖYLE:** "Sentetik veri uydurmadık: organizatörün dağılımında aday seçtik, dağılımı
  kapılarla kanıtladık, son sözü bağımsız bir yargıca ve kör insan denetimine bıraktık."
- **GÖRSEL:** Huni diyagramı: aday havuzu → dağılım kapıları (iki eşik rozetiyle) → LLM-judge
  182k → donmuş 145.848; huninin yanında "0.987" mühür rozeti.

### S6 — Model seçimi: neden 306M alan-içi encoder?
- Aynı donmuş veri, aynı düzenek, ön-kayıtlı ">0.012 fark" kuralıyla ölçülmüş tablo (DEV OOF):

  | Model | macro-F1 | Not |
  |---|---:|---|
  | **e-ticaret ön-eğitimli 306M encoder (seçilen)** | **0.8553** | hız 1.4× ceza, 3× değil |
  | BERTurk-128k (184M) | 0.8205 | fark 0.032 ≫ 0.012 taban |
  | mdeberta / e5-large | ~0.81 | açıkça altta |
  | 0.6B LLM metin kulesi | 0.7660 | 4× yavaş, en zayıf S/I |
  | 12B LLM (S/I fine-tune, referans çıtası) | 0.8724 | ~60 satır/s → teslim edilemez |
- Alan ön-eğitimi kazandırıyor: e-ticaret metniyle ön-eğitim BERTurk'e +0.032; parametrenin
  ~%63'ü embedding tablosunda → hesap katmanları küçük, hız cezası ölçülen **1.4×**.
- Veri katkıları backbone'dan bağımsız doğrulandı (+0.018 ekstralar BERTurk'te de) → model ve
  veri kararları birbirini kirletmiyor.
- **SÖYLE:** "Daha büyük model daha iyi model değil: puan matematiğinde 12B'nin +0.017 F1'i,
  30 kat hız kaybını asla geri ödeyemiyor — 306M'lik alan-içi encoder bu problemin doğru boyu."
- **GÖRSEL:** F1 (y) – hız (x, log ölçek) saçılımı; sağ üst Pareto köşesinde bizim model
  işaretli; 12B sol üstte "hız duvarı" çizgisinin arkasında.

### S7 — Asıl savaş: semi/irrelevant sınırı
- Oracle ayrıştırması: mükemmel bir R-kapısı +0.050 getirirken **mükemmel bir S/I-kapısı
  +0.110** getiriyor — 2.2× headroom; dönüşüm ölçülü: Δmacro-F1 ≈ 0.87 × ΔAUC(S|I).
- Bu yüzden tüm veri/mimari kararları tek metriğe hizalandı: AUC(S|I).
- Karar kuralı ordinal: olasılıklardan tek scalar `x = p_semi + 2·p_rel`, iki eşikle üç sınıf;
  zero-shot 12B üzerinde bile üç sınıflı generation'ı +0.018 geçiyor.
- **Her etiket yalnız kendi satırından** — private 90k tek batch işlenmeyeceği için batch
  istatistiği kullanan hiçbir post-processing teslimde yok.
- Eşik overfit'i ölçüldü: eşikleri taze query-disjoint yarıda yeniden fit etmenin maliyeti
  0.0031 ± 0.0024 → eşik fitting overfit kaynağı değil.
- **SÖYLE:** "Problemi bir boyuta indirdik: model tek bir alaka skoru üretir, iki eşik üç sınıfı
  ayırır — bu hem en yüksek ölçülen doğruluk, hem de jüriye tek bakışta gösterilebilen bir karar."
- **GÖRSEL:** Solda iki bar (+0.050 R-kapısı vs +0.110 S/I-kapısı); sağda sayı doğrusu üzerinde
  üç sınıfın scalar dağılımları (renkli yoğunluklar) ve iki eşik çizgisi.

### S8 — Hız: resmî kronometreyi mühendislemek
- Kronometre container start → çıktı CSV. Jüri-tarzı koşularda duvar saati **17 s → 14 s → 13 s**
  (7.5k public; her adım ölçümle).
- 13 saniyenin anatomisi: ~6.5 s yapısal container maliyeti (GPU hook 3.7 + create 0.3 +
  harness ~2.5 — tüm takımlarda aynı) + ~6.1 s bizim işimiz (model hazırlığı 3.9 s veri yoluyla
  örtüşük, saf inference 2.2 s).
- Kazanımlar **parity-kapılı** (çıktı byte-identical değilse optimizasyon reddedilir):
  transformers'sız model portu (fark tam 0), lazy import 5.9 → 0.16 s, süreç-paralel veri yolu
  (90k'da ~17.9 → 11.9 s), uzunluk-kovali batching.
- 90k projeksiyonu ~45–50 s ⟦FREEZE: SPS=…⟧ — 150 dakikalık limitin ~%0.5'i.
- Ölçülüp reddedilenler: torch.compile (derleme süresi skorlanan sürenin içinde → net negatif),
  derinlik budama, iki-model kademesi.
- **SÖYLE:** "Sürenin neredeyse yarısı model bile değildi — hız puanını boru hattında kazandık
  ve hiçbir optimizasyonu, çıktı byte'ı byte'ına aynı kalmadan kabul etmedik."
- **GÖRSEL:** Yatay waterfall: 0→13 s tek çubuk, renkli segmentler (yapısal container / veri
  yolu / model hazırlığı / inference); üstünde "17 → 14 → 13" ok dizisi.

### S9 — Validasyon disiplini: 7.500 satıra overfit etmemek
- Elimizdeki tek gerçek-dağılım etiketli veri: 7.500 satırlık public. Private: 90.000, görünmez.
  Public'e bakılan her seçim bir overfit fırsatı — disiplin baştan kuruldu.
- **Query-disjoint DEV/HOLDOUT** (3.750/3.750, sha256'lı bölme): tüm fitting ve tüm
  karşılaştırmalar DEV'de; HOLDOUT karantinada, **asla model seçmez**.
- HOLDOUT protokolü: milestone başına tek okuma, yalnız DEV'in seçtiği şampiyonda. Okumalar:
  +0.002, −0.006, son okuma ⟦FREEZE: HOLDOUT=…⟧ — DEV dürüst çıktı.
- Gürültü tabanı ölçüldü: bootstrap %95 CI ±0.012 → ön-kayıtlı kural: fark >0.012 değilse
  kazanan ilan edilmez, basit olan kalır. Seed duyarlılığı da ölçüldü (yayılım ~0.004).
- Kill kuralları deney koşulmadan önce yazıldı (canary ≥0.85, el denetimi ≥0.85, probe +0.010);
  metrik fonksiyonu organizatörün `evaluate.py`'ı ile bit-identical doğrulandı.
- **SÖYLE:** "Görünmez 90 bin satırda yarışıyoruz; bu yüzden her sayının yanında bir gürültü
  tabanı, her seçimin önünde önceden yazılmış bir kural var — hikâyenin hiçbir yerinde
  cherry-picking'e yer bırakmadık."
- **GÖRSEL:** Şema: public 7.5k → DEV (açık kutu: "tüm kararlar burada") / HOLDOUT (kilitli
  kasa ikonu: 3 tarihli okuma etiketi); altta ±0.012'lik gürültü cetveli.

### S10 — Çalışmayanlar (ölçtük, reddettik)  *(güvenilirlik slaydı)*
- Cosine-band negative mining → irrelevant verimi %6–21 → ret.
- "Hard band" aday seçimi → organizatör profilinin tam tersi (adversarial AUC 0.9101) →
  hat baştan tasarlandı.
- Yargıçsız kural etiketi (61.8k satır, ~%11 beklenen kontaminasyon) → küme çöpe;
  "yargıçtan geçmeyen satır eğitime giremez" kuralı buradan doğdu.
- Distillation hattı → etiket kalitesi düzelince gereksizleşti; 12B yalnız referans çıtası
  olarak kaldı. Basit olan kazandı.
- Kademe, derinlik budama, torch.compile, 4. epoch, yardımcı loss, prompt probe'ları,
  global recalibration → hepsi sayısıyla ölçülüp reddedildi (tam liste raporda).
- **SÖYLE:** "Bu, sunumun en pahalı slaydı — bu yollara gerçek gün ve para harcadık; neyin
  çalışmadığını sayısıyla bilmek, çalışan sayılara güvenmenizin sebebidir."
- **GÖRSEL:** Üç sütunlu tablo: Fikir | Ölçüm | Karar — 6 satır, her satırda kırmızı "ret"
  damgası; distillation satırında "basit olan kazandı" vurgusu.

### S11 — Sonuçlar (donmuş sayılar)
- Doğruluk: DEV OOF ⟦FREEZE: DEV=…⟧ · HOLDOUT ⟦FREEZE: HOLDOUT=…⟧ · teslim imajından ölçülen
  public macro-F1 ⟦FREEZE: PUBLIC-F1⟧ (+ sorgu-bazlı bootstrap CI).
- Hız: jüri bayraklarıyla zamanlı koşu ⟦FREEZE: SPS=…⟧; ⟦FREEZE: GPU-PROVA⟧.
- Sert eşikler: ≥0.85 macro-F1 ✓ · 150 dk limitine karşı ~1 dk mertebesi ✓.
- İmaj: digest-pinli base, tüm ağırlıklar içeride, runtime'da ağ yok, read-only dosya sistemi
  ve jüri bayraklarının tamamı altında test edildi.
- Her sayı tek komutla yeniden üretilebilir: development dump + sha256 manifest'ler.
- **SÖYLE:** "Bu sayılar taslaktan değil, jürinin koşturacağı imajın kendisinden ölçüldü —
  teslim ettiğimiz şey bir model değil, doğrulanmış bir ölçüm zinciri."
- **GÖRSEL:** Büyük skor kartı: iki dev rakam (macro-F1 ve SPS) + iki yeşil onay işareti
  (eşikler); alt bantta küçük künye (imaj boyutu ⟦FREEZE: IMAJ⟧, checksum).

### S12 — Canlı demo geçişi: modelin kendisi açıklıyor
- Kural (şartnameyle aynı): ekrandaki her şey **tahmini üreten checkpoint'in kendi
  ağırlıklarından** — ikinci bir model ya da anlatıcı LLM yok.
- Birincil açıklama: class-specific Integrated Gradients — axiomatic; completeness sapması
  her açıklamada ekranda basılır ("açıklama, tahmine matematiksel olarak bağlı").
- Ordinal sayı doğrusu: scalar + iki eşik + en yakın sınıra mesafe → "model bu satırı neden
  semi'de bıraktı" tek bakışta.
- Counterfactual: sorgudaki her tek-kelime silme aynı modelle yeniden skorlanır; etiketi
  çeviren kelimeler kart olarak listelenir. Attention yalnız "destekleyici görünüm" etiketiyle.
- Aynı imajın 2. modu; ngrok ile servis; yedek plan hazır (yerel tarayıcı + önceden alınmış
  ekran kaydı — yalnız ağ/tünel felaketinde).
- **SÖYLE:** "Şimdi sözü modele veriyorum: kararını kendisi açıklayacak, ben sadece
  soracağım — dilerseniz sorguyu jüriden biri yazsın."

**60 saniyelik demo akışı (ezber):**
1. *(0–10 s)* Public test'ten aranabilir listeden bir satır seç → ürün kartı + modelin gördüğü
   normalize girdi ekranda ("ham veri ile modelin gördüğü aynı şey değil — bilinçli").
2. *(10–25 s)* "Açıkla" → tahmin rozeti + olasılık çubukları + sayı doğrusu: scalar'ın iki
   eşiğe göre konumu — "işte kararın kendisi, tek boyutta."
3. *(25–40 s)* IG token ısısı + alan çubukları: karara en çok katkı veren sorgu/ürün
   token'ları; completeness sapması ekranda → "bu açıklama süslü bir tahmin değil, muhasebesi
   tutan bir ayrıştırma."
4. *(40–55 s)* Sorguyu elle değiştir (ör. markayı değiştir) → aynı model yeniden skorlar:
   etiket, sayı doğrusu ve açıklama birlikte değişir.
5. *(55–60 s)* Counterfactual kartı: "şu tek kelime silinince etiket dönüyor." Kapanış
   cümlesi: "Gördüğünüz her sayı tahmini üreten checkpoint'ten geliyor — ikinci model yok."
   → Ardından klavye jüriye: istedikleri row_id + istedikleri sorgu.
- **GÖRSEL:** Arayüzün ekran görüntüsü (sol: aranabilir row_id listesi; ana panel: ürün kartı,
  düzenlenebilir sorgu kutusu, tahmin rozeti, sayı doğrusu SVG, IG ısısı, counterfactual
  kartları, metot dipnotu).

### S13 — Ekip, maliyet, kaynak
- Donanım: organizatörün sağladığı **1× A100 40GB** (geliştirme ve test, jüri ortamıyla aynı
  spec); Stage-1 tek bir RTX 4080 üzerinde geliştirilmişti.
- Dış harcama: LLM-judge etiketleme **≈105 USD** — tamamı build-time, faturayla mutabık;
  başka hiçbir dış servis yok. Runtime'da internet yok, tüm ağırlıklar imajda.
- Geliştirme agentic araçlarla (organizatör açıkça serbest bıraktı); her sayı insan denetimli
  protokollerle (kör denetim, ön-kayıtlı kapılar) doğrulandı.
- Teslim: tek imaj (iki mod) + development dump (tüm veri üretim/eğitim/validasyon kodu ve
  verileri) + 14 başlıklı teknik rapor.
- **SÖYLE:** "Bütün sistem bir A100 ve ~105 dolarlık etiket bütçesiyle kuruldu — bütçeyle
  değil, disiplinle ölçeklendik."
- **GÖRSEL:** Dört mini künye kartı: GPU · ~105 USD · imaj ⟦FREEZE: IMAJ⟧ · 813k eğitim satırı
  (145.848'i yargıçlı irrelevant).

### S14 — Kapanış: neden 90k'da da ayakta kalır?
- Sızıntı kapatıldı (%46.9 → %0.45) → model kopya kâğıdını değil gerçek sinyali öğrendi.
- Eksik sınıf organizatör dağılımında üretildi ve bağımsız denetimden 0.987 ile geçti.
- Karar satır-bazlı, eşik taşınımı ölçülü, HOLDOUT protokolü DEV'i doğruladı → public'e
  değil dağılıma fit olduk.
- Üç cümlelik özet: doğruluk ölçüldü, hız ölçüldü, her iddia yeniden üretilebilir.
- **SÖYLE:** "Public sette güzel sayı üretmek kolaydır; biz bu sistemi görünmeyen 90 bin
  satırda ayakta kalsın diye inşa ettik — sorularınızı bekliyoruz."
- **GÖRSEL:** Üç ikonlu özet şerit (kalkan = sızıntı kapalı · huni = denetimli veri ·
  terazi = validasyon disiplini); teşekkür + takım iletişimi.

---

## 3. Q&A hazırlığı — jüriden beklenen 10 zor soru

1. **"12B model daha doğruydu; neden onu teslim etmediniz?"**
   Puan matematiği: hız MaxNorm'unda ~60 satır/s'lik bir 12B her senaryoda kaybediyor; +0.017
   F1 ≈ +0.6 puan, hız kaybı bunun katbekat üstünü siliyor. 12B'yi referans çıtası olarak
   tuttuk (0.8724) ve öğrencimiz ~30× hızda o çıtanın dibine geldi.

2. **"Ürettiğiniz irrelevant'ların organizatörünkine benzediğini nereden biliyorsunuz?"**
   Üç bağımsız kanıt: adversarial probe bizim kümeyi gerçeğinden 0.7185 AUC ile ayırabiliyor —
   organizatörün kendi train→test tabanının (0.748) bile altında; zorluk profili gerçek bantta
   (12B margin medyanı −1.5 vs −1.0); ve transfer kapıyla birlikte hareket etti (gerçek-I
   isabeti 0.374 → 0.605 → S/I AUC 0.904).

3. **"LLM-judge kullanmak bir sızıntı ya da hile değil mi?"**
   Yargıç yalnız eğitim verisini build-time'da etiketledi — runtime'da tek bir çağrı yok, test
   tahminine asla dokunmadı. Kendisi de denetlendi: 8 aday aynı DEV satırlarında paired yarıştı
   (kazanan 0.924) ve çıktısı kör 3-okuyuculu insan denetiminden 0.987 precision ile geçti.

4. **"Private setin dağılımı farklı çıkarsa ne olur?"**
   Test'in üç sınıfı da aynı yeni export'tan geliyor; normalizasyonumuz snapshot artefaktlarını
   iki taraftan da siliyor. Tüm seçimler query-disjoint DEV'de yapıldı, HOLDOUT okumaları DEV'i
   doğruladı (±0.006) ve karar satır-bazlı — dağılım kayarsa bile batch-istatistiği kırılganlığımız yok.

5. **"Eşikleriniz 7.500 satırlık public'e overfit değil mi?"**
   Bunu ölçtük: eşikleri taze bir query-disjoint yarıda yeniden fit etmenin maliyeti yalnızca
   0.0031 ± 0.0024. Parametresiz alternatifler de ölçüldü (argmax 0.832 vs eşikli 0.853) —
   +0.021'lik katkının taşınım iyimserliği sadece 0.003.

6. **"Neden distillation yapmadınız? Büyük modelden öğretmek standart değil mi?"**
   Ölçüm asıl darboğazın kapasite değil etiket kalitesi olduğunu gösterdi: her satır yargıçtan
   geçince düz etiketli öğrenci tek başına 12B çıtasının yakınına geldi. Daha basit boru hattı =
   daha az kırılganlık; karmaşıklığı sayı getirmeden almıyoruz.

7. **"Daha küçük bir modelle çok daha fazla hız puanı almaz mıydınız?"**
   Ölçtük: en hızlı adayımız 0.8205 F1'de kalıyor — ≥0.85 sert eşiğinin altı, yani hız puanı
   sıfırlanma riski. Derinlik budama da +%10 hıza karşılık 0.6–1.6 F1 puanı yedi; sürenin
   çoğu modelde değil veri yolunda olduğundan küçülmek hız bile getirmiyor.

8. **"Public etiketlerini eğitimde kullanmak serbestti; neden kullanmadınız?"**
   Model ağırlıkları public'i hiç görmedi; public yalnız eşik fit'i ve doğrulamada kullanıldı.
   Böylece elimizde gerçek bir genelleme okuması kaldı — 7.500 satırı yemek kısa vadeli kâr,
   90k'da körlük demekti.

9. **"Açıklamalarınızın gerçekten modeli yansıttığının garantisi ne?"**
   IG axiomatic: attribution'ların toplamı = f(girdi) − f(baseline) ve completeness sapması her
   ekranda basılıyor; counterfactual'lar aynı checkpoint'in gerçek yeniden skorlamaları.
   Arayüzdeki preprocessing, serving koduyla byte-identical assert'li — ikinci model yok.

10. **"Tek seed, tek koşu — sonuçlarınız şans eseri olamaz mı?"**
    Seed duyarlılığını ayrıca ölçtük (aynı reçete iki seed: fark ~0.004, yayılım gürültü
    tabanının içinde) ve karar kuralımız gereği >0.012 olmayan hiçbir fark "kazanım" sayılmadı.
    Manşet sayılar tekil şanslı koşular değil; merdivenin her basamağı ablationla bağımsız kanıtlı.

---

## 4. Sunum geneli notlar

- **Ton:** ölçüm dili; her iddianın yanında sayısı. "Çok iyi çalışıyor" yok, "0.7994 → 0.8149" var.
- **Bir slayt = bir mesaj.** Jüri mühendis; metin az, sayı büyük, kaynak raporda.
- Internal kod adları/deney etiketleri hiçbir slaytta yok; model adları herkese açık adlarıyla.
- **Demo provası:** en az 3 tam tekrar (biri kötü ağ senaryosuyla); ngrok linki sunum öncesi
  ayakta ve `/readyz` yeşil; yedek: yerel tarayıcı + ekran kaydı.
- FREEZE alanları rapor donduğunda tek geçişte doldurulur; sunum ve rapor sayıları
  **birebir aynı** olmak zorunda (jüri ikisini yan yana görecek).
- 5–6 Eylül'de organizatörün açıklayacağı kalem listesi gelince bu iskeletle eşleştir;
  eksik kalem varsa S2/S13'e madde eklenir, akış bozulmaz.

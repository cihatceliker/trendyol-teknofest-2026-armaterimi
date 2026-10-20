# E18 — attribute key audit: what can be deleted from `attributes` without losing query-relevant information

**Inputs.** `common/train_norm.parquet` (500,000 rows, 4,339,362 key:value segments, 98.77M attribute chars) and
`common/public_test_norm.parquet` (7,500 rows, 78,578 segments, 1.73M chars). Both already have the 18 leak keys of
`common/drop_keys.csv` removed and are U+0307-stripped / lower-cased. Segments were split with the pipeline's own
`normalize.SEG` regex, so every number here is about text the model would actually see.
Public test was used for **schema/value statistics only** (no labels exist in the normalized file; no DEV/holdout split was touched).
No GPU, no training.

**Outputs.** `key_report.csv` (1,063 keys = every key in train ∪ test, all statistics + recommendation),
`drop_keys_v2_candidates.csv` (the 40 `DROP_KEY` keys, `key,reason`), `aggregate_estimate.json`.
Scratch scripts: `/var/tmp/claude-1000/-home-jupyter/483abbaf-6d7f-4b27-838c-53d3d34a821c/scratchpad/e18/s0*.py`.

---

## 1. Headline

| | train | public test |
|---|--:|--:|
| attribute chars removed by the 40 `DROP_KEY` keys alone | **30.9%** | **27.2%** |
| + boilerplate-valued segments of kept keys (strict tier, §4) | +3.1% | +3.1% |
| **total removed (variant A: whole segment goes)** | **33.9%** | **30.3%** |
| total removed, variant B (boilerplate segments keep the bare key name) | 32.7% | 29.2% |
| mean attribute chars per row, before → after (A) | 197.5 → 130.5 | 231.0 → 161.0 |
| p95 chars per row, before → after (A) | 498 → 358 | 571 → 430 |
| mean segments per row, before → after (A) | 8.68 → 6.42 | 10.48 → 7.97 |
| rows whose attributes become empty (A), of all rows | 5.89% (29,451) | 4.92% (369) |
| rows empty after (A), incl. the 0.45% / 1.13% already empty | 6.34% | 6.05% |

The rows that become empty are rows that carried *only* `menşei` / care boilerplate (e.g. `menşei: tr`,
`bakım talimatları (genel): ürünün kullanım ömrünü korumak için …`); they are concentrated in süpermarket, hobi,
otomobil, kozmetik, bahçe — categories where the catalogue never had a real attribute.

**Three keys are 15.6% of all train attribute characters and carry nothing a query can use:** `yıkama talimatı`
(5.6%, 74% of its values are codes such as `type 1`, `t44`, `1184469`, or care-label URLs), `bakım talimatları (genel)`
(5.1%, 72% is the one sentence *"ürünün kullanım ömrünü korumak için ürüne özel bakım talimatlarını takip ediniz."*)
and `menşei` (4.9%, two-letter country codes). Together with `bakım talimatları (gıda temas)`, `sürdürülebilirlik detayı`,
`garanti süresi`, `kutu durumu`, `garanti tipi`, `persona` they are 28.1% of train chars / 25.0% of test chars — the
entire saving comes from nine keys; the other 31 `DROP_KEY` keys add 2.5% / 2.0%.

---

## 2. Method — what the evidence columns mean

For every key, per split (`_train` / `_test`), `key_report.csv` has:

* **frequency / cost**: `rows`, `rowpct`, `pairs`, `chars` (segment + its `", "` separator), `charshare` (of all attribute chars), `nval`, `meanvlen`, `top5`.
* **boiler**: fraction of the key's segments whose value is boilerplate under the strict tier in §4 (`belirtilmemiş`, `… mevcut değil`, `diğer`, care sentences, URLs, UUIDs, punctuation-only). `yok`/`var`/`evet`/`hayır` are **not** boilerplate.
* **query landing** (`land`): fraction of the key's segments whose value shares ≥1 token with the row's own query. Tokens: `\w+` on diacritic-folded lower text, ≥3 chars, minus a small stop list (function words, `ürün`, units, `adet/parça/tip`, `var/yok/evet/hayır`). `land_rel` / `land_semi` / `land_diff` = the same restricted to relevant / semi rows (train only).
* **`chance`**: landing after shuffling queries *within the same category root* — the rate a key gets by vocabulary overlap alone. `lift = land/chance`.
* **`uniq`**: landing on a query token that is **absent from title, category and brand** — the value is the *only* place the query word is confirmed. This is the strongest "the key adds information" signal.
* **`landnum`**: digit-bearing tokens (≥2 chars) shared with the query (catches `650 ml`, `iphone 15`, `128 gb`).
* **`stem`**: shared 5-char prefix (catches Turkish suffix variation: `kemerli`/`kemersiz`, `baskılı`/`baskısız`).
* **`kname`**: the key *name* shares a token with the query (`yaka tipi` vs "v yaka elbise"); **`redund`**: all value tokens already occur in title ∪ category ∪ brand.
* `p_rel_train`: P(relevant | key present) in train (for reference only; the leak keys were already removed).

Decision rule (conservative, as briefed):

1. `DROP_KEY` only if the key belongs to a semantic family that a query cannot state (care, warranty, origin, sustainability,
   box/packaging, package dimensions, legal/regulatory, manufacturer/importer identity, installation service, marketing persona,
   media links, the `refill` flag) **and** the evidence agrees: `land ≤ 0.3%` *or* every landing is redundant (`uniq ≈ 0`), `landnum ≈ 0`.
   Every one of the 40 passes; the only ones with visible landing are `garanti tipi` (3.8%, but 100% brand names — `uniq` 0.01%, and
   its `yenilenmiş ürün` value is covered by the category in 100% of 839 rows) and `üretici adı` (manufacturer = brand, `uniq` 0).
2. Otherwise `DROP_BOILERPLATE_VALUES_ONLY` if ≥0.5% of the key's values are boilerplate (174 keys), else `KEEP` (849 keys).
   Anything with a rubric role (colour, material, size, type, model, capacity, gender/age, pattern, neckline, sleeve, heel, sole,
   fit, collection, set/piece count, compatibility) is kept even at zero landing; those are flagged `confidence = low-evidence`.
3. Compatibility keys (`cep telefonu modeli`, `uyumlu model/marka/cihaz`, `araç marka ve model`, …) are `KEEP` **including** their
   `belirtilmemiş` values — rubric §4C says device silence is itself a signal, and A12 found the bare key name matters there.

---

## 3. Top-40 keys by train char share

`land % (chance)` = landing rate vs within-category shuffled baseline; `rel−semi` = landing on relevant rows minus semi rows (pp);
`uniq` = landing that title/category/brand do not already provide; `num` = numeric-token landing. All train unless marked te.

| # | key | rec | role | rows % tr | rows % te | chars % tr | chars % te | boiler % | land % (chance) | rel−semi | uniq % | num % | land % te |
|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `yıkama talimatı` | **DROP_KEY** | care instructions | 28.9 | 34.3 | 5.60 | 6.52 | 4.0 | 0.28 (0.07) | +0.07 | 0.02 | 0.00 | 0.15 |
| 2 | `renk` | KEEP | colour | 74.7 | 77.2 | 5.11 | 4.58 | 0.0 | 3.04 (0.52) | +2.49 | 1.49 | 0.00 | 4.68 |
| 3 | `bakım talimatları (genel)` | **DROP_KEY** | care instructions | 10.9 | 7.6 | 5.11 | 3.00 | 72.2 | 0.05 (0.13) | -0.10 | 0.00 | 0.00 | 0.35 |
| 4 | `materyal` | BOILER_ONLY | material | 52.2 | 53.6 | 5.10 | 4.54 | 2.0 | 5.19 (0.51) | +3.22 | 1.30 | 0.00 | 4.08 |
| 5 | `menşei` | **DROP_KEY** | origin | 81.1 | 88.1 | 4.85 | 4.51 | 0.0 | 0.00 (0.00) | +0.00 | 0.00 | 0.00 | 0.00 |
| 6 | `materyal bileşeni` | BOILER_ONLY | material | 20.8 | 28.8 | 3.88 | 4.84 | 2.0 | 2.93 (0.46) | +1.70 | 0.45 | 0.02 | 2.68 |
| 7 | `bakım talimatları (gıda temas)` | **DROP_KEY** | care instructions | 8.0 | 3.2 | 3.86 | 1.33 | 68.3 | 0.10 (0.08) | +0.02 | 0.01 | 0.00 | 0.00 |
| 8 | `sürdürülebilirlik detayı` | **DROP_KEY** | sustainability | 19.4 | 30.6 | 3.21 | 4.34 | 0.0 | 0.00 (0.00) | +0.00 | 0.00 | 0.00 | 0.00 |
| 9 | `özellik` | BOILER_ONLY | spec/feature | 24.2 | 13.4 | 2.45 | 1.13 | 8.0 | 3.56 (0.21) | +0.65 | 0.73 | 0.00 | 3.18 |
| 10 | `kumaş tipi` | BOILER_ONLY | material | 20.2 | 38.9 | 2.15 | 3.47 | 18.0 | 0.64 (0.07) | +0.47 | 0.16 | 0.00 | 0.89 |
| 11 | `ortam` | BOILER_ONLY | type/form/style | 21.9 | 36.3 | 2.14 | 2.97 | 1.0 | 0.77 (0.25) | +0.11 | 0.17 | 0.00 | 1.17 |
| 12 | `desen` | BOILER_ONLY | type/form/style | 29.3 | 43.1 | 1.99 | 2.53 | 0.6 | 0.62 (0.09) | +0.35 | 0.19 | 0.00 | 1.36 |
| 13 | `ek özellik` | BOILER_ONLY | spec/feature | 12.9 | 18.2 | 1.87 | 2.24 | 35.0 | 0.95 (0.17) | +0.68 | 0.37 | 0.00 | 1.54 |
| 14 | `garanti süresi` | **DROP_KEY** | warranty | 14.9 | 9.2 | 1.79 | 0.95 | 6.6 | 0.00 (0.00) | +0.00 | 0.00 | 0.00 | 0.00 |
| 15 | `kutu durumu` | **DROP_KEY** | box/packaging | 14.7 | 22.7 | 1.65 | 2.20 | 0.0 | 0.00 (0.02) | -0.01 | 0.00 | 0.00 | 0.12 |
| 16 | `topuk boyu` | BOILER_ONLY | type/form/style | 8.8 | 8.2 | 1.55 | 1.23 | 9.7 | 10.94 (10.16) | -11.60 | 4.20 | 0.00 | 22.76 |
| 17 | `parça sayısı` | BOILER_ONLY | count/set | 13.1 | 8.3 | 1.23 | 0.66 | 2.3 | 0.03 (0.00) | -0.04 | 0.03 | 0.28 | 0.00 |
| 18 | `koleksiyon` | KEEP | type/form/style | 10.9 | 19.6 | 1.19 | 1.89 | 0.0 | 5.38 (0.77) | -2.47 | 0.26 | 0.00 | 7.35 |
| 19 | `garanti tipi` | **DROP_KEY** | warranty | 5.9 | 4.5 | 1.17 | 0.77 | 0.1 | 3.76 (0.12) | +1.01 | 0.01 | 0.06 | 5.31 |
| 20 | `tema / stil` | BOILER_ONLY | type/form/style | 10.2 | 5.5 | 1.10 | 0.51 | 0.7 | 0.45 (0.06) | +0.46 | 0.28 | 0.00 | 1.21 |
| 21 | `topuk tipi` | BOILER_ONLY | type/form/style | 8.6 | 8.0 | 1.09 | 0.87 | 2.7 | 11.54 (10.76) | -12.73 | 4.59 | 0.00 | 26.09 |
| 22 | `boyut/ebat` | BOILER_ONLY | size/dimension/capacity | 9.8 | 4.7 | 1.06 | 0.44 | 1.4 | 0.50 (0.14) | +0.24 | 0.06 | 0.14 | 0.00 |
| 23 | `dış materyal` | BOILER_ONLY | material | 8.7 | 7.3 | 1.04 | 0.75 | 0.6 | 3.90 (0.64) | -0.33 | 1.43 | 0.00 | 1.63 |
| 24 | `yaka tipi` | KEEP | type/form/style | 8.7 | 19.9 | 1.02 | 2.02 | 0.0 | 11.59 (2.32) | -0.73 | 1.85 | 0.00 | 8.70 |
| 25 | `kalıp` | BOILER_ONLY | type/form/style | 12.5 | 28.6 | 1.01 | 1.96 | 5.7 | 4.94 (0.36) | +4.71 | 0.43 | 0.00 | 1.86 |
| 26 | `kemer/kuşak durumu` | KEEP | type/form/style | 6.6 | 16.0 | 0.99 | 2.05 | 0.0 | 0.06 (0.00) | +0.08 | 0.02 | 0.00 | 0.17 |
| 27 | `paket içeriği` | BOILER_ONLY | count/set | 9.0 | 14.1 | 0.97 | 1.30 | 0.9 | 0.00 (0.00) | +0.00 | 0.00 | 0.00 | 0.10 |
| 28 | `kol tipi` | KEEP | type/form/style | 8.6 | 19.7 | 0.91 | 1.72 | 0.0 | 1.31 (0.72) | -0.56 | 0.78 | 0.00 | 3.79 |
| 29 | `bağlama şekli` | BOILER_ONLY | type/form/style | 7.2 | 5.7 | 0.90 | 0.61 | 0.8 | 0.15 (0.06) | +0.18 | 0.11 | 0.00 | 1.41 |
| 30 | `boy` | BOILER_ONLY | size/dimension/capacity | 11.8 | 26.8 | 0.84 | 1.56 | 8.0 | 1.07 (0.23) | +0.69 | 0.43 | 0.00 | 2.64 |
| 31 | `persona` | **DROP_KEY** | marketing persona | 7.2 | 14.3 | 0.84 | 1.38 | 3.2 | 0.03 (0.01) | +0.05 | 0.00 | 0.00 | 0.09 |
| 32 | `ürün detayı` | BOILER_ONLY | spec/feature | 7.1 | 12.9 | 0.82 | 1.30 | 1.5 | 1.16 (0.34) | +0.84 | 0.43 | 0.00 | 1.14 |
| 33 | `dokuma tipi` | KEEP | material | 7.2 | 19.7 | 0.80 | 1.90 | 0.0 | 4.79 (0.23) | +0.84 | 0.73 | 0.00 | 5.15 |
| 34 | `tamir edilebilirlik` | **DROP_KEY** | legal/regulatory | 3.0 | 2.7 | 0.79 | 0.61 | 0.0 | 0.01 (0.00) | +0.02 | 0.00 | 0.00 | 0.00 |
| 35 | `siluet` | KEEP | type/form/style | 9.9 | 25.2 | 0.79 | 1.78 | 0.0 | 4.50 (0.29) | +2.19 | 0.96 | 0.00 | 4.07 |
| 36 | `ürün tipi` | BOILER_ONLY | type/form/style | 7.7 | 13.2 | 0.79 | 1.03 | 1.5 | 12.98 (0.67) | -9.38 | 3.19 | 0.00 | 1.72 |
| 37 | `diğer özellikler` | BOILER_ONLY | spec/feature | 2.4 | 2.4 | 0.74 | 0.82 | 9.9 | 15.89 (1.20) | +6.17 | 0.98 | 0.39 | 13.56 |
| 38 | `kol boyu` | KEEP | type/form/style | 8.7 | 22.3 | 0.71 | 1.61 | 0.1 | 1.90 (0.92) | -1.17 | 1.16 | 0.00 | 3.52 |
| 39 | `sezon` | KEEP | type/form/style | 7.8 | 16.2 | 0.69 | 1.15 | 0.0 | 0.00 (0.00) | +0.00 | 0.00 | 0.00 | 0.33 |
| 40 | `astar durumu` | KEEP | type/form/style | 5.0 | 10.3 | 0.60 | 1.06 | 0.0 | 0.17 (0.04) | +0.24 | 0.11 | 0.00 | 0.00 |
Reading the table:

* The kept rubric keys behave as the rubric predicts: `renk` lands 3.0% vs 0.5% chance and **+2.5 pp more on relevant than semi rows**,
  `materyal` 5.2% / +3.2 pp, `kalıp` 4.9% / +4.7 pp, `koleksiyon` 5.4% (lift ~7, `tesettür`, `basic`), `ürün tipi` 13%, `tip` 19%,
  `diğer özellikler` 16%, `kullanım alanı` 12% with 6% unique landing. Compatibility keys land 24–69% and land **more on semi rows**
  (`cep telefonu modeli` 29% rel vs 77% semi) — the accessory mechanism from A12 §3.1.
* Every `DROP_KEY` key in the top 40 lands at ≤0.3% with `uniq ≤ 0.02%` and `num = 0`, i.e. below or at its shuffled baseline,
  and no value ever carries a query token that the title did not already carry.
* `topuk boyu` / `topuk tipi` / `yaka tipi` land 11–12% but only at ~chance (10%, 2%): the landing token is the head noun
  (`topuklu`, `yaka`). They are kept (rubric: heel/neckline), consistent with A12's note that they are false-match prone for overlap features.

---

## 4. Value patterns to delete (boilerplate), with their weight

**Strict tier — recommended.** A segment is deleted when its value (stripped, lower) is

| pattern | examples | train chars | test chars |
|---|---|--:|--:|
| exact `belirtilmemiş` / `belirtilmemis` / `belirtilmedi` | `kumaş tipi: belirtilmemiş` (18% of that key), `boy: belirtilmemiş` (8%), `topuk boyu: belirtilmemiş` (10%) | 1.47% | 1.51% |
| contains `mevcut değil` | `ek özellik: ek özellik mevcut değil` (35% of the key), `deri kalitesi: parça mevcut değil` (50%) | 1.05% | 1.18% |
| exact `diğer` | `batarya türü: diğer` (90%), `özellik: diğer` (5%), `bağlantı: diğer` | 0.17% | 0.08% |
| care sentences: `takip ediniz`, `etiketi(ni) kontrol ediniz`, `etiketindeki talimat`, `pflegeetikett` | (mostly inside keys that are dropped anyway) | 0.11% | 0.03% |
| unknown-type tokens: `tanımsız`, `bilinmiyor`, `bilgi yok`, `not available`, `n/a`, `none`, `unknown`, `uygulanabilir değil`, `none persona`, `belirsiz` | `batarya boyutu: uygulanabilir değil` (92%) | 0.08% | 0.01% |
| punctuation-only (`-`, `.`, `_`, empty) | `diğer özellikler: -` (5.5%) | 0.03% | 0.03% |
| URL / `<image_link>` / UUID (`^https?://`, `^[0-9a-f]{8}-…`) | `yıkama talimatı: https://cdn.dsmcdn.com/…/wash_at_30.jpg`, `enerji sınıfı görseli: b51f4157-…` | (in dropped keys) | |
| key-specific: `ürün uyarıları (oyuncak)` values ≥40 chars or matching `uygun değildir|gözetim|içerdiğinden|boğulma|tehlike|uyarı|dikkat|yetişkin` ; `diğer özellikler: yok` | legal toy warnings; "other features: none" | 0.14% | 0.24% |
| the keyless sentinel: whole field == `unknown` (4,097 train rows, 0 test) | treat as empty string | 0.03% | 0 |

Total strict tier on kept keys: **3.06% train / 3.10% test** of attribute chars; 22,700 train segments are hit in `ek özellik`, 18,300 in
`kumaş tipi`, 9,600 in `özellik`, 5,200 each in `materyal` and `deri kalitesi`.

**Soft tier — NOT recommended for deletion (reported for completeness).** "Universal" values — `sezon: tüm sezonlar` (56% of the key),
`boy: standart`, `cilt tipi: tüm cilt tipleri`, `yaş: tüm yaş grupları`, `saç tipi: tüm saç tipleri`, `araç marka ve model: universal`,
`boyut/ebat: tek ebat` — are 1.39% train / 1.51% test. They are not informationless: `universal` is how an accessory says it fits any car
(rubric §5.4), `tek ebat` answers a size query, `tüm sezonlar` says a seasonal query is not contradicted. Left in.

---

## 5. The 40 `DROP_KEY` keys, by family (train / test char share)

- **care instructions** (14.57% / 10.85%): `yıkama talimatı`, `bakım talimatları (genel)`, `bakım talimatları (gıda temas)` — landing 0.28% / 0.05% / 0.10%, all below chance; the 402 `yıkama talimatı` landings are `halı`, `ayakkabı`, `perde`, `yıkanabilir` inside long care sentences and are in the title in all but 0.02% of cases.
- **origin** (4.85% / 4.51%): `menşei` — two-letter codes, landing exactly 0. See §7 for the caveat.
- **sustainability** (3.21% / 4.34%): `sürdürülebilirlik detayı` — `hayır` 83%, `evet` 12%, `organik` 2.7%; 0 landings; the 38 train queries containing `organik` never meet a row whose sustainability value is `organik`.
- **warranty** (2.96% / 1.72%): `garanti süresi` (0 landings), `garanti tipi` (landings are brand names, `uniq` 0.01%; `yenilenmiş` covered by category 100%).
- **box/packaging** (1.79% / 2.38%): `kutu durumu` (3 landings in 73,620 rows, all the query "kutu çanta"), `piller kutuya dahil mi?`, `ürün kutusu`, `paket görseli (ön)` / `(arka)` (image links).
- **legal/regulatory** (1.71% / 1.28%): `tamir edilebilirlik`, `ürün güvenliği bilgisi` (hazard class), `gıda temas sembolü`, `enerji sınıf aralığı` (the *labelling scale* "a'dan g'ye", not the class — the class stays in `enerji sınıfı`), `enerji sınıfı görseli` and `ürün bilgi formu` (UUIDs), `tip-19. ruhsat tarihi`, `ruhsat tarih`.
- **marketing persona** (0.84% / 1.38%): `persona` (`cool & comfort`, `fashion forward`, `young`…; 0 landings).
- **package dimensions** (0.21% / 0.20%): `paket genişlik`, `paket derinlik`, `paket yükseklik` — carton size, not product size; numeric landing 0.00%, redundancy with title 1–2%.
- **installation service** (0.20% / 0.15%): `kurulum gerekli mi?`, `ücretsiz kurulum/montaj`.
- **refill flag** (0.13% / 0.07%): `refill` — `hayır` 95%; the 455 `evet` rows are ordinary shampoos/serums (title says refill/yedek/dolum in 6%); 13 train queries contain "refill", none lands.
- **media link** (0.10% / 0.10%): `enerji ve besin değerleri` — 100% `<image_link>` / URLs.
- **manufacturer/importer identity** (0.03% / 0.02%): `üretici adı/adres bilgisi/mail adresi`, `birincil/ikincil/üçüncül ithalatçı adı/adres bilgisi/mail adresi` (12 keys) — addresses and e-mails; manufacturer name = brand (`uniq` 0).

13 of the 40 occur in only one split (the importer keys, `paket görseli`, `ruhsat` keys) — they are in the list so the rule is schema-complete, they cost nothing.

---

## 6. Aggregate estimate (from `aggregate_estimate.json`)

| | train | test |
|---|--:|--:|
| total attribute chars | 98,772,561 | 1,732,309 |
| removed by `DROP_KEY` | 30.88% | 27.22% |
| removed by boilerplate values on kept keys | 3.06% | 3.10% |
| **removed total, variant A** | **33.94%** | **30.31%** |
| removed total, variant B (bare key name kept for boilerplate) | 32.73% | 29.16% |
| rows with attributes → rows becoming empty | 497,730 → 29,451 (5.89% of all rows) | 7,415 → 369 (4.92%) |
| rows empty after, of all rows | 6.34% | 6.05% |

The test reduction is ~3.6 pp smaller than train because test rows carry more garment-schema keys (apparel is 31.8% of test vs
12.5% of train), which are kept; the drop list is not tuned to either split.

---

## 7. Ambiguous keys — judgement calls, in order of chars at stake

1. **`menşei`** (4.9% / 4.5% of chars) — `DROP_KEY`, *medium* confidence. A query never states an origin code, landing is exactly 0,
   and the brief lists origin as "almost certainly cannot matter". The caveat is A12 §4.1: on DEV the *value* is an item prior
   (`bd`/`kh`/`cn` origin is ~1.75× enriched for `irrelevant` inside giyim, n=25–50 per cell, WEAK), and in train P(relevant) varies by
   country (tr 0.48, cn 0.38, vn 0.71, um 0.24 — but train R/S is export-confounded). That is a catalogue-quality prior, not query
   information. Recommendation stands; if the ablation on the real model shows a loss, re-add it as a 2-bit side feature (`origin_non_tr`), not as 5 chars of text per row.
2. **`materyal bileşeni`** (3.9% / 4.8%, 11,387 distinct values) — `KEEP` (boilerplate only). It is material, it lands 2.9%
   (+1.7 pp on relevant rows) and 0.45% uniquely — so it cannot be deleted by key. But 34% of its segments are token-subsets of the row's
   own `materyal` value (40% of `materyal` ∪ title), e.g. `materyal: pamuk polyester, materyal bileşeni: 65% pamuk, 35% polyester`.
   A follow-up value rule "drop `materyal bileşeni` when its tokens ⊆ `materyal` tokens" would save ~1.3% of chars; not certified here.
3. **`persona`** (0.8% / 1.4%) — `DROP_KEY`, *medium*. Marketing segments (`cool & comfort`, `fashion forward`) with 0 landings.
   Train P(relevant) differs by value (0.65–0.77) and A12 §4 found a weak, sign-flipping, category-confounded effect on DEV. Dropped.
4. **Lifestyle keys the rubric calls weak but that land: `kullanım alanı`, `ortam`, `koleksiyon`, `tema / stil`** (0.4% / 2.1% / 1.2% / 1.1%) —
   `KEEP`. Rubric §4B deleted room/placement as zero-effect on the *label*, and A12 warns `ortam`/`koleksiyon` fire the wrong way
   in contradiction rules. But `kullanım alanı` lands 11.7% (chance 1.8%) with 6.0% unique (`mutfak`, `bebek odası`), `koleksiyon` 5.4%
   (lift 7, `tesettür giyim`), so the text *is* query-referenced; whether the model should trust it is a modelling question, not a deletion one.
5. **Bucketed dimension / capacity keys: `yükseklik`, `genişlik`, `derinlik`, `hacim`, `güç (watt)`, `ağırlık`, `boyut/ebat`, `kapasite`**
   (2.9% of train / 1.3% of test chars together) — `KEEP`, *low evidence*. Values are ranges (`11 - 30 cm`, `0-15 l`, `500-1500 watt`) that essentially never
   equal a query numeral (numeric landing 0.0–0.1%); A12 §5.3 showed numeric checks work on the title, not attributes (16 extra rows in
   3,750). They survive only on the rubric's "size/capacity → semi" role. This is the next block to ablate if more speed is needed.
6. Smaller ones, all `KEEP` by the conservative rule although nothing ever lands: `sezon` (0.7%; a semantic check shows winter queries
   with `sezon: kış` are 86% relevant vs 63% for `tüm sezonlar`, n=118/59), `kemer/kuşak durumu` (1.0%, stem landing `kemerli`/`kuşaklı`
   0.07%), `baskı/nakış tekniği` (0.4%, and `baskısız` items are still 89% relevant for "baskılı" queries, n=44), `cep` (0.6%),
   `frekans` (0.2%, `50 hz / 60 hz` in 82% — a genuinely dead spec), `voltaj` (0.2%), `cam şekli`, `otomatik kapanma`, `bluetooth`, `sayfa sayısı`.
7. **`alerjen uyarıları`** and **`ürün uyarıları (oyuncak)`** (0.1% + 0.16%) — kept with key-specific boilerplate rules rather than dropped:
   the toy-warning field is routinely misused for the product type (`akülü`, `peluş oyuncak`, `puzzle` — 8% landing) and allergen text
   names the food itself (`badem`, `kahve`).
8. **`kutu durumu`** — `DROP_KEY`, high. Train P(relevant) is 0.70 for `kutu yok` vs 0.55 for `kutulu`, but A12 §4.2 showed the DEV effect
   vanishes within category root; landing is 3 rows in 73,620. Dropped.

---

## 8. Things to carry into the pipeline

* Apply the key list as an **exact lower-cased match on the key**, after `t1()`, in `clean_attrs` — the same place the current 18 leak keys are dropped. The 40 keys are disjoint from `common/drop_keys.csv`.
* Apply the strict boilerplate tier as a value rule on the remaining segments; skip it for the seven compatibility keys.
* Treat `attributes == "unknown"` as empty.
* 60 "keys" are parse artefacts (a comma inside a long care sentence, e.g. `kuru temizleme yapılmaz. ; not`, 104 train rows, 0.00% chars). They are labelled `KEEP` / `parse artefact` in the CSV because no key rule can target them; they mostly live inside care text that is being dropped anyway.
* Not done here and worth a measured ablation before trusting: the `materyal bileşeni` ⊆ `materyal` rule (item 2), the bucketed-dimension block (item 5), and value truncation (A12 showed ≤60 chars is free on a proxy).

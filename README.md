# Medical_Data_Processing
Repo for my university thesis.

Szakdolgozat csinálós részeében:
- Kipróbálom az az architektúrák zászlós hajóit
- Amelyik a legjobban teljesít, abba az irányba tovább menve, tervezek egyet finomhangolva (már amennyire lehet)
- Ez a szóban forgó finomhangolás akár csak az is lehet, hogy több osztály helyett átállok a "melanoma igen/nem, hány %-ban", kimenetre

Próba architektúrák
- ConvNext
- SwimTransformer
- CoAtNet

Konkrét versenyeztetett hálók (Base modellek, amennyire lehet egymáshoz közeli paraméterszámmal):

ConvNeXt-B, Swin-B, CoAtNet-2 (Note: A small modellek egymáshoz hasonlóbb paraméterűek)



Le kéne írni a hálót

Megírni a tanítási folyamatot
- Configurálhatónak kéne lennie, hogy 
    - ha megszakad: ugyanonnan folytassa
    - ugorja át, ha már be van tanítva

Egy alapos kiértékelés 
- metrikák a tanulás közben is nagyon hasznosak

Végén validáció



LossFN:

Use CrossEntropyLoss when:
- exactly one class is correct
- classes are mutually exclusive
- model outputs raw logits of shape [B, C]

Binary classification
- CrossEntropyLoss works
- but many people prefer BCEWithLogitsLoss
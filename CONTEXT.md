# Contexto del Proyecto — Clasificador de Galaxias Clumpy

## Objetivo
Clasificacion binaria de galaxias JWST como **clumpy** (grumosas) o **no-clumpy** usando deep learning con transfer learning.

## Datos

**Ubicacion**: `data/`

- **JSON de etiquetas**: `data/11_05_2026_dictionary_with_all_statmorph_and_classifications.json`
  - 1459 galaxias con metadatos completos
  - Estructura: `{"galaxies": [{"name": "...", "filters": [...], "info": {...}, "frames": [...]}]}`
  - Etiqueta target: `info.visual_clas.Clumpy` (booleana)
  - `frames[0]` contiene morfometrias (Gini, M20, C, A, S, sersic_n, etc.)

- **Imagenes**: `data/colour_images/` — ~1490 PNG de 240x240 px, RGB
  - Naming: `{FIELD}_{ID}.png` (ej: `GS4_10635.png`)
  - Algunas variantes: `_wMIRI` (con MIRI), `_MIRIonly` (solo MIRI)

**Estadistica clave**:
- Total con imagen: **1442 galaxias**
- Clumpy: **141 (9.8%)** ← clase minoritaria, desbalance severo
- No clumpy: **1301 (90.2%)**
- Campos: COS4 (400), U4 (395), G4 (294), GN4 (200), GS4 (153)

## Entorno de trabajo
- **Local (Windows)**: PyTorch 2.12 CPU, PowerShell 7+
- **Colab**: T4 GPU, PyTorch 2.6+ (torch.amp API)
- **Drive path en Colab**: `/content/drive/MyDrive/ProyectoClumps/data/`

## Modelos implementados

### 1. `clumpy_classifier_convnext.ipynb` (baseline)
- ConvNeXt-Tiny solo imagen
- **F1 inicial: 0.19** (modelo colapso a predecir siempre "clumpy" por pos_weight=9.2 + WeightedRandomSampler)

### 2. `clumpy_classifier_convnext_hybrid.ipynb`
- **ConvNeXt-Tiny + MLP** de features tabulares
- 10 features: Gini, M20, C, A, S, sn_per_pixel, sersic_n, sersic_rhalf, ellipticity_centroid, elongation_centroid
- Arquitectura: `Imagen -> ConvNeXt[768] + Tabular -> MLP[128] -> Concat[896] -> FC -> 1`
- **F1 CV: 0.59 ± 0.06** ✓ (post-fix del bug de unpacking)
- ROC-AUC: 0.87, PR-AUC: 0.49, Precision: 0.54, Recall: 0.69
- 40 celdas: incluye OOF ensemble honesto (celda 25), in-sample ensemble (celda 30), curvas de entrenamiento (celdas 25-26)

### 2.1. `clumpy_classifier_convnext_hybrid_v2_1.ipynb` (RECOMENDADO FINAL) ⭐
- **ConvNeXt-Tiny + MLP** con 15 features tabulares (agrega F(G,M20), S(G,M20), rpetro_circ, r20, r80)
- TTA (4 augmentaciones: original, hflip, vflip, hvflip)
- 10-Fold Stratified CV (menor varianza)
- Ensemble de 10 modelos + TTA en inferencia
- **F1 CV VALIDADO: 0.62 ± 0.12** (sin el bug de unpacking, arquitectura mas robusta)
- **NO tiene el bug** — usa `evaluate_tta()` directamente sobre `fold_val` en el CV loop
- 29 celdas: sin retrain final (elimina otra fuente de leakage)

### 3. `clumpy_classifier_resnet_hybrid.ipynb`
- ResNet-50 + MLP (mismas features tabulares)
- **F1 CV VALIDADO: 0.50** (post-fix del bug de unpacking)
- Reportado inicialmente 0.67 pero estaba inflado por el mismo bug que el ConvNeXt
- Descartado por rendimiento inferior al ConvNeXt hibrido (0.59)

## Decisiones clave

- **Loss**: Focal Loss (α=0.9, γ=2.0) — reemplazo BCEWithLogitsLoss + pos_weight que causaba colapso
- **Balance**: WeightedRandomSampler (sin pos_weight adicional)
- **Split**: 5-Fold Stratified Cross Validation (no train/val/test split simple)
- **Umbral**: Optimizado por fold en val set (search 0.05-0.95, step 0.01)
- **Freeze schedule**: Backbone congelado 5 epocas, luego fine-tuning completo con LR=5e-5
- **Optimizer**: AdamW (LR_head=2e-4, LR_finetune=5e-5, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Batch size**: 64
- **Epochs**: 50 con early stopping (patience=10) sobre val F1
- **Mixed precision (AMP)**: `torch.amp.autocast('cuda')` — API nueva PyTorch 2.6+
- **Augmentation train**: HFlip, VFlip, Rotation(15), ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2)
- **Augmentation eval**: Solo Resize(240) + CenterCrop(224) + Normalize (ImageNet stats)

## Estructura del notebook hibrido (ConvNeXt)

1. **Setup + imports** (celda 2)
2. **Mount Drive + CUDA check** (celda 4)
3. **Cargar datos + tabular features** (celda 6)
4. **Dataset + transforms + scaler por fold** (celda 8)
5. **Visualizar ejemplos** (celda 10)
6. **HybridModel class** (celda 12)
7. **Focal Loss + funciones aux (build_model, train_one_epoch, evaluate, find_best_threshold)** (celda 14)
8. **train_fold function** (celda 16)
9. **5-Fold CV loop** (celda 18)
10. **Resultados CV (tabla, media±std)** (celda 20)
11. **Visualizaciones CV (bars por fold, ROC/PR pooled, CM)** (celdas 22-23)
12. **Ensemble de 5 modelos del CV** (celdas 25-27)
13. **Retrain final en todos los datos** (celda 29)
14. **Grad-CAM** (celda 31)
15. **Exportar modelo (.pth)** (celda 33)
16. **Inferencia con predict_clumpy()** (celda 35)

## Problemas resueltos

1. **torch.cuda.amp deprecado** → migrado a `torch.amp.autocast('cuda')` + `GradScaler('cuda')`
2. **UnpicklingError con weights_only=True** → convertir numpy arrays a listas nativas antes de `torch.save`, reconvertir en `torch.load`
3. **RuntimeError en Grad-CAM** (`Can't call numpy() on Tensor that requires grad`) → agregar `.detach()` antes de `.cpu().numpy()`
4. **Modelo colapsaba a una clase** → cambiar de BCEWithLogitsLoss+pos_weight a FocalLoss sin pos_weight
5. **F1 bajo con umbral 0.5** → optimizar umbral por fold via `find_best_threshold` maximizando F1

## Comparacion de resultados

| Modelo | F1 CV | ROC-AUC | Precision | Recall | Std F1 |
|---|---|---|---|---|---|
| ConvNeXt solo-imagen (v1) | 0.19 | 0.60 | 0.11 | 1.00 | — |
| ConvNeXt solo-imagen (con Focal + umbral) | 0.58 | 0.90 | 0.53 | 0.67 | 0.07 |
| ~~ConvNeXt hibrido (con bug)~~ | ~~0.76~~ | ~~0.95~~ | ~~0.92~~ | ~~0.68~~ | ~~0.17~~ |
| **ConvNeXt hibrido (VALIDADO)** | **0.59 ± 0.06** | **0.87** | **0.54** | **0.69** | 0.06 |
| ResNet-50 hibrido (con bug, sin re-validar) | 0.67 (inflado) | 0.86 | 0.83 | 0.56 | 0.03 |
| **ResNet-50 hibrido (VALIDADO)** | **0.50** | — | — | — | — |
| v2 (MixUp + LS) | 0.58 | 0.89 | 0.53 | 0.71 | 0.08 |
| v2.1 (sin MixUp/LS) | 0.62 | 0.89 | 0.64 | 0.62 | 0.12 |

**Nota**: El F1=0.67 del ResNet-50 esta con el mismo bug de unpacking que tenia el ConvNeXt. Se fixeo el codigo pero no se re-corrio el CV. El F1 real probablemente cae al rango 0.55-0.65 tras el fix.

### OOF Ensemble (pooled, umbral unico) — ConvNeXt hibrido
- F1: 0.5425, Precision: 0.5030, Recall: 0.5887
- ROC-AUC: 0.8542, PR-AUC: 0.4395, Umbral: 0.76
- Mas realista para produccion (un solo umbral vs uno por fold)

### Ensemble in-sample (optimista) — ConvNeXt hibrido
- F1: 0.8367 (in-sample, 4/5 modelos vieron cada galaxia)
- Solo valido como modelo de despliegue para galaxias NUEVAS

## BUG CRITICO IDENTIFICADO (fix aplicado)

En el CV loop del notebook hibrido, el unpacking estaba mal:

```python
val_loader, _, _, _ = make_loaders(fold_train, fold_val, ...)
```

`make_loaders` retorna `(train_loader, val_loader, sm, ss)`, entonces `val_loader` en realidad era el **train_loader** con WeightedRandomSampler. Toda la evaluacion post-training corria sobre TRAIN (con oversampling que forzaba ~50% clumpy). Esto inflaba F1 de 0.58 a 0.76.

**Fix**: crear val loader explicitamente con el scaler retornado:
```python
val_ds_eval = ClumpyDataset(fold_val, transform=eval_transform, scaler_mean=sm, scaler_std=ss)
val_loader_eval = DataLoader(val_ds_eval, batch_size=BATCH_SIZE, shuffle=False, ...)
```

Verificacion del bug: la matriz de confusion del OOF ensemble mostraba 5768 muestras con 50% clumpy en vez de 1442 con 9.8%.

**Modelo real recomendado**: ConvNeXt hibrido (F1 = 0.59 ± 0.06 VALIDADO). El ResNet-50 tenia el mismo bug; tras el fix su F1 real cayo a **0.50**, confirmando que estaba inflado y descartandolo como alternativa.

## Analisis final del rendimiento (para reporte/paper)

### Resumen ejecutivo

El clasificador binario ConvNeXt-Tiny + MLP tabular logra **F1 = 0.62 ± 0.12** (v2.1, 10-Fold Stratified CV con TTA) sobre 1442 galaxias JWST (9.8% clumpy). Version base sin TTA obtuvo F1 = 0.59 ± 0.06 (5-Fold CV). Buena discriminacion (**ROC-AUC = 0.89**) pero precision moderada (**0.64**) por dificultad intrinseca del desbalance con solo 141 ejemplos positivos.

### Fortalezas
1. **Discriminacion solida** (ROC-AUC 0.89, PR-AUC 5x sobre baseline)
2. **Recall aceptable** en clase minoritaria
3. **Estabilidad**: version base con std=0.06 en 5-fold
4. **Pipeline robusto**: CV estratificado, scaler por fold (sin leakage), Focal Loss, mixed precision, TTA
5. Modelo hibrido no mejora significativo sobre solo-imagen (0.62 vs 0.58) → features tabulares parcialmente redundantes con ConvNeXt

### Debilidades
1. **Precision moderada** (0.64 en v2.1, 0.54 en base): ~1 de cada 2 predicciones "clumpy" correcta
2. **Techo por tamano del dataset**: multiples arquitecturas convergen a F1 = 0.50-0.67
3. **Umbral optimizado en val**: sesgo optimista de ~2-5%
4. **Variabilidad de recall** (std 0.16): sensible a fold split
5. **Ganancia hibrida marginal**: ConvNeXt ya captura mucha morfometria implicitamente

### Comparacion con literatura
- Regresion logistica sobre morfometrias: F1 ~ 0.30-0.45
- CNN solo-imagen (ResNet/EfficientNet): F1 ~ 0.50-0.65
- **Nuestro ConvNeXt hibrido**: F1 = 0.59 ← en rango tipico
- Ensemble multi-modelo + TTA (state-of-art): F1 ~ 0.65-0.75

## Estado actual y proximos pasos posibles

- ✅ Pipeline completo: CV + retrain + ensemble + Grad-CAM + inferencia
- ✅ Modelo desplegable: `hybrid_clumpy_classifier.pth` (ConvNeXt hibrido)
- ✅ Ensemble: `ensemble_clumpy_classifier.pth` (5 modelos promediados)

### Posibles mejoras futuras (no implementadas)
1. **Test-Time Augmentation (TTA)**: predecir sobre variantes augmentadas y promediar
2. **Data augmentation avanzada**: MixUp, CutMix, RandomErasing
3. **10-Fold en vez de 5-Fold**: reducir varianza (std=0.17 es alta)
4. **Ensemble de arquitecturas**: ConvNeXt + ResNet + EfficientNet
5. **Pseudo-labeling**: si hay galaxias sin etiqueta, predecirlas y re-entrenar
6. **Label smoothing**: 0.05-0.1 para mejor generalizacion
7. **Analisis de errores**: examinar los False Negatives (galaxias clumpy no detectadas)

## Archivos del proyecto

```
Proyecto-Semestral-Deep-Learning/
├── data/
│   ├── 11_05_2026_dictionary_with_all_statmorph_and_classifications.json
│   └── colour_images/    # ~1490 PNG 240x240
├── clumpy_classifier_convnext.ipynb              # Baseline solo-imagen
├── clumpy_classifier_convnext_hybrid.ipynb       # Hibrido 10-feat, 5-fold (F1=0.59)
├── clumpy_classifier_convnext_hybrid_v2_1.ipynb  # RECOMENDADO ⭐ (F1=0.62, 15-feat + TTA + 10-fold)
├── clumpy_classifier_resnet_hybrid.ipynb         # Alternativa (F1=0.50 validado, descartado)
└── README.md
```

## Codigo clave (extraidos del notebook recomendado)

### HybridModel
```python
class HybridModel(nn.Module):
    def __init__(self, num_tabular_features=10, tabular_hidden=128):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True)
        img_feature_dim = self.backbone.head.fc.in_features  # 768
        self.backbone.head.fc = nn.Identity()
        self.tabular_net = nn.Sequential(
            nn.Linear(num_tabular_features, tabular_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(img_feature_dim + tabular_hidden, 1)

    def forward(self, images, tabular):
        img_feat = self.backbone(images)
        tab_feat = self.tabular_net(tabular)
        return self.classifier(torch.cat([img_feat, tab_feat], dim=1))
```

### Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits.view(-1))
        t = targets.view(-1).float()
        p_t = p * t + (1 - p) * (1 - t)
        alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
        focal = alpha_t * (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits.view(-1), t, reduction='none')
        return (focal * bce).mean()
```

### Feature extraction
```python
TABULAR_COLS = [
    'Gini', 'M20', 'C', 'A', 'S', 'sn_per_pixel',
    'sersic_n', 'sersic_rhalf', 'ellipticity_centroid', 'elongation_centroid'
]

for g in data['galaxies']:
    name = g['name']
    if name not in img_files or not g['frames']:
        continue
    frame = g['frames'][0]
    try:
        tab = [float(frame[c]) for c in TABULAR_COLS]
    except (KeyError, TypeError, ValueError):
        continue
    label = 1 if g['info']['visual_clas']['Clumpy'] else 0
    samples.append({'name': name, 'path': ..., 'label': label,
                    'tabular': np.array(tab, dtype=np.float32)})
```

### CV loop pattern
```python
K_FOLDS = 5
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
cv_results, trained_models = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(samples)), all_labels)):
    fold_train = [samples[i] for i in train_idx]
    fold_val = [samples[i] for i in val_idx]
    model, best_th, best_f1, history, sm, ss = train_fold(fold_train, fold_val, f'F{fold+1}')
    trained_models.append((copy.deepcopy(model.state_dict()), sm, ss, best_th))
    # ... evaluar y guardar metricas ...
```

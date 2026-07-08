# Figuras comparativas del poster

Este directorio debe contener **4 imágenes PNG comparativas** generadas desde los 3 notebooks de modelos.

## Imágenes requeridas

| Archivo | Descripción | Origen |
|---|---|---|
| `comparativa_loss_curves.png` | Curvas de loss (train/val) para los 3 modelos | Generar con script comparativo |
| `comparativa_roc_curves.png` | Curvas ROC superpuestas para los 3 modelos | Generar con script comparativo |
| `comparativa_pr_curves.png` | Curvas Precision-Recall superpuestas | Generar con script comparativo |
| `comparativa_confusion_matrices.png` | 3 matrices de confusión (una por modelo) | Generar con script comparativo |

## Figuras individuales existentes (referencia)

Cada notebook genera sus propias figuras individuales:

### ConvNeXt solo imagen (`clumpy_classifier_convnext.ipynb`)
- `cv_metrics.png` — barras de métricas por fold
- `cv_curves.png` — ROC, PR, matriz de confusión
- `gradcam_examples.png` — mapas de calor Grad-CAM

### ConvNeXt híbrido v2.1 (`clumpy_classifier_convnext_hybrid_v2_1.ipynb`)
- `v2_1_cv_metrics.png` — barras de métricas por fold
- `v2_1_cv_curves.png` — ROC, PR, matriz de confusión

### ResNet-50 híbrido (`clumpy_classifier_resnet_hybrid.ipynb`)
- `resnet_cv_metrics.png` — barras de métricas por fold
- `resnet_cv_curves.png` — ROC, PR, matriz de confusión

## Cómo generar las figuras comparativas

### Opción A: Script Python comparativo (recomendado)

Crear un script `generar_comparativas.py` que:
1. Cargue los resultados de los 3 notebooks (OOF predictions, ROC/PR curves)
2. Genere las 4 figuras comparativas con matplotlib
3. Guarde en `poster/figs/`

Estructura esperada:
```python
import matplotlib.pyplot as plt
import numpy as np

# Cargar predicciones OOF de cada modelo
# (extraer de los notebooks o de archivos .npy guardados)

# 1. Loss curves
fig, ax = plt.subplots()
ax.plot(epochs, convnext_val_loss, label='ConvNeXt solo imagen')
ax.plot(epochs, hybrid_val_loss, label='ConvNeXt híbrido v2.1')
ax.plot(epochs, resnet_val_loss, label='ResNet-50 híbrido')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.legend()
plt.savefig('figs/comparativa_loss_curves.png', dpi=150)

# 2. ROC curves
fig, ax = plt.subplots()
ax.plot(fpr_convnext, tpr_convnext, label=f'ConvNeXt solo (AUC=0.90)')
ax.plot(fpr_hybrid, tpr_hybrid, label=f'ConvNeXt híbrido (AUC=0.89)')
ax.plot(fpr_resnet, tpr_resnet, label=f'ResNet-50 (AUC=0.86)')
ax.plot([0,1], [0,1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
plt.savefig('figs/comparativa_roc_curves.png', dpi=150)

# 3. PR curves
fig, ax = plt.subplots()
ax.plot(recall_convnext, precision_convnext, label=f'ConvNeXt solo (AP=0.49)')
ax.plot(recall_hybrid, precision_hybrid, label=f'ConvNeXt híbrido (AP=0.57)')
ax.plot(recall_resnet, precision_resnet, label=f'ResNet-50 (AP=0.38)')
ax.axhline(y=0.098, color='k', linestyle='--', alpha=0.5, label='Baseline (9.8%)')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend()
plt.savefig('figs/comparativa_pr_curves.png', dpi=150)

# 4. Confusion matrices (3 subplots)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Plot each confusion matrix
plt.savefig('figs/comparativa_confusion_matrices.png', dpi=150)
```

### Opción B: Extraer de los notebooks existentes

Si los notebooks ya tienen las figuras individuales, puedes:
1. Abrir los 3 notebooks en Colab
2. Extraer las curvas ROC/PR de cada `*_cv_curves.png`
3. Combinarlas manualmente en una figura comparativa con matplotlib

### Opción C: Placeholder temporal

Si aún no tienes las figuras comparativas, puedes compilar el poster con placeholders:

```powershell
magick -size 1200x800 xc:lightgray -pointsize 40 -gravity center -annotate 0 "comparativa_loss_curves.png" figs/comparativa_loss_curves.png
magick -size 1200x800 xc:lightgray -pointsize 40 -gravity center -annotate 0 "comparativa_roc_curves.png" figs/comparativa_roc_curves.png
magick -size 1200x800 xc:lightgray -pointsize 40 -gravity center -annotate 0 "comparativa_pr_curves.png" figs/comparativa_pr_curves.png
magick -size 1500x500 xc:lightgray -pointsize 40 -gravity center -annotate 0 "comparativa_confusion_matrices.png" figs/comparativa_confusion_matrices.png
```

## Paleta de colores para las figuras

Usar los mismos colores del poster para consistencia:
- **ConvNeXt solo imagen**: `#4FC3F7` (astrocyan)
- **ConvNeXt híbrido v2.1**: `#FFD700` (astrogold) — modelo destacado
- **ResNet-50 híbrido**: `#E63946` (clumpyred)

## Notas

- El poster comparativo **no usa** las figuras individuales `v2_1_cv_metrics.png` ni `v2_1_cv_curves.png`
- Todas las figuras deben ser generadas desde las predicciones OOF (out-of-fold) de cada modelo
- Las métricas en las figuras deben coincidir con la tabla del poster:
  - ConvNeXt solo: F1=0.58, ROC-AUC=0.90, PR-AUC=0.49
  - ConvNeXt híbrido v2.1: F1=0.62, ROC-AUC=0.89, PR-AUC=0.57
  - ResNet-50: F1=0.50, ROC-AUC=0.86, PR-AUC=0.38

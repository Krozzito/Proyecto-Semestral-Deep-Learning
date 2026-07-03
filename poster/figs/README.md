# Placeholders para imagenes del poster

Este directorio debe contener **2 imagenes PNG** generadas desde el notebook
`clumpy_classifier_convnext_hybrid_v2_1.ipynb`.

## Imagenes esperadas

| Archivo | Origen en Colab (Drive) | Descripcion |
|---|---|---|
| `cv_metrics.png` | `v2_1_cv_metrics.png` (celda 11 del notebook) | Bars de F1/Prec/Rec/AUC/PR-AUC/threshold por fold |
| `cv_curves.png`  | `v2_1_cv_curves.png` (celda 11 del notebook) | 3 graficos: ROC, PR, matriz confusion |

## Como obtenerlas

Tras ejecutar el notebook completo en Colab, las imagenes se guardan
automaticamente en Google Drive:

```
/content/drive/MyDrive/ProyectoClumps/v2_1_cv_metrics.png
/content/drive/MyDrive/ProyectoClumps/v2_1_cv_curves.png
```

Descargalas del Drive, **renombralas** como en la tabla y colocalas aqui
en `poster/figs/`:

```
poster/figs/cv_metrics.png
poster/figs/cv_curves.png
```

## Nota

El notebook v2.1 **no tiene celda de Grad-CAM** (a diferencia del hibrido base),
por lo que el poster no incluye seccion de mapas de calor. Si quieres agregar
Grad-CAM al poster, puedes generar los mapas ejecutando la celda
correspondiente del notebook `clumpy_classifier_convnext_hybrid.ipynb`
(v1 base) sobre el modelo v2.1 cargado, o dejarlo como trabajo futuro.

## Placeholder temporal

Si aun no tienes las imagenes y quieres compilar el poster para ver el layout,
puedes:

1. **Comentar temporalmente** las lineas `\includegraphics{...}` en `poster.tex`
2. **Crear placeholders vacios** con ImageMagick:

```powershell
magick -size 1200x800 xc:lightgray -pointsize 40 -gravity center -annotate 0 "cv_metrics.png" figs/cv_metrics.png
magick -size 1800x600 xc:lightgray -pointsize 40 -gravity center -annotate 0 "cv_curves.png"  figs/cv_curves.png
```

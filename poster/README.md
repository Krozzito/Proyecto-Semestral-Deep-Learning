# Poster A0 vertical - Comparativa de Arquitecturas para Clasificación de Galaxias Clumpy

Poster científico en LaTeX (formato A0 vertical, 841 x 1189 mm) para la defensa del proyecto semestral de Deep Learning.

**Enfoque:** Comparativa empírica de 3 arquitecturas:
1. ConvNeXt-Tiny (solo imagen)
2. ConvNeXt-Tiny + MLP híbrido v2.1 (recomendado)
3. ResNet-50 + MLP híbrido

**Modelo destacado:** ConvNeXt híbrido v2.1 con **F1 = 0.62 ± 0.12**, ROC-AUC = 0.89

## Archivos

- `poster.tex` — código fuente principal LaTeX (versión comparativa)
- `figs/` — carpeta con imágenes PNG comparativas (ver `figs/README.md`)
- `README.md` — este archivo

## Antes de compilar: preparativos

### 1. Preparar imágenes comparativas (4 archivos)

Generar manualmente las siguientes figuras y colocarlas en `figs/`:

| Archivo | Descripción |
|---|---|
| `comparativa_loss_curves.png` | Curvas de loss train/val para los 3 modelos |
| `comparativa_roc_curves.png` | Curvas ROC superpuestas (AUC: 0.90, 0.89, 0.86) |
| `comparativa_pr_curves.png` | Curvas Precision-Recall superpuestas (AP: 0.49, 0.57, 0.38) |
| `comparativa_confusion_matrices.png` | 3 matrices de confusión (una por modelo) |

Ver `figs/README.md` para detalles sobre cómo generarlas.

### 2. Personalizar campos

Editar `poster.tex` y reemplazar los placeholders:

```latex
Guía: [NOMBRE DEL GUÍA]       → nombre del guía/asesor
```

El autor ya está completado: Daniel Ignacio Soto Salgado

### 3. Agregar logo de la universidad (opcional)

Colocar el logo en `figs/logo_universidad.png` y descomentar la línea en `\titlegraphic{}`:

```latex
\titlegraphic{
    \includegraphics[width=8cm]{figs/logo_universidad.png}
}
```

### 4. Verificar URL del QR code

En la sección "Código, Datos y Reproducibilidad", la URL del QR apunta a:
```latex
\qrcode[height=5.5cm]{https://github.com/Krozzito/Proyecto-Semestral-Deep-Learning}
```

Ajustar si tu repo tiene otro nombre/usuario.

## Cómo compilar

### Opción A: Overleaf (recomendado, sin instalar nada)

1. Ir a https://www.overleaf.com y crear cuenta
2. **New Project** → **Upload Project** → subir toda la carpeta `poster/`
3. Asegurarse de que `poster.tex` sea el main document
4. Compilar con **Recompile** (usa PDFLaTeX por defecto)
5. Descargar el PDF resultante

### Opción B: Compilar local con TeX Live / MiKTeX

```bash
cd poster
pdflatex poster.tex
pdflatex poster.tex   # 2 pasadas para referencias cruzadas
```

Requiere paquetes: `tikzposter`, `qrcode`, `babel-spanish`, `siunitx`, `booktabs`, `colortbl`, `seqsplit`. Todos vienen en TeX Live full.

## Estructura visual del poster

```
+----------------------------------------------------------+
|                    TÍTULO GRANDE                         |
|            (dorado sobre azul con estrellas)             |
+---------------------+------------------------------------+
| RESUMEN COMPARATIVO |      MÉTRICAS FINALES              |
| (55% ancho)         |      Tabla F1/AUC/PRAUC x 3 modelos|
+---------------------+------------------------------------+
| INTRODUCCIÓN | MODELO 1: ConvNeXt  | MODELO 2: ConvNeXt  |
| + DATASET    | solo imagen         | híbrido v2.1 ★      |
+--------------+---------------------+---------------------+
| MODELO 3: ResNet-50 | COMPARATIVA DETALLADA | CURVAS LOSS|
| (diagrama + métricas)| (tabla 6 métricas)    | (gráfico)  |
+---------------------+----------------------+-------------+
|  RESULTADOS VISUALES (ROC + PR + Confusion Matrices)     |
|                                            | DISCUSIÓN   |
|                                            | COMPARATIVA |
+--------------------------------------------+-------------+
|    CONCLUSIONES + TRABAJO FUTURO   | REFERENCIAS         |
|                                    | + QR REPO GITHUB    |
+------------------------------------+---------------------+
```

## Contenido alineado con comparativa

El poster describe **específicamente** la comparativa de 3 modelos:

### Modelo 1: ConvNeXt-Tiny (solo imagen)
- Solo backbone convolucional, sin componentes tabulares
- 5-Fold Stratified CV
- F1 = 0.58 ± 0.07, ROC-AUC = 0.90, PR-AUC = 0.49

### Modelo 2: ConvNeXt híbrido v2.1 ⭐ (recomendado)
- ConvNeXt-Tiny + MLP con 15 features tabulares
- 10-Fold Stratified CV + TTA (4 augmentaciones)
- F1 = 0.62 ± 0.12, ROC-AUC = 0.89, PR-AUC = 0.57

### Modelo 3: ResNet-50 híbrido
- ResNet-50 + MLP con 10 features tabulares
- 5-Fold Stratified CV
- F1 = 0.50, ROC-AUC = 0.86, PR-AUC = 0.38

## Paleta de colores

- **Azul oscuro** `#0A0E27` — fondo título
- **Azul medio** `#1B2A5C` — bordes de bloques
- **Dorado** `#FFD700` — título, acentos, modelo destacado
- **Cyan** `#4FC3F7` — subtítulo, ConvNeXt solo imagen
- **Rojo** `#E63946` — clumpy, ResNet-50
- **Verde** `#2A9D8F` — no-clumpy

## Solución de problemas

**Error: `File 'tikzposter.cls' not found`**
- Overleaf lo trae por defecto. En local: `tlmgr install tikzposter`.

**Error: `File 'qrcode.sty' not found`**
- Overleaf lo trae. En local: `tlmgr install qrcode`.

**Las imágenes no aparecen**
- Verificar que las 4 figuras comparativas existan en `figs/`.
- Ver `figs/README.md` para instrucciones.

**Texto se sale de las cajas**
- Reducir tamaño de fuente base cambiando `22pt` en la primera línea a `20pt`.
- O acortar textos de los bloques.

**El poster sale muy grande al imprimir**
- Verificar que la imprenta usa formato A0 (841 x 1189 mm).
- No usar "ajustar a página" en el visor de PDF antes de imprimir.

## Impresión

- **Formato**: A0 vertical (841 x 1189 mm)
- **Resolución recomendada**: 300 DPI
- **Papel**: mate satinado (evita reflejos bajo luz del auditorio)
- **Servicios**: PhotoBox, Vistaprint, imprentas locales de plotters

## Tiempo estimado

- Preparar 4 imágenes comparativas: 30 min
- Personalizar campos: 5 min
- Primera compilación + ajustes: 30 min
- Revisión final: 30 min
- **Total: ~1.5 horas**

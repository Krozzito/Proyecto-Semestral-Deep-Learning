# Poster A0 vertical - Clasificacion de Galaxias Clumpy

Poster cientifico en LaTeX (formato A0 vertical, 841 x 1189 mm) para la
defensa del proyecto semestral de Deep Learning.

**Alineado con el notebook final:** `clumpy_classifier_convnext_hybrid_v2_1.ipynb`

Metricas destacadas: **F1 = 0.62 ± 0.12**, ROC-AUC = 0.89 (10-Fold CV + TTA + ensemble)

## Archivos

- `poster.tex` — codigo fuente principal LaTeX
- `figs/` — carpeta con imagenes PNG del notebook (ver `figs/README.md`)
- `README.md` — este archivo

## Antes de compilar: preparativos

### 1. Preparar imagenes (2 archivos)

Copiar 2 imagenes PNG a `figs/` desde Google Drive tras correr el notebook:

- `v2_1_cv_metrics.png` → renombrar a `figs/cv_metrics.png`
- `v2_1_cv_curves.png`  → renombrar a `figs/cv_curves.png`

Ver `figs/README.md` para detalles.

### 2. Personalizar campos

Editar `poster.tex` y reemplazar los placeholders:

```latex
Autor: [TU NOMBRE]            → tu nombre completo
Guia: [NOMBRE DEL GUIA]       → nombre del guia/asesor
[DEPARTAMENTO / CARRERA]      → tu departamento
[UNIVERSIDAD]                 → nombre de la universidad
```

### 3. Agregar logo de la universidad (opcional)

Colocar el logo en `figs/logo_universidad.png` y descomentar la linea en
`\titlegraphic{}`:

```latex
\titlegraphic{
    \includegraphics[width=8cm]{figs/logo_universidad.png}
}
```

### 4. Actualizar URL del QR code

En la seccion "Codigo, Datos y Reproducibilidad", ajustar la URL si tu
repo tiene otro nombre/usuario:

```latex
\qrcode[height=6cm]{https://github.com/anomalyco/Proyecto-Semestral-Deep-Learning}
```

## Como compilar

### Opcion A: Overleaf (recomendado, sin instalar nada)

1. Ir a https://www.overleaf.com y crear cuenta gratis
2. **New Project** → **Upload Project** → subir toda la carpeta `poster/`
3. Asegurarse de que `poster.tex` sea el main document
4. Compilar con **Recompile** (usa PDFLaTeX por defecto)
5. Descargar el PDF resultante

### Opcion B: Compilar local con TeX Live / MiKTeX

```bash
cd poster
pdflatex poster.tex
pdflatex poster.tex   # 2 pasadas para referencias cruzadas
```

Requiere paquetes: `tikzposter`, `qrcode`, `babel-spanish`, `siunitx`,
`booktabs`. Todos vienen en TeX Live full.

## Estructura visual del poster

```
+----------------------------------------------------------+
|                    TITULO GRANDE                         |
|                    (dorado sobre azul)                   |
+---------------------+------------------------------------+
| RESUMEN (v2.1)      |      METRICAS FINALES              |
| (65% ancho)         |      Tabla F1/AUC/Prec/Rec         |
+---------------------+------------------------------------+
| INTRODUCCION | ARQUITECTURA (TikZ)  | RESULTADOS POR FOLD |
| DATASET      | ENTRENAMIENTO v2.1   | CURVAS ROC/PR/CM    |
+--------------+----------------------+--------------------+
|  COMPARACION DE MODELOS            |    DISCUSION        |
|  (baseline vs hibrido vs v2.1 vs   |    (fortalezas /    |
|   ResNet, con nota de bugs fixed)  |     limitaciones)   |
+------------------------------------+---------------------+
|    CONCLUSIONES + TRABAJO FUTURO   | REFERENCIAS         |
|                                    | + QR REPO GITHUB    |
+------------------------------------+---------------------+
```

## Contenido alineado con v2.1

El poster describe **especificamente v2.1** con:
- 15 features tabulares (10 originales + `F(G,M20)`, `S(G,M20)`, `rpetro_circ`, `r20`, `r80`)
- 10-Fold Stratified CV
- TTA con 4 augmentaciones
- Focal Loss sin label smoothing
- Ensemble de 10 modelos + TTA
- Sin retrain final (usa directamente el ensemble)
- Mencion explicita del bug de leakage detectado y corregido

## Paleta de colores

- **Azul oscuro** `#0A0E27` — fondo titulo
- **Azul medio** `#1B2A5C` — bordes de bloques
- **Dorado** `#FFD700` — titulo, acentos
- **Cyan** `#4FC3F7` — subtitulo
- **Rojo** `#E63946` — clumpy
- **Verde** `#2A9D8F` — no-clumpy

Para cambiar la paleta, editar el bloque de `\definecolor{...}` al inicio
de `poster.tex`.

## Solucion de problemas

**Error: `File 'tikzposter.cls' not found`**
- Overleaf lo trae por defecto. En local: `tlmgr install tikzposter`.

**Error: `File 'qrcode.sty' not found`**
- Overleaf lo trae. En local: `tlmgr install qrcode`.

**Las imagenes no aparecen**
- Verificar que `figs/cv_metrics.png` y `figs/cv_curves.png` existan.
- Ver `figs/README.md` para instrucciones.

**Texto se sale de las cajas**
- Reducir tamaño de fuente base cambiando `25pt` en la primera linea a `24pt`.
- O acortar textos de los bloques.

**El poster sale muy grande al imprimir**
- Verificar que la imprenta usa formato A0 (841 x 1189 mm).
- No usar "ajustar a pagina" en el visor de PDF antes de imprimir.

## Impresion

- **Formato**: A0 vertical (841 x 1189 mm)
- **Resolucion recomendada**: 300 DPI
- **Papel**: mate satinado (evita reflejos bajo luz del auditorio)
- **Servicios**: PhotoBox, Vistaprint, imprentas locales de plotters

## Tiempo estimado

- Preparar 2 imagenes: 10 min
- Personalizar campos: 10 min
- Primera compilacion + ajustes: 30 min
- Revision final: 30 min
- **Total: ~1.5 horas**


# Brazil Severe Weather Outlook

Sistema automático de previsão convectiva severa para o Brasil, inspirado conceitualmente em produtos probabilísticos/categóricos de centros de previsão de tempo severo, mas com identidade visual e metodologia próprias.

> **Status:** arquitetura/MVP em desenvolvimento. Nenhuma saída deste repositório deve ser tratada como produto operacional de segurança pública até que exista calibração e verificação histórica suficientes.

## Objetivo

Construir um pipeline multimodelo que ingere campos atmosféricos brutos e perfis verticais, calcula diagnósticos termodinâmicos/cinemáticos/sinóticos, estima iniciação e modo convectivo, produz probabilidades de hazards e gera outlooks em GeoJSON/mapas para o Brasil e América do Sul adjacente.

Produtos planejados:

- Categorical Convective Outlook: `TSTM`, `MRGL`, `SLGT`, `ENH`, `MDT`, `HIGH`
- Thunderstorm Probability
- Severe Weather Probability
- Tornado Probability + Significant Tornado
- Hail Probability + Significant Hail
- Wind Probability + Significant Wind
- Supercell Probability
- QLCS Probability
- Convective Initiation Probability
- Forecast Confidence
- Multi-model spread
- Discussão meteorológica técnica automática

## Princípios

O sistema não deve usar regras simples do tipo `CAPE + shear => categoria`.

A previsão deverá combinar, entre outros:

- instabilidade e CIN;
- umidade e perfis verticais;
- forcing e iniciação convectiva;
- cisalhamento, SRH e storm-relative flow;
- modo convectivo esperado;
- estrutura sinótica;
- consenso e spread de ensembles;
- climatologia regional como prior estatístico;
- calibração/verificação por eventos históricos.

Os índices compostos (STP, SCP, SHIP, EHI etc.) serão tratados como **features**, nunca como decisão isolada.

## Domínio

Processamento sugerido: `15°N–60°S, 90°W–25°W`.

A interface poderá destacar o Brasil, mas o pipeline deve processar também Argentina, Uruguai, Paraguai, Bolívia, Peru, leste do Chile, sul da Colômbia e Atlântico adjacente.

## Modelos

Arquitetura baseada em adapters:

- GFS
- GEFS
- ECMWF
- ECMWF Ensemble, quando disponível/licenciado
- ICON
- WRF (ingestão de runs existentes e, futuramente, execução local)

Cada adapter converte o formato nativo para um dataset interno padronizado.

## Stack proposta

### Backend

- Python
- FastAPI
- xarray / dask
- cfgrib / ecCodes
- MetPy
- NumPy / SciPy / pandas
- GeoPandas / Shapely / pyproj
- PostgreSQL + PostGIS
- MinIO/S3 para grids e arquivos grandes

### Frontend

- React / Next.js / TypeScript
- MapLibre GL JS ou Leaflet

## Estrutura alvo

```text
backend/
  app/
    ingestion/
    models/
    processing/
    diagnostics/
    synoptic/
    convection/
    hazards/
    ensemble/
    calibration/
    verification/
    outlook/
    api/
    database/
configs/
frontend/
workers/
tests/
docker/
```

## Pipeline operacional

```text
DOWNLOAD MODEL DATA
        ↓
VALIDATE FILES
        ↓
DECODE GRIB/NETCDF
        ↓
CROP SOUTH AMERICA
        ↓
STANDARDIZE VARIABLES
        ↓
VERTICAL PROFILE PROCESSING
        ↓
CALCULATE DIAGNOSTICS
        ↓
REGRID
        ↓
MODEL-SPECIFIC ANALYSIS
        ↓
ENSEMBLE / CONSENSUS
        ↓
CONVECTIVE INITIATION
        ↓
CONVECTIVE MODE
        ↓
HAZARD MODELS
        ↓
CALIBRATION
        ↓
PROBABILITIES
        ↓
CATEGORICAL RISK
        ↓
POLYGON GENERATION
        ↓
MAPS / GEOJSON / API
        ↓
ARCHIVE + VERIFICATION
```

## Saída GeoJSON

Os endpoints de outlook/hazard deverão retornar `FeatureCollection` com propriedades padronizadas, por exemplo:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": []
      },
      "properties": {
        "product": "tornado",
        "probability": 0.05,
        "distance_km": 40,
        "valid_start": "2026-08-29T12:00:00Z",
        "valid_end": "2026-08-30T12:00:00Z",
        "confidence": "moderate",
        "significant": false
      }
    }
  ]
}
```

## Automação

Estados planejados de cada ciclo:

- `WAITING_FOR_DATA`
- `DOWNLOADING`
- `PROCESSING`
- `GENERATING_OUTLOOK`
- `READY`
- `FAILED`

Outlooks só deverão ser publicados quando o conjunto mínimo de variáveis críticas estiver completo.

## Calibração e verificação

Separar sempre:

1. `raw_environment_score`
2. `calibrated_probability`

Métodos iniciais preferidos: regressão logística e gradient boosting, com evolução apenas quando justificada pela base histórica.

Métricas:

- Brier Score / Brier Skill Score
- ROC AUC
- reliability diagram
- POD
- FAR
- CSI
- ETS

## Licença e dados

As fontes de dados meteorológicos possuem licenças/termos próprios. O projeto deve respeitar explicitamente as condições de cada provedor, especialmente dados ECMWF.

---

O primeiro marco técnico é entregar um backend executável com adapters padronizados, configuração centralizada, GeoJSON válido, endpoints de status e um pipeline sintético de teste antes de conectar downloads operacionais de modelos.

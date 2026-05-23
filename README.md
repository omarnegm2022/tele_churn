# tele_churn
```mermaid
graph TD
    %% 1. Ingestion Layer
    subgraph Ingestion [1. Ingestion Layer]
        A[OpenCellID Towers <br/> Africa_towers.csv] -->|Ingest Raw Towers Data| C[Raw Ingestion Layer]
        B[Expresso Users <br/> expresso.csv] -->|Ingest Raw Users Data| C
    end

    %% 2. Processing Layer
    subgraph Processing [2. Processing Layer]
        C --> D[Run Processing Scripts]
        D -->|Generate Sample Dataset| E[telecom_churn_100k.csv]
        D -->|Generate Full Dataset| F[telecom_churn.csv - 2M rows]
    end

    %% 3. Modeling Layer
    subgraph Modeling [3. Modeling Layer]
        E & F --> G[Exploratory Data Analysis - EDA]
        G --> H["Train ML Pipelines (SMOTE / Target Encoding)"]
        
        %% Model Matrix Evaluation Node
        H --> Matrix{Model Selection Matrix <br/> Evaluation}
        Matrix -->|F1: 0.52 / ROC-AUC: 0.74| M1[Logistic Regression]
        Matrix -->|F1: 0.64 / ROC-AUC: 0.89| M2[Random Forest]
        Matrix -->|F1: 0.68 / ROC-AUC: 0.91| M3[XGBoost]
        Matrix -.->|F1: 0.72 / ROC-AUC: 0.93+ <br/> Champion Selected| H_Champ[LightGBM Champion]
        
        M1 & M2 & M3 -->|Discarded| Discard[Baseline Benchmarks]
        H_Champ -->|Saves Trained Artifact| I[churn_model.joblib]
        H_Champ -->|Outputs Predictions| J[churn_predictions.csv]
    end

    %% 4. Serving & Deployment Layer
    subgraph Serving [4. Serving & Deployment]
        I & J --> K[Streamlit Dashboard]
        I --> M[Flask REST API Endpoint]
        I & J --> P[Azure Dashboard]
        
        K & M & P --> R[Deploy Codebase]
        R --> L[Streamlit Cloud] 
        R --> Q[Azure Cloud]
        
        %% Real-time event simulation trigger
        N[Arrival of New Ingestion Data] -->|Airflow Event Trigger| O[Send Notification Mail]
    end

    %% Structural Aesthetics and Thematic Coloring
    style Ingestion fill:#f0f7ff,stroke:#0284c7,stroke-width:2px,rx:8px,ry:8px
    style Processing fill:#f0dfec,stroke:#b71540,stroke-width:2px,rx:8px,ry:8px
    style Modeling fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,rx:8px,ry:8px
    style Serving fill:#faf5ff,stroke:#9333ea,stroke-width:2px,rx:8px,ry:8px
    
    %% Highlight Nodes
    style H_Champ fill:#1e293b,stroke:#deff9a,stroke-width:2.5px,color:#deff9a
    style O fill:#fef08a,stroke:#ca8a04,stroke-width:2.5px,stroke-dasharray: 5 5

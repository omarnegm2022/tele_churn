# =============================================================================
# Telecom Customer Churn Prediction - Streamlit Dashboard
# =============================================================================

import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Telecom Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #0b0b12;
    color: white;
}
.kpi-card {
    background: #151522;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #2b2b40;
}
.kpi-value {
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def classify_risk(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"


def calculate_metrics(df):
    if "actual_churn" not in df.columns:
        return None

    y_true = df["actual_churn"]
    y_pred = df["churn_prediction"]
    y_prob = df["churn_probability"]

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    payload = joblib.load(path)
    return payload["model"], payload["metadata"]


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙ Dashboard Settings")

# Resolve repo root regardless of where the app was launched from.
# On Streamlit Cloud the cwd is /mount/src/<repo> and __file__ is at
# /mount/src/<repo>/dashboards/churn_dashboard_app.py. Locally, the cwd
# could be either the repo root or dashboards/.
_DASH_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DASH_ROOT)


def _first_existing(*candidates: str) -> str:
    """Return the first existing path; falls back to the last candidate."""
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]


_default_csv = _first_existing(
    # Locally — the freshly-predicted file (gitignored, may not exist on Cloud)
    os.path.join(_REPO_ROOT, "outputs", "predictions", "churn_predictions.csv"),
    # Canonical baseline committed to the repo (always exists on Cloud)
    os.path.join(_REPO_ROOT, "outputs", "predictions", "churn_predictions_notebook.csv"),
)
_default_model = _first_existing(
    os.path.join(_REPO_ROOT, "models", "churn_model.joblib"),
    os.path.join(_REPO_ROOT, "dashboards", "models", "churn_model.joblib"),
)

csv_path = st.sidebar.text_input("Predictions Path", value="./churn_predictions.csv")
model_path = st.sidebar.text_input("Model Path", value="./churn_model.joblib")

# Defensive loaders — Streamlit Cloud shows a redacted "Oh no" page when an
# uncaught exception fires during module-load. We surface the real reason here.
try:
    df = load_data(csv_path)
except FileNotFoundError as exc:
    st.error(
        f"Predictions CSV not found at `{csv_path}`.\n\n"
        f"Either commit it to the repo, or override the path in the sidebar."
    )
    st.code(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Failed to read predictions CSV: {exc}")
    st.stop()

try:
    model, metadata = load_model(model_path)
except FileNotFoundError as exc:
    st.error(
        f"Model joblib not found at `{model_path}`.\n\n"
        f"Either commit `models/churn_model.joblib` to the repo "
        f"(small enough at ~5 MB), or override the path in the sidebar."
    )
    st.code(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Failed to load model joblib: {exc}")
    st.code(str(exc))
    st.stop()

threshold = st.sidebar.slider(
    "Classification Threshold",
    0.1,
    0.9,
    float(metadata.get("threshold", 0.42)),
    0.01
)

risk_filter = st.sidebar.multiselect(
    "Risk Segment",
    options=df["risk_segment"].unique(),
    default=df["risk_segment"].unique()
)

df = df[df["risk_segment"].isin(risk_filter)].copy()
df["churn_prediction"] = (
    df["churn_probability"] >= threshold
).astype(int)

# =========================================================
# HEADER
# =========================================================
st.title("📈 Telecom Customer Churn Prediction Dashboard")
st.caption("Executive dashboard for customer churn monitoring")

# =========================================================
# KPI SECTION
# =========================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Predicted Churn", f"{df['churn_prediction'].sum():,}")
col3.metric("Retention", f"{len(df)-df['churn_prediction'].sum():,}")
col4.metric("Avg Probability", f"{df['churn_probability'].mean():.2%}")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Trends",
    "🌍 Geography",
    "🔍 Predict",
    "📈 Model"
])

# =========================================================
# TAB 1 - TRENDS
# =========================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="churn_probability",
            color="risk_segment",
            nbins=40,
            title="Churn Probability Distribution"
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        fig = px.pie(
            df,
            names="risk_segment",
            title="Risk Segment Distribution"
        )
        st.plotly_chart(fig, width='stretch')

    if "revenue" in df.columns:
        fig = px.scatter(
            df,
            x="revenue",
            y="churn_probability",
            color="risk_segment",
            title="Revenue vs Churn Risk"
        )
        st.plotly_chart(fig, width='stretch')

# =========================================================
# TAB 2 - GEOGRAPHY (Senegal regions)
# =========================================================
with tab2:
    st.subheader("Churn risk by Senegal region")

    # Find a regions GeoJSON next to the project's datasets folder.
    geojson_path = None
    for candidate in (
        "../datasets/telecom_churn/senegal_regions.geojson",
        "datasets/telecom_churn/senegal_regions.geojson",
        "/opt/airflow/project/datasets/telecom_churn/senegal_regions.geojson",
    ):
        if os.path.exists(candidate):
            geojson_path = candidate
            break

    if "region" in df.columns:
        agg = (
            df.groupby("region", as_index=False)
              .agg(
                  customers=("churn_probability", "size"),
                  avg_churn_prob=("churn_probability", "mean"),
                  predicted_churn=("churn_prediction", "sum"),
              )
        )
        agg["region_norm"] = agg["region"].astype(str).str.strip().str.upper()

        c1, c2 = st.columns([2, 1])

        with c1:
            if geojson_path:
                import json
                with open(geojson_path, "r", encoding="utf-8") as f:
                    geojson = json.load(f)
                # Senegal GeoJSONs use NAME_1 (or "name") for the region label.
                key = "properties.NAME_1"
                if geojson.get("features"):
                    props = geojson["features"][0].get("properties", {})
                    if "NAME_1" not in props:
                        for cand in ("name", "NAME_2", "NOMREG"):
                            if cand in props:
                                key = f"properties.{cand}"
                                break
                fig_map = px.choropleth_mapbox(
                    agg,
                    geojson=geojson,
                    locations="region_norm",
                    featureidkey=key,
                    color="avg_churn_prob",
                    color_continuous_scale="Reds",
                    range_color=(0, max(0.01, agg["avg_churn_prob"].max())),
                    mapbox_style="carto-positron",
                    zoom=5.5,
                    center={"lat": 14.5, "lon": -14.5},
                    opacity=0.7,
                    hover_data={
                        "region_norm": True,
                        "avg_churn_prob": ":.2%",
                        "customers": ":,",
                        "predicted_churn": ":,",
                    },
                    title="Average churn probability by region",
                )
                fig_map.update_layout(height=520, margin={"l": 0, "r": 0, "t": 40, "b": 0})
                st.plotly_chart(fig_map, width='stretch')
            else:
                fig_bar = px.bar(
                    agg.sort_values("avg_churn_prob", ascending=False),
                    x="region_norm", y="avg_churn_prob", color="avg_churn_prob",
                    color_continuous_scale="Reds",
                    title="Average churn probability by region",
                )
                st.plotly_chart(fig_bar, width='stretch')

        with c2:
            st.markdown("##### Top regions by predicted churn")
            top = agg.sort_values("predicted_churn", ascending=False).head(10)
            top["avg_churn_prob"] = top["avg_churn_prob"].map(lambda x: f"{x:.1%}")
            st.dataframe(
                top[["region_norm", "customers", "predicted_churn", "avg_churn_prob"]]
                .rename(columns={
                    "region_norm": "Region",
                    "customers": "Customers",
                    "predicted_churn": "Pred. Churn",
                    "avg_churn_prob": "Avg P(churn)",
                }),
                hide_index=True,
            )
    else:
        st.warning("`region` column missing in predictions — geography tab disabled.")


# =========================================================
# TAB 3 - PREDICTION
# =========================================================
with tab3:
    st.subheader("Single Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.header("💼 Business Features")
        montant = st.number_input("Montant (Recharge)", min_value=0.0, value=100.0, step=100.0)
        frequence_rech = st.number_input("Frequence Rech", min_value=0.0, value=1.0, step=1.0)
        # revenue = st.number_input("Revenue", min_value=0.0, value=50.0, step=100.0)
        # arpu_segment = st.number_input("ARPU Segment", min_value=0.0, value=30.0, step=100.0)
        # frequence = st.number_input("Frequence", min_value=0.0, value=1.0, step=1.0)
        data_volume = st.number_input("Data Volume", min_value=0.0, value=0.0, step=500.0)
        on_net = st.number_input("On Net", min_value=0.0, value=0.0, step=10.0)
        orange = st.number_input("Orange", min_value=0.0, value=0.0, step=10.0)
        tigo = st.number_input("Tigo", min_value=0.0, value=0.0, step=10.0)
        
        regularity = st.slider("Regularity", min_value=1.0, max_value=30.0, value=1.0, step=1.0)
        freq_top_pack = st.number_input("Freq Top Pack", min_value=0.0, value=0.0, step=1.0)

    with col2:
        st.header("🌐 Technical & Regional Features")
        region_tower_count = st.number_input("Region Tower Count", min_value=0.0, value=5.0, step=1.0)
        # region_avg_range = st.number_input("Region Avg Range", min_value=0.0, value=1200.0, step=100.0)
        region_avg_samples = st.number_input("Region Avg Samples", min_value=0.0, value=350.0, step=50.0)
        # region_coverage_index = st.number_input("Region Coverage Index", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
        
        # region_network_quality_score = st.number_input("Region Network Quality Score", min_value=0.0, max_value=1.0, value=0.10, step=0.05)
        # arr_network_quality_score = st.number_input("Arrondissement Network Quality Score", min_value=0.0, max_value=1.0, value=0.85, step=0.05)

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        try:
            # 2. تحويل البيانات المدخلة مباشرة إلى DataFrame
            input_dict = {
                
                "montant": float(montant), #remain it at 100.0, as either direction of change it increases the risk.
                "frequence_rech": float(frequence_rech), 
                #NO effect
                "revenue": float(50.0),            "arpu_segment": float(30.0), "frequence": float(1.0), 
                
                "data_volume": float(data_volume),#Increase the risk
                "on_net": float(on_net), #Decreases the risk
                "orange": float(orange), # >=120.0  increases the risk by 0.68 once.
                "tigo": float(tigo), # above 0.0, increases the risk by 0.68 once.
                "regularity": float(regularity),
                "freq_top_pack": float(freq_top_pack), # above 1.0, increases the risk by 0.68 once.
                
                "region_tower_count": float(region_tower_count),#TODO: inspect its strange effect.
                
                #NO effect
                "region_avg_range": float(1200.0), "region_coverage_index": float(0.45), 
                "region_network_quality_score": float(0.10), "arr_network_quality_score": float(0.85),
               
                "region_avg_samples": float(region_avg_samples)#At 0.0, increases the risk by 0.68
                
            }
            df = pd.DataFrame([input_dict])
            
            # 3. حساب الـ Feature Engineering داخلياً
            df['log_montant'] = np.log1p(df['montant'])
            df['log_revenue'] = np.log1p(df['revenue'])
            df['log_data_volume'] = np.log1p(df['data_volume'])
            df['log_on_net'] = np.log1p(df['on_net'])
            df['log_orange'] = np.log1p(df['orange'])
            df['log_tigo'] = np.log1p(df['tigo'])
            df['log_freq_top_pack'] = np.log1p(df['freq_top_pack'])
            
            df['avg_recharge_amount'] = np.where(df['frequence_rech'] <= 0, 0.0, df['montant'] / df['frequence_rech'])
            df['log_avg_recharge_amount'] = np.log1p(df['avg_recharge_amount'])
            
            #NOTE: revenue has NO effect on the prediction.
            df['avg_revenue_per_tx'] = np.where(df['frequence'] <= 0, 0.0, df['revenue'] / df['frequence'])
            df['log_avg_revenue_per_tx'] = np.log1p(df['avg_revenue_per_tx'])
            
            df['is_data_user'] = np.where(df['data_volume'] > 0, 1.0, 0.0)
            df['no_data_flag'] = np.where(df['data_volume'] <= 0, 1.0, 0.0)
            df['is_loyal'] = np.where(df['regularity'] > 15, 1.0, 0.0)
            df['engagement_score'] = (df['frequence'] + df['frequence_rech'] + df['regularity'])
            df['network_quality_delta'] = df['region_network_quality_score'] - df['arr_network_quality_score']
            
            df['top_pack'] = 'NO_PACK'
            df['region'] = 'UNKNOWN'

            # if expected_features:
            #     for col in expected_features:
            #         if col not in df.columns:
            #             df[col] = 0.0
            #     df = df[expected_features]

            # 4. التوقع الحقيقي المباشر
            probability = float(model.predict_proba(df)[0][1])
            
            st.metric(label="Churn Probability", value=f"{probability:.2%}")    
            # if probability >= 0.25:
            #     st.error("⚠️ **Prediction:** High Risk")
            #     st.warning("💡 تحذير: العميل معرض للمغادرة! نقترح تقديم عرض مخصص لحفظ العميل.")
            # else:
            #     st.success("✅ **Prediction:** Low Risk")
            #     st.info("💡 العميل مستقر حالياً، لا توجد خطورة.")
        # except:
        #     probability = 0.65
        except Exception as e:
            st.error(f"حدث خطأ أثناء الحساب الرياضي: {str(e)}")
#NOTE: MFarouk code done!
        risk = classify_risk(probability)
        st.info(f"Risk Level: {risk}")

# =========================================================
# TAB 4 - MODEL
# =========================================================
with tab4:
    metrics = calculate_metrics(df)

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['Precision']:.2%}")
        c3.metric("Recall", f"{metrics['Recall']:.2%}")
        c4.metric("F1", f"{metrics['F1']:.2%}")
        c5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2%}")

        cm = confusion_matrix(
            df["actual_churn"],
            df["churn_prediction"]
        )

        fig_cm = px.imshow(cm, text_auto=True)
        st.plotly_chart(fig_cm, width='stretch')

        fpr, tpr, _ = roc_curve(
            df["actual_churn"],
            df["churn_probability"]
        )

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr))
        fig_roc.update_layout(title="ROC Curve")

        st.plotly_chart(fig_roc, width='stretch')

st.caption("Telecom Customer Churn Dashboard")

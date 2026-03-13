"""VP-SYS - Brain Analytics

Módulo de inteligência preditiva para Firestore.

Objetivo:
- Ler dados das coleções: vendas, produtos, assistencia
- Prever risco de esgotamento de estoque por item
- Publicar insights na coleção: insights_preditivos

Dependências:
- firebase-admin
- pandas
- scikit-learn

Uso:
    python brain_analytics.py

Pré-requisitos de autenticação:
- Defina FIREBASE_SERVICE_ACCOUNT_PATH com o caminho do JSON da service account
  ou use GOOGLE_APPLICATION_CREDENTIALS.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin
from sklearn.linear_model import LinearRegression


# ---------------------------- Configurações ----------------------------
LOOKBACK_DAYS = 90
COLLECTION_VENDAS = "vendas"
COLLECTION_PRODUTOS = "produtos"
COLLECTION_ASSISTENCIA = "assistencia"
COLLECTION_INSIGHTS = "insights_preditivos"


ITEM_CANDIDATES = ["item_nome", "item", "produto", "nome", "nome_produto", "descricao"]
QTY_CANDIDATES = ["quantidade", "qtd", "qty", "consumo", "unidades"]
DATE_CANDIDATES = ["data", "data_venda", "created_at", "timestamp", "data_movimento"]
STOCK_CANDIDATES = ["estoque_atual", "estoque", "saldo", "quantidade_estoque", "qtd_estoque"]


@dataclass
class Insight:
    item_nome: str
    data_estimada_esgotamento: Optional[str]
    nivel_de_urgencia: str
    sugestao_de_compra: str
    dias_restantes_estimados: Optional[int]
    consumo_medio_diario_90d: float


# ---------------------------- Firebase ----------------------------
def init_firestore_client() -> firestore.Client:
    if not firebase_admin._apps:
        cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            initialize_app(cred)
        else:
            # Fallback para ADC se ambiente já estiver configurado
            initialize_app()
    return firestore.client()


def read_collection(db: firestore.Client, collection_name: str) -> pd.DataFrame:
    docs = list(db.collection(collection_name).stream())
    rows = []
    for doc in docs:
        data = doc.to_dict() or {}
        data["_doc_id"] = doc.id
        rows.append(data)
    return pd.DataFrame(rows)


# ---------------------------- Utilidades de parsing ----------------------------
def _pick_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return value.strip("-")[:100] or "item-sem-nome"


# ---------------------------- Engenharia de dados ----------------------------
def build_consumption_daily(
    vendas_df: pd.DataFrame,
    assistencia_df: pd.DataFrame,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=lookback_days - 1)

    def normalize_consumption(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["item_nome", "data", "consumo"])

        item_col = _pick_existing_column(df, ITEM_CANDIDATES)
        qty_col = _pick_existing_column(df, QTY_CANDIDATES)
        date_col = _pick_existing_column(df, DATE_CANDIDATES)

        if not item_col or not qty_col or not date_col:
            return pd.DataFrame(columns=["item_nome", "data", "consumo"])

        normalized = pd.DataFrame()
        normalized["item_nome"] = df[item_col].astype(str).str.strip()
        normalized["consumo"] = _safe_numeric(df[qty_col]).abs()
        normalized["data"] = _to_datetime_utc(df[date_col]).dt.date

        normalized = normalized.dropna(subset=["data"])
        normalized = normalized[normalized["item_nome"] != ""]
        normalized = normalized[normalized["data"] >= start_date]
        normalized = normalized[normalized["data"] <= today]
        return normalized

    vendas_norm = normalize_consumption(vendas_df)
    assist_norm = normalize_consumption(assistencia_df)

    merged = pd.concat([vendas_norm, assist_norm], ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=["item_nome", "data", "consumo_dia"])

    grouped = (
        merged.groupby(["item_nome", "data"], as_index=False)["consumo"]
        .sum()
        .rename(columns={"consumo": "consumo_dia"})
    )

    # Garantir série diária completa por item para os últimos N dias
    all_days = pd.date_range(start=start_date, end=today, freq="D").date
    out_frames = []
    for item in grouped["item_nome"].unique():
        base = pd.DataFrame({"data": all_days})
        item_data = grouped[grouped["item_nome"] == item][["data", "consumo_dia"]]
        filled = base.merge(item_data, on="data", how="left").fillna({"consumo_dia": 0.0})
        filled["item_nome"] = item
        out_frames.append(filled)

    return pd.concat(out_frames, ignore_index=True)[["item_nome", "data", "consumo_dia"]]


def build_stock_snapshot(produtos_df: pd.DataFrame) -> pd.DataFrame:
    if produtos_df.empty:
        return pd.DataFrame(columns=["item_nome", "estoque_atual"])

    item_col = _pick_existing_column(produtos_df, ITEM_CANDIDATES)
    stock_col = _pick_existing_column(produtos_df, STOCK_CANDIDATES)

    if not item_col or not stock_col:
        return pd.DataFrame(columns=["item_nome", "estoque_atual"])

    stock = pd.DataFrame()
    stock["item_nome"] = produtos_df[item_col].astype(str).str.strip()
    stock["estoque_atual"] = _safe_numeric(produtos_df[stock_col]).clip(lower=0)
    stock = stock[stock["item_nome"] != ""]

    # Se houver duplicatas de produto, soma estoque
    stock = stock.groupby("item_nome", as_index=False)["estoque_atual"].sum()
    return stock


# ---------------------------- Modelagem ----------------------------
def _urgency_from_days(days_left: Optional[int]) -> str:
    if days_left is None:
        return "Baixo"
    if days_left <= 15:
        return "Crítico"
    if days_left <= 45:
        return "Médio"
    return "Baixo"


def _suggest_purchase(avg_daily: float, urgency: str) -> str:
    if avg_daily <= 0:
        return "Sem consumo relevante nos últimos 90 dias; revisar item antes de comprar."

    coverage_days = 30
    if urgency == "Crítico":
        coverage_days = 60
    elif urgency == "Médio":
        coverage_days = 45

    qty = max(1, math.ceil(avg_daily * coverage_days))
    return f"Comprar aproximadamente {qty} unidades para cobertura de {coverage_days} dias."


def predict_stockout_for_item(item_name: str, estoque_atual: float, consumo_diario: pd.Series) -> Insight:
    consumo = consumo_diario.fillna(0.0).astype(float)
    avg_daily = float(consumo.mean())  # média dos últimos 90 dias (com dias zerados)

    # Estimar histórico de estoque para regressão linear
    # série ordenada do mais antigo -> mais recente
    future_consumption = consumo.iloc[::-1].cumsum().iloc[::-1] - consumo
    estoque_historico = estoque_atual + future_consumption.values

    X = pd.DataFrame({"dia": range(len(consumo))})
    y = pd.Series(estoque_historico)

    model = LinearRegression()
    model.fit(X, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    days_left = None
    if slope < 0:
        x_zero = -intercept / slope
        days_left_reg = int(math.ceil(x_zero - (len(consumo) - 1)))
        days_left_reg = max(0, days_left_reg)

        # ancora na média de consumo dos últimos 90 dias (requisito)
        if avg_daily > 0:
            days_left_avg = int(math.ceil(estoque_atual / avg_daily))
            days_left = max(0, int(round((days_left_reg + days_left_avg) / 2)))
        else:
            days_left = days_left_reg
    elif avg_daily > 0:
        days_left = int(math.ceil(estoque_atual / avg_daily))

    if days_left is not None:
        depletion_date = (datetime.now(timezone.utc).date() + timedelta(days=days_left)).isoformat()
    else:
        depletion_date = None

    urgency = _urgency_from_days(days_left)
    suggestion = _suggest_purchase(avg_daily, urgency)

    return Insight(
        item_nome=item_name,
        data_estimada_esgotamento=depletion_date,
        nivel_de_urgencia=urgency,
        sugestao_de_compra=suggestion,
        dias_restantes_estimados=days_left,
        consumo_medio_diario_90d=round(avg_daily, 4),
    )


def generate_insights(
    produtos_df: pd.DataFrame,
    consumo_daily_df: pd.DataFrame,
) -> list[Insight]:
    estoque_df = build_stock_snapshot(produtos_df)
    if estoque_df.empty:
        return []

    insights: list[Insight] = []
    for _, row in estoque_df.iterrows():
        item = row["item_nome"]
        estoque_atual = float(row["estoque_atual"])

        serie = consumo_daily_df[consumo_daily_df["item_nome"] == item]["consumo_dia"]
        if serie.empty:
            serie = pd.Series([0.0] * LOOKBACK_DAYS)

        insight = predict_stockout_for_item(item, estoque_atual, serie)
        insights.append(insight)

    return insights


# ---------------------------- Persistência Firestore ----------------------------
def write_insights(db: firestore.Client, insights: list[Insight]) -> int:
    batch = db.batch()
    count = 0

    for insight in insights:
        doc_id = _slug(insight.item_nome)
        ref = db.collection(COLLECTION_INSIGHTS).document(doc_id)
        payload = {
            "item_nome": insight.item_nome,
            "data_estimada_esgotamento": insight.data_estimada_esgotamento,
            "nivel_de_urgencia": insight.nivel_de_urgencia,
            "sugestao_de_compra": insight.sugestao_de_compra,
            "dias_restantes_estimados": insight.dias_restantes_estimados,
            "consumo_medio_diario_90d": insight.consumo_medio_diario_90d,
            "atualizado_em": firestore.SERVER_TIMESTAMP,
            "modelo": "linear_regression_90d",
        }
        batch.set(ref, payload, merge=True)
        count += 1

    if count:
        batch.commit()
    return count


# ---------------------------- Pipeline principal ----------------------------
def run_pipeline() -> dict:
    db = init_firestore_client()

    vendas_df = read_collection(db, COLLECTION_VENDAS)
    produtos_df = read_collection(db, COLLECTION_PRODUTOS)
    assist_df = read_collection(db, COLLECTION_ASSISTENCIA)

    consumo_daily = build_consumption_daily(vendas_df, assist_df, lookback_days=LOOKBACK_DAYS)
    insights = generate_insights(produtos_df, consumo_daily)
    written = write_insights(db, insights)

    # Tendência de faturamento (resumo para log/expansão futura)
    faturamento_info = {}
    if not vendas_df.empty:
        date_col = _pick_existing_column(vendas_df, DATE_CANDIDATES)
        total_col = _pick_existing_column(vendas_df, ["valor_total", "faturamento", "valor", "total"])
        if date_col and total_col:
            tmp = vendas_df[[date_col, total_col]].copy()
            tmp["data"] = _to_datetime_utc(tmp[date_col]).dt.date
            tmp["valor"] = _safe_numeric(tmp[total_col])
            tmp = tmp.dropna(subset=["data"])
            últimos_90 = datetime.now(timezone.utc).date() - timedelta(days=89)
            tmp = tmp[tmp["data"] >= últimos_90]
            if not tmp.empty:
                daily = tmp.groupby("data", as_index=False)["valor"].sum().sort_values("data")
                X = pd.DataFrame({"dia": range(len(daily))})
                y = daily["valor"].values
                model = LinearRegression().fit(X, y)
                slope = float(model.coef_[0])
                tendencia = "Alta" if slope > 0 else ("Queda" if slope < 0 else "Estável")
                faturamento_info = {
                    "tendencia_faturamento_90d": tendencia,
                    "inclinacao_diaria": round(slope, 4),
                }

    return {
        "colecoes_lidas": {
            "vendas": len(vendas_df),
            "produtos": len(produtos_df),
            "assistencia": len(assist_df),
        },
        "insights_gravados": written,
        "faturamento": faturamento_info,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("Pipeline concluído:")
    print(result)

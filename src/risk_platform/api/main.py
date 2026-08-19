from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from risk_platform.api.dependencies import verify_api_key
from risk_platform.audit.repository import AuditRepository
from risk_platform.config import settings
from risk_platform.explainability.shap_explainer import Explainer
from risk_platform.logging_config import configure_logging
from risk_platform.models.predictor import Predictor
from risk_platform.monitoring.metrics import (
    MODEL_READY,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTIONS,
    RISK_SCORE,
)
from risk_platform.monitoring.tracing import configure_tracing
from risk_platform.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    Driver,
    PredictionRequest,
    PredictionResponse,
)

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise AI Risk Decision Platform",
    version="0.1.0",
    description="Governed reference service for real-time and batch ML risk decisions.",
)

configure_tracing(app)

_predictor: Predictor | None = None
_explainer: Explainer | None = None
_audit: AuditRepository | None = None


def _ensure_runtime() -> tuple[Predictor, Explainer, AuditRepository]:
    global _predictor, _explainer, _audit
    if _predictor is None:
        path = Path(settings.model_path)
        if not path.exists():
            MODEL_READY.set(0)
            raise HTTPException(status_code=503, detail=f"Model artifact not found: {path}")
        _predictor = Predictor(path)
        _explainer = Explainer(_predictor)
        MODEL_READY.set(1)
    if _audit is None:
        _audit = AuditRepository(settings.database_url)
    return _predictor, _explainer, _audit


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Enterprise AI Risk Decision Platform</title>
  <style>
    :root { --ink:#10213a; --muted:#64748b; --line:#dbe3ed; --panel:#f8fafc; --accent:#2563eb; --good:#047857; --warn:#b45309; --bad:#b91c1c; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:#fff; }
    header { border-bottom:1px solid var(--line); padding:32px 24px 24px; }
    .wrap { max-width:1180px; margin:auto; }
    .eyebrow { color:var(--accent); font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:12px; }
    h1 { margin:8px 0 8px; font-size:clamp(28px,4vw,46px); letter-spacing:-.035em; }
    .sub { color:var(--muted); max-width:850px; font-size:16px; line-height:1.6; }
    main { padding:28px 24px 60px; }
    .grid { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(320px,.85fr); gap:24px; }
    .card { border:1px solid var(--line); border-radius:18px; padding:22px; background:#fff; box-shadow:0 8px 30px rgba(15,23,42,.05); }
    .card h2 { margin:0 0 16px; font-size:20px; }
    .form-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
    label { display:block; color:#334155; font-size:12px; font-weight:700; margin-bottom:6px; }
    input { width:100%; border:1px solid #cbd5e1; border-radius:10px; padding:10px 11px; font-size:14px; color:var(--ink); }
    input:focus { outline:2px solid #bfdbfe; border-color:var(--accent); }
    button { margin-top:18px; border:0; border-radius:11px; padding:12px 17px; background:var(--accent); color:white; font-weight:800; cursor:pointer; font-size:14px; }
    button:disabled { opacity:.55; cursor:wait; }
    .result-top { display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .prob { font-size:46px; font-weight:850; letter-spacing:-.04em; margin:8px 0 0; }
    .badge { padding:7px 10px; border-radius:999px; font-size:12px; font-weight:800; background:#e2e8f0; color:#334155; }
    .bar { height:12px; background:#e2e8f0; border-radius:999px; overflow:hidden; margin:16px 0 20px; }
    .bar > div { height:100%; width:0%; background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444); transition:width .35s ease; }
    .meta { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:20px; }
    .meta div { background:var(--panel); padding:10px 12px; border-radius:10px; }
    .meta span { display:block; font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
    .meta strong { display:block; margin-top:4px; font-size:14px; }
    .drivers { display:grid; gap:8px; }
    .driver { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:8px; align-items:center; padding:10px 12px; border:1px solid var(--line); border-radius:10px; }
    .driver small { color:var(--muted); }
    .positive { color:var(--bad); font-weight:800; }
    .negative { color:var(--good); font-weight:800; }
    .status { margin-top:16px; font-size:13px; color:var(--muted); min-height:20px; }
    .architecture { margin-top:24px; display:grid; grid-template-columns:repeat(5,1fr); gap:8px; }
    .architecture div { text-align:center; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 6px; font-size:11px; font-weight:700; }
    footer { margin-top:24px; color:var(--muted); font-size:12px; line-height:1.5; }
    footer a { color:var(--accent); }
    @media(max-width:850px){ .grid{grid-template-columns:1fr}.architecture{grid-template-columns:repeat(2,1fr)} }
    @media(max-width:520px){ .form-grid,.meta{grid-template-columns:1fr} }
  </style>
</head>
<body>
<header><div class="wrap">
  <div class="eyebrow">Production ML / MLOps portfolio demo</div>
  <h1>Enterprise AI Risk Decision Platform</h1>
  <div class="sub">Interactive risk scoring with XGBoost, native Tree-SHAP explanations, FastAPI serving, input validation, Prometheus-compatible metrics and auditable decision logging. All data shown here are synthetic.</div>
</div></header>
<main><div class="wrap">
  <div class="grid">
    <section class="card">
      <h2>Transaction risk input</h2>
      <div class="form-grid" id="form"></div>
      <button id="score">Score transaction</button>
      <div class="status" id="status">Ready.</div>
    </section>
    <section class="card">
      <div class="result-top"><h2>Decision</h2><div class="badge" id="riskBand">Not scored</div></div>
      <div class="prob" id="prob">—</div>
      <div class="bar"><div id="barFill"></div></div>
      <div class="meta">
        <div><span>Decision</span><strong id="decision">—</strong></div>
        <div><span>Threshold</span><strong id="threshold">—</strong></div>
        <div><span>Backend</span><strong id="backend">—</strong></div>
        <div><span>Model version</span><strong id="version">—</strong></div>
      </div>
      <h2 style="font-size:16px">Top Tree-SHAP drivers</h2>
      <div class="drivers" id="drivers"><div style="color:var(--muted);font-size:13px">Run a prediction to see local feature contributions.</div></div>
    </section>
  </div>
  <div class="architecture">
    <div>Validated input</div><div>Feature pipeline</div><div>XGBoost</div><div>Tree-SHAP</div><div>FastAPI</div>
    <div>Real-time API</div><div>Batch API</div><div>SQLite audit</div><div>Metrics</div><div>Model governance</div>
  </div>
  <footer>Portfolio/reference implementation, not a credit, fraud, AML or customer eligibility decision system. The hosted free-tier SQLite audit store is ephemeral and can reset when the service sleeps or redeploys. <a href="/docs">Open API documentation</a> · <a href="/v1/model">Model metadata</a> · <a href="/health">Health</a></footer>
</div></main>
<script>
const fields = [
  ["transaction_amount","Transaction amount",2400,"number","0.01"],
  ["account_age_days","Account age (days)",38,"number","1"],
  ["transactions_24h","Transactions in 24h",17,"number","1"],
  ["avg_amount_30d","30-day average amount",145,"number","0.01"],
  ["international","International (0/1)",1,"number","1"],
  ["high_risk_country","High-risk country (0/1)",1,"number","1"],
  ["device_new","New device (0/1)",1,"number","1"],
  ["failed_logins_24h","Failed logins in 24h",3,"number","1"],
  ["transaction_hour","Transaction hour (0–23)",2,"number","1"],
  ["customer_tenure_years","Customer tenure (years)",0.1,"number","0.1"]
];
const form = document.getElementById('form');
fields.forEach(([id,label,value,type,step])=>{const d=document.createElement('div');d.innerHTML=`<label for="${id}">${label}</label><input id="${id}" type="${type}" step="${step}" value="${value}">`;form.appendChild(d)});
function tx(){const t={};fields.forEach(([id])=>{const el=document.getElementById(id);t[id]=['account_age_days','transactions_24h','international','high_risk_country','device_new','failed_logins_24h','transaction_hour'].includes(id)?parseInt(el.value,10):parseFloat(el.value)});return t}
const btn=document.getElementById('score');
btn.onclick=async()=>{btn.disabled=true;document.getElementById('status').textContent='Scoring…';try{const r=await fetch('/v1/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({transaction:tx(),explain:true})});const j=await r.json();if(!r.ok)throw new Error(j.detail||JSON.stringify(j));const pct=(j.risk_probability*100);document.getElementById('prob').textContent=pct.toFixed(1)+'%';document.getElementById('barFill').style.width=Math.min(100,pct)+'%';document.getElementById('riskBand').textContent=j.risk_band.toUpperCase()+' RISK';document.getElementById('decision').textContent=j.decision;document.getElementById('threshold').textContent=(j.threshold*100).toFixed(0)+'%';document.getElementById('backend').textContent=j.model_backend;document.getElementById('version').textContent=j.model_version;const drivers=document.getElementById('drivers');drivers.innerHTML='';j.top_drivers.forEach(d=>{const e=document.createElement('div');e.className='driver';const cls=d.contribution>=0?'positive':'negative';e.innerHTML=`<div><strong>${d.feature}</strong><br><small>${d.direction.replaceAll('_',' ')}</small></div><div class="${cls}">${d.contribution>=0?'+':''}${d.contribution.toFixed(3)}</div>`;drivers.appendChild(e)});document.getElementById('status').textContent='Prediction completed and audit event recorded.'}catch(e){document.getElementById('status').textContent='Error: '+e.message}finally{btn.disabled=false}};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    predictor, _, _ = _ensure_runtime()
    return {"status": "ready", "model_version": predictor.version, "backend": predictor.backend}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _predict_records(records: list[dict], explain: bool) -> list[PredictionResponse]:
    predictor, explainer, audit = _ensure_runtime()
    raw = pd.DataFrame(records)
    started = time.perf_counter()
    try:
        probabilities = predictor.predict_proba(raw)
        drivers = explainer.top_drivers(raw) if explain else [[] for _ in records]
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failure")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    total_latency = time.perf_counter() - started
    per_record_ms = total_latency * 1000 / max(len(records), 1)
    responses: list[PredictionResponse] = []
    audit_rows: list[dict] = []
    for record, probability, row_drivers in zip(records, probabilities, drivers, strict=True):
        request_id = str(uuid.uuid4())
        decision = "manual_review" if probability >= predictor.threshold else "approve"
        PREDICTIONS.labels(decision, predictor.version, predictor.backend).inc()
        PREDICTION_LATENCY.observe(total_latency / max(len(records), 1))
        RISK_SCORE.observe(float(probability))
        audit_rows.append(
            {
                "request_id": request_id,
                "model_version": predictor.version,
                "model_backend": predictor.backend,
                "record": record,
                "risk_probability": float(probability),
                "decision": decision,
                "latency_ms": per_record_ms,
            }
        )
        responses.append(
            PredictionResponse(
                request_id=request_id,
                risk_probability=round(float(probability), 6),
                risk_band=predictor.risk_band(float(probability)),
                decision=decision,
                threshold=predictor.threshold,
                model_backend=predictor.backend,
                model_version=predictor.version,
                top_drivers=[Driver(**d) for d in row_drivers],
            )
        )
    try:
        audit.write_many(audit_rows)
    except Exception:
        logger.exception("Audit batch write failed")
    return responses


@app.post("/v1/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def predict(request: PredictionRequest) -> PredictionResponse:
    return _predict_records([request.transaction.model_dump()], request.explain)[0]


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_api_key)])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = _predict_records([r.model_dump() for r in request.transactions], request.explain)
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)


@app.get("/v1/model", dependencies=[Depends(verify_api_key)])
def model_metadata() -> dict:
    predictor, _, _ = _ensure_runtime()
    return predictor.metadata

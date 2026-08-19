# Free browser deployment on Render

This repo includes a free-tier deployment profile for the public portfolio demo.

## What the hosted demo runs

- FastAPI browser dashboard and REST API
- XGBoost inference
- XGBoost native Tree-SHAP (`pred_contribs`) explanations
- Pydantic input validation
- SQLite audit events (ephemeral on free hosting)
- Prometheus-compatible `/metrics`
- Pre-trained synthetic-data model committed at `models/risk_model.joblib`

The full repository still documents PyTorch, MLflow, Docker, Kubernetes, PostgreSQL,
Prometheus/Grafana and OpenTelemetry. Those are intentionally excluded from the
512 MB free web-service runtime.

## Deploy

1. Commit `render.yaml`, `.python-version`, `requirements-render.txt` and the source changes.
2. Push to GitHub.
3. In Render choose **New > Blueprint** and select this repository, or use the Deploy to Render link in the README.
4. Apply the Blueprint.
5. When deployment finishes, open the generated `*.onrender.com` URL.

## Free-tier caveat

Render free web services sleep after inactivity and use an ephemeral filesystem,
so the local SQLite audit database is reset on service restart, spin-down or redeploy.
This is suitable for a portfolio demo, not durable production data.

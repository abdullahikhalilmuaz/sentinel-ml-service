# Sentinel ML Service (Python)

This folder is a drop-in replacement for the existing `ml-service` folder. It keeps the Node integration route `POST /internal/score`, health route `GET /health`, model status route `GET /internal/model/status`, and the feature field names already used by the Node backend.

## Important behaviour and limits

- The default training command generates a **synthetic demo model** so the API contract can be tested locally. It is not evidence of real identity recognition and is not production-validated.
- Without a user baseline, scoring mode is `population_demo`: it scores unusual behaviour relative to the population model, not whether the current person is the account owner.
- Personalized scoring is supported when Node sends a ready per-user baseline using the `baseline` request field described below. The backend currently does not send or calculate that baseline, so that integration must be changed in the later backend pass.
- The anomaly score is a policy index from 0 to 1, **not a probability**. Risk thresholds must be evaluated against genuine/impostor test data before treating them as security guarantees.
- No passwords or raw typed text should be collected. Only feature summaries should be sent.

## Install on Windows

Open Command Prompt or PowerShell in your existing `Authentication-system\ml-service` folder. Back up the current files, then copy the files from this package over the same folder structure. Do not replace the parent frontend/backend folders.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Keep your existing `.env` with `PORT=8000`; do not overwrite it with `.env.example`. Ensure `.env` is in the `ml-service` root, beside `app.py`.

## Train the artifact

From the `ml-service` root, with the virtual environment active:

```powershell
python -m src.train
```

This overwrites `artifacts/isolation_forest.pkl` with the new versioned artifact. It creates `artifacts` automatically. The existing pickle from the earlier script is not compatible with this calibrated artifact format, so retraining is required.

For tests:

```powershell
python -m pytest -q
```

Start the service:

```powershell
python app.py
```

The service binds to `127.0.0.1:8000` by default. Keep it local during development.

## Smoke checks

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/internal/model/status
```

A health response of `status: ok` means Flask is reachable; check model status separately. Model status should show `status: loaded`, `modelAvailable: true`, and (initially) `trainingDataKind: synthetic_demo`.

Example score request in PowerShell:

```powershell
$body = @{
  schemaVersion = 1
  sessionId = "local-smoke-test"
  features = @{
    typingIntervalMean = 220.0
    typingIntervalStd = 65.0
    typingIntervalMedian = 215.0
    pointerSpeedMean = 350.0
    pointerSpeedStd = 120.0
    clickIntervalMean = 850.0
    clickIntervalStd = 300.0
    idleRatio = 0.2
    navigationCount = 2
    interactionCount = 28
  }
} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/internal/score -ContentType "application/json" -Body $body
```

The response will use `scoreMode: population_demo` until Node sends an enrolled user baseline.

## Personalized baseline contract for the backend pass

Node should only send `baseline` after a user has enough clean enrolled samples (this implementation requires at least 20). The expected request shape is:

```json
{
  "schemaVersion": 1,
  "sessionId": "the-current-session-id",
  "features": { "...": "the ten schema-v1 features" },
  "baseline": {
    "status": "ready",
    "sampleCount": 30,
    "featureStats": {
      "mean": { "...": "mean for each of the ten features" },
      "std": { "...": "standard deviation for each of the ten features" }
    }
  }
}
```

The literal `...` values above are explanatory placeholders, not valid request values. `mean` and `std` must include each feature name defined in `src/feature_schema.py`. The backend should calculate those statistics from the user's enrolled feature batches; do not take a profile baseline from the client browser.

The schema assumes interval values are milliseconds, pointer speed is pixels/second, `idleRatio` is from 0 to 1, and counts are whole numbers per observation window. The frontend feature collector must use the same units. If its implementation uses different units, update both the feature contract and training data generator together before integrating.

## Training on real data later

After collecting a legitimate, consented dataset, prepare a CSV with one row per genuine observation window and these exact columns:

```text
typingIntervalMean,typingIntervalStd,typingIntervalMedian,pointerSpeedMean,pointerSpeedStd,clickIntervalMean,clickIntervalStd,idleRatio,navigationCount,interactionCount
```

Use only vetted genuine-session rows for this Isolation Forest fit. Do not include passwords, raw keystrokes, account identifiers, or labels as model features. Then run:

```powershell
python -m src.train --csv .\data\genuine_sessions.csv
```

The script requires at least 100 valid rows. The status response will still report `productionReady: false` until a separate evaluation process marks the artifact as validated; simply training on observed genuine data is not sufficient. Use a separate labelled genuine/impostor holdout dataset to compute FAR, FRR, precision, recall, latency, and detection rate.

## Optional internal API key

For local integration right now, leave `ML_SERVICE_API_KEY` blank because the current Node service does not send the header yet. When the backend is updated, set the same random secret in both services and have Node send it as `X-ML-Service-Key`. Never expose that key to the frontend.

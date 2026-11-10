from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

root = Path(__file__).resolve().parent
metadata = json.loads((root / 'artifacts/metadata.json').read_text(encoding='utf-8'))
raw = pd.read_csv(root / 'artifacts/reload_probe.csv')[metadata['features']]
preprocessor = joblib.load(root / 'artifacts/preprocessor.joblib')
model = keras.models.load_model(root / 'artifacts/lab8_model.keras')
prediction = model.predict(preprocessor.transform(raw).astype('float32'), verbose=0)
expected = np.load(root / 'artifacts/reload_expected.npy')
np.testing.assert_allclose(prediction, expected, rtol=1e-5, atol=1e-6)
result = {'passed': True, 'fresh_process': True, 'rows': len(raw),
          'max_absolute_difference': float(np.abs(prediction-expected).max()),
          'fit_or_adapt_called': False, 'predictions': prediction.reshape(-1).tolist()}
(root / 'artifacts/reload_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))

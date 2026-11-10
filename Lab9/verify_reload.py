"""Fresh-process reload verification; no training or data-source dependency."""
from pathlib import Path
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
import json
import numpy as np
from tensorflow import keras

root = Path(__file__).resolve().parent
probe = np.load(root / 'results/reload_probe.npz')
model = keras.models.load_model(root / 'food_cnn.keras')
actual = model.predict(probe['images'],verbose=0)
np.testing.assert_allclose(actual,probe['probabilities'],rtol=1e-5,atol=1e-6)
result = {'passed':True,'fresh_process':True,'fit_called':False,'rows':len(actual),
          'max_absolute_difference':float(np.abs(actual-probe['probabilities']).max())}
(root / 'results/reload_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)

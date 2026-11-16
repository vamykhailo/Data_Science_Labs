from pathlib import Path
import json
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
from tensorflow import keras

root=Path(__file__).resolve().parent
out=root/'lab10_results'
with np.load(out/'reload_probe.npz') as data:
    images=data['images']
    expected=data['probabilities']
model=keras.models.load_model(out/'selected_model.keras')
actual=model(images,training=False).numpy()
np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
result={'passed':True,'fresh_process':True,'fit_called':False,'rows':len(actual),
        'max_absolute_difference':float(np.abs(actual-expected).max())}
(out/'reload_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)

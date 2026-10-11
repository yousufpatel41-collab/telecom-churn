import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import api
from train import build_model

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.customer = api.CustomerFeatures(gender="Male", Partner="Yes", Dependents="No", tenure=0,
            PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic", OnlineSecurity="No",
            OnlineBackup="Yes", DeviceProtection="No", TechSupport="No", StreamingTV="Yes",
            StreamingMovies="Yes", Contract="Two year", PaperlessBilling="Yes", PaymentMethod="Mailed check",
            MonthlyCharges=80, TotalCharges=0)
        a=api.preprocess_input(self.customer)
        b=a.copy(); b["Contract"]="Month-to-month"; b["gender"]="Female"
        self.X=pd.concat([a,b]*10, ignore_index=True)
        self.model=build_model(); self.model.set_params(classifier__n_estimators=2)
        self.model.fit(self.X, [0,1]*10)
        self.old=api.model; api.model=self.model
        self.addCleanup(setattr, api, "model", self.old)

    def test_categories_are_distinct(self):
        enc=self.model.named_steps["preprocess"]
        self.assertFalse(np.array_equal(enc.transform(self.X.iloc[[0]]),enc.transform(self.X.iloc[[1]])))

    def test_api_uses_training_pipeline(self):
        expected=round(float(self.model.predict_proba(api.preprocess_input(self.customer))[0,1]),4)
        self.assertEqual(api.predict_churn(self.customer).churn_probability,expected)
        self.assertEqual(api.predict_batch([self.customer])["predictions"][0]["churn_probability"],expected)

    def test_unavailable_model_returns_503(self):
        api.model=None
        with self.assertRaises(api.HTTPException) as error:
            api.predict_churn(self.customer)
        self.assertEqual(error.exception.status_code,503)

    def test_missing_charges_and_unknown_category(self):
        row=self.X.iloc[[0]].copy()
        row["TotalCharges"]=np.nan
        row["PaymentMethod"]="new method"
        self.assertTrue(np.isfinite(self.model.predict_proba(row)).all())

if __name__ == "__main__":
    unittest.main()

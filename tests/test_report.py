"""Check that published metric tables agree with recorded holdout counts."""
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
class ReportTests(unittest.TestCase):
    def test_metric_consistency(self):
        m=json.loads((ROOT/'outputs/metrics.json').read_text())
        results=m.get('models',{'xgboost':m})
        if 'models' in m:
            selected=max(results,key=lambda n:results[n]['cv_auc_mean'])
            self.assertEqual(selected,m['selected_by_training_cv_auc'])
        for result in results.values():
            (tn,fp),(fn,tp)=result['confusion_matrix']
            self.assertEqual(tn+fp+fn+tp,m['test_rows'])
            self.assertAlmostEqual((tn+tp)/m['test_rows'],result['accuracy'])
            self.assertAlmostEqual(tp/(tp+fn),result['recall'])
            self.assertAlmostEqual(tp/(tp+fp),result['precision'])
            self.assertTrue(0<=result['roc_auc']<=1)

if __name__=='__main__':unittest.main()

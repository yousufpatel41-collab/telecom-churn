"""Render the selected model's recorded holdout confusion matrix as SVG."""
import json
from html import escape
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def build():
    metrics=json.loads((ROOT/'outputs/metrics.json').read_text())
    name=metrics.get('selected_by_training_cv_auc','xgboost')
    result=metrics['models'][name] if 'models' in metrics else metrics
    matrix=result['confusion_matrix']
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="330" viewBox="0 0 640 330" role="img" aria-labelledby="title desc">',
           '<title id="title">Held-out confusion matrix</title>',
           '<desc id="desc">Rows are actual class; columns are predicted class. Values are held-out record counts.</desc>',
           '<rect width="640" height="330" rx="8" fill="#f3f5f1"/>',
           '<g font-family="Arial,sans-serif" fill="#20382c">',
           '<text x="28" y="35" font-size="19" font-weight="bold">Held-out evaluation</text>',
           f'<text x="28" y="58" font-size="12">{escape(name.replace("_"," "))} · {metrics["test_rows"]} records · threshold {metrics["threshold"]}</text>',
           '<text x="168" y="94" font-size="12">Predicted 0</text><text x="288" y="94" font-size="12">Predicted 1</text>']
    for row in range(2):
        parts.append(f'<text x="28" y="{147+row*76}" font-size="13">Actual {row}</text>')
        for col in range(2):
            x=139+col*120;y=108+row*76
            color='#d3e6d8' if row==col else '#f0dcc8'
            parts.append(f'<rect x="{x}" y="{y}" width="110" height="65" rx="4" fill="{color}"/><text x="{x+55}" y="{y+42}" text-anchor="middle" font-size="26">{matrix[row][col]}</text>')
    for i,(label,val) in enumerate([('ROC-AUC',f'{result["roc_auc"]:.4f}'),('Accuracy',f'{result["accuracy"]:.2%}'),('Recall',f'{result["recall"]:.2%}')]):
        parts.append(f'<text x="414" y="{120+i*49}" font-size="12">{label}</text><text x="414" y="{144+i*49}" font-size="22">{val}</text>')
    parts.extend(['<text x="28" y="296" font-size="11">One stratified split; training CV selected the model before test evaluation.</text>','</g></svg>'])
    (ROOT/'outputs/evaluation.svg').write_text(''.join(parts)+'\n',encoding='utf-8')

if __name__=='__main__':build()

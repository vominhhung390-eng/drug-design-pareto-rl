#!/usr/bin/env python
"""ExtraTrees robustness grid for VEGFR2 high-confidence classification."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from benchmark_predictor_round2 import featurize

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'/'predictor_retraining_v3_20260731'/'data'/'vegfr2'
OUT=ROOT/'results'/'predictor_v41_20260802'/'vegfr2_forest_grid'

def select(frame):
    data=frame[(frame.pactivity<=5.75)|(frame.pactivity>=7.25)].reset_index(drop=True)
    return data,(data.pactivity>=7.25).astype(int).to_numpy()

def metrics(y,p):
    return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)),'brier':float(brier_score_loss(y,p))}

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];predictions={}
    specs=[(mf,leaf,criterion) for mf in ('sqrt',0.05,0.1,0.2,0.4) for leaf in (1,2,4,8) for criterion in ('gini','entropy')]
    for fold in ('fold_a','fold_b'):
        train,ytr=select(pd.read_csv(DATA/'single_protein_wt_or_unspecified'/fold/'train.csv'))
        val,y=select(pd.read_csv(DATA/'single_protein_assay_ge5'/fold/'validation.csv'))
        xtr,xval=featurize(train.smiles),featurize(val.smiles)
        for mf,leaf,criterion in specs:
            name=f"mf{mf}_leaf{leaf}_{criterion}";model=ExtraTreesClassifier(n_estimators=700,max_features=mf,min_samples_leaf=leaf,
                   criterion=criterion,class_weight='balanced',n_jobs=-1,random_state=42)
            model.fit(xtr,ytr);p=model.predict_proba(xval)[:,1]
            rows.append({'fold':fold,'model':name,'max_features':mf,'min_samples_leaf':leaf,'criterion':criterion,'n':len(y),**metrics(y,p)})
        print(fold,'complete',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'fold_metrics.csv',index=False)
    summary=(frame.groupby(['model','max_features','min_samples_leaf','criterion'],as_index=False).agg(mean_auroc=('auroc','mean'),
             worst_auroc=('auroc','min'),mean_auprc=('auprc','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'),mean_brier=('brier','mean'))
             .sort_values(['worst_auroc','mean_auroc'],ascending=False))
    summary.to_csv(OUT/'summary.csv',index=False);print(summary.head(20).to_string(index=False),flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python
"""Systematic EGFR local-similarity classifier grid on frozen V4 folds."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'/'predictor_retraining_v3_20260731'/'data'/'egfr'/'single_protein_assay_ge10'
OUT=ROOT/'results'/'predictor_v41_20260802'/'egfr_similarity_grid'

def metrics(y,p):
    return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)),'brier':float(brier_score_loss(y,p))}

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];ks=(5,10,20,40,80);powers=(1,3,6)
    for fold in ('fold_a','fold_b'):
        train=pd.read_csv(DATA/fold/'train.csv');val=pd.read_csv(DATA/fold/'validation.csv')
        train=train[(train.pactivity<=5.5)|(train.pactivity>=7.5)].reset_index(drop=True)
        val=val[(val.pactivity<=5.5)|(val.pactivity>=7.5)].reset_index(drop=True)
        ytr=(train.pactivity>=7.5).astype(int).to_numpy();y=(val.pactivity>=7.5).astype(int).to_numpy()
        for radius in (1,2,3,4):
            for count in (False,True):
                gen=rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=2048)
                getter=gen.GetCountFingerprint if count else gen.GetFingerprint
                tf=[getter(Chem.MolFromSmiles(s)) for s in train.smiles]
                predictions={(k,p):[] for k in ks for p in powers}
                for s in val.smiles:
                    sims=np.asarray(DataStructs.BulkTanimotoSimilarity(getter(Chem.MolFromSmiles(s)),tf));order=np.argsort(sims)[::-1]
                    for k in ks:
                        idx=order[:k]
                        for power in powers:predictions[(k,power)].append(np.average(ytr[idx],weights=np.maximum(sims[idx],1e-6)**power))
                for (k,power),values in predictions.items():
                    p=np.asarray(values);name=f"r{radius}_{'count' if count else 'bit'}_k{k}_p{power}"
                    rows.append({'fold':fold,'model':name,'radius':radius,'count':count,'k':k,'power':power,'n':len(y),**metrics(y,p)})
        print(fold,'complete',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'fold_metrics.csv',index=False)
    summary=(frame.groupby(['model','radius','count','k','power'],as_index=False).agg(mean_auroc=('auroc','mean'),worst_auroc=('auroc','min'),
             mean_auprc=('auprc','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'),mean_brier=('brier','mean'))
             .sort_values(['worst_auroc','mean_auroc'],ascending=False))
    summary.to_csv(OUT/'summary.csv',index=False);print(summary.head(20).to_string(index=False),flush=True)

if __name__=='__main__':main()

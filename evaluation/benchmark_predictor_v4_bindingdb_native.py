#!/usr/bin/env python
"""Native BindingDB temporal benchmark for EGFR high-confidence classification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from benchmark_predictor_round2 import featurize


ROOT=Path(__file__).resolve().parents[1]
V3=ROOT/'results'/'predictor_retraining_v3_20260731'
OUT=ROOT/'results'/'predictor_v4_90plus_20260731'/'bindingdb_native'


def metrics(y,p):
    return {'n':len(y),'n_active':int(y.sum()),'n_inactive':int(len(y)-y.sum()),
            'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)),'brier':float(brier_score_loss(y,p))}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(V3/'bindingdb_augmented'/'bindingdb_egfr_exact_ic50.csv')
    data=(raw.groupby('smiles',as_index=False).agg(pactivity=('pactivity','median'),low=('pactivity','min'),high=('pactivity','max'),
                                                   first_year=('document_year','min'),last_year=('document_year','max'),n=('pactivity','size')))
    data=data[(data.high-data.low)<=1.0]
    data=data[(data.pactivity<=5.5)|(data.pactivity>=7.5)].copy();data['label']=(data.pactivity>=7.5).astype(int)
    train=data[data.first_year<=2023].reset_index(drop=True);test=data[data.first_year>=2024].reset_index(drop=True)
    xtr,xte=featurize(train.smiles),featurize(test.smiles);ytr=train.label.to_numpy();y=test.label.to_numpy();ratio=(ytr==0).sum()/max(1,(ytr==1).sum())
    models={'extratrees':ExtraTreesClassifier(n_estimators=1200,max_features='sqrt',min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42),
            'randomforest':RandomForestClassifier(n_estimators=1000,max_features='sqrt',min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42),
            'xgb':XGBClassifier(n_estimators=1000,learning_rate=.03,max_depth=7,min_child_weight=6,subsample=.85,colsample_bytree=.65,
                                reg_lambda=8,reg_alpha=.1,scale_pos_weight=ratio,tree_method='hist',device='cuda',eval_metric='auc',n_jobs=-1,random_state=42)}
    predictions={}
    for name,model in models.items():model.fit(xtr,ytr);predictions[name]=model.predict_proba(xte)[:,1]
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);tf=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.smiles];knn=[]
    for s in test.smiles:
        sims=np.asarray(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)),tf));idx=np.argsort(sims)[::-1][:20]
        knn.append(np.average(ytr[idx],weights=np.maximum(sims[idx],1e-6)**3))
    predictions['knn20']=np.asarray(knn);predictions['et_knn50']=.5*predictions['extratrees']+.5*predictions['knn20']
    results={name:metrics(y,p) for name,p in predictions.items()}
    test.assign(**{name:p for name,p in predictions.items()}).to_csv(OUT/'predictions.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps({'train_n':len(train),'test_n':len(test),'models':results},indent=2),encoding='utf-8')
    print(json.dumps({'train_n':len(train),'test_n':len(test),'models':results},indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python
"""Latest-year stress tests for frozen V4 high-confidence candidates."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from benchmark_predictor_round2 import featurize


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results'/'predictor_v4_90plus_20260731'/'latest_stress'
V3=ROOT/'results'/'predictor_retraining_v3_20260731'/'data'
WRAPPER=ROOT/'evaluation'/'run_chemprop_utf8.py'


def metrics(y,p):
    if len(np.unique(y))<2:
        return {'n':len(y),'n_active':int(y.sum()),'n_inactive':int(len(y)-y.sum()),'auroc':None}
    return {'n':len(y),'n_active':int(y.sum()),'n_inactive':int(len(y)-y.sum()),
            'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)),'brier':float(brier_score_loss(y,p))}


def prepare_egfr():
    source=V3/'egfr'/'single_protein_assay_ge10'; out=RUN/'egfr_data'; out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(source/'development_through_2023.csv'); test=pd.read_csv(source/'exploratory_2024plus.csv')
    train=train[(train.pactivity<=5.5)|(train.pactivity>=7.5)].copy(); train['active_label']=(train.pactivity>=7.5).astype(int)
    # Latest data lack strong negatives, so retain the full 6.5 endpoint as a declared stress test.
    test=test.copy(); test['active_label']=(test.pactivity>=6.5).astype(int)
    train[['smiles','active_label']].to_csv(out/'train.csv',index=False)
    test[['smiles','active_label']].to_csv(out/'test.csv',index=False)
    return train.reset_index(drop=True),test.reset_index(drop=True),out


def chemprop(train_dir,variant):
    out=RUN/'egfr_models'/variant; out.mkdir(parents=True,exist_ok=True); pred=out/'model_0'/'test_predictions.csv'
    cmd=[sys.executable,'-X','utf8',str(WRAPPER),'train','-i',str(train_dir/'train.csv'),str(train_dir/'test.csv'),str(train_dir/'test.csv'),
         '-o',str(out),'--smiles-columns','smiles','--target-columns','active_label','--task-type','classification','--metrics','roc','prc',
         '--class-balance','--accelerator','gpu','--devices','1','--num-workers','0','--batch-size','256','--epochs','50','--patience','10',
         '--warmup-epochs','2','--data-seed','42','--pytorch-seed','42','--message-hidden-dim','300','--depth','3','--ffn-hidden-dim','300','--ffn-num-layers','1']
    if variant=='dmpnn_morgan':cmd += ['--molecule-featurizers','morgan_binary']
    if not pred.exists():
        env=os.environ.copy();env.update({'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','RICH_FORCE_TERMINAL':'false'})
        with (out/'launcher.log').open('w',encoding='utf-8') as log:
            proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError((out/'launcher.log').read_text(encoding='utf-8',errors='replace')[-5000:])
    return pd.read_csv(pred).active_label.to_numpy(float)


def egfr_test():
    train,test,data_dir=prepare_egfr(); ytr=train.active_label.to_numpy(); y=test.active_label.to_numpy()
    dmpnn=chemprop(data_dir,'dmpnn'); morgan=chemprop(data_dir,'dmpnn_morgan')
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048); tf=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.smiles];knn=[]
    for s in test.smiles:
        sims=np.asarray(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)),tf));idx=np.argsort(sims)[::-1][:20]
        knn.append(np.average(ytr[idx],weights=np.maximum(sims[idx],1e-6)**3))
    probability=.7*dmpnn+.1*morgan+.2*np.asarray(knn)
    test.assign(prediction=probability,dmpnn=dmpnn,dmpnn_morgan=morgan,knn=knn).to_csv(RUN/'egfr_2024plus_full_threshold.csv',index=False)
    result={'target':'EGFR','task':'full_threshold_stress_after_margin_training',**metrics(y,probability)}
    print(result,flush=True);return result


def vegfr2_test():
    train=pd.read_csv(V3/'vegfr2'/'single_protein_wt_or_unspecified'/'development_through_2023.csv')
    test=pd.read_csv(V3/'vegfr2'/'single_protein_assay_ge5'/'exploratory_2024plus.csv')
    train=train[(train.pactivity<=5.75)|(train.pactivity>=7.25)].copy(); ytr=(train.pactivity>=7.25).astype(int)
    test=test[(test.pactivity<=5.75)|(test.pactivity>=7.25)].copy(); y=(test.pactivity>=7.25).astype(int).to_numpy()
    model=ExtraTreesClassifier(n_estimators=1200,max_features='sqrt',min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42)
    model.fit(featurize(train.smiles),ytr); probability=model.predict_proba(featurize(test.smiles))[:,1]
    test.assign(prediction=probability,label=y).to_csv(RUN/'vegfr2_2024plus_margin075.csv',index=False)
    result={'target':'VEGFR2','task':'confidence_margin_0_75',**metrics(y,probability)}
    print(result,flush=True);return result


def main():
    RUN.mkdir(parents=True,exist_ok=True);results=[egfr_test(),vegfr2_test()]
    (RUN/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()

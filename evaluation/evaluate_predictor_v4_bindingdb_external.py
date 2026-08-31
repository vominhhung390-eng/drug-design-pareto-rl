#!/usr/bin/env python
"""External BindingDB 2024+ test for the frozen EGFR V4 ensemble."""
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
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


ROOT=Path(__file__).resolve().parents[1]
V3=ROOT/'results'/'predictor_retraining_v3_20260731'
V4=ROOT/'results'/'predictor_v4_90plus_20260731'
OUT=V4/'bindingdb_external'; WRAPPER=ROOT/'evaluation'/'run_chemprop_utf8.py'


def predict(model_dir,input_csv,output_csv):
    env=os.environ.copy();env.update({'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','RICH_FORCE_TERMINAL':'false'})
    command=[sys.executable,'-X','utf8',str(WRAPPER),'predict','-q','--test-path',str(input_csv),
             '--smiles-columns','smiles','--model-paths',str(model_dir),'--preds-path',str(output_csv),
             '--batch-size','256','--num-workers','0','--accelerator','gpu','--devices','1']
    if model_dir.name=='dmpnn_morgan':
        command += ['--molecule-featurizers','morgan_binary']
    subprocess.run(command,
                   check=True,cwd=ROOT,env=env)
    return pd.read_csv(output_csv).active_label.to_numpy(float)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bd=pd.read_csv(V3/'bindingdb_augmented'/'bindingdb_egfr_exact_ic50.csv')
    ch=pd.read_csv(V3/'data'/'egfr'/'single_protein_assay_ge10'/'development_through_2023.csv')
    group=(bd[bd.document_year>=2024].groupby('smiles',as_index=False)
           .agg(pactivity=('pactivity','median'),low=('pactivity','min'),high=('pactivity','max'),
                first_document_year=('document_year','min'),n_measurements=('pactivity','size')))
    group=group[((group.high-group.low)<=1.0)&(~group.smiles.isin(set(ch.smiles)))]
    test=group[(group.pactivity<=5.5)|(group.pactivity>=7.5)].reset_index(drop=True)
    test['label']=(test.pactivity>=7.5).astype(int); input_csv=OUT/'input.csv';test[['smiles']].to_csv(input_csv,index=False)
    models=V4/'latest_stress'/'egfr_models'
    dmpnn=predict(models/'dmpnn',input_csv,OUT/'dmpnn_predictions.csv')
    morgan=predict(models/'dmpnn_morgan',input_csv,OUT/'morgan_predictions.csv')
    train=ch[(ch.pactivity<=5.5)|(ch.pactivity>=7.5)].reset_index(drop=True); ytr=(train.pactivity>=7.5).astype(int).to_numpy()
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);tf=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.smiles];knn=[];similarity=[]
    for s in test.smiles:
        sims=np.asarray(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)),tf));idx=np.argsort(sims)[::-1][:20]
        knn.append(np.average(ytr[idx],weights=np.maximum(sims[idx],1e-6)**3));similarity.append(sims[idx[0]])
    probability=.7*dmpnn+.1*morgan+.2*np.asarray(knn);y=test.label.to_numpy()
    result={'n':len(y),'n_active':int(y.sum()),'n_inactive':int(len(y)-y.sum()),
            'auroc':float(roc_auc_score(y,probability)),'auprc':float(average_precision_score(y,probability)),
            'balanced_accuracy':float(balanced_accuracy_score(y,probability>=.5)),
            'brier':float(brier_score_loss(y,probability))}
    test.assign(prediction=probability,dmpnn=dmpnn,dmpnn_morgan=morgan,knn=knn,max_train_similarity=similarity).to_csv(OUT/'predictions.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

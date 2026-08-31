#!/usr/bin/env python
"""Train five-member EGFR D-MPNN ensembles on frozen V4 time folds."""
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
V3=ROOT/'results'/'predictor_retraining_v3_20260731'/'data'/'egfr'/'single_protein_assay_ge10'
V4=ROOT/'results'/'predictor_v4_90plus_20260731'
DATA=V4/'chemprop_classification'/'data'/'egfr'/'confidence_margin_1_0'
OUT=ROOT/'results'/'predictor_v41_20260802'/'egfr_seed_ensemble'
WRAPPER=ROOT/'evaluation'/'run_chemprop_utf8.py'


def score(y,p):
    return {'n':len(y),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)),'brier':float(brier_score_loss(y,p))}


def train(fold,variant):
    data=DATA/fold;out=OUT/fold/variant;out.mkdir(parents=True,exist_ok=True)
    expected=out/'model_4'/'test_predictions.csv'
    cmd=[sys.executable,'-X','utf8',str(WRAPPER),'train','-i',str(data/'train.csv'),str(data/'validation.csv'),str(data/'validation.csv'),
         '-o',str(out),'--smiles-columns','smiles','--target-columns','active_label','--task-type','classification',
         '--metrics','roc','prc','accuracy','f1','--class-balance','--accelerator','gpu','--devices','1','--num-workers','0',
         '--batch-size','256','--epochs','60','--patience','12','--warmup-epochs','2','--ensemble-size','5',
         '--data-seed','42','--pytorch-seed','42','--message-hidden-dim','300','--depth','3','--ffn-hidden-dim','300','--ffn-num-layers','1']
    if variant=='dmpnn_morgan':cmd += ['--molecule-featurizers','morgan_binary']
    if not expected.exists():
        env=os.environ.copy();env.update({'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','RICH_FORCE_TERMINAL':'false'})
        with (out/'launcher.log').open('w',encoding='utf-8') as log:
            proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError((out/'launcher.log').read_text(encoding='utf-8',errors='replace')[-6000:])
    truth=pd.read_csv(data/'validation.csv');y=truth.active_label.to_numpy(int);individual=[];rows=[]
    for index in range(5):
        p=pd.read_csv(out/f'model_{index}'/'test_predictions.csv').active_label.to_numpy(float);individual.append(p)
        rows.append({'fold':fold,'variant':variant,'member':index,**score(y,p)})
    mean=np.mean(individual,axis=0);rows.append({'fold':fold,'variant':variant,'member':'ensemble',**score(y,mean)})
    truth.assign(prediction=mean,prediction_std=np.std(individual,axis=0)).to_csv(out/'ensemble_predictions.csv',index=False)
    print(fold,variant,'ensemble',score(y,mean),flush=True);return rows,mean


def knn(fold):
    train=pd.read_csv(V3/fold/'train.csv');val=pd.read_csv(V3/fold/'validation.csv')
    train=train[(train.pactivity<=5.5)|(train.pactivity>=7.5)].reset_index(drop=True)
    val=val[(val.pactivity<=5.5)|(val.pactivity>=7.5)].reset_index(drop=True)
    ytr=(train.pactivity>=7.5).astype(int).to_numpy();y=(val.pactivity>=7.5).astype(int).to_numpy()
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);tf=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.smiles];p=[]
    for s in val.smiles:
        sims=np.asarray(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)),tf));idx=np.argsort(sims)[::-1][:20]
        p.append(np.average(ytr[idx],weights=np.maximum(sims[idx],1e-6)**3))
    return y,np.asarray(p)


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];fold_predictions={}
    for fold in ('fold_a','fold_b'):
        fold_predictions[fold]={}
        for variant in ('dmpnn','dmpnn_morgan'):
            member_rows,p=train(fold,variant);rows.extend(member_rows);fold_predictions[fold][variant]=p
        y,k=knn(fold);fold_predictions[fold]['knn']=k
        candidates={
            'dmpnn70_morgan10_knn20':.7*fold_predictions[fold]['dmpnn']+.1*fold_predictions[fold]['dmpnn_morgan']+.2*k,
            'dmpnn60_morgan20_knn20':.6*fold_predictions[fold]['dmpnn']+.2*fold_predictions[fold]['dmpnn_morgan']+.2*k,
            'dmpnn50_morgan25_knn25':.5*fold_predictions[fold]['dmpnn']+.25*fold_predictions[fold]['dmpnn_morgan']+.25*k,
        }
        for name,p in candidates.items():
            rows.append({'fold':fold,'variant':name,'member':'fixed_ensemble',**score(y,p)})
            print(fold,name,score(y,p),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'fold_metrics.csv',index=False)
    fixed=frame[frame.member.eq('fixed_ensemble')]
    summary=(fixed.groupby('variant',as_index=False).agg(mean_auroc=('auroc','mean'),worst_auroc=('auroc','min'),
             mean_auprc=('auprc','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'),mean_brier=('brier','mean'))
             .sort_values(['worst_auroc','mean_auroc'],ascending=False))
    summary.to_csv(OUT/'summary.csv',index=False);print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()

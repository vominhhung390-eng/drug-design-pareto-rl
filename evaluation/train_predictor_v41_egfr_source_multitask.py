#!/usr/bin/env python
"""Two-head EGFR model: ChEMBL primary task plus BindingDB auxiliary task."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


ROOT=Path(__file__).resolve().parents[1]
V3=ROOT/'results'/'predictor_retraining_v3_20260731'
OUT=ROOT/'results'/'predictor_v41_20260802'/'egfr_source_multitask_v2'
WRAPPER=ROOT/'evaluation'/'run_chemprop_utf8.py'


def make_data(fold):
    cutoff={'fold_a':2019,'fold_b':2021}[fold];out=OUT/'data'/fold;out.mkdir(parents=True,exist_ok=True)
    base=V3/'data'/'egfr'/'single_protein_assay_ge10'/fold
    ch=pd.read_csv(base/'train.csv');val=pd.read_csv(base/'validation.csv')
    ch=ch[(ch.pactivity<=5.5)|(ch.pactivity>=7.5)].copy();ch['chembl_label']=(ch.pactivity>=7.5).astype(float);ch['bindingdb_label']=np.nan
    val=val[(val.pactivity<=5.5)|(val.pactivity>=7.5)].copy();val['chembl_label']=(val.pactivity>=7.5).astype(float)
    # The auxiliary value in the final test file is a CLI compatibility placeholder
    # only; it is never used for fitting, early stopping, selection, or reporting.
    val['bindingdb_label']=val['chembl_label']
    bd=pd.read_csv(V3/'bindingdb_augmented'/'bindingdb_egfr_exact_ic50.csv');bd=bd[(bd.document_year<=cutoff)&(~bd.smiles.isin(set(val.smiles)))]
    bd=(bd.groupby('smiles',as_index=False).agg(pactivity=('pactivity','median'),low=('pactivity','min'),high=('pactivity','max')))
    bd=bd[((bd.high-bd.low)<=1.0)&((bd.pactivity<=5.5)|(bd.pactivity>=7.5))].copy();bd['bindingdb_label']=(bd.pactivity>=7.5).astype(float);bd['chembl_label']=np.nan
    ch_internal=ch.sample(frac=0.10,random_state=42);ch=ch.drop(ch_internal.index)
    bd_internal=bd.sample(frac=0.10,random_state=42);bd=bd.drop(bd_internal.index)
    # Merge identical structures without forcing the two databases to share labels.
    merged={}
    for row in pd.concat([ch[['smiles','chembl_label','bindingdb_label']],bd[['smiles','chembl_label','bindingdb_label']]],ignore_index=True).itertuples(index=False):
        item=merged.setdefault(row.smiles,{'smiles':row.smiles,'chembl_label':np.nan,'bindingdb_label':np.nan})
        if pd.notna(row.chembl_label):item['chembl_label']=row.chembl_label
        if pd.notna(row.bindingdb_label):item['bindingdb_label']=row.bindingdb_label
    train=pd.DataFrame(merged.values())
    internal=pd.concat([ch_internal[['smiles','chembl_label','bindingdb_label']],
                        bd_internal[['smiles','chembl_label','bindingdb_label']]],ignore_index=True)
    train.to_csv(out/'train.csv',index=False);internal.to_csv(out/'internal_validation.csv',index=False)
    val[['smiles','chembl_label','bindingdb_label']].to_csv(out/'test.csv',index=False)
    return out,len(ch),len(bd),len(train)


def metrics(y,p):
    return {'n':len(y),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)),'brier':float(brier_score_loss(y,p))}


def train_one(fold,variant):
    data,n_ch,n_bd,n_union=make_data(fold);out=OUT/fold/variant;out.mkdir(parents=True,exist_ok=True)
    expected=out/'model_2'/'test_predictions.csv'
    cmd=[sys.executable,'-X','utf8',str(WRAPPER),'train','-i',str(data/'train.csv'),str(data/'internal_validation.csv'),str(data/'test.csv'),
         '-o',str(out),'--smiles-columns','smiles','--target-columns','chembl_label','bindingdb_label','--task-type','classification',
         '--metrics','roc','prc','accuracy','f1','--class-balance','--task-weights','1.0','0.5','--accelerator','gpu','--devices','1',
         '--num-workers','0','--batch-size','256','--epochs','60','--patience','12','--warmup-epochs','2','--ensemble-size','3',
         '--data-seed','42','--pytorch-seed','42','--message-hidden-dim','300','--depth','3','--ffn-hidden-dim','300','--ffn-num-layers','1']
    if variant=='dmpnn_morgan':cmd += ['--molecule-featurizers','morgan_binary']
    elif variant=='chemeleon':cmd += ['--from-foundation','CHEMELEON']
    if not expected.exists():
        env=os.environ.copy();env.update({'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','RICH_FORCE_TERMINAL':'false'})
        with (out/'launcher.log').open('w',encoding='utf-8') as log:
            proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError((out/'launcher.log').read_text(encoding='utf-8',errors='replace')[-7000:])
    truth=pd.read_csv(data/'test.csv');y=truth.chembl_label.to_numpy(int);pred=[];rows=[]
    for i in range(3):
        p=pd.read_csv(out/f'model_{i}'/'test_predictions.csv').chembl_label.to_numpy(float);pred.append(p)
        rows.append({'fold':fold,'variant':variant,'member':i,'chembl_train_n':n_ch,'bindingdb_train_n':n_bd,'union_n':n_union,**metrics(y,p)})
    mean=np.mean(pred,axis=0);rows.append({'fold':fold,'variant':variant,'member':'ensemble','chembl_train_n':n_ch,'bindingdb_train_n':n_bd,'union_n':n_union,**metrics(y,mean)})
    truth.assign(prediction=mean,prediction_std=np.std(pred,axis=0)).to_csv(out/'ensemble_predictions.csv',index=False)
    print(fold,variant,metrics(y,mean),{'chembl':n_ch,'bindingdb':n_bd,'union':n_union},flush=True);return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for fold in ('fold_a','fold_b'):
        for variant in ('dmpnn','dmpnn_morgan','chemeleon'):rows.extend(train_one(fold,variant))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'fold_metrics.csv',index=False)
    ens=frame[frame.member.eq('ensemble')]
    summary=(ens.groupby('variant',as_index=False).agg(mean_auroc=('auroc','mean'),worst_auroc=('auroc','min'),mean_auprc=('auprc','mean'),
             mean_balanced_accuracy=('balanced_accuracy','mean'),mean_brier=('brier','mean')).sort_values(['worst_auroc','mean_auroc'],ascending=False))
    summary.to_csv(OUT/'summary.csv',index=False);print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()

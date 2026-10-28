"""Figures 1-7 and Table 2 of the paper, from the outputs of scripts/run_analyses.py.

Example
-------
    python scripts/make_figures.py --results results --out results/figures
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing the package
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from mammo_routing.config import SEEDS
from mammo_routing.results_io import wide
from mammo_routing.stats import TARGET_SENSITIVITIES as TARGETS
from mammo_routing.stats import auc, operating_point, threshold_for_sensitivity

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", type=Path, default=Path("results"))
ap.add_argument("--out", type=Path, default=Path("results/figures"))
args = ap.parse_args()
OUT = str(args.out); os.makedirs(OUT, exist_ok=True)
AN = args.results / "analysis"
CONDITIONS_DIR, CONDITIONS_25, CONDITIONS_AVG = AN / "conditions" / "clip4.0", AN / "conditions" / "clip2.5", AN / "conditions" / "clip_average"
MECHANISM_DIR, AFFINITY_DIR = AN / "mechanism", AN / "affinity"
PREDICTIONS_DIR = args.results / "predictions" / "cbis" / "clip4.0"
AFFINITY_FILE = PREDICTIONS_DIR / "ensembles_part_a" / "affinity_matrix_used.npy"
EXTERNAL_AUC = AN / "external" / "clip4.0" / "inbreast_mass_auc.csv"
EXTERNAL_COMPARISONS = AN / "external" / "clip4.0" / "inbreast_mass_comparisons.csv"
SETTING_NAMES = {"clip4.0": "clip 4.0 (primary)", "clip2.5": "clip 2.5", "clip_average": "average over clips 1–4"}


def simple_ensemble_aucs():
    frames = []
    for folder, name in SETTING_NAMES.items():
        d = pd.read_csv(AN / "simple_ensembles" / folder / "afage_vs_simple_ensembles.csv")
        frames.append(d.assign(setting=name))
    return pd.concat(frames, ignore_index=True)


def simple_ensemble_differences(setting):
    d = simple_ensemble_aucs()
    return d[(d.kind == "D") & (d.setting == setting)].set_index("model")


plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','font.family':'DejaVu Sans'})
CB=['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#F0E442','#000000']
FC={'Mass':'#0072B2','Calcification':'#E69F00'}
W2=7.0  # double-column width (inches)
BB={'resnet50':'ResNet-50','efficientnet_b0':'EfficientNet-B0','densenet121':'DenseNet-121','convnext_tiny':'ConvNeXt-Tiny','vit_base_patch16_224':'ViT-B/16'}
CNNs=['resnet50','efficientnet_b0','densenet121','convnext_tiny']
COND={'C1':'C1','C1_aux':'C1+Aux','C2_gate':'C2-Gate','C3_rand':'C3-Rand','C3_beta0':'C3 β=0','C3_beta025':'C3 β=0.25','C3_beta05':'C3 β=0.5',
      'C3_beta075':'C3 β=0.75','C3_beta1':'C3 β=1','C3_oracle':'C3-Oracle','C3_static':'C3-Static','C3_vit5bb':'C3+ViT-B/16'}
def letter(ax,l,x=-0.13,y=1.03): ax.text(x,y,l,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')

def letters(fig, pairs, dx=0.006, dy=0.008):
    # Put each panel letter at the top-left of the panel's full extent (including tick labels),
    # so letters never overlap labels of neighbouring panels.
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, l in pairs:
        bb = inv.transform(ax.get_tightbbox(r).get_points())
        fig.text(bb[0, 0] - dx, bb[1, 1] + dy, l, fontweight='bold', fontsize=10, ha='left', va='bottom')

def save(fig,name): fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png',dpi=300); plt.close(fig); print('saved',name)

# predictions (for U4/U8 per-seed AUCs and operating points)
fs=[f for f in glob.glob(str(PREDICTIONS_DIR/'**'/'predictions'/'*.csv*'),recursive=True) if os.path.isfile(f)]
fs+=[x for f in glob.glob(str(PREDICTIONS_DIR/'**'/'predictions'/'*.csv*'),recursive=True) if os.path.isdir(f) for x in glob.glob(f+'/*.csv')]
P=pd.concat([pd.read_csv(f) for f in fs]); P['roi']=P.roi_path.str.split('cbis_rois/').str[-1]
NXT={42:123,123:456,456:42}
def scores(split):
    W,M=wide(P,split); out={}
    for s in SEEDS:
        u4=W[[(f'U_{b}',s) for b in CNNs]].mean(axis=1).values; u4n=W[[(f'U_{b}',NXT[s]) for b in CNNs]].mean(axis=1).values
        out[s]={'U4':u4,'U8':(u4+u4n)/2,**{k:W[(k,s)].values for k in COND}}
    return out,M
St,Mt=scores('test'); Sv,Mv=scores('val'); yt,yv,ct=Mt.label_idx.values,Mv.label_idx.values,Mt.category.values

# ---- FIGURE 1 — STUDY DESIGN
# ---------------- Fig 1: study design ----------------
fig=plt.figure(figsize=(W2,3.9)); ax=fig.add_axes([0,0,1,1]); ax.axis('off'); ax.set_xlim(0,14); ax.set_ylim(0,7.8)
def box(x,y,w,h,title,body,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.12',fc=fc,ec='#555555',lw=0.8))
    ax.text(x+w/2,y+h-0.22,title,ha='center',va='top',fontweight='bold',fontsize=7.6)
    ax.text(x+w/2,y+h-0.62,body,ha='center',va='top',fontsize=6.6,linespacing=1.3)
def arrow(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1,color='#333333'))
box(0.1,4.3,2.5,3.3,'Data','CBIS-DDSM ROIs\nre-extracted and\nprovenance-verified\n\n1,728 / 442 / 497 ROIs\n877 / 220 / 266 patients\nno patient overlap','#DCEBF7')
box(3.1,4.3,3.6,3.3,'Models (×3 seeds, one recipe)','Unified single-backbone (4)\nFinding-type specialists (5×2)\n→ affinity matrix A\n\nEnsemble conditions (12):\nC1, C1+Aux, C2-Gate,\nC3 β∈{0,0.25,0.5,0.75,1},\nC3-Rand, -Oracle, -Static, +ViT','#E6F4EA')
box(7.2,4.3,3.0,3.3,'Comparators','Jointly trained\nensemble (C1)\n\nSimple averages of\nindependent models:\nU4 (4 models)\nU8 (8 models)','#FFF1D6')
box(10.7,4.3,3.2,3.3,'Evaluation','Deterministic inference\nPaired bootstrap:\nseeds × patients\nAchieved sensitivity\nPartial AUC (sens ≥ 0.8)\nContrast settings ×3\nExternal: INbreast','#F5E3EF')
for x1,x2 in [(2.6,3.1),(6.7,7.2),(10.2,10.7)]: arrow(x1,5.95,x2,5.95)
# AFAGE schematic (bottom row)
ax.text(0.1,3.85,'AFAGE (C3)',fontweight='bold',fontsize=8)
small=lambda x,y,w,t,fc: (ax.add_patch(FancyBboxPatch((x,y),w,0.75,boxstyle='round,pad=0.03,rounding_size=0.1',fc=fc,ec='#555555',lw=0.7)),ax.text(x+w/2,y+0.375,t,ha='center',va='center',fontsize=6.6))
small(0.1,2.3,1.5,'ROI\n224×224','#FFFFFF')
for i,b in enumerate(['ResNet-50','EfficientNet-B0','DenseNet-121','ConvNeXt-Tiny']):
    small(2.1,0.2+i*0.85,1.9,b,'#DCEBF7'); arrow(1.6,2.67,2.1,0.57+i*0.85)
small(4.6,2.75,2.4,'Auxiliary finding-type\npredictor  p̂  (Eq. 1)','#E6F4EA')
small(4.6,1.55,2.4,'Affinity prior\ng_aff from A and p̂  (Eq. 3)','#FFF1D6')
small(4.6,0.35,2.4,'Gate G([p̂ ‖ proj(f_b)])\n→ g_raw','#F5E3EF')
for y in [3.12,1.92,0.72]: arrow(4.0,1.9,4.6,y)
small(7.5,1.55,2.9,'Backbone weights w  (Eq. 4)\nsoftmax((g_raw + β g_aff) / τ)','#EEEEEE')
for y in [3.12,1.92,0.72]: arrow(7.0,y,7.5,1.92)
small(10.8,1.55,1.6,'Weighted\nfusion (Eq. 5)','#EEEEEE'); arrow(10.4,1.92,10.8,1.92)
small(12.7,1.55,1.2,'Benign /\nmalignant','#FFFFFF'); arrow(12.4,1.92,12.7,1.92)
ax.text(7.5,0.55,'β = 0: A not used (gate learns from p̂)\nβ = 1: explicit affinity prior',fontsize=6.4,va='center',color='#444444')
save(fig,'fig1_study_design')

# ---- FIGURE 2 — AFFINITY
sa=pd.read_csv(f'{AFFINITY_DIR}/specialist_auc.csv'); it=pd.read_csv(f'{AFFINITY_DIR}/interaction_terms.csv'); cf=pd.read_csv(f'{AFFINITY_DIR}/cross_finding.csv',index_col=0)
fig=plt.figure(figsize=(W2,3.3))
gs=GridSpec(1,3,figure=fig,width_ratios=[1.45,1.0,1.0],left=0.08,right=0.92,bottom=0.30,top=0.90,wspace=1.05)
axa=fig.add_subplot(gs[0]); bbs=list(BB); x=np.arange(len(bbs))
for j,(split,mk) in enumerate([('val','o'),('test','s')]):
    for f,dx in [('Mass',-0.12),('Calcification',0.12)]:
        d=sa[(sa.split==split)&(sa.finding==f)].set_index('backbone').loc[bbs]
        axa.errorbar(x+dx+(j-0.5)*0.1,d['mean'],yerr=d['sd'],fmt=mk,ms=3.5,capsize=1.5,lw=0.8,color=FC[f],mfc=FC[f] if split=='test' else 'white',
                     label=f"{f}, {'validation' if split=='val' else 'test'}")
axa.set_xticks(x); axa.set_xticklabels([BB[b] for b in bbs],rotation=35,ha='right'); axa.set_ylabel('Specialist AUC (own finding type)'); axa.set_ylim(0.5,0.92)
axa.legend(fontsize=6,loc='upper center',bbox_to_anchor=(0.5,-0.36),ncol=2,frameon=False,handletextpad=0.3,columnspacing=0.8)
axb=fig.add_subplot(gs[1]); d=it[it.finding=='Calcification'].set_index('backbone').loc[CNNs]; yy=np.arange(4)[::-1]
axb.errorbar(d.estimate,yy,xerr=[d.estimate-d.ci_lo,d.ci_hi-d.estimate],fmt='o',ms=3.5,capsize=2,color=CB[0],lw=0.9)
axb.axvline(0,color='k',lw=0.7); axb.set_yticks(yy); axb.set_yticklabels([BB[b] for b in CNNs]); axb.set_ylim(-0.6,3.6)
axb.set_xlabel('Interaction term\n(relative calcification advantage)')
axc=fig.add_subplot(gs[2]); cf=cf[['Mass','Calcification']]; rows=[BB[b] for b in bbs]
M_=np.vstack([cf.loc[[f'Mass ({b})' for b in bbs]].values, cf.loc[[f'Calcification ({b})' for b in bbs]].values])
im=axc.imshow(M_,cmap='RdYlGn',vmin=0.45,vmax=0.85,aspect='auto')
for i in range(M_.shape[0]):
    for j in range(2): axc.text(j,i,f'{M_[i,j]:.2f}',ha='center',va='center',fontsize=6)
axc.set_xticks([0,1]); axc.set_xticklabels(['Masses','Calcif.']); axc.set_xlabel('Evaluated on')
axc.set_yticks(range(10)); axc.set_yticklabels(rows+rows,fontsize=6); axc.axhline(4.5,color='k',lw=0.8)
axc.text(-0.98,0.75,'Mass specialists',rotation=90,ha='center',va='center',fontsize=6.3,transform=axc.transAxes)
axc.text(-0.98,0.25,'Calcification specialists',rotation=90,ha='center',va='center',fontsize=6.3,transform=axc.transAxes)
cax=fig.add_axes([0.94,0.38,0.012,0.45]); cb=fig.colorbar(im,cax=cax); cb.ax.tick_params(labelsize=6); cb.set_label('Test AUC',fontsize=6.5)
letters(fig,[(axa,'a'),(axb,'b'),(axc,'c')])
save(fig,'fig2_affinity')

# ---- FIGURE 3 — AUC OVERVIEW
# ---------------- Fig 3: AUC overview incl. U4/U8 ----------------
models=['C1','C1_aux','C2_gate','C3_rand','C3_beta0','C3_beta025','C3_beta05','C3_beta075','C3_beta1','C3_oracle','C3_static','C3_vit5bb','U4','U8']
rows=[]
for s in SEEDS:
    for k in models:
        sc=St[s][k]; rows.append({'m':k,'all':auc(yt,sc),'mass':auc(yt[ct=='Mass'],sc[ct=='Mass']),'calc':auc(yt[ct=='Calcification'],sc[ct=='Calcification'])})
g=pd.DataFrame(rows).groupby('m').agg(['mean','std']).loc[models]
fig,ax=plt.subplots(figsize=(W2,2.5)); x=np.arange(len(models)); w=0.27
for i,(c,nm,col,al) in enumerate([('all','All ROIs','#444444',0.65),('mass','Mass',FC['Mass'],0.9),('calc','Calcification',FC['Calcification'],0.9)]):
    ax.bar(x+(i-1)*w,g[(c,'mean')],w,yerr=g[(c,'std')],capsize=1.2,color=col,alpha=al,label=nm,error_kw=dict(lw=0.6))
ax.axvspan(11.5,13.5,color='#FFF1D6',zorder=0); ax.text(12.5,0.865,'simple\naverages',ha='center',fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([COND.get(k,k) for k in models],rotation=30,ha='right'); ax.set_ylim(0.55,0.9); ax.set_ylabel('Test AUC (mean ± s.d., 3 seeds)')
ax.legend(ncol=3,loc='upper left',frameon=False)
save(fig,'fig3_auc_overview')

# ---- FIGURE 4 — PAIRED DIFFERENCES
# ---------------- Fig 4: paired differences ----------------
cv=pd.read_csv(f'{CONDITIONS_DIR}/comparisons_vs_C1.csv'); a=cv[cv.endpoint=='dAUC'].set_index('condition').loc[[k for k in COND if k!='C1']]
d26=simple_ensemble_differences('clip 4.0 (primary)')
fig,axes=plt.subplots(1,2,figsize=(W2,2.9),gridspec_kw=dict(wspace=0.75))
ax=axes[0]; yy=np.arange(len(a))[::-1]; ax.axvspan(-0.02,0.02,color='grey',alpha=0.15)
for yi,(c,r) in zip(yy,a.iterrows()):
    ax.errorbar(r.estimate,yi,xerr=[[r.estimate-r.ci_lo],[r.ci_hi-r.estimate]],fmt='o',ms=3.5,capsize=2,lw=0.9,color=CB[5] if c in('C3_beta0','C3_beta1') else CB[0])
ax.axvline(0,color='k',lw=0.7); ax.set_yticks(yy); ax.set_yticklabels([COND[c] for c in a.index]); ax.set_xlabel('ΔAUC vs C1 (95% CI)'); letter(ax,'a',-0.42)
ax=axes[1]; order=['C3_beta0 − U4','C3_beta1 − U4','C1 − U4','C3_beta0 − U8','C3_beta1 − U8','C1 − U8']; d=d26.loc[order]; yy=np.arange(len(d))[::-1]
ax.axvspan(-0.02,0.02,color='grey',alpha=0.15)
for yi,(c,r) in zip(yy,d.iterrows()):
    ax.errorbar(r.estimate,yi,xerr=[[r.estimate-r.ci_lo],[r.ci_hi-r.estimate]],fmt='o',ms=3.5,capsize=2,lw=0.9,color=CB[7] if c.startswith('C1') else CB[2])
ax.axvline(0,color='k',lw=0.7); ax.set_yticks(yy); ax.set_yticklabels([c.replace('C3_beta0','C3 β=0').replace('C3_beta1','C3 β=1') for c in order]); ax.set_xlabel('ΔAUC vs simple average (95% CI)'); letter(ax,'b',-0.42)
save(fig,'fig4_paired_differences')

# ---- FIGURE 5 — FALSE POSITIVES AT ACHIEVED SENSITIVITY
# ---------------- Fig 5: FP vs achieved sensitivity ----------------
fig,ax=plt.subplots(figsize=(3.6,2.8))
for i,k in enumerate(['C1','C3_beta0','C3_beta1','U8']):
    pts=[]
    for t in TARGETS:
        fp,se=[],[]
        for s in SEEDS:
            thr=threshold_for_sensitivity(yv,Sv[s][k],t); op=operating_point(yt,St[s][k],thr); fp.append(op['fp']); se.append(op['sens'])
        pts.append((np.mean(se),np.mean(fp),t))
    pts=np.array(pts); ax.plot(pts[:,0],pts[:,1],'o-',ms=3.5,lw=1,color=[CB[7],CB[5],CB[0],CB[2]][i],label=COND.get(k,k))
    if k in ('C1','U8'):
        for se,fp,t in pts: ax.annotate(f'{int(t*100)}%',(se,fp),fontsize=5.8,xytext=(3,-8 if k=='U8' else 3),textcoords='offset points')
ax.set_xlabel('Achieved test sensitivity'); ax.set_ylabel('False positives (test, n = 497)'); ax.legend(frameon=False)
save(fig,'fig5_false_positives')

# ---- FIGURE 6 — MECHANISM
hd=pd.read_csv(f'{MECHANISM_DIR}/headroom.csv'); ca=pd.read_csv(f'{CONDITIONS_DIR}/condition_auc.csv').groupby('condition').auc.mean()
u8=np.mean([auc(yt,St[s]['U8']) for s in SEEDS])
hv=pd.Series({'C1':hd.C1_auc.mean(),'C3 β=0':ca['C3_beta0'],'U8':u8,'C3-Oracle':ca['C3_oracle'],
              'Best per finding\n(validation)':hd[hd['mode'].str.startswith('honest')].composite_auc.mean(),
              'Best per finding\n(test, bound)':hd[hd['mode'].str.startswith('oracle')].composite_auc.mean()})
A0=np.load(AFFINITY_FILE); sm=np.exp(A0)/np.exp(A0).sum(0); gw=pd.read_csv(f'{MECHANISM_DIR}/gate_weights.csv')
fig=plt.figure(figsize=(W2,5.6))
top=GridSpec(1,3,figure=fig,width_ratios=[1.7,1,1],left=0.08,right=0.98,top=0.93,bottom=0.60,wspace=0.35)
bot=GridSpec(1,2,figure=fig,left=0.08,right=0.98,top=0.42,bottom=0.12,wspace=0.28)
axa=fig.add_subplot(top[0]); bars=axa.bar(range(len(hv)),hv.values,color=[CB[7],CB[5],CB[2],CB[3],CB[4],'#AAAAAA'],alpha=0.85)
for r,v in zip(bars,hv.values): axa.text(r.get_x()+r.get_width()/2,v+0.0015,f'{v:.3f}',ha='center',fontsize=6)
axa.set_xticks(range(len(hv))); axa.set_xticklabels(hv.index,fontsize=6,rotation=40,ha='right'); axa.set_ylim(0.74,0.815); axa.set_ylabel('Test AUC')
hm=[]
for j,(M_,nm) in enumerate([(A0,'A (specialist\nvalidation AUC)'),(sm,'softmax over\nbackbones')]):
    ax=fig.add_subplot(top[1+j]); lo,hi=(0.5,1.0) if j==0 else (0.0,0.5)
    ax.imshow(M_,cmap='viridis',aspect='auto',vmin=lo,vmax=hi)
    for i in range(4):
        for k in range(2): ax.text(k,i,f'{M_[i,k]:.3f}',ha='center',va='center',color='w' if (M_[i,k]-lo)/(hi-lo)<0.7 else 'k',fontsize=6)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Mass','Calc.'],fontsize=6.5); ax.set_yticks(range(4))
    ax.set_yticklabels([BB[b] for b in CNNs] if j==0 else [],fontsize=6.3); ax.set_title(nm,fontsize=6.8); hm.append(ax)
axc=[]
for j,k in enumerate(['C3_beta0','C3_beta1']):
    ax=fig.add_subplot(bot[j]); d=gw[gw.condition==k].groupby('backbone')[['w_mass','w_calc','per_roi_sd']].mean().loc[CNNs]; x=np.arange(4)
    ax.bar(x-0.18,d.w_mass,0.36,yerr=d.per_roi_sd,capsize=1.5,color=FC['Mass'],label='Mass ROIs',error_kw=dict(lw=0.6))
    ax.bar(x+0.18,d.w_calc,0.36,yerr=d.per_roi_sd,capsize=1.5,color=FC['Calcification'],label='Calcification ROIs',error_kw=dict(lw=0.6))
    ax.set_xticks(x); ax.set_xticklabels([BB[b] for b in CNNs],fontsize=6.3,rotation=20,ha='right'); ax.set_ylim(0,0.5)
    ax.set_ylabel('Mean gate weight' if j==0 else ''); ax.set_title(COND[k],fontsize=7)
    if j==0: ax.legend(frameon=False,fontsize=6,loc='upper right')
    axc.append(ax)
letters(fig,[(axa,'a'),(hm[0],'b'),(axc[0],'c')])
save(fig,'fig6_mechanism')

# ---- FIGURE 7 — CONTRAST SENSITIVITY
# ---------------- Fig 7: contrast sensitivity ----------------
r26=simple_ensemble_aucs(); r26=r26[(r26.kind=='AUC')&(r26.subset=='all')]
aux={st: pd.read_csv(p).groupby('condition').auc.mean()['C1_aux'] for st,p in [('clip 4.0 (primary)',f'{CONDITIONS_DIR}/condition_auc.csv'),('clip 2.5',f'{CONDITIONS_25}/condition_auc.csv'),('average over clips 1–4',f'{CONDITIONS_AVG}/condition_auc.csv')]}
mods=['C1','C1_aux','C3_beta0','C3_beta1','U4','U8']; fig,ax=plt.subplots(figsize=(3.9,2.7)); x=np.arange(len(mods))
for i,(st,mk) in enumerate([('clip 4.0 (primary)','o'),('clip 2.5','s'),('average over clips 1–4','^')]):
    vals=[aux[st] if m=='C1_aux' else r26[(r26.setting==st)&(r26.model==m)].estimate.iloc[0] for m in mods]
    ax.plot(x+(i-1)*0.18,vals,mk,ms=4.5,ls='none',color=CB[i],label=st[0].upper()+st[1:])
ax.set_xticks(x); ax.set_xticklabels([COND.get(m,m) for m in mods]); ax.set_ylabel('Test AUC (mean of 3 seeds)'); ax.legend(frameon=False,fontsize=6.3,loc='lower right')
save(fig,'fig7_contrast_sensitivity')

# ---- TABLE 1 — PRIMARY COMPARISONS AND INBREAST (LaTeX + CSV)
cv = pd.read_csv(f'{CONDITIONS_DIR}/comparisons_vs_C1.csv'); cv = cv[cv.endpoint == 'dAUC'].set_index('condition')
d26 = simple_ensemble_differences('clip 4.0 (primary)')
def eqm(lo, hi):
    for m in (0.02, 0.03, 0.05):
        if lo > -m and hi < m: return f'$\\pm{m:.2f}$'
    return '--'
rows = []
for name, r, adj in [('C3 $\\beta=0$ $-$ C1$^{\\dagger}$', cv.loc['C3_beta0'], True), ('C3 $\\beta=1$ $-$ C1$^{\\dagger}$', cv.loc['C3_beta1'], True),
                     ('C3 $\\beta=0$ $-$ U8', d26.loc['C3_beta0 − U8'], False), ('C3 $\\beta=1$ $-$ U8', d26.loc['C3_beta1 − U8'], False), ('C1 $-$ U8', d26.loc['C1 − U8'], False)]:
    p = min(1, r.p_boot * 2) if adj else r.p_boot
    rows.append((name, r.estimate, r.ci_lo, r.ci_hi, p, eqm(r.ci90_lo, r.ci90_hi), 2.8 * r.se))
ea = pd.read_csv(EXTERNAL_AUC).set_index('quantity')
ec = pd.read_csv(EXTERNAL_COMPARISONS).set_index('quantity')
f = lambda v: '$0.000$' if abs(v) < 0.0005 else f'${v:+.3f}$'
pf = lambda p: f'{p:.3f}' if p < 0.01 else f'{p:.2f}'
L = [r'\begin{table}[h]', r'\caption{Primary comparisons on CBIS-DDSM and external validation on INbreast masses (clip limit 4.0).}\label{tab:primary}%',
     r'\begin{tabular}{@{}lrrrcr@{}}', r'\toprule', r'\multicolumn{6}{@{}l}{\textit{CBIS-DDSM test set (497 ROIs, 266 patients)}}\\',
     r'Comparison & $\Delta$AUC & 95\% CI & $p$ & Equivalent within & MDE\\', r'\midrule']
for n, e, lo, hi, p, eq, mde in rows:
    L.append(f'{n} & {f(e)} & ${lo:.3f}$ to ${hi:.3f}$ & {pf(p)} & {eq} & {mde:.3f}\\\\'.replace('$-0.','$-0.'))
L += [r'\midrule', r'\multicolumn{6}{@{}l}{\textit{INbreast masses (103 ROIs, 43 patients)}}\\', r'Model or comparison & AUC or $\Delta$AUC & 95\% CI & $p$ & & \\', r'\midrule']
for k, n in [('C1', 'C1'), ('C3_beta0', 'C3 $\\beta=0$'), ('C3_beta1', 'C3 $\\beta=1$'), ('U8', 'U8')]:
    r = ea.loc[k]; L.append(f'{n} & {r.estimate:.3f} & {r.ci_lo:.3f} to {r.ci_hi:.3f} & & & \\\\')
for k, n in [('C3_beta0 − C1', 'C3 $\\beta=0$ $-$ C1'), ('C3_beta1 − C1', 'C3 $\\beta=1$ $-$ C1')]:
    r = ec.loc[k]; L.append(f'{n} & {f(r.estimate)} & ${r.ci_lo:.3f}$ to ${r.ci_hi:.3f}$ & {pf(r.p_boot)} & & \\\\')
L += [r'\botrule', r'\end{tabular}', r'\footnotetext{$^{\dagger}$Primary comparisons; $p$ Bonferroni-adjusted for two comparisons. Other $p$-values unadjusted. Equivalence: narrowest margin among $\pm0.02$, $\pm0.03$ and $\pm0.05$ AUC within which the 90\% interval lies. MDE, minimum detectable effect (2.8 $\times$ bootstrap s.e.). Means over three seeds; intervals from the paired seed $\times$ patient bootstrap.}', r'\end{table}']
open(f'{OUT}/table2_primary.tex', 'w').write('\n'.join(L))
pd.DataFrame(rows, columns=['comparison', 'dAUC', 'ci_lo', 'ci_hi', 'p', 'equivalent_within', 'MDE']).to_csv(f'{OUT}/table2_primary.csv', index=False)
print('\n'.join(L))


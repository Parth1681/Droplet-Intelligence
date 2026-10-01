from pathlib import Path as _P; _ROOT = _P(__file__).resolve().parents[1]  # repo root (droplet-intelligence/)
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ['/root/.fonts/InterVariable.ttf']: fm.fontManager.addfont(f)
V=str(_ROOT/'Droplet_Intelligence_v2')
INK,INK2,MUTED,GRID='#0b0b0b','#52514e','#8a8984','#e6e5e0'
B,O,A='#2a78d6','#eb6834','#1baf7a'
plt.rcParams.update({'font.family':'Inter Variable','font.size':8.5,'axes.edgecolor':MUTED,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':.6,'axes.axisbelow':True,'axes.linewidth':.7,
 'axes.titlesize':9,'axes.titleweight':'semibold','axes.titlelocation':'left','axes.titlecolor':INK,'legend.frameon':False,'savefig.dpi':300})
bm=json.load(open(f'{V}/results/benchmark.json')); M={m['model']:m for m in bm['metrics']}
ab=json.load(open('ablation.json')); sem=json.load(open(f'{V}/results/sem/summary.json'))
nest=json.load(open(f'{V}/results/nested.json')); lg=json.load(open(f'{V}/results/legacy_gpr_vs_xgb.json'))
base=M['baseline']['loso_rmse']
rows=[('Matérn 5/2 GP',M['matern52']['loso_rmse']-base,M['matern52']['delta_vs_baseline_ci95']),
 ('Laan backbone + residual GP',M['physics_residual']['loso_rmse']-base,M['physics_residual']['delta_vs_baseline_ci95']),
 ('SEM CNN descriptors + GP',sem['loso_rmse']-base,sem['delta_vs_baseline_ci95']),
 ('Fluid–surface interaction GP',M['structured']['loso_rmse']-base,M['structured']['delta_vs_baseline_ci95']),
 ('Nested model selection',nest['nested_rmse']-base,nest['delta_vs_baseline_ci95']),
 ('GP without surface descriptors',ab['blind3']['loso_rmse']-base,[-ab['baseline_minus_blind3_ci'][1],-ab['baseline_minus_blind3_ci'][0]]),
 ('XGBoost, same five inputs',lg['XGB']['LOSO']-lg['GPR']['LOSO'],[-lg['LOSO_GPR_minus_XGB_CI'][1],-lg['LOSO_GPR_minus_XGB_CI'][0]])]
json.dump(rows,open('forest_rows.json','w'))
# Fig 1 forest
fig,ax=plt.subplots(figsize=(6.3,2.6)); y=np.arange(len(rows))[::-1]
for yi,(n,d,ci) in zip(y,rows):
    sig=ci[0]>0 or ci[1]<0
    ax.plot(ci,[yi,yi],color=INK2,lw=1.4,solid_capstyle='round')
    ax.plot(d,yi,'o',ms=6,color=O if sig else B,mec='white',mew=1.2,zorder=3)
    ax.text(0.0305,yi,f'{d*1000:+.1f}  [{ci[0]*1000:+.1f}, {ci[1]*1000:+.1f}]',va='center',ha='left',fontsize=7.5,color=INK2)
ax.axvline(0,color=MUTED,lw=.9,ls='--'); ax.set_yticks(y,[r[0] for r in rows]); ax.tick_params(axis='y',length=0); ax.grid(axis='y',visible=False)
ax.set_xlim(-0.016,0.030); ax.set_xticks(np.arange(-0.015,0.031,0.005)); ax.xaxis.set_major_formatter(lambda v,p:'0' if abs(v)<1e-9 else f'{v*1000:+.0f}')
ax.set_xlabel('Change in unseen-surface RMSE vs baseline GP (×10⁻³ β units; left = better)')
ax.text(0.0305,len(rows)-0.35,'Δ×10⁻³  [95% CI]',fontsize=7.5,color=INK,weight='semibold',ha='left')
fig.savefig('fig1_forest.png',bbox_inches='tight',facecolor='white'); plt.close()
# Fig 2 per-surface
order=['S50','S100','S200','S400','S600','S800','D50','D100','D200','D400','D600','D800']
ser=[('Baseline GP (5 inputs)',M['baseline']['per_surface'],B,'o'),('GP without surface descriptors',ab['blind3']['per_surface'],A,'s'),('XGBoost, same five inputs',lg['XGB']['per_surface'],O,'D')]
fig,ax=plt.subplots(figsize=(6.3,2.7)); x=np.arange(len(order))
for i,(n,d,c,mk) in enumerate(ser):
    ax.plot(x+(i-1)*0.18,[d[s] for s in order],mk,ms=5.5,color=c,mec='white',mew=.8,label=n,zorder=3)
ax.axvline(5.5,color=MUTED,lw=.8); ax.text(2.5,0.121,'Shallow channels (6 µm)',ha='center',fontsize=7.5,color=INK2); ax.text(8.5,0.121,'Deep channels (25 µm)',ha='center',fontsize=7.5,color=INK2)
ax.set_xticks(x,order); ax.grid(axis='x',visible=False); ax.set_ylim(0,0.128); ax.set_ylabel('RMSE when surface held out')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=3,fontsize=7.5,handletextpad=.3)
fig.savefig('fig2_per_surface.png',bbox_inches='tight',facecolor='white'); plt.close()
# Fig 3 parity
d=pd.read_csv(f'{V}/data/experiments.csv'); t=d[d.split=='textured'].set_index('row_id'); r=d[d.split=='REF-H'].reset_index(drop=True)
lo=pd.read_csv(f'{V}/results/loso_predictions.csv'); lo=lo[lo.model=='baseline']; lo['gly']=t.loc[lo.row_id,'glycerol'].values
rf=pd.read_csv(f'{V}/results/refh_predictions.csv'); rf=rf[rf.model=='baseline'].reset_index(drop=True); rf['gly']=r.glycerol.values
glys=[0,20,60,78,91]; cm=plt.get_cmap('Blues'); col={g:cm(0.35+0.6*i/4) for i,g in enumerate(glys)}
fig,axs=plt.subplots(1,2,figsize=(6.3,3.0),sharex=True,sharey=True)
for ax,df,ttl in [(axs[0],lo,'a  Unseen textured surface (LOSO, n = 1,498)'),(axs[1],rf,'b  Smooth reference REF-H (n = 125)')]:
    ax.plot([1.35,3.45],[1.35,3.45],color=MUTED,lw=.8,ls='--',zorder=1)
    for g in glys:
        m=df.gly==g; ax.scatter(df.beta[m],df.prediction[m],s=9,color=col[g],edgecolor='white',linewidth=.3,label=f'{g} wt%',zorder=2)
    e=df.prediction-df.beta; ax.text(0.04,0.96,f'RMSE {np.sqrt((e**2).mean()):.3f}\nMAPE {(e.abs()/df.beta).mean()*100:.1f}%',transform=ax.transAxes,va='top',fontsize=7.5,color=INK2)
    ax.set_title(ttl,fontsize=8.5); ax.set_xlabel('Measured β_max'); ax.set_aspect('equal')
axs[0].set_ylabel('Predicted β_max'); axs[0].set_xlim(1.35,3.45); axs[0].set_ylim(1.35,3.45)
axs[1].legend(title='Glycerol',title_fontsize=7.5,fontsize=7,loc='lower right',handletextpad=.1,markerscale=1.4)
fig.savefig('fig3_parity.png',bbox_inches='tight',facecolor='white'); plt.close()
# Fig 4 by fluid
z=1.6448536; lo['cov']=(lo.beta>=np.exp(lo.mu_log-z*lo.sd_log))&(lo.beta<=np.exp(lo.mu_log+z*lo.sd_log)); rf['cov']=(rf.beta>=rf.lower)&(rf.beta<=rf.upper)
S=[]
for g in glys:
    a,bb=lo[lo.gly==g],rf[rf.gly==g]
    S.append(dict(gly=g,loso_rmse=float(np.sqrt(((a.prediction-a.beta)**2).mean())),refh_rmse=float(np.sqrt(((bb.prediction-bb.beta)**2).mean())),loso_cov=float(a['cov'].mean()),refh_cov=float(bb['cov'].mean()),mu=float(t[t.glycerol==g].mu.iloc[0])))
json.dump(S,open('by_fluid.json','w'),indent=1)
fig,axs=plt.subplots(1,2,figsize=(6.3,2.5)); x=np.arange(5); w=.36
for ax,k,yl,tt in [(axs[0],'rmse','RMSE of β_max','a  Error by fluid'),(axs[1],'cov','Share inside 90% interval','b  Interval coverage by fluid')]:
    for j,(lab,c,key) in enumerate([('Unseen textured (LOSO)',B,'loso_'),('Smooth REF-H',O,'refh_')]):
        v=[s[key+k] for s in S]; ax.bar(x+(j-.5)*w,v,w*.9,color=c,label=lab,zorder=2)
    ax.set_xticks(x,[f'{g}%' for g in glys]); ax.grid(axis='x',visible=False); ax.set_ylabel(yl); ax.set_title(tt); ax.set_xlabel('Glycerol (wt%) — viscosity rises left to right')
axs[1].axhline(.9,color=INK2,lw=.9,ls='--'); axs[1].text(-0.55,.925,'90% target',ha='left',fontsize=7,color=INK2); axs[1].set_ylim(0,1.05); axs[1].yaxis.set_major_formatter(lambda v,p:f'{v:.0%}')
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=7.5)
fig.tight_layout(w_pad=2,rect=(0,0.06,1,1)); fig.savefig('fig4_by_fluid.png',bbox_inches='tight',facecolor='white'); plt.close()
print(json.dumps(S,indent=0))

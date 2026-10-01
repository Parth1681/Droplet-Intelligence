from pathlib import Path as _P; _ROOT = _P(__file__).resolve().parents[2]  # repo root (droplet-intelligence/)
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for w in (400,500,600): fm.fontManager.addfont(f'fonts/ibm-plex-sans-latin-{w}-normal.ttf')
V=str(_ROOT/'Droplet_Intelligence_v2'); N=json.load(open(str(_ROOT/'paper/numbers.json')))
INK,INK2,GRID='#3b2f27','#5e5249','#e6dfd7'; BR,TAN,DOT='#7a4f32','#c9a27a','#a08872'
plt.rcParams.update({'font.family':['IBM Plex Sans','DejaVu Sans'],'font.size':29,'axes.edgecolor':'#9a8e84','axes.labelcolor':INK2,
 'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':1.1,
 'axes.axisbelow':True,'axes.linewidth':1.3,'svg.fonttype':'path','legend.frameon':False,'mathtext.fontset':'dejavusans'})
MM=1/25.4
def save(fig,n): fig.savefig(n+'.svg',transparent=True,bbox_inches='tight',pad_inches=0.12); plt.close(fig)

# 1 models
z=pd.DataFrame(N['zoo']).sort_values('loso',ascending=False)
fig,ax=plt.subplots(figsize=(215*MM,172*MM)); y=np.arange(len(z))
for yi,(_,r) in zip(y,z.iterrows()):
    gp=r['name']=='Gaussian process'
    ax.hlines(yi,0,r.loso,color=BR if gp else GRID,lw=5 if gp else 3.2,zorder=1)
    ax.plot(r.loso,yi,'o',ms=17 if gp else 12,color=BR if gp else DOT,mec='white',mew=1.8,zorder=3)
    ax.text(r.loso+.006,yi,f'{r.loso:.3f}',va='center',fontsize=23.5,color=INK if gp else INK2,weight=600 if gp else 400)
ax.set_yticks(y,z['name'],fontsize=24.8); ax.tick_params(axis='y',length=0)
for t in ax.get_yticklabels():
    if t.get_text()=='Gaussian process': t.set_color(INK); t.set_fontweight(600)
ax.grid(axis='y',visible=False); ax.set_xlim(0,.3); ax.set_xlabel('RMSE, held-out surface')
save(fig,'f_models')

# 2 coverage by fluid
S=N['by_fluid']; x=np.arange(len(S)); w=.38
fig,ax=plt.subplots(figsize=(250*MM,118*MM))
for j,(lab,c,k) in enumerate([('Unseen textured surface',BR,'loso_cov'),('Smooth plate REF-H',TAN,'refh_cov')]):
    v=[s[k] for s in S]; ax.bar(x+(j-.5)*w,v,w*.92,color=c,label=lab,zorder=2)
    for xi,vi in zip(x+(j-.5)*w,v): ax.text(xi,.03,f'{vi:.0%}',ha='center',va='bottom',fontsize=17,color='white' if j==0 else INK,weight=600)
ax.axhline(.9,color=INK,lw=1.6,ls='--',zorder=3); ax.text(-.62,.915,'90% target',fontsize=20.7,color=INK,ha='left',va='bottom')
ax.set_xticks(x,[f"{s['gly']}%" for s in S]); ax.set_xlabel('Glycerol (wt%)')
ax.set_ylim(0,1.13); ax.set_yticks([0,.25,.5,.75,1]); ax.yaxis.set_major_formatter(lambda v,p:f'{v:.0%}')
ax.set_ylabel('Inside 90% interval'); ax.grid(axis='x',visible=False)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2,fontsize=22.1,handlelength=1)
save(fig,'f_fluid')

# 3 REF-H parity
d=pd.read_csv(f'{V}/data/experiments.csv'); r=d[d.split=='REF-H'].reset_index(drop=True)
rf=pd.read_csv(f'{V}/results/refh_predictions.csv'); rf=rf[rf.model=='baseline'].reset_index(drop=True); rf['gly']=r.glycerol.values
fig,ax=plt.subplots(figsize=(200*MM,165*MM))
ax.plot([1.35,3],[1.35,3],color=INK2,lw=1.6,ls='--')
cols=['#e2c9ab','#c9a27a','#a8744c','#7a4f32','#3b2f27']
for i,g in enumerate([0,20,60,78,91]):
    m=rf.gly==g; ax.scatter(rf.beta[m],rf.prediction[m],s=95,color=cols[i],edgecolor='white',linewidth=.7,label=f'{g}%',zorder=3)
ax.set_xlim(1.35,3); ax.set_ylim(1.35,3); ax.set_xlabel('Measured $\\beta_{\\rm max}$'); ax.set_ylabel('Predicted $\\beta_{\\rm max}$')
ax.legend(title='Glycerol',title_fontsize=26.6,fontsize=19.3,loc='lower right',handletextpad=.1,labelspacing=.25)
save(fig,'f_parity')

# 4 forest
rows=N['forest']
fig,ax=plt.subplots(figsize=(185*MM,118*MM)); y=np.arange(len(rows))[::-1]
for yi,(n,dd,ci) in zip(y,rows):
    sig=ci[0]>0 or ci[1]<0
    ax.plot([ci[0]*1e3,ci[1]*1e3],[yi,yi],color=INK2,lw=2.6,solid_capstyle='round')
    ax.plot(dd*1e3,yi,'o',ms=15,color=TAN if not sig else BR,mec='white',mew=1.6,zorder=3)
ax.axvline(0,color=INK2,lw=1.4,ls='--')
ax.set_yticks(y,[r[0] for r in rows],fontsize=22.1); ax.tick_params(axis='y',length=0); ax.grid(axis='y',visible=False)
ax.set_xlim(-6,29); ax.set_xlabel('Δ RMSE vs baseline GP (×10⁻³)',fontsize=23.5)
ax.text(-5.6,len(rows)-.3,'← better',fontsize=20.7,color=INK2); ax.text(28.6,len(rows)-.3,'worse →',fontsize=20.7,color=INK2,ha='right')
save(fig,'f_forest')
print('ok')

# 5 CNN phi vs geometry phi
fd=json.load(open(f'{V}/results/sem/summary.json'))['fold_descriptors']
geo={s['surface']:s['phi'] for s in N['surfaces']}
refh_cnn=json.load(open(f'{V}/models/sem_gp.json'))['surface_descriptors']['REF-H']['phi']
fig,ax=plt.subplots(figsize=(118*MM,112*MM))
ax.plot([0,1.05],[0,1.05],color=INK2,lw=1.5,ls='--')
for s,v in fd.items():
    deep=s.startswith('D'); ax.plot(geo[s],v[0],'o' if deep else 's',ms=15,color=INK if deep else TAN,mec='white',mew=1.5,zorder=3)
ax.plot(1.0,refh_cnn,'D',ms=17,color=BR,mec='white',mew=1.5,zorder=4)
ax.annotate('REF-H read\nas $\\varphi$ = %.2f'%refh_cnn,(1.0,refh_cnn),(0.45,0.12),fontsize=22,color=BR,arrowprops=dict(arrowstyle='-',color=BR,lw=1.4))
ax.plot([],[],'o',ms=13,color=INK,label='deep'); ax.plot([],[],'s',ms=13,color=TAN,label='shallow')
ax.legend(loc='upper left',fontsize=21,title='held out',title_fontsize=20,handletextpad=.1,borderaxespad=.1,alignment='left')
ax.tick_params(labelsize=22)
ax.set_xlim(-.03,1.07); ax.set_ylim(-.03,1.07); ax.set_aspect('equal')
ax.set_xticks([0,.5,1]); ax.set_yticks([0,.5,1])
ax.set_xlabel('$\\varphi$ from geometry',fontsize=24); ax.set_ylabel('$\\varphi$ read by CNN',fontsize=24)
save(fig,'f_cnn')

# 6 per held-out surface
ab=json.load(open(str(_ROOT/'paper/ablation.json'))); lg=json.load(open(f'{V}/results/legacy_gpr_vs_xgb.json'))
bm={m['model']:m for m in json.load(open(f'{V}/results/benchmark.json'))['metrics']}
order=['S50','S100','S200','S400','S600','S800','D50','D100','D200','D400','D600','D800']
ser=[('GP',bm['baseline']['per_surface'],BR,'o'),('GP without $\\varphi$, $V_{\\rm tex}$',ab['blind3']['per_surface'],TAN,'s'),('XGBoost',lg['XGB']['per_surface'],INK,'D')]
fig,ax=plt.subplots(figsize=(262*MM,112*MM)); x=np.arange(12)
for i,(n,dd,c,mk) in enumerate(ser):
    ax.plot(x+(i-1)*.22,[dd[s] for s in order],mk,ms=13,color=c,mec='white',mew=1.3,label=n,zorder=3)
ax.axvline(5.5,color=INK2,lw=1.3)
ax.text(2.5,.126,'shallow · 6 µm',ha='center',fontsize=22,color=INK2); ax.text(8.5,.126,'deep · 25 µm',ha='center',fontsize=22,color=INK2)
ax.set_xticks(x,[s[1:] for s in order],fontsize=21); ax.tick_params(axis='y',labelsize=22); ax.yaxis.label.set_size(24); ax.set_xlabel('Channel pitch of the held-out surface (µm)',fontsize=24); ax.grid(axis='x',visible=False)
ax.set_ylim(0,.135); ax.set_yticks([0,.04,.08,.12]); ax.set_ylabel('RMSE')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.3),ncol=3,fontsize=21,handletextpad=.2,columnspacing=1.2)
save(fig,'f_surface')
print('ok2')

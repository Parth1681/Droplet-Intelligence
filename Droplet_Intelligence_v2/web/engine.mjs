// Float64 implementation of the exported GP posterior.
export function featureVector(p,r){
 if(!p||typeof p!=='object'||Array.isArray(p))throw Error('Input must be an object');
 const allowed=['D_mm','V','rho','sigma','mu','surface','spacing','depth','phi','texvol','model'];
 for(const k of Object.keys(p))if(!allowed.includes(k))throw Error(`Unknown field: ${k}`);
 for(const k of ['D_mm','V','rho','sigma','mu'])if(typeof p[k]!=='number'||!Number.isFinite(p[k])||p[k]<=0)throw Error(`${k} must be a finite positive number`);
 const surface=p.surface??'D200';let g;
 if(r.surfaces[surface]){if(['spacing','depth','phi','texvol'].some(k=>k in p))throw Error('Use surface=custom when supplying geometry');g=r.surfaces[surface]}
 else if(surface==='custom'){g={};for(const k of ['spacing','depth','phi','texvol']){if(typeof p[k]!=='number'||!Number.isFinite(p[k])||p[k]<0)throw Error(`Custom ${k} must be finite and nonnegative`);g[k]=p[k]}if(g.phi>1)throw Error('phi must be at most 1')}
 else throw Error('Unknown surface');
 const D=p.D_mm*.001,Re=p.rho*p.V*D/p.mu,We=p.rho*p.V*p.V*D/p.sigma,Oh=Math.sqrt(We)/Re,P=We*Math.pow(Re,-.4),x=[Math.log(Re),Math.log(We),Math.log(p.D_mm),g.phi,g.texvol];
 if(![...x,Re,We,Oh,P].every(Number.isFinite))throw Error('Inputs exceed numerical limits');
 return {x,Re,We,Oh,P,D,V:p.V,rho:p.rho,sigma:p.sigma,mu:p.mu,...g,surface};
}
function m32(r){const z=Math.sqrt(3)*r;return (1+z)*Math.exp(-z)}
export class GP{
 constructor(meta,L){this.m=meta;this.L=L}
 posterior(f,variance=true){
  const m=this.m,x=f.x.map((v,i)=>(v-m.xmean[i])/m.xscale[i]),ks=new Float64Array(m.n),s=m.kernel;let sum=0;
  for(let j=0;j<m.n;j++){let rf=0,rs=0;for(let i=0;i<5;i++){const z=(x[i]-m.X[j][i])/s.length_scale[i];if(i<3)rf+=z*z;else rs+=z*z}let k;
   if(s.type==='structured'){const a=m32(Math.sqrt(rf)),b=m32(Math.sqrt(rs));k=s.amplitudes[0]*a+s.amplitudes[1]*b+s.amplitudes[2]*a*b}
   else{const v=Math.sqrt(rf+rs);k=s.amplitude*(s.nu===1.5?m32(v):(1+Math.sqrt(5)*v+5*v*v/3)*Math.exp(-Math.sqrt(5)*v))}ks[j]=k;sum+=k*m.alpha[j];}
  let mu=sum*m.yscale+m.ymean;if(m.A!==null){const p=Math.sqrt(f.P);mu+=Math.log(Math.pow(f.Re,.2)*p/(m.A+p))}if(!variance)return {mu,sd:0};
  const v=new Float64Array(m.n);let norm=0,offset=0;for(let i=0;i<m.n;i++){let a=ks[i];for(let j=0;j<i;j++)a-=this.L[offset+j]*v[j];v[i]=a/this.L[offset+i];norm+=v[i]*v[i];offset+=i+1}
  const diag=s.type==='structured'?s.amplitudes.reduce((a,b)=>a+b,0):s.amplitude;return {mu,sd:Math.sqrt(Math.max(diag+m.noise-norm,1e-12))*m.yscale};
 }
 predict(p,r){
  const f=featureVector(p,r),m=this.m,reasons=[];
  if(m.kind==='sem_gp'){const d=m.surface_descriptors[f.surface];if(d){f.phi=d.phi;f.texvol=d.texvol;f.x[3]=d.phi;f.x[4]=d.texvol}reasons.push('Experimental SEM branch: nominal GP interval excludes image-encoder uncertainty and is not conformal-calibrated')}
  const {mu,sd}=this.posterior(f);
  for(const k of ['D','V','Re','We','rho','sigma','mu','phi','texvol']){const [lo,hi]=r.support[k];if(f[k]<lo-1e-12||f[k]>hi+1e-12)reasons.push(`${k} outside measured training range`)}
  if(f.surface==='REF-H'||f.phi>=.999)reasons.push('Smooth surface: historical interval undercoverage; wettability is not modeled');if(f.surface==='custom')reasons.push('Custom surface: supplied descriptors are not independently validated');
  const od=r.ood,z=f.x.map((v,i)=>(v-od.mean[i])/od.scale[i]);let d=0;for(let i=0;i<5;i++)for(let j=0;j<5;j++)d+=z[i]*od.precision[i][j]*z[j];d=Math.sqrt(Math.max(d,0));
  let lo=0,hi=od.sorted_distances.length;while(lo<hi){const mid=(lo+hi)>>1;if(od.sorted_distances[mid]<d)lo=mid+1;else hi=mid}const pct=lo/od.sorted_distances.length*100;
  if(pct>=95)reasons.push('Feature-space anomaly score at or above training 95th percentile');
  const band=m.q*sd,wide=Math.max(band,m.abs_width_log),v=[mu,mu-band,mu+band,mu-wide,mu+wide].map(Math.exp);if(!v.every(Number.isFinite))throw Error('Prediction exceeds numerical limits');
  return {model_version:r.version,model:m.kind,beta_max:v[0],point_estimator:'lognormal median',interval:{lower:v[1],upper:v[2],nominal_level:.9,method:m.kind==='sem_gp'?'nominal GP, conditional on CNN descriptors':'empirical CV calibration'},wide_band:{lower:v[3],upper:v[4],method:'empirical sensitivity band, no coverage guarantee'},mu_log:mu,sd_log:sd,physics:{Re:f.Re,We:f.We,Oh:f.Oh,P:f.P,phi:f.phi,texvol:f.texvol},reliability:{status:m.kind==='sem_gp'?(reasons.length>1?'extrapolation':'experimental'):(reasons.length?'extrapolation':'within_measured_support'),reasons,ood_percentile:pct}};
 }
}

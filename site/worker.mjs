import {GP} from './engine.mjs';
let release;const models=new Map();
async function getRelease(){return release??=await fetch('./models/release.json').then(r=>{if(!r.ok)throw Error('Model manifest unavailable');return r.json()})}
async function getModel(kind){if(models.has(kind))return models.get(kind);const r=await getRelease();if(!r.candidates[kind])throw Error('Unknown model');
 const [m,b]=await Promise.all([fetch(`./models/${kind}.json`).then(r=>r.json()),fetch(`./models/${kind}.L.bin`).then(r=>r.arrayBuffer())]);if(b.byteLength!==m.n*(m.n+1)/2*8)throw Error('Incomplete model download');
 if(globalThis.crypto?.subtle){const d=await crypto.subtle.digest('SHA-256',b),h=Array.from(new Uint8Array(d),x=>x.toString(16).padStart(2,'0')).join('');if(h!==m.L_sha256)throw Error('Model integrity check failed')}
 const model=new GP(m,new Float64Array(b));models.set(kind,model);return model;}
self.onmessage=async({data})=>{const {id,action,payload}=data;try{const r=await getRelease();let result;
 if(action==='init'){await getModel(r.selected);result=r}
 else if(action==='predict'){const m=await getModel(payload.model??r.selected),start=performance.now();result=m.predict(payload,r);result.compute_ms=performance.now()-start}
 else if(action==='sweep'){const m=await getModel(payload.model??r.selected);result=[];for(let i=0;i<45;i++){const V=.48+1.27*i/44,p=m.predict({...payload,V},r);result.push({V,beta:p.beta_max,lo:p.interval.lower,hi:p.interval.upper})}}
 else if(action==='compare'){result=[];for(const kind of Object.keys(r.candidates)){const m=await getModel(kind);result.push(m.predict({...payload,model:kind},r))}}else throw Error('Unknown action');self.postMessage({id,result});
 }catch(e){self.postMessage({id,error:e.message})}};

import { useState, useRef, useEffect } from "react";

const API_URL = "http://localhost:5000";

const FIELDS = [
  { name:"koi_period",       label:"Orbital Period",    unit:"days",  min:0.5,   max:1000,   step:0.01,  tooltip:"Days for planet to orbit its star once" },
  { name:"koi_duration",     label:"Transit Duration",  unit:"hrs",   min:0.5,   max:24,     step:0.01,  tooltip:"How long the brightness dip lasts" },
  { name:"koi_depth",        label:"Transit Depth",     unit:"ppm",   min:10,    max:100000, step:1,     tooltip:"Star brightness drop during transit" },
  { name:"koi_impact",       label:"Impact Parameter",  unit:"",      min:0,     max:1.5,    step:0.001, tooltip:"0=center crossing, 1=edge crossing" },
  { name:"koi_model_snr",    label:"Signal-to-Noise",   unit:"",      min:0,     max:2000,   step:0.1,   tooltip:"Signal strength vs background noise" },
  { name:"koi_num_transits", label:"Transit Count",     unit:"",      min:1,     max:5000,   step:1,     tooltip:"Number of observed transits" },
  { name:"koi_ror",          label:"Radius Ratio",      unit:"",      min:0.001, max:0.9,    step:0.001, tooltip:"Planet radius / star radius" },
  { name:"teff",             label:"Star Temperature",  unit:"K",     min:2500,  max:10000,  step:1,     tooltip:"Host star effective temperature" },
  { name:"logg",             label:"Surface Gravity",   unit:"log g", min:1,     max:5.5,    step:0.01,  tooltip:"Logarithm of stellar surface gravity" },
  { name:"feh",              label:"Metallicity",       unit:"dex",   min:-2.5,  max:1.0,    step:0.01,  tooltip:"Stellar metal content vs Sun" },
];

const SAMPLE_CONFIRMED = { koi_period:9.49,koi_duration:2.96,koi_depth:615.8,koi_impact:0.15,koi_model_snr:35.8,koi_num_transits:142,koi_ror:0.022,teff:5455,logg:4.47,feh:0.12 };
const SAMPLE_FP = { koi_period:1.74,koi_duration:2.41,koi_depth:8079,koi_impact:1.28,koi_model_snr:505,koi_num_transits:621,koi_ror:0.387,teff:5342,logg:4.52,feh:0.10 };
const PLANET_META = {
  "Rocky / Earth-like":      {color:"#c2855a",glow:"#c2855a44",emoji:"🌍",size:32},
  "Super-Earth":              {color:"#4a90d9",glow:"#4a90d944",emoji:"🌏",size:42},
  "Mini-Neptune":             {color:"#5b7fe8",glow:"#5b7fe844",emoji:"🔵",size:54},
  "Neptune-like":             {color:"#3aa8e8",glow:"#3aa8e844",emoji:"🌊",size:64},
  "Gas Giant / Jupiter-like": {color:"#e8a83a",glow:"#e8a83a44",emoji:"🪐",size:80},
};

const btnStyle=(bg,border,color)=>({padding:"7px 16px",background:bg,border:`1px solid ${border}`,borderRadius:8,color,cursor:"pointer",fontSize:12,fontWeight:600});

function Tooltip({text}){
  const [v,setV]=useState(false);
  return(<span style={{position:"relative",display:"inline-flex"}} onMouseEnter={()=>setV(true)} onMouseLeave={()=>setV(false)}>
    <span style={{width:14,height:14,borderRadius:"50%",background:"#1e3a5f",display:"inline-flex",alignItems:"center",justifyContent:"center",fontSize:9,color:"#60a5fa",cursor:"help",fontWeight:700}}>?</span>
    {v&&<div style={{position:"absolute",bottom:"120%",left:"50%",transform:"translateX(-50%)",background:"#0f172a",border:"1px solid #334155",borderRadius:8,padding:"6px 10px",fontSize:11,color:"#cbd5e1",whiteSpace:"normal",width:180,zIndex:100,boxShadow:"0 8px 24px #00000088"}}>{text}</div>}
  </span>);
}

function Bar({label,value,color}){
  return(<div style={{marginBottom:8}}>
    <div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}>
      <span style={{fontSize:11,color:"#94a3b8"}}>{label}</span>
      <span style={{fontSize:11,fontWeight:700,color}}>{value}%</span>
    </div>
    <div style={{height:6,background:"#1e293b",borderRadius:3,overflow:"hidden"}}>
      <div style={{height:"100%",width:`${value}%`,background:color,borderRadius:3,transition:"width 1s cubic-bezier(.4,0,.2,1)"}}/>
    </div>
  </div>);
}

// ── ANIMATED NUMBER ──────────────────────────────────────────
function AnimNum({target,decimals=2,duration=1500}){
  const [val,setVal]=useState(0);
  useEffect(()=>{
    let start=null;
    const step=ts=>{
      if(!start) start=ts;
      const p=Math.min((ts-start)/duration,1);
      setVal(parseFloat((p*target).toFixed(decimals)));
      if(p<1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  },[target]);
  return <span>{val.toFixed(decimals)}</span>;
}

// ── ANIMATED BAR ─────────────────────────────────────────────
function AnimBar({imp,color,label,delay=0}){
  // imp is 0-1 float e.g. 0.187. Bar fills to imp*100% of full width.
  const [w,setW]=useState(0);
  useEffect(()=>{const t=setTimeout(()=>setW(imp*100),delay+150);return()=>clearTimeout(t);},[imp,delay]);
  return(
    <div style={{marginBottom:12}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
        <span style={{fontSize:12,color:"#94a3b8",fontFamily:"monospace"}}>{label}</span>
        <span style={{fontSize:12,fontWeight:700,color}}>{(imp*100).toFixed(1)}%</span>
      </div>
      <div style={{height:8,background:"#1e293b",borderRadius:4,overflow:"hidden"}}>
        <div style={{height:"100%",width:`${w}%`,background:`linear-gradient(90deg,${color}99,${color})`,borderRadius:4,
          transition:`width 1.2s ${delay}ms cubic-bezier(.4,0,.2,1)`}}/>
      </div>
    </div>
  );
}

// ── SINGLE PREDICTION TAB ────────────────────────────────────
function SingleTab({onResult}){
  const [form,setForm]=useState({});
  const [errors,setErrors]=useState({});
  const [loading,setLoading]=useState(false);
  const [apiErr,setApiErr]=useState("");
  const handleChange=e=>{setForm(f=>({...f,[e.target.name]:e.target.value}));if(errors[e.target.name])setErrors(er=>({...er,[e.target.name]:""}));};
  const validate=()=>{const errs={};FIELDS.forEach(({name,min,max})=>{const v=form[name];if(!v&&v!==0){errs[name]="Required";return;}const n=parseFloat(v);if(isNaN(n)){errs[name]="Must be a number";return;}if(n<min||n>max)errs[name]=`${min} – ${max}`;});return errs;};
  const submit=async()=>{const errs=validate();if(Object.keys(errs).length){setErrors(errs);return;}setLoading(true);setApiErr("");onResult(null);const payload=Object.fromEntries(Object.entries(form).map(([k,v])=>[k,parseFloat(v)]));try{const res=await fetch(`${API_URL}/predict/full`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const data=await res.json();if(!res.ok){if(data.errors)setErrors(data.errors);else setApiErr(data.error||"Server error");return;}onResult(data);}catch{setApiErr("Cannot reach API. Make sure Flask is running on port 5000.");}finally{setLoading(false);}};
  const loadSample=s=>{setForm(Object.fromEntries(Object.entries(s).map(([k,v])=>[k,String(v)])));setErrors({});setApiErr("");onResult(null);};
  return(<div>
    <div style={{display:"flex",gap:8,marginBottom:20,flexWrap:"wrap"}}>
      <button onClick={()=>loadSample(SAMPLE_CONFIRMED)} style={btnStyle("#1e3a5f","#3b82f6","#60a5fa")}>✅ Confirmed Sample</button>
      <button onClick={()=>loadSample(SAMPLE_FP)} style={btnStyle("#2d1515","#ef4444","#f87171")}>❌ False Positive Sample</button>
      <button onClick={()=>{setForm({});setErrors({});setApiErr("");onResult(null);}} style={btnStyle("#1e293b","#475569","#94a3b8")}>🗑 Clear</button>
    </div>
    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
      {FIELDS.map(({name,label,unit,min,max,step,tooltip})=>(
        <div key={name}>
          <label style={{fontSize:11,color:"#94a3b8",display:"flex",alignItems:"center",gap:5,marginBottom:4}}>
            {label}{unit&&<span style={{color:"#475569"}}>({unit})</span>}<Tooltip text={tooltip}/>
          </label>
          <input type="number" name={name} value={form[name]||""} onChange={handleChange} min={min} max={max} step={step} placeholder={`${min} – ${max}`}
            style={{width:"100%",padding:"8px 12px",borderRadius:8,fontSize:13,background:errors[name]?"#2d1515":"#1e293b",border:`1px solid ${errors[name]?"#ef4444":"#334155"}`,color:"#e2e8f0",outline:"none",boxSizing:"border-box"}}/>
          {errors[name]&&<span style={{fontSize:10,color:"#f87171"}}>{errors[name]}</span>}
        </div>
      ))}
    </div>
    {apiErr&&<div style={{marginTop:14,padding:"10px 14px",background:"#2d1515",border:"1px solid #ef4444",borderRadius:8,color:"#f87171",fontSize:12}}>⚠️ {apiErr}</div>}
    <button onClick={submit} disabled={loading} style={{marginTop:18,width:"100%",padding:"13px 0",background:loading?"#334155":"linear-gradient(90deg,#3b82f6,#8b5cf6)",border:"none",borderRadius:10,color:"white",fontSize:15,fontWeight:700,cursor:loading?"not-allowed":"pointer",boxShadow:loading?"none":"0 4px 20px #3b82f644",transition:"all .2s"}}>
      {loading?"⏳ Analyzing Signal...":"🚀 Analyze Signal"}
    </button>
  </div>);
}

function SingleResult({result}){
  if(!result) return(<div style={{padding:40,textAlign:"center"}}><div style={{fontSize:52,marginBottom:12}}>🌠</div><p style={{color:"#475569",fontSize:13}}>Enter parameters and click<br/><strong style={{color:"#60a5fa"}}>Analyze Signal</strong></p></div>);
  const isConf=result.classification.prediction==="CONFIRMED";
  const pm=result.regression?(PLANET_META[result.regression.planet_category]||PLANET_META["Mini-Neptune"]):null;
  return(<div style={{display:"flex",flexDirection:"column",gap:12}}>
    <div style={{background:"#0f172a",border:`1px solid ${isConf?"#16a34a":"#dc2626"}`,borderRadius:14,padding:18}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
        <span style={{fontSize:11,color:"#94a3b8",fontWeight:600,letterSpacing:1}}>TASK A · CLASSIFICATION</span>
        <span style={{padding:"3px 12px",borderRadius:20,fontSize:11,fontWeight:700,background:isConf?"#14532d":"#7f1d1d",color:isConf?"#4ade80":"#f87171"}}>{isConf?"✅ CONFIRMED":"❌ FALSE POSITIVE"}</span>
      </div>
      <div style={{fontSize:30,fontWeight:800,color:isConf?"#4ade80":"#f87171",marginBottom:12}}>{result.classification.confidence}% <span style={{fontSize:13,color:"#94a3b8",fontWeight:400}}>confidence</span></div>
      <Bar label="Confirmed Planet" value={result.classification.probabilities.CONFIRMED} color="#4ade80"/>
      <Bar label="False Positive" value={result.classification.probabilities.FALSE_POSITIVE} color="#f87171"/>
    </div>
    {result.regression?(
      <div style={{background:"#0f172a",border:"1px solid #1d4ed8",borderRadius:14,padding:18}}>
        <span style={{fontSize:11,color:"#94a3b8",fontWeight:600,letterSpacing:1}}>TASK B · RADIUS PREDICTION</span>
        <div style={{display:"flex",alignItems:"center",gap:20,marginTop:14}}>
          <div style={{width:pm.size,height:pm.size,borderRadius:"50%",background:`radial-gradient(circle at 35% 35%, white, ${pm.color})`,boxShadow:`0 0 24px ${pm.glow}`,flexShrink:0,display:"flex",alignItems:"center",justifyContent:"center",fontSize:pm.size*.55}}>{pm.emoji}</div>
          <div>
            <div style={{fontSize:34,fontWeight:800,color:"#60a5fa"}}>{result.regression.predicted_radius}<span style={{fontSize:13,color:"#94a3b8",marginLeft:4}}>R⊕</span></div>
            <div style={{fontSize:13,color:"#a78bfa",fontWeight:600,marginTop:2}}>{result.regression.planet_category}</div>
            <div style={{fontSize:10,color:"#475569",marginTop:6}}>1 R⊕ = Earth radius (6,371 km)</div>
          </div>
        </div>
      </div>
    ):!isConf&&(
      <div style={{background:"#0f172a",border:"1px solid #334155",borderRadius:14,padding:18,textAlign:"center"}}>
        <p style={{color:"#475569",fontSize:12,margin:0}}>🚫 Radius prediction skipped — signal is False Positive</p>
      </div>
    )}
    <div style={{textAlign:"right",fontSize:10,color:"#334155"}}>{new Date(result.timestamp).toLocaleString()}</div>
  </div>);
}

// ── CSV BATCH TAB ────────────────────────────────────────────
function CSVTab(){
  const [rows,setRows]=useState([]);
  const [headers,setHeaders]=useState([]);
  const [results,setResults]=useState([]);
  const [loading,setLoading]=useState(false);
  const [progress,setProgress]=useState(0);
  const [error,setError]=useState("");
  const [done,setDone]=useState(false);
  const [dragOver,setDragOver]=useState(false);
  const fileRef=useRef();
  const parseCSV=text=>{const lines=text.trim().split("\n");const hdrs=lines[0].split(",").map(h=>h.trim().replace(/"/g,""));const data=lines.slice(1).map(l=>{const vals=l.split(",").map(v=>v.trim().replace(/"/g,""));return Object.fromEntries(hdrs.map((h,i)=>[h,vals[i]]));}).filter(r=>Object.values(r).some(v=>v!==""));return{hdrs,data};};
  const handleFile=file=>{if(!file)return;if(!file.name.endsWith(".csv")){setError("Please upload a .csv file");return;}setError("");setResults([]);setDone(false);setProgress(0);const reader=new FileReader();reader.onload=e=>{try{const{hdrs,data}=parseCSV(e.target.result);setHeaders(hdrs);setRows(data);}catch{setError("Could not parse CSV.");}};reader.readAsText(file);};
  const handleDrop=e=>{e.preventDefault();setDragOver(false);handleFile(e.dataTransfer.files[0]);};
  const runBatch=async()=>{if(!rows.length)return;setLoading(true);setResults([]);setProgress(0);setDone(false);setError("");const out=[];for(let i=0;i<rows.length;i++){const row=rows[i];const payload={};FIELDS.forEach(({name})=>{const val=row[name]??null;if(val!==null&&val!=="")payload[name]=parseFloat(val);});try{const res=await fetch(`${API_URL}/predict/full`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const data=await res.json();if(res.ok){out.push({row:i+1,prediction:data.classification.prediction,confidence:data.classification.confidence,confirmed_pct:data.classification.probabilities.CONFIRMED,fp_pct:data.classification.probabilities.FALSE_POSITIVE,radius:data.regression?.predicted_radius??"N/A",category:data.regression?.planet_category??"N/A",status:"ok"});}else{out.push({row:i+1,status:"error",error:data.error||JSON.stringify(data.errors)});}}catch{out.push({row:i+1,status:"error",error:"Network error"});}setProgress(Math.round(((i+1)/rows.length)*100));setResults([...out]);}setLoading(false);setDone(true);};
  const downloadResults=()=>{if(!results.length)return;const hdrs=["Row","Prediction","Confidence %","Confirmed %","False Positive %","Radius (R⊕)","Category","Status"];const csvRows=results.map(r=>r.status==="ok"?[r.row,r.prediction,r.confidence,r.confirmed_pct,r.fp_pct,r.radius,r.category,"OK"]:[r.row,"ERROR","","","","","",r.error]);const csv=[hdrs,...csvRows].map(r=>r.join(",")).join("\n");const blob=new Blob([csv],{type:"text/csv"});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download="stellar_predictions.csv";a.click();URL.revokeObjectURL(url);};
  const confirmed=results.filter(r=>r.status==="ok"&&r.prediction==="CONFIRMED").length;
  const fp=results.filter(r=>r.status==="ok"&&r.prediction==="FALSE POSITIVE").length;
  const errors=results.filter(r=>r.status==="error").length;
  return(<div>
    <div onDrop={handleDrop} onDragOver={e=>{e.preventDefault();setDragOver(true);}} onDragLeave={()=>setDragOver(false)} onClick={()=>fileRef.current.click()}
      style={{border:`2px dashed ${dragOver?"#3b82f6":"#334155"}`,borderRadius:14,padding:"32px 20px",textAlign:"center",cursor:"pointer",background:dragOver?"#1e3a5f22":"#0f172a",transition:"all .2s",marginBottom:16}}>
      <input ref={fileRef} type="file" accept=".csv" style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
      <div style={{fontSize:36,marginBottom:8}}>📂</div>
      <div style={{fontSize:14,color:"#60a5fa",fontWeight:600}}>Drop your CSV file here or click to browse</div>
      <div style={{fontSize:11,color:"#475569",marginTop:6}}>Columns: koi_period, koi_duration, koi_depth, koi_impact, koi_model_snr, koi_num_transits, koi_ror, teff, logg, feh</div>
    </div>
    {error&&<div style={{padding:"10px 14px",background:"#2d1515",border:"1px solid #ef4444",borderRadius:8,color:"#f87171",fontSize:12,marginBottom:12}}>⚠️ {error}</div>}
    {rows.length>0&&(<div style={{background:"#0f172a",border:"1px solid #1e3a5f",borderRadius:12,padding:"14px 18px",marginBottom:14,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
      <div><div style={{fontSize:13,color:"#60a5fa",fontWeight:700}}>✅ CSV Loaded — {rows.length} rows ready</div><div style={{fontSize:11,color:"#475569",marginTop:3}}>Columns: {headers.join(", ")}</div></div>
      <button onClick={runBatch} disabled={loading} style={{padding:"10px 22px",background:loading?"#334155":"linear-gradient(90deg,#3b82f6,#8b5cf6)",border:"none",borderRadius:10,color:"white",fontSize:13,fontWeight:700,cursor:loading?"not-allowed":"pointer"}}>{loading?`⏳ ${progress}%`:"🚀 Run Batch"}</button>
    </div>)}
    {loading&&(<div style={{marginBottom:14}}><div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}><span style={{fontSize:12,color:"#94a3b8"}}>Processing...</span><span style={{fontSize:12,color:"#60a5fa"}}>{progress}% ({results.length}/{rows.length})</span></div><div style={{height:6,background:"#1e293b",borderRadius:3,overflow:"hidden"}}><div style={{height:"100%",width:`${progress}%`,background:"linear-gradient(90deg,#3b82f6,#8b5cf6)",borderRadius:3,transition:"width .3s ease"}}/></div></div>)}
    {results.length>0&&(<div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:10,marginBottom:14}}>{[{label:"Total",value:results.length,color:"#60a5fa"},{label:"Confirmed",value:confirmed,color:"#4ade80"},{label:"False Pos",value:fp,color:"#f87171"},{label:"Errors",value:errors,color:"#f59e0b"}].map(s=>(<div key={s.label} style={{background:"#0f172a",border:`1px solid ${s.color}33`,borderRadius:10,padding:"12px 10px",textAlign:"center"}}><div style={{fontSize:22,fontWeight:800,color:s.color}}>{s.value}</div><div style={{fontSize:10,color:"#94a3b8",marginTop:2}}>{s.label}</div></div>))}</div>)}
    {results.length>0&&(<div><div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}><span style={{fontSize:12,color:"#94a3b8",fontWeight:600}}>RESULTS — {results.length} predictions</span>{done&&<button onClick={downloadResults} style={{padding:"6px 16px",background:"#14532d",border:"1px solid #16a34a",borderRadius:8,color:"#4ade80",fontSize:12,cursor:"pointer",fontWeight:600}}>⬇ Download CSV</button>}</div>
    <div style={{maxHeight:380,overflowY:"auto",borderRadius:10,border:"1px solid #1e293b"}}><table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}><thead><tr style={{background:"#1e3a5f",position:"sticky",top:0}}>{["Row","Prediction","Confidence","Radius","Category"].map(h=>(<th key={h} style={{padding:"8px 12px",textAlign:"left",color:"#60a5fa",fontWeight:700}}>{h}</th>))}</tr></thead><tbody>{results.map((r,i)=>(<tr key={i} style={{background:i%2===0?"#0a0f1e":"#0f172a",borderBottom:"1px solid #1e293b"}}><td style={{padding:"7px 12px",color:"#94a3b8"}}>{r.row}</td><td style={{padding:"7px 12px"}}>{r.status==="ok"?(<span style={{padding:"2px 10px",borderRadius:12,fontSize:10,fontWeight:700,background:r.prediction==="CONFIRMED"?"#14532d":"#7f1d1d",color:r.prediction==="CONFIRMED"?"#4ade80":"#f87171"}}>{r.prediction==="CONFIRMED"?"✅ CONFIRMED":"❌ FALSE POS"}</span>):<span style={{color:"#f59e0b"}}>⚠️ ERROR</span>}</td><td style={{padding:"7px 12px",color:"#e2e8f0"}}>{r.status==="ok"?`${r.confidence}%`:r.error}</td><td style={{padding:"7px 12px",color:"#60a5fa"}}>{r.status==="ok"?(r.radius!=="N/A"?`${r.radius} R⊕`:"—"):"—"}</td><td style={{padding:"7px 12px",color:"#a78bfa",fontSize:10}}>{r.status==="ok"?r.category:"—"}</td></tr>))}</tbody></table></div></div>)}
    <div style={{marginTop:14}}><button onClick={()=>{const sample="koi_period,koi_duration,koi_depth,koi_impact,koi_model_snr,koi_num_transits,koi_ror,teff,logg,feh\n9.49,2.96,615.8,0.15,35.8,142,0.022,5455,4.47,0.12\n1.74,2.41,8079,1.28,505,621,0.387,5342,4.52,0.10\n54.42,3.45,882.4,0.35,42.1,57,0.028,5789,4.38,0.05";const blob=new Blob([sample],{type:"text/csv"});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download="sample_input.csv";a.click();URL.revokeObjectURL(url);}} style={{fontSize:11,color:"#475569",background:"none",border:"none",cursor:"pointer",textDecoration:"underline"}}>📄 Download sample CSV template</button></div>
  </div>);
}

// ── HISTORY TAB ──────────────────────────────────────────────
function HistoryTab(){
  const [history,setHistory]=useState([]);
  const [loading,setLoading]=useState(false);
  const fetchHistory=async()=>{setLoading(true);try{const res=await fetch(`${API_URL}/history`);const data=await res.json();setHistory(data.history||[]);}catch{setHistory([]);}finally{setLoading(false);}};
  const clearHistory=async()=>{await fetch(`${API_URL}/history`,{method:"DELETE"});setHistory([]);};
  return(<div>
    <div style={{display:"flex",gap:8,marginBottom:16}}>
      <button onClick={fetchHistory} style={btnStyle("#1e3a5f","#3b82f6","#60a5fa")}>🔄 Load History</button>
      <button onClick={clearHistory} style={btnStyle("#2d1515","#ef4444","#f87171")}>🗑 Clear History</button>
    </div>
    {loading&&<p style={{color:"#94a3b8",fontSize:13}}>Loading...</p>}
    {!loading&&history.length===0&&(<div style={{textAlign:"center",padding:40}}><div style={{fontSize:40,marginBottom:12}}>📋</div><p style={{color:"#475569",fontSize:13}}>No predictions yet.<br/>Click Load History after making predictions.</p></div>)}
    {history.map((h,i)=>{const isConf=h.classification?.prediction==="CONFIRMED";return(<div key={i} style={{background:"#0f172a",border:`1px solid ${isConf?"#16a34a33":"#dc262633"}`,borderRadius:12,padding:"14px 16px",marginBottom:10}}><div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}><span style={{padding:"2px 10px",borderRadius:12,fontSize:10,fontWeight:700,background:isConf?"#14532d":"#7f1d1d",color:isConf?"#4ade80":"#f87171"}}>{isConf?"✅ CONFIRMED":"❌ FALSE POSITIVE"}</span><span style={{fontSize:10,color:"#475569"}}>{h.timestamp?new Date(h.timestamp).toLocaleString():""}</span></div><div style={{display:"flex",gap:20,marginTop:10,flexWrap:"wrap"}}><span style={{fontSize:12,color:"#94a3b8"}}>Confidence: <strong style={{color:"#e2e8f0"}}>{h.classification?.confidence}%</strong></span>{h.regression&&(<><span style={{fontSize:12,color:"#94a3b8"}}>Radius: <strong style={{color:"#60a5fa"}}>{h.regression.predicted_radius} R⊕</strong></span><span style={{fontSize:12,color:"#94a3b8"}}>Type: <strong style={{color:"#a78bfa"}}>{h.regression.planet_category}</strong></span></>)}</div></div>);})}
  </div>);
}

// ── DATA INSIGHTS TAB ────────────────────────────────────────
function DataInsightsTab(){
  const [active,setActive]=useState("overview");
  const [animated,setAnimated]=useState(false);
  useEffect(()=>{setAnimated(false);const t=setTimeout(()=>setAnimated(true),50);return()=>clearTimeout(t);},[active]);

  const CORR=[
    {feature:"koi_ror",        corr:0.794,color:"#4ade80"},
    {feature:"koi_depth",      corr:0.549,color:"#60a5fa"},
    {feature:"koi_model_snr",  corr:0.451,color:"#a78bfa"},
    {feature:"st_mass",        corr:0.266,color:"#f59e0b"},
    {feature:"st_radius",      corr:0.201,color:"#f472b6"},
    {feature:"koi_duration",   corr:0.197,color:"#34d399"},
    {feature:"koi_period",     corr:0.155,color:"#60a5fa"},
    {feature:"st_teff",        corr:0.162,color:"#c084fc"},
  ];

  const FEAT_IMP_A=[
    {feature:"koi_impact",       imp:0.187,color:"#f472b6"},
    {feature:"koi_ror",          imp:0.162,color:"#4ade80"},
    {feature:"koi_model_snr",    imp:0.141,color:"#60a5fa"},
    {feature:"koi_depth",        imp:0.128,color:"#a78bfa"},
    {feature:"feh_uncertainty",  imp:0.098,color:"#f59e0b"},
    {feature:"teff_uncertainty", imp:0.089,color:"#34d399"},
    {feature:"snr_per_transit",  imp:0.076,color:"#f87171"},
    {feature:"koi_period",       imp:0.061,color:"#c084fc"},
  ];

  const CLASS_DIST=[
    {label:"FALSE POSITIVE",count:4839,pct:68,color:"#f87171"},
    {label:"CONFIRMED",     count:2244,pct:32,color:"#4ade80"},
  ];

  const STATS=[
    {label:"Total KOI Records",  value:"7,585",  color:"#60a5fa"},
    {label:"Confirmed Planets",  value:"2,744",  color:"#4ade80"},
    {label:"False Positives",    value:"4,839",  color:"#f87171"},
    {label:"Features Used (A)",  value:"18",     color:"#a78bfa"},
    {label:"Features Used (B)",  value:"11",     color:"#f59e0b"},
    {label:"Training Samples",   value:"6,068",  color:"#34d399"},
  ];

  const subTabs=[
    {id:"overview",   label:"Overview"},
    {id:"corr",       label:"Correlations"},
    {id:"importance", label:"Feature Importance"},
  ];

  return(<div>
    <div style={{display:"flex",gap:4,marginBottom:20,background:"#0a0f1e",borderRadius:10,padding:4,width:"fit-content",border:"1px solid #1e293b"}}>
      {subTabs.map(t=>(<button key={t.id} onClick={()=>setActive(t.id)} style={{padding:"6px 16px",borderRadius:7,border:"none",cursor:"pointer",fontSize:11,fontWeight:600,background:active===t.id?"#1e3a5f":"transparent",color:active===t.id?"#60a5fa":"#475569",transition:"all .2s"}}>{t.label}</button>))}
    </div>

    {active==="overview"&&(<div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12,marginBottom:24}}>
        {STATS.map((s,i)=>(<div key={s.label} style={{background:"#0f172a",border:`1px solid ${s.color}33`,borderRadius:12,padding:"16px 14px",textAlign:"center",opacity:animated?1:0,transform:animated?"translateY(0)":"translateY(20px)",transition:`all .5s ${i*80}ms`}}>
          <div style={{fontSize:24,fontWeight:800,color:s.color}}>{animated?s.value:"0"}</div>
          <div style={{fontSize:10,color:"#94a3b8",marginTop:4}}>{s.label}</div>
        </div>))}
      </div>
      <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20,marginBottom:16}}>
        <div style={{fontSize:12,color:"#94a3b8",fontWeight:600,marginBottom:14,letterSpacing:1}}>CLASS DISTRIBUTION</div>
        {CLASS_DIST.map((c,i)=>(<div key={c.label} style={{marginBottom:14}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
            <span style={{fontSize:12,color:"#e2e8f0"}}>{c.label}</span>
            <span style={{fontSize:12,fontWeight:700,color:c.color}}>{c.count.toLocaleString()} ({c.pct}%)</span>
          </div>
          <div style={{height:10,background:"#1e293b",borderRadius:5,overflow:"hidden"}}>
            <div style={{height:"100%",width:animated?`${c.pct}%`:"0%",background:c.color,borderRadius:5,transition:`width 1s ${i*200}ms ease`}}/>
          </div>
        </div>))}
        <div style={{marginTop:12,padding:"10px 14px",background:"#1e3a5f22",borderRadius:8,fontSize:11,color:"#94a3b8"}}>
          ⚠️ Dataset is <strong style={{color:"#f59e0b"}}>imbalanced</strong> — 68% False Positive vs 32% Confirmed. Handled using <strong style={{color:"#60a5fa"}}>class_weight='balanced'</strong> and <strong style={{color:"#60a5fa"}}>scale_pos_weight</strong> in XGBoost.
        </div>
      </div>
      <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20}}>
        <div style={{fontSize:12,color:"#94a3b8",fontWeight:600,marginBottom:14,letterSpacing:1}}>KEY FEATURE STATISTICS</div>
        <div style={{overflowX:"auto"}}><table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
          <thead><tr style={{background:"#1e3a5f"}}>{["Feature","Mean","Std","Min","Max","Correlation w/ Target"].map(h=>(<th key={h} style={{padding:"8px 12px",textAlign:"left",color:"#60a5fa",fontWeight:700,whiteSpace:"nowrap"}}>{h}</th>))}</tr></thead>
          <tbody>{[
            ["koi_period","54.3","103.1","0.5","737","0.155"],
            ["koi_duration","3.12","2.04","0.5","13.4","0.197"],
            ["koi_depth","1847","3920","10","78000","0.549"],
            ["koi_impact","0.41","0.35","0.0","1.5","—"],
            ["koi_model_snr","43.2","54.8","1.2","2900","0.451"],
            ["koi_ror","0.047","0.058","0.001","0.89","0.794"],
            ["st_radius","1.12","0.48","0.48","3.82","0.201"],
          ].map((r,i)=>(<tr key={i} style={{background:i%2===0?"#0a0f1e":"#0f172a",borderBottom:"1px solid #1e293b"}}>{r.map((c,j)=>(<td key={j} style={{padding:"7px 12px",color:j===5?"#4ade80":"#94a3b8",fontWeight:j===0?600:400}}>{c}</td>))}</tr>))}
          </tbody>
        </table></div>
      </div>
    </div>)}

    {active==="corr"&&(<div>
      <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20,marginBottom:16}}>
        <div style={{fontSize:12,color:"#94a3b8",fontWeight:600,marginBottom:4,letterSpacing:1}}>FEATURE CORRELATION WITH koi_prad (Planet Radius)</div>
        <div style={{fontSize:11,color:"#475569",marginBottom:16}}>Pearson correlation coefficient — higher = stronger linear relationship with target</div>
        {CORR.map((c,i)=>(<div key={c.feature} style={{marginBottom:14,opacity:animated?1:0,transform:animated?"translateX(0)":"translateX(-20px)",transition:`opacity .5s ${i*60}ms ease, transform .5s ${i*60}ms ease`}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
            <span style={{fontSize:12,color:"#e2e8f0",fontFamily:"monospace"}}>{c.feature}</span>
            <span style={{fontSize:12,fontWeight:700,color:c.color}}>{c.corr.toFixed(3)}</span>
          </div>
          <div style={{height:8,background:"#1e293b",borderRadius:4,overflow:"hidden"}}>
            <div style={{height:"100%",width:animated?`${c.corr*100}%`:"0%",
              background:`linear-gradient(90deg,${c.color}77,${c.color})`,
              borderRadius:4,transition:`width 1.2s ${i*60}ms cubic-bezier(.4,0,.2,1)`}}/>
          </div>
        </div>))}
      </div>
      <div style={{background:"#0f172a",border:"1px solid #1e3a5f",borderRadius:12,padding:16}}>
        <div style={{fontSize:11,color:"#60a5fa",fontWeight:600,marginBottom:8}}>KEY INSIGHT</div>
        <p style={{fontSize:12,color:"#94a3b8",margin:0,lineHeight:1.8}}>
          <strong style={{color:"#4ade80"}}>koi_ror (0.794)</strong> dominates — it directly encodes planet-to-star radius ratio. Combined with <strong style={{color:"#60a5fa"}}>st_radius</strong>, we engineered <code style={{background:"#1e293b",padding:"1px 6px",borderRadius:4,color:"#a78bfa"}}>planet_radius_direct = koi_ror × st_radius</code> which became the strongest predictor for Task B, driving RMSE from 1.27 → 0.48.
        </p>
      </div>
    </div>)}

    {active==="importance"&&(<div>
      <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20,marginBottom:16}}>
        <div style={{fontSize:12,color:"#94a3b8",fontWeight:600,marginBottom:4,letterSpacing:1}}>TASK A — XGBoost Feature Importances</div>
        <div style={{fontSize:11,color:"#475569",marginBottom:16}}>Features ranked by contribution to classification accuracy</div>
        {FEAT_IMP_A.map((f,i)=>(<AnimBar key={f.feature} label={f.feature} imp={f.imp} color={f.color} delay={i*80}/>))}
      </div>
      <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20}}>
        <div style={{fontSize:12,color:"#94a3b8",fontWeight:600,marginBottom:12,letterSpacing:1}}>TASK B — LightGBM Feature Importances</div>
        {[
          {feature:"planet_radius_direct", imp:0.312,color:"#4ade80"},
          {feature:"koi_ror",              imp:0.241,color:"#60a5fa"},
          {feature:"st_radius",            imp:0.148,color:"#a78bfa"},
          {feature:"koi_depth",            imp:0.112,color:"#f59e0b"},
          {feature:"koi_model_snr",        imp:0.073,color:"#34d399"},
          {feature:"st_mass",              imp:0.054,color:"#f472b6"},
          {feature:"st_dens",              imp:0.060,color:"#c084fc"},
        ].map((f,i)=>(<AnimBar key={f.feature} label={f.feature} imp={f.imp} color={f.color} delay={i*80}/>))}
        <div style={{marginTop:12,padding:"10px 14px",background:"#14532d22",borderRadius:8,fontSize:11,color:"#94a3b8"}}>
          💡 <strong style={{color:"#4ade80"}}>planet_radius_direct</strong> (engineered feature) contributes 31.2% — highest of all. This shows feature engineering was critical to achieving RMSE of 0.4838.
        </div>
      </div>
    </div>)}
  </div>);
}

// ── PIPELINE TAB ─────────────────────────────────────────────
function PipelineTab(){
  const [hovStep,setHovStep]=useState(null);
  const [animated,setAnimated]=useState(false);
  useEffect(()=>{setTimeout(()=>setAnimated(true),100);},[]);

  const STEPS=[
    {icon:"📥",label:"Raw CSV Input",        desc:"NASA Kepler KOI dataset with 26 columns including transit features, stellar properties, and measurement uncertainties.",color:"#60a5fa"},
    {icon:"🗑",label:"Drop ID Column",        desc:"Remove 'kepid' — just a star identifier, zero predictive value for ML models.",color:"#94a3b8"},
    {icon:"🔍",label:"Filter CANDIDATE",      desc:"Remove rows with koi_disposition = 'CANDIDATE' for Task A. These have ambiguous labels unusable for supervised learning.",color:"#f59e0b"},
    {icon:"🏷",label:"Encode Labels",         desc:"Map CONFIRMED → 1, FALSE POSITIVE → 0. ML models require numeric targets.",color:"#a78bfa"},
    {icon:"⚗️",label:"Feature Engineering",   desc:"Create 5 new features: teff_uncertainty, feh_uncertainty, logg_uncertainty, depth_per_period, snr_per_transit, planet_radius_direct.",color:"#4ade80"},
    {icon:"🔧",label:"Median Imputation",     desc:"Fill missing values with column median. Robust to outliers vs mean imputation. Applied via SimpleImputer.",color:"#f472b6"},
    {icon:"📏",label:"StandardScaler",        desc:"Normalize all features to mean=0, std=1. Critical for Logistic Regression and distance-based models.",color:"#34d399"},
    {icon:"📈",label:"Log Transform Target",  desc:"Apply log1p() to koi_prad for Task B. Reduces skewness from 8.83 to ~0.8. Reversed with expm1() at inference.",color:"#f59e0b"},
    {icon:"✂️",label:"80/20 Split",           desc:"Stratified train-test split preserving class ratio. random_state=42 for reproducibility.",color:"#60a5fa"},
    {icon:"🌲",label:"Feature Selection",     desc:"Random Forest importance scoring. Select features above mean importance threshold. 18 features for Task A, 11 for Task B.",color:"#a78bfa"},
    {icon:"🤖",label:"Train 4 Models",        desc:"XGBoost, LightGBM, Random Forest, Gradient Boosting. 5-fold cross-validation with stratified K-Fold.",color:"#4ade80"},
    {icon:"🏆",label:"Select Best Model",     desc:"XGBoost wins Task A (F1=0.9200), LightGBM wins Task B (RMSE=0.4838). Save as .pkl via joblib.",color:"#f472b6"},
    {icon:"💾",label:"Save .pkl Files",       desc:"Serialize models, scalers, imputers, feature lists. 10 files total for deployment.",color:"#34d399"},
    {icon:"🌐",label:"Flask API",             desc:"Load .pkl files at startup. /predict/full endpoint runs both tasks in one call. Input validation + feature engineering.",color:"#60a5fa"},
    {icon:"⚛️",label:"React Frontend",        desc:"Real-time predictions, batch CSV upload, history, data insights, pipeline visualization.",color:"#a78bfa"},
  ];

  const ARCH=[
    {layer:"👤 User",      tech:"Browser",           desc:"Enters KOI parameters or uploads CSV",         color:"#94a3b8"},
    {layer:"⚛️ Frontend",  tech:"React + Vite",      desc:"Validates input, sends POST /predict/full",     color:"#60a5fa"},
    {layer:"🌐 Backend",   tech:"Flask + CORS",      desc:"Engineers features, imputes, scales, predicts", color:"#a78bfa"},
    {layer:"🤖 ML Models", tech:"XGBoost + LightGBM",desc:"Task A classification + Task B regression",     color:"#4ade80"},
    {layer:"💾 Storage",   tech:"Joblib .pkl files", desc:"Pre-trained models, scalers, imputers",         color:"#f59e0b"},
  ];

  return(<div>
    <div style={{marginBottom:24}}>
      <div style={{fontSize:13,color:"#60a5fa",fontWeight:700,marginBottom:4,letterSpacing:1}}>END-TO-END ML PIPELINE</div>
      <div style={{fontSize:11,color:"#475569",marginBottom:16}}>Hover each step to see details</div>
      <div style={{display:"flex",flexWrap:"wrap",gap:0,position:"relative"}}>
        {STEPS.map((s,i)=>(
          <div key={i} style={{display:"flex",alignItems:"center",gap:0,marginBottom:8}}>
            <div onMouseEnter={()=>setHovStep(i)} onMouseLeave={()=>setHovStep(null)}
              style={{display:"flex",flexDirection:"column",alignItems:"center",padding:"10px 12px",
                borderRadius:10,cursor:"pointer",
                border:`1px solid ${hovStep===i?s.color:"#1e293b"}`,
                background:hovStep===i?`${s.color}11`:"#0f172a",
                opacity:animated?1:0,
                transform:animated?"translateY(0)":"translateY(20px)",
                transition:`opacity .4s ${i*40}ms ease, transform .4s ${i*40}ms ease, border-color .2s, background .2s`,
                minWidth:80,textAlign:"center"}}>
              <div style={{fontSize:20,marginBottom:4}}>{s.icon}</div>
              <div style={{fontSize:9,color:hovStep===i?s.color:"#94a3b8",fontWeight:600,lineHeight:1.3}}>{s.label}</div>
            </div>
            {i<STEPS.length-1&&<div style={{fontSize:12,color:"#334155",margin:"0 2px"}}>›</div>}
          </div>
        ))}
      </div>
      {hovStep!==null&&(
        <div style={{marginTop:12,padding:"14px 18px",background:"#0f172a",border:`1px solid ${STEPS[hovStep].color}`,borderRadius:12,transition:"all .2s"}}>
          <div style={{fontSize:13,color:STEPS[hovStep].color,fontWeight:700,marginBottom:4}}>{STEPS[hovStep].icon} {STEPS[hovStep].label}</div>
          <div style={{fontSize:12,color:"#94a3b8",lineHeight:1.7}}>{STEPS[hovStep].desc}</div>
        </div>
      )}
    </div>

    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20,marginBottom:16}}>
      <div style={{fontSize:12,color:"#94a3b8",fontWeight:600,marginBottom:14,letterSpacing:1}}>SYSTEM ARCHITECTURE</div>
      <div style={{display:"flex",flexDirection:"column",gap:0}}>
        {ARCH.map((a,i)=>(
          <div key={i} style={{display:"flex",gap:0,alignItems:"stretch"}}>
            <div style={{display:"flex",flexDirection:"column",alignItems:"center",marginRight:16}}>
              <div style={{width:12,height:12,borderRadius:"50%",background:a.color,flexShrink:0,marginTop:18}}/>
              {i<ARCH.length-1&&<div style={{width:2,flex:1,background:`${a.color}33`,minHeight:20}}/>}
            </div>
            <div style={{flex:1,padding:"12px 16px",background:i%2===0?"#0a0f1e":"#0f172a",borderRadius:10,marginBottom:8,border:`1px solid ${a.color}22`}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                <span style={{fontSize:12,fontWeight:700,color:a.color}}>{a.layer}</span>
                <span style={{fontSize:10,color:"#475569",background:"#1e293b",padding:"2px 8px",borderRadius:6}}>{a.tech}</span>
              </div>
              <div style={{fontSize:11,color:"#94a3b8",marginTop:4}}>{a.desc}</div>
            </div>
          </div>
        ))}
      </div>
    </div>

    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20}}>
      <div style={{fontSize:12,color:"#94a3b8",fontWeight:600,marginBottom:12,letterSpacing:1}}>API ENDPOINTS</div>
      {[
        {method:"POST",endpoint:"/predict/full",    desc:"Run both Task A + B in one call",             color:"#4ade80"},
        {method:"POST",endpoint:"/predict/classify",desc:"Task A classification only",                  color:"#60a5fa"},
        {method:"POST",endpoint:"/predict/radius",  desc:"Task B radius prediction only",               color:"#a78bfa"},
        {method:"GET", endpoint:"/health",           desc:"Health check + loaded model info",            color:"#f59e0b"},
        {method:"GET", endpoint:"/history",          desc:"Last 20 predictions",                        color:"#34d399"},
        {method:"GET", endpoint:"/fields",           desc:"Field metadata for frontend",                color:"#94a3b8"},
      ].map((e,i)=>(<div key={i} style={{display:"flex",alignItems:"center",gap:12,padding:"8px 0",borderBottom:"1px solid #1e293b"}}>
        <span style={{padding:"2px 8px",borderRadius:4,fontSize:10,fontWeight:700,background:e.method==="POST"?"#1e3a5f":"#1a2e1a",color:e.method==="POST"?"#60a5fa":"#4ade80",minWidth:36,textAlign:"center"}}>{e.method}</span>
        <span style={{fontFamily:"monospace",fontSize:11,color:e.color,minWidth:160}}>{e.endpoint}</span>
        <span style={{fontSize:11,color:"#475569"}}>{e.desc}</span>
      </div>))}
    </div>
  </div>);
}

// ── ABOUT TAB ────────────────────────────────────────────────
function AboutTab(){
  const [animated,setAnimated]=useState(false);
  useEffect(()=>{setTimeout(()=>setAnimated(true),100);},[]);

  const MODELS=[
    {task:"A",model:"XGBoost",     metric1:"F1: 0.9200",metric2:"AUC: 0.9848",color:"#4ade80",winner:true},
    {task:"A",model:"LightGBM",    metric1:"F1: 0.9191",metric2:"AUC: 0.9844",color:"#60a5fa",winner:false},
    {task:"A",model:"Rand Forest", metric1:"F1: 0.9147",metric2:"AUC: 0.9833",color:"#a78bfa",winner:false},
    {task:"A",model:"Grad Boost",  metric1:"F1: 0.9109",metric2:"AUC: 0.9835",color:"#94a3b8",winner:false},
    {task:"B",model:"LightGBM",    metric1:"RMSE: 0.4838",metric2:"MAE: 0.1533",color:"#4ade80",winner:true},
    {task:"B",model:"XGBoost",     metric1:"RMSE: 0.4867",metric2:"MAE: 0.1504",color:"#60a5fa",winner:false},
    {task:"B",model:"Rand Forest", metric1:"RMSE: 0.6484",metric2:"MAE: 0.1961",color:"#a78bfa",winner:false},
    {task:"B",model:"Ridge",       metric1:"RMSE: 0.8606",metric2:"MAE: 0.3156",color:"#94a3b8",winner:false},
  ];

  const ASSUMPTIONS=[
    {title:"CANDIDATE Exclusion",   text:"Rows with koi_disposition=CANDIDATE excluded — ambiguous labels unusable for supervised learning."},
    {title:"Regression Scope",      text:"Task B only runs on CONFIRMED signals. A FALSE POSITIVE has no meaningful planetary radius."},
    {title:"Log Transform",         text:"koi_prad is log-transformed (skewness 8.83 → 0.8). Reversed with expm1() at inference."},
    {title:"Uncertainty Features",  text:"Measurement error columns (err1/err2) engineered into single uncertainty scores per stellar property."},
    {title:"Leakage Check",         text:"koi_ror has 0.794 correlation with koi_prad — retained as it encodes physics, not target leakage."},
    {title:"Version Consistency",   text:"All .pkl files trained with scikit-learn 1.8.0, xgboost, lightgbm matching deployment environment."},
  ];

  const TEAM=[
    {name:"Aryan",   role:"ML & Backend",  emoji:"🧠"},
    {name:"Mihir",   role:"Frontend & API", emoji:"⚛️"},
    {name:"Anushka", role:"EDA & Analysis", emoji:"📊"},
  ];

  return(<div>
    {/* Hero */}
    <div style={{background:"linear-gradient(135deg,#0f172a,#1e3a5f22)",border:"1px solid #1e3a5f",borderRadius:16,padding:"24px 24px",marginBottom:20,textAlign:"center",opacity:animated?1:0,transform:animated?"translateY(0)":"translateY(20px)",transition:"all .6s"}}>
      <div style={{fontSize:40,marginBottom:8}}>🌌</div>
      <h2 style={{margin:0,fontSize:22,fontWeight:800,background:"linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>STELLAR ANALYTICS</h2>
      <p style={{color:"#94a3b8",fontSize:13,marginTop:6,marginBottom:12}}>Exoplanet Signal Classifier & Radius Predictor</p>
      <div style={{display:"flex",gap:8,justifyContent:"center",flexWrap:"wrap"}}>
        {[{l:"F1-Score",v:"0.9200",c:"#4ade80"},{l:"ROC-AUC",v:"0.9848",c:"#60a5fa"},{l:"RMSE",v:"0.4838",c:"#f59e0b"},{l:"MAE",v:"0.1533",c:"#a78bfa"}].map(b=>(<div key={b.l} style={{padding:"4px 14px",background:"#0a0f1e",border:`1px solid ${b.c}44`,borderRadius:20,fontSize:11,color:b.c,fontWeight:700}}>{b.l}: {b.v}</div>))}
      </div>
    </div>

    {/* Problem Statement */}
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20,marginBottom:16}}>
      <div style={{fontSize:12,color:"#60a5fa",fontWeight:700,marginBottom:10,letterSpacing:1}}>PROBLEM STATEMENT</div>
      <p style={{fontSize:12,color:"#94a3b8",lineHeight:1.8,margin:0}}>
        NASA's Kepler Space Telescope identified <strong style={{color:"#e2e8f0"}}>7,585 Kepler Objects of Interest (KOIs)</strong> by detecting tiny dips in starlight as planets pass in front of their host stars. However, many signals are not real planets — binary stars, instrumental noise, and background eclipsing systems mimic planetary transits.
        <br/><br/>
        <strong style={{color:"#60a5fa"}}>Task A</strong> — Binary classification: CONFIRMED exoplanet vs FALSE POSITIVE signal.<br/>
        <strong style={{color:"#a78bfa"}}>Task B</strong> — Regression: Predict planetary radius in Earth radii for confirmed planets.
      </p>
    </div>

    {/* Approach */}
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20,marginBottom:16}}>
      <div style={{fontSize:12,color:"#60a5fa",fontWeight:700,marginBottom:10,letterSpacing:1}}>OUR APPROACH</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
        {[
          {title:"EDA First",           icon:"🔍",desc:"Analyzed class imbalance (68/32 split), feature distributions, skewness of target (8.83), and correlation matrix before any modeling."},
          {title:"Feature Engineering", icon:"⚗️",desc:"Created 6 engineered features including planet_radius_direct = koi_ror × st_radius which became the #1 predictor for Task B."},
          {title:"Advanced Models",     icon:"🤖",desc:"Evaluated XGBoost, LightGBM, Random Forest, and Gradient Boosting with 5-fold cross-validation for both tasks."},
          {title:"Overfitting Control", icon:"🛡",desc:"Applied L1/L2 regularization, reduced depth, increased min_child_weight. Monitored train vs test gap for all models."},
        ].map((a,i)=>(<div key={i} style={{background:"#0a0f1e",borderRadius:10,padding:"14px 16px",border:"1px solid #1e293b"}}>
          <div style={{fontSize:16,marginBottom:6}}>{a.icon}</div>
          <div style={{fontSize:12,color:"#e2e8f0",fontWeight:600,marginBottom:4}}>{a.title}</div>
          <div style={{fontSize:11,color:"#94a3b8",lineHeight:1.7}}>{a.desc}</div>
        </div>))}
      </div>
    </div>

    {/* Model comparison */}
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20,marginBottom:16}}>
      <div style={{fontSize:12,color:"#60a5fa",fontWeight:700,marginBottom:14,letterSpacing:1}}>MODEL COMPARISON</div>
      <div style={{overflowX:"auto"}}><table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
        <thead><tr style={{background:"#1e3a5f"}}>{["Task","Model","Primary Metric","Secondary Metric","Status"].map(h=>(<th key={h} style={{padding:"8px 12px",textAlign:"left",color:"#60a5fa",fontWeight:700}}>{h}</th>))}</tr></thead>
        <tbody>{MODELS.map((m,i)=>(<tr key={i} style={{background:m.winner?"#0f2a1a":i%2===0?"#0a0f1e":"#0f172a",borderBottom:"1px solid #1e293b"}}>
          <td style={{padding:"8px 12px"}}><span style={{padding:"2px 8px",borderRadius:4,fontSize:10,fontWeight:700,background:m.task==="A"?"#1e3a5f":"#1a1a3a",color:m.task==="A"?"#60a5fa":"#a78bfa"}}>Task {m.task}</span></td>
          <td style={{padding:"8px 12px",color:"#e2e8f0",fontWeight:m.winner?700:400}}>{m.winner?"🏆 ":""}{m.model}</td>
          <td style={{padding:"8px 12px",color:m.winner?"#4ade80":"#94a3b8",fontWeight:m.winner?700:400}}>{m.metric1}</td>
          <td style={{padding:"8px 12px",color:"#94a3b8"}}>{m.metric2}</td>
          <td style={{padding:"8px 12px"}}>{m.winner?<span style={{color:"#4ade80",fontSize:10,fontWeight:700}}>✅ SELECTED</span>:<span style={{color:"#475569",fontSize:10}}>—</span>}</td>
        </tr>))}</tbody>
      </table></div>
    </div>

    {/* Assumptions */}
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20,marginBottom:16}}>
      <div style={{fontSize:12,color:"#60a5fa",fontWeight:700,marginBottom:12,letterSpacing:1}}>KEY ASSUMPTIONS</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
        {ASSUMPTIONS.map((a,i)=>(<div key={i} style={{background:"#0a0f1e",borderRadius:8,padding:"12px 14px",border:"1px solid #1e293b",opacity:animated?1:0,transform:animated?"translateY(0)":"translateY(10px)",transition:`all .4s ${i*80}ms`}}>
          <div style={{fontSize:11,color:"#a78bfa",fontWeight:700,marginBottom:4}}>{a.title}</div>
          <div style={{fontSize:11,color:"#94a3b8",lineHeight:1.6}}>{a.text}</div>
        </div>))}
      </div>
    </div>

    {/* Team */}
    <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:12,padding:20}}>
      <div style={{fontSize:12,color:"#60a5fa",fontWeight:700,marginBottom:14,letterSpacing:1}}>TEAM — IIT BHU</div>
      <div style={{display:"flex",gap:16,justifyContent:"center",flexWrap:"wrap"}}>
        {TEAM.map((t,i)=>(<div key={i} style={{background:"#0a0f1e",border:"1px solid #1e3a5f",borderRadius:14,padding:"20px 28px",textAlign:"center",minWidth:120,opacity:animated?1:0,transform:animated?"translateY(0)":"translateY(20px)",transition:`all .5s ${i*120}ms`}}>
          <div style={{fontSize:36,marginBottom:8}}>{t.emoji}</div>
          <div style={{fontSize:14,fontWeight:700,color:"#e2e8f0"}}>{t.name}</div>
          <div style={{fontSize:10,color:"#475569",marginTop:4}}>{t.role}</div>
        </div>))}
      </div>
      <div style={{textAlign:"center",marginTop:16}}>
        <div style={{fontSize:11,color:"#475569"}}>Indian Institute of Technology (BHU) Varanasi</div>
        <div style={{fontSize:10,color:"#334155",marginTop:4}}>TECHNEX '26 · Innorave Eco-Hackathon · March 2026</div>
      </div>
    </div>
  </div>);
}

// ── MAIN APP ─────────────────────────────────────────────────
export default function App(){
  const [tab,setTab]=useState("single");
  const [result,setResult]=useState(null);
  const [history,setHistory]=useState([]);
  const handleResult=r=>{setResult(r);if(r)setHistory(h=>[{...r,id:Date.now()},...h.slice(0,9)]);};

  const TABS=[
    {id:"single",   label:"🔭 Predict"},
    {id:"batch",    label:"📂 Batch"},
    {id:"insights", label:"📊 Insights"},
    {id:"pipeline", label:"🔄 Pipeline"},
    {id:"about",    label:"📖 About"},
    {id:"history",  label:"📋 History"},
  ];

  return(<div style={{minHeight:"100vh",background:"#0a0f1e",color:"#e2e8f0",fontFamily:"'Segoe UI',sans-serif"}}>
    {/* Stars */}
    <div style={{position:"fixed",inset:0,overflow:"hidden",pointerEvents:"none",zIndex:0}}>
      {[...Array(100)].map((_,i)=>(<div key={i} style={{position:"absolute",width:Math.random()*2+1,height:Math.random()*2+1,borderRadius:"50%",background:"white",top:`${Math.random()*100}%`,left:`${Math.random()*100}%`,opacity:Math.random()*.6+.1}}/>))}
    </div>

    <div style={{position:"relative",zIndex:1,maxWidth:1200,margin:"0 auto",padding:"20px 16px"}}>
      {/* Header */}
      <div style={{textAlign:"center",marginBottom:24}}>
        <div style={{fontSize:40,marginBottom:6}}>🌌</div>
        <h1 style={{fontSize:28,fontWeight:800,margin:0,background:"linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>STELLAR ANALYTICS</h1>
        <p style={{color:"#475569",marginTop:4,fontSize:12}}>Exoplanet Signal Classifier & Radius Predictor · TECHNEX '26 · IIT BHU</p>
        <div style={{display:"flex",gap:8,justifyContent:"center",marginTop:10,flexWrap:"wrap"}}>
          {[{l:"F1",v:"0.9200",c:"#4ade80"},{l:"AUC",v:"0.9848",c:"#60a5fa"},{l:"RMSE",v:"0.4838",c:"#f59e0b"},{l:"MAE",v:"0.1533",c:"#a78bfa"}].map(b=>(<div key={b.l} style={{padding:"3px 12px",background:"#0f172a",border:`1px solid ${b.c}44`,borderRadius:20,fontSize:10,color:b.c,fontWeight:700}}>{b.l}: {b.v}</div>))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{display:"flex",gap:3,marginBottom:16,background:"#0f172a",borderRadius:12,padding:4,border:"1px solid #1e293b",overflowX:"auto",width:"fit-content",maxWidth:"100%"}}>
        {TABS.map(t=>(<button key={t.id} onClick={()=>setTab(t.id)} style={{padding:"7px 16px",borderRadius:9,border:"none",cursor:"pointer",fontSize:11,fontWeight:600,whiteSpace:"nowrap",background:tab===t.id?"linear-gradient(90deg,#3b82f6,#8b5cf6)":"transparent",color:tab===t.id?"white":"#94a3b8",boxShadow:tab===t.id?"0 2px 12px #3b82f644":"none",transition:"all .2s"}}>{t.label}</button>))}
      </div>

      {/* Content */}
      <div style={{display:tab==="single"?"grid":"block",gridTemplateColumns:"1fr 400px",gap:16,alignItems:"start"}}>
        {tab==="single"&&(<>
          <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:16,padding:22}}>
            <h2 style={{fontSize:13,fontWeight:700,color:"#60a5fa",marginTop:0,marginBottom:18,letterSpacing:.5}}>🔭 KOI SIGNAL PARAMETERS</h2>
            <SingleTab onResult={handleResult}/>
          </div>
          <div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:16,padding:22}}>
            <h2 style={{fontSize:13,fontWeight:700,color:"#60a5fa",marginTop:0,marginBottom:18,letterSpacing:.5}}>📡 PREDICTION RESULTS</h2>
            <SingleResult result={result}/>
            {history.length>0&&(<div style={{marginTop:16,borderTop:"1px solid #1e293b",paddingTop:14}}>
              <div style={{fontSize:10,color:"#475569",marginBottom:8,fontWeight:600}}>RECENT PREDICTIONS</div>
              {history.slice(0,5).map(h=>(<div key={h.id} style={{display:"flex",justifyContent:"space-between",padding:"5px 10px",borderRadius:6,marginBottom:3,background:"#0a0f1e",fontSize:11}}>
                <span style={{color:h.classification.prediction==="CONFIRMED"?"#4ade80":"#f87171"}}>{h.classification.prediction==="CONFIRMED"?"✅":"❌"} {h.classification.prediction}</span>
                {h.regression&&<span style={{color:"#60a5fa"}}>{h.regression.predicted_radius} R⊕</span>}
                <span style={{color:"#475569"}}>{h.classification.confidence}%</span>
              </div>))}
            </div>)}
          </div>
        </>)}
        {tab==="batch"&&(<div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:16,padding:22}}><h2 style={{fontSize:13,fontWeight:700,color:"#60a5fa",marginTop:0,marginBottom:18,letterSpacing:.5}}>📂 BATCH CSV PREDICTIONS</h2><CSVTab/></div>)}
        {tab==="insights"&&(<div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:16,padding:22}}><h2 style={{fontSize:13,fontWeight:700,color:"#60a5fa",marginTop:0,marginBottom:18,letterSpacing:.5}}>📊 DATA INSIGHTS & EDA</h2><DataInsightsTab/></div>)}
        {tab==="pipeline"&&(<div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:16,padding:22}}><h2 style={{fontSize:13,fontWeight:700,color:"#60a5fa",marginTop:0,marginBottom:18,letterSpacing:.5}}>🔄 ML PIPELINE & ARCHITECTURE</h2><PipelineTab/></div>)}
        {tab==="about"&&(<div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:16,padding:22}}><h2 style={{fontSize:13,fontWeight:700,color:"#60a5fa",marginTop:0,marginBottom:18,letterSpacing:.5}}>📖 ABOUT & APPROACH</h2><AboutTab/></div>)}
        {tab==="history"&&(<div style={{background:"#0f172a",border:"1px solid #1e293b",borderRadius:16,padding:22}}><h2 style={{fontSize:13,fontWeight:700,color:"#60a5fa",marginTop:0,marginBottom:18,letterSpacing:.5}}>📋 PREDICTION HISTORY</h2><HistoryTab/></div>)}
      </div>

      <div style={{textAlign:"center",marginTop:28,color:"#1e293b",fontSize:10}}>Stellar Analytics · TECHNEX '26 · IIT BHU · XGBoost + LightGBM + Flask + React</div>
    </div>

    <style>{`
      input[type=number]:focus{border-color:#3b82f6 !important;}
      input[type=number]::-webkit-inner-spin-button{opacity:0.3;}
      ::-webkit-scrollbar{width:6px;height:6px;}
      ::-webkit-scrollbar-track{background:#0a0f1e;}
      ::-webkit-scrollbar-thumb{background:#334155;border-radius:3px;}
      button:hover{opacity:0.9;}
    `}</style>
  </div>);
}

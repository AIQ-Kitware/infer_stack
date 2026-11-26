const {make,C,src,bench,repo}=require('../layout.cjs');
const {contexts,labels,ep,v,fmt}=require('../metrics.cjs');
module.exports=async function(){
const {slide,cover,note,card,code,strip,flow,step,txt,rect,line,dot,write}=make('evidence','Measuring inference',C.orange);
const series=[{id:ep.q,n:'HyperQwen',c:C.grey},{id:ep.ql,n:'HyperQwen Long',c:C.blue},{id:ep.qh,n:'HyperQwen Huge',c:C.orange},{id:ep.ml,n:'Maya Long',c:C.teal}];
function legend(s,ss,x=9.83,y=2.42){ss.forEach((a,i)=>{line(s,x,y+i*.43,x+.36,y+i*.43,a.c,3);txt(s,a.n,x+.46,y+i*.43-.15,2.48,.29,{fontSize:14});});}
function chart(s,phase,key,max,ticks,ss=series){const x=1.16,y=2.6,w=8.15,h=3.58;for(const t of ticks){const yy=y+h*(1-t/max);line(s,x,yy,x+w,yy,C.soft,.7);txt(s,String(t),.57,yy-.14,.48,.28,{fontSize:12,color:C.muted,align:'right'});}contexts.forEach((n,i)=>{const xx=x+w*i/5;txt(s,labels[i],xx-.33,y+h+.15,.66,.3,{fontSize:13,color:C.muted,align:'center'});});for(const a of ss){let last=null;contexts.forEach((n,i)=>{const val=v(a.id,n,phase,key);if(val==null){last=null;return;}const xx=x+w*i/5,yy=y+h*(1-val/max);if(last)line(s,last[0],last[1],xx,yy,a.c,2.8);dot(s,xx-.055,yy-.055,.11,a.c);last=[xx,yy];});}legend(s,ss);}
{
const s=cover('Performance case study / 10–15 minutes','Which model configuration\nshould I actually run?','A repeatable experiment on one RTX 3090 reveals why the answer depends on the workload.','Six configurations • 96 successful requests • November 25, 2026');
note(s,'Frame this as measured serving performance, not intelligence or a comprehensive hardware evaluation. The original export contains 112 ledger records including acquisition and release events, 96 successful trial records and four context-headroom skips.',bench);
}
{
const s=slide('Measure requests, not just peak output speed.','The same synthetic code-review workload is run at each eligible context target.','Method');
flow(s,[['Fresh','New user prefix'],['Exact prefix','Repeat same prompt'],['Extended prefix','Append more code']],2.33);
step(s,'01','Inputs','1, 8, 32, 64, 128 and 160 Ki target tokens; 512 output-token budget.',.57,4.19,12.13,1.05);
step(s,'02','Measurements','Time to first token, stream rate, end-to-end rate, cached tokens and hardware identity.',.57,5.45,12.13,1.05);
note(s,'One repeat per phase. Context targets are approximate; actual tokenization is API-reported. Fresh means a new user prefix, not a cold process or a globally flushed cache. TTFT includes transport, queueing and processing. Stream rate is an estimate from observed streaming, not native-engine decode counters. Extended prompt is longer than fresh.',[...src('docs/benchmark.md'),...bench]);
}
{
const s=slide('The comparison is on the same physical GPU.','All six endpoint configurations reached readiness on a single RTX 3090.','Experimental controls');
rect(s,.55,2.22,5.96,3.67,C.dark);txt(s,'RTX 3090',.79,2.53,5.47,.8,{fontSize:39,bold:true,color:C.white});txt(s,'24 GiB VRAM',.81,3.42,5.4,.52,{fontSize:24,color:C.light});txt(s,'PCIe Gen3 ×2 observed',.81,4.19,5.4,.55,{fontSize:24,bold:true,color:C.cyan});
card(s,6.81,2.22,5.97,3.67,'What stays separate','HyperQwen standard / long / huge\nMaya standard / long / huge\n\nDifferent images and runtime profiles.',C.orange);
strip(s,'Same card ≠ identical model, quantization, speculative decoding or software.');
note(s,'The observed link and upstream bridge information are from configurations.json and summary.csv. A GPU’s own capability is not the effective upstream bandwidth. No other GPU was measured in this sweep, so do not attribute differences to PCIe. All six configurations used one physical card; identifiers are intentionally omitted.',bench);
}
{
const s=slide('Maya sustains long-context throughput.','Fresh phase, estimated output tokens per second.','Result 1');
chart(s,'fresh','stream_tps',200,[0,50,100,150,200]);
const a=v(ep.ml,163840,'fresh','stream_tps'),b=v(ep.qh,163840,'fresh','stream_tps');
rect(s,9.75,4.4,3.03,1.74,C.tealLight);txt(s,`${fmt(a/b,2)}×`,9.98,4.57,2.56,.65,{fontSize:37,bold:true,color:C.teal});txt(s,`${fmt(a)} vs ${fmt(b)} t/s at 160K`,9.98,5.35,2.56,.51,{fontSize:16,color:C.ink});
txt(s,'Target context (Ki tokens). Missing targets are omitted, not treated as zero.',.57,6.72,12.1,.21,{fontSize:12,color:C.muted});
note(s,'HyperQwen standard is fastest at the 1K target. Maya Long leads the shown variants at 32K and above. The other Maya variants sustain approximately 87–93 tokens/s at 160K. The biggest supported context was not fully searched: the benchmark ends at 160K. One trial per point.',bench);
}
{
const s=slide('But a repeated prefix changes the outcome.','At 160K, the fresh-to-extended-prefix TTFT contrast is the key observation.','Result 2');
const panels=[{key:'ttft_s',t:'Time to first token',unit:'Seconds · lower is better',max:220},{key:'e2e_tps',t:'End-to-end output rate',unit:'Tokens/s · higher is better',max:24}];
const pair=[series[2],series[3]];
panels.forEach((p,i)=>{const x=.55+i*6.22;rect(s,x,2.22,6.01,3.77,C.white,C.soft);txt(s,p.t,x+.2,2.43,5.61,.42,{fontSize:23,bold:true});txt(s,p.unit,x+.2,2.88,5.61,.26,{fontSize:13,color:C.muted});['fresh','extended-prefix'].forEach((ph,j)=>{let yy=3.22+j*1.25;txt(s,ph==='fresh'?'Fresh':'Extended prefix',x+.2,yy,5.61,.27,{fontSize:14,bold:true});pair.forEach((a,k)=>{let n=v(a.id,163840,ph,p.key),w=4.34*n/p.max,by=yy+.34+k*.34;rect(s,x+.2,by,w,.23,a.c);txt(s,fmt(n,n<10?2:1),x+.28+w,by-.025,1,.29,{fontSize:14,bold:true,color:a.c});});});});
line(s,.72,6.29,1.13,6.29,C.orange,3);txt(s,'HyperQwen Huge',1.25,6.13,2.55,.33,{fontSize:16});line(s,4.02,6.29,4.43,6.29,C.teal,3);txt(s,'Maya Long',4.55,6.13,2.3,.33,{fontSize:16});txt(s,'512-token generation budget',7.54,6.13,5.22,.33,{fontSize:15,color:C.muted,align:'right'});
note(s,'At 160K the observed extended-prefix TTFT is about 2.13 s vs 99.4 s. HyperQwen Huge reported 163,968 cached tokens; Maya Long reported zero and its TTFT stayed similar to fresh. This is evidence of no observed reuse benefit for Maya here, not proof that its implementation can never reuse prefixes. The extended-prefix input is slightly longer than the fresh input.',bench);
}
{
const s=slide('For agents, request length changes the winner.','Illustrative model: elapsed time ≈ TTFT + output tokens / stream rate.','Interpretation');
const t1=v(ep.qh,163840,'extended-prefix','ttft_s'),r1=v(ep.qh,163840,'extended-prefix','stream_tps'),t2=v(ep.ml,163840,'extended-prefix','ttft_s'),r2=v(ep.ml,163840,'extended-prefix','stream_tps');
const cross=(t2-t1)/(1/r1-1/r2);const x=1.15,y=2.55,w=8.08,h=3.57,m=390;
for(let n=0;n<=375;n+=75){let yy=y+h*(1-n/m);line(s,x,yy,x+w,yy,C.soft,.7);txt(s,`${n}s`,.57,yy-.14,.48,.28,{fontSize:12,color:C.muted,align:'right'});}
for(let k=0;k<=8192;k+=2048){let xx=x+w*k/8192;txt(s,k===0?'0':`${k/1024}K`,xx-.31,y+h+.16,.62,.28,{fontSize:13,color:C.muted,align:'center'});}
for(const [t,r,c] of [[t1,r1,C.orange],[t2,r2,C.teal]])line(s,x,y+h*(1-t/m),x+w,y+h*(1-(t+8192/r)/m),c,3);
const cx=x+w*cross/8192,cy=y+h*(1-(t1+cross/r1)/m);line(s,cx,cy,cx,y+h,C.grey,1.3);dot(s,cx-.075,cy-.075,.15,C.ink);
legend(s,[series[2],series[3]],9.8,2.73);rect(s,9.72,4.1,3.06,1.8,C.orangeLight);txt(s,`~${(Math.round(cross/10)*10).toLocaleString('en-US')}`,9.92,4.28,2.65,.65,{fontSize:34,bold:true,color:'9E551A'});txt(s,'Output tokens at estimated crossover',9.92,5.04,2.65,.66,{fontSize:16,color:C.ink,valign:'top'});
txt(s,'Generated tokens per request  ·  Extrapolation, not measured long-output trials.',.57,6.71,12.1,.22,{fontSize:12,color:C.muted});
note(s,'Uses extended-prefix TTFT and stream rates from the 512-token measurements. Assumes constant rates and startup latency as output length grows. The crossing is an approximation, not a benchmark at 3K+ output tokens. Long reasoning or post-compaction may favor Maya; short cached tool-call turns may favor HyperQwen. Evaluate time-to-solved task and quality separately.',bench);
}
{
const s=slide('Run the experiment on your own hardware.','The benchmark takes responsibility for acquiring, measuring and releasing each endpoint.','Reproduce');
code(s,"infer-stack catalog endpoint list\ninfer-stack benchmark run \\\n  'qwen3.8-*' 'inkling-2-mini-maya*' \\\n  --contexts 1024,32768,131072,163840 \\\n  --repeats 3 --allowed-gpus 1",.55,2.2,12.23,2.8,20);
step(s,'01','Watch','Open the printed report.html while measurements arrive.',.57,5.15,6,1.1);
step(s,'02','Compare','Save the JSONL ledger; repeat on a different card with the same profiles.',6.81,5.15,5.97,1.1);
note(s,'Example assumes the endpoint families are registered in the local catalog and GPU index 1 is the intended card. Validate actual indices before running. Repeat with a different allowed index, hold endpoint configuration and workload constant, and record physical GPU identity. The current runner is local Compose/LiteLLM, not remote or KubeAI. The report updates incrementally and is self-contained. Runtime may be substantial.',src('docs/benchmark.md','infer_stack/benchmark/runner.py'));
}
{
const s=slide('Make the next experiment more useful.','The benchmark is a starting point for selecting and refining serving profiles.','Next experiments');
card(s,.55,2.2,3.9,3.62,'Hardware','Same configuration, different GPU link; more repeats and sampled telemetry.',C.blue);
card(s,4.72,2.2,3.9,3.62,'Serving','Investigate prefix reuse, prefill time and sustained generation at longer contexts.',C.teal);
card(s,8.88,2.2,3.9,3.62,'Task success','Measure time to a correct solution under real compaction and tool-call patterns.',C.orange);
strip(s,'No single throughput score establishes the best model or serving configuration.');
note(s,'These experiments are proposals, not completed results. For real workload evaluation, record request length distributions, actual context, cache behavior, solution success and elapsed wall time. Preserve evidence of serving identity; configured provenance is not loaded-weight attestation.',[...src('docs/benchmark.md'),...bench]);
}
await write();
};

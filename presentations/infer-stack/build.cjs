// Source for the 2026-11-25 infer-stack technical overview.
// Run `npm ci && npm run build` in this directory.
const pptxgen = require('pptxgenjs');
const fs = require('node:fs');
const path = require('node:path');
const rows = require('./data/benchmark-2026-11-25.json');

const outputDir = path.join(__dirname, 'build');
const outputFile = path.join(outputDir, 'infer-stack-technical-overview.pptx');
fs.mkdirSync(outputDir, { recursive: true });

const pptx=new pptxgen(); pptx.layout='LAYOUT_WIDE'; pptx.author='Infer-stack'; pptx.subject='Technical introduction and benchmark case study'; pptx.title='infer-stack | On-demand inference infrastructure'; pptx.lang='en-US';
const W=40/3,H=7.5; const C={bg:'F7F8F6',white:'FFFFFF',ink:'172B38',muted:'536675',soft:'DCE5E5',teal:'087E80',tealLight:'D7EFEB',blue:'3569A7',blueLight:'E4ECF8',orange:'D17A29',orangeLight:'F7EBD9',dark:'102834',dark2:'193C49',cyan:'80D9CB',grey:'8C9BA5',light:'D2E4E8'};
const F='Aptos', M='Liberation Mono'; const SH=pptx.ShapeType;
function text(s,t,x,y,w,h,o={}){s.addText(t,{x,y,w,h,fontFace:F,fontSize:18,color:C.ink,margin:0,breakLine:false,valign:'mid',...o});}
function rect(s,x,y,w,h,fill=C.white,line=null,r=0){s.addShape(r?SH.roundRect:SH.rect,{x,y,w,h,rectRadius:r,fill:{color:fill},line:{color:line||fill,transparency:line?0:100,width:0.8},radius:r});}
function line(s,x1,y1,x2,y2,col=C.soft,width=1,arrow=false,dash=null){s.addShape(SH.line,{x:x1,y:y1,w:x2-x1,h:y2-y1,line:{color:col,width,beginArrowType:undefined,endArrowType:arrow?'triangle':undefined,dashType:dash||undefined}});}
function circle(s,x,y,d,fill,lineColor=null){s.addShape(SH.ellipse,{x,y,w:d,h:d,fill:{color:fill},line:{color:lineColor||fill,transparency:lineColor?0:100,width:.8}});}
function tag(s,t,x,y,w,fill=C.tealLight,fg=C.teal){rect(s,x,y,w,.35,fill);text(s,t,x+.11,y,w-.22,.35,{fontSize:11,bold:true,color:fg,charSpacing:.5});}
function header(s,k,title,subtitle){s.background={color:C.bg};text(s,k.toUpperCase(),.55,.30,12.2,.30,{fontSize:11,bold:true,color:C.teal,charSpacing:1.5});text(s,title,.55,.77,12.2,.67,{fontSize:33,bold:true});if(subtitle)text(s,subtitle,.57,1.51,12.15,.52,{fontSize:17,color:C.muted,valign:'top'});}
function footer(s,n,section,dark=false){const col=dark?C.light:C.muted;line(s,.55,7.03,12.78,7.03,dark?'3C5A65':C.soft,.7);text(s,'infer-stack  /  '+section,.55,7.12,10,.20,{fontSize:10,color:col});text(s,String(n).padStart(2,'0'),12.05,7.12,.72,.20,{fontSize:10,color:col,align:'right'});}
function slide(k,t,st){let s=pptx.addSlide();header(s,k,t,st);footer(s,pptx._slides.length,k);return s;}
function notes(s,body,src){s.addNotes(`${body}\n\n[Sources]\n${src.map(x=>'- '+x).join('\n')}\n[/Sources]`);}
const repo='https://github.com/AIQ-Kitware/infer_stack';
const src=(...p)=>p.map(x=>`Project source / ${x} (snapshot 0b637511df60 plus supplied Nov 25 overlays; ${repo})`);
const benchmark=['User-provided benchmark export: summary.csv, report.md, configurations.json (Nov 25, 2026). 6 endpoint acquisitions, 96 successful measured requests, 4 context-headroom skips; 112 total ledger records.'];
function code(s,t,x,y,w,h,fs=17){rect(s,x,y,w,h,C.dark);text(s,t,x+.22,y+.15,w-.44,h-.30,{fontFace:M,fontSize:fs,color:'E2F4F0',valign:'top',breakLine:false});}
function block(s,title,desc,x,y,w,h,accent=C.teal,fill=C.white){rect(s,x,y,w,h,fill,C.soft);rect(s,x,y,.07,h,accent);text(s,title,x+.23,y+.19,w-.46,.42,{fontSize:22,bold:true});text(s,desc,x+.23,y+.76,w-.46,h-.9,{fontSize:17,color:C.muted,valign:'top'});}
function labelBox(s,t,sub,x,y,w,h,fill=C.white,accent=C.teal){rect(s,x,y,w,h,fill,C.soft);text(s,t,x+.15,y+.11,w-.3,.38,{fontSize:18,bold:true,color:accent,align:'center'});text(s,sub,x+.12,y+.59,w-.24,h-.67,{fontSize:14,color:C.muted,align:'center',valign:'top'});}
function bullets(s,arr,x,y,w,h,fs=19){const step=h/arr.length;arr.forEach((a,i)=>{circle(s,x,y+i*step+.15,.09,C.teal);text(s,a,x+.25,y+i*step,w-.25,step-.07,{fontSize:fs,valign:'top'});});}
function data(endpoint,target,phase,key){let r=rows.find(a=>a.endpoint===endpoint&&a.target===target&&a.phase===phase&&a.ok===1);return r?r[key]:null;}
const ep={q:'qwen3.8-27b-dbirks-hyperqwen',ql:'qwen3.8-27b-dbirks-hyperqwen-long',qh:'qwen3.8-27b-dbirks-hyperqwen-huge',m:'inkling-2-mini-maya',ml:'inkling-2-mini-maya-long',mh:'inkling-2-mini-maya-huge'};
const contexts=[1024,8192,32768,65536,131072,163840];const labels=['1K','8K','32K','64K','128K','160K'];
// 1 - Title
{
 const s=pptx.addSlide();s.background={color:C.dark};rect(s,0,0,.16,H,C.cyan);text(s,'TECHNICAL OVERVIEW  /  NOVEMBER 2026',.65,.49,8,.35,{fontSize:12,bold:true,color:C.cyan,charSpacing:2});
 text(s,'infer-stack',.62,1.48,8.2,1.10,{fontSize:64,bold:true,color:C.white});text(s,'On-demand inference\ninfrastructure',.65,2.71,7.6,1.45,{fontSize:34,bold:true,color:C.white,valign:'top'});text(s,'Discover, acquire, serve, release, benchmark.',.67,4.52,7.6,.52,{fontSize:21,color:C.light});text(s,'One catalog. One API. Multiple serving engines.',.67,5.26,7.7,.5,{fontSize:17,color:C.cyan});
 const x=9.04,w=3.55;for(const [i,a] of [['01','Declare','catalog.yaml'],['02','Realize','GPU placement + leases'],['03','Consume','OpenAI-compatible API']].entries()){let yy=1.5+i*1.55;rect(s,x,yy,w,1.24,C.dark2);text(s,a[0],x+.20,yy+.19,.52,.38,{fontSize:15,color:C.cyan,bold:true});text(s,a[1],x+.85,yy+.13,w-1.05,.46,{fontSize:23,bold:true,color:C.white});text(s,a[2],x+.85,yy+.66,w-1.05,.32,{fontSize:14,color:C.light});}
 footer(s,1,'overview',true);notes(s,'Audience: engineers who run or evaluate local language models. Introduce infer-stack as lifecycle infrastructure rather than a new inference engine or model. The deck reflects the supplied source and updates through Nov 25, 2026, not a claim that each configuration is production-validated.',src('README.md','pyproject.toml','docs/index.md'));
}
// 2 - Motivation
{
 const s=slide('Motivation','The model is not the whole serving stack','A workstation or cluster may run different models over the course of one day.');
 const y=2.17;block(s,'Manual serving','Choose images and flags.\n\nTrack GPU assignment.\n\nRestart services and reconnect clients.',.55,y,5.97,3.60,C.orange);block(s,'With infer-stack','Declare named endpoints.\n\nAcquire resources when needed.\n\nKeep one client-facing API.',6.81,y,5.97,3.60,C.teal);
 text(s,'The goal: spend time on the workload, not repeatedly wiring the serving environment.',.57,6.08,12.15,.64,{fontSize:22,bold:true,color:C.teal});notes(s,'This is the motivation, not a measured time-savings claim. The project is an orchestration/control layer. It does not replace the model runtime.',src('README.md','docs/operations.md'));
}
// 3 - workflow
{
 const s=slide('Quickstart','From a clean environment to a running endpoint','Linux, Python 3.10+, Docker Compose; NVIDIA runtime for GPU inference.');
 const cols=[.55,4.77,9.0];const widths=[4.04,4.04,3.78];
 const steps=[['1  Install & configure',"pip install 'infer-stack[tui]'\numask 077\ninfer-stack config init \\\n  --backend compose\ninfer-stack config set ui false"],['2  Discover & acquire',"infer-stack catalog suggest \\\n  --apply\ninfer-stack catalog endpoint list\ninfer-stack acquire smollm2-135m \\\n  --env-file lease.env"],['3  Verify & release',"infer-stack test smollm2-135m\n\ninfer-stack release \\\n  --env-file lease.env"]];
 steps.forEach((a,i)=>{text(s,a[0],cols[i],2.13,widths[i],.45,{fontSize:20,bold:true});code(s,a[1],cols[i],2.72,widths[i],2.65,12.5);});
 rect(s,.55,5.66,12.23,.97,C.tealLight);text(s,'No GPU?',.75,5.83,1.46,.30,{fontSize:19,bold:true,color:C.teal});text(s,'Use catalog suggest --simulator --apply and acquire mock-smol to exercise the same workflow.',2.26,5.78,10.26,.62,{fontSize:17,color:C.ink});
 notes(s,'Show these commands in a shell if live demo is possible. The UI is disabled in this quickstart to avoid publishing an unauthenticated WebUI; enable it deliberately on a trusted, firewalled host. The default gateway API remains available. Choose another catalog endpoint if the small smoke-test endpoint is unavailable; confirm deployment changes when prompted. The simulator produces mock text, not real inference. Never commit lease.env.',src('README.md','docs/mock-endpoints.md','docs/planning/known-limitations.md'));
}
// 4 - API client contract
{
 const s=slide('Client contract','Keep the client independent of the engine','Use the endpoint alias as the model name through the normal OpenAI-compatible gateway.');
 text(s,'Shell / lease ownership',.55,2.14,5.95,.42,{fontSize:20,bold:true});text(s,'client.py',6.82,2.14,5.96,.42,{fontSize:20,bold:true});
 code(s,"pip install openai\n\ninfer-stack run \\\n  --endpoint smollm2-135m -- \\\n  python client.py",.55,2.73,5.96,2.63,16);
 code(s,"import os\nfrom openai import OpenAI\n\nclient = OpenAI()\nresponse = client.chat.completions.create(\n    model=os.environ['INFER_STACK_MODELS'],\n    messages=[{'role': 'user',\n               'content': 'Say hello.'}],\n)\nprint(response.choices[0].message.content)",6.82,2.73,5.96,3.66,13.1);
 rect(s,.55,5.59,5.96,.80,C.tealLight);text(s,'The wrapper acquires, exports connection info, runs the child, then releases its lease.',.72,5.70,5.62,.57,{fontSize:16,color:C.ink});
 text(s,'The SDK reads OPENAI_BASE_URL and OPENAI_API_KEY from the child environment.',.57,6.62,12.18,.26,{fontSize:13,color:C.muted});notes(s,'This single-endpoint example uses the standard Python OpenAI client and the environment supplied by infer-stack run. The SDK is an additional install. The alias, not the model repository, is the request model ID. For bundles, choose the appropriate individual alias rather than treating a comma-separated list as one model. The example assumes normal local gateway settings and a catalog containing smollm2-135m. No credentials are embedded in the script.',src('README.md','docs/operations.md','docs/mock-endpoints.md'));
}

// 5 - Architecture
{
 const s=slide('Architecture','One control plane, two paths','Management operations decide what runs; inference requests use a stable front door.');
 tag(s,'MANAGEMENT',.56,2.08,1.74);tag(s,'INFERENCE',.56,4.67,1.46,C.blueLight,C.blue);
 const boxY=2.58;const bx=[.57,3.50,6.42,9.34];const bw=[2.53,2.53,2.53,3.43];
 [['CLI / TUI','Acquire, release, inspect'],['Catalog + ledger','Definitions & desired state'],['Controller','Admit, reconcile, probe'],['Serving backend','Compose or KubeAI']].forEach((p,i)=>labelBox(s,p[0],p[1],bx[i],boxY,bw[i],1.30));
 for(let i=0;i<3;i++)line(s,bx[i]+bw[i]+.03,boxY+.64,bx[i+1]-.04,boxY+.64,C.teal,2,true);
 const yy=5.13;labelBox(s,'Application','OpenAI-compatible client',.57,yy,2.53,1.25,C.white,C.blue);labelBox(s,'LiteLLM','Alias + API routing',4.08,yy,2.74,1.25,C.white,C.blue);labelBox(s,'Model servers','vLLM / Ollama / custom*',9.34,yy,3.43,1.25,C.white,C.blue);
 line(s,3.15,yy+.61,4.02,yy+.61,C.blue,2,true);line(s,6.87,yy+.61,9.28,yy+.61,C.blue,2,true);line(s,11.05,3.90,11.05,5.08,C.teal,2,true);text(s,'realizes',11.19,4.24,1.4,.28,{fontSize:12,color:C.muted});
 text(s,'* Custom launchers and Ollama: Compose only. External OpenAI-compatible servers can also be routed.',.57,6.54,12.20,.25,{fontSize:12,color:C.muted});notes(s,'The architecture diagram is conceptual. Catalog defines possible serving configurations; the ledger records demand and desired deployments. Controller applies the selected backend and verifies generation. LiteLLM is the normal API gateway, not the scheduling authority. Optional Open WebUI is a client. See backend parity for differences and external endpoint access semantics.',src('docs/backend-parity.md','docs/catalog.md','infer_stack/leasing/controller.py'));
}
// 6 - objects
{
 const s=slide('Architecture','Four objects make the lifecycle explicit','The model, endpoint, deployment and lease are different things.');
 const arr=[['MODEL','Weight source','HF repository, revision, dtype, quantization'],['ENDPOINT','Serving recipe','Public alias, runtime, context, placement'],['DEPLOYMENT','Running realization','Compatible engine process or KubeAI Model'],['LEASE','Demand','A consumer’s claim on an endpoint; TTL and release']];
 arr.forEach((a,i)=>{let x=.55+i*3.10;rect(s,x,2.38,2.92,3.55,C.white,C.soft);tag(s,a[0],x+.17,2.59,2.58);text(s,a[1],x+.17,3.12,2.58,.62,{fontSize:23,bold:true,valign:'top'});text(s,a[2],x+.17,3.98,2.58,1.52,{fontSize:18,color:C.muted,valign:'top'});});
 text(s,'Multiple compatible consumers can share a deployment; different runtime configurations need not share.',.57,6.19,12.18,.51,{fontSize:18,bold:true,color:C.teal});notes(s,'A lease is not a container and it is not necessarily an exclusive allocation. A deployment can have multiple compatible served aliases and consumers. Incompatible settings lead to distinct deployment identity. The default persistent disk caches are independent of process residency.',src('docs/catalog.md','docs/operations.md','docs/backend-parity.md','infer_stack/leasing/catalog.py'));
}
// 7 - model catalog
{
 const s=slide('Configuration','Serving profiles are data, not scripts','Different endpoint recipes can refer to the same weights.');
 code(s,`models:\n  smol135:\n    source: >-\n      hf://HuggingFaceTB/SmolLM2-135M-Instruct\n\nendpoints:\n  smol135-1:\n    engine: vllm\n    model: smol135\n    runtime: {max_model_len: 8192}`, .55,2.15,7.13,4.32,16.2);
 text(s,'Catalog separates:',8.04,2.27,4.72,.45,{fontSize:23,bold:true});bullets(s,['Weight identity and public model alias','Runtime image, launch flags and context','GPU count, VRAM floor and device constraints','Sharing and reclaim policy'],8.04,2.99,4.70,3.10,18);
 notes(s,'This is a real compact catalog example from docs/catalog.md. Additional runtime settings, placement and sharing/reclaim options are declared in the catalog; a single model can have multiple named endpoint configurations. Config and catalog are user-authoritative; internal recovery snapshots keep approved publication state consistent across crashes.',src('docs/catalog.md','docs/adr/0001-user-config-is-authoritative.md'));
}
// 8 - registry
{
 const s=slide('Discovery','Start with a recipe; keep room for experimentation','Hardware-aware suggestions and full registry browsing serve different needs.');
 block(s,'Suggest','Propose a curated set for the GPUs detected.\n\nApply adds entries without pruning the existing catalog.',.55,2.15,5.96,2.85,C.teal);block(s,'Browse models','Inspect the complete suggestion registry, including hardware mismatches.\n\nImport, edit, or acquire an eligible profile.',6.81,2.15,5.97,2.85,C.blue);
 code(s,'infer-stack catalog suggest --simulate-hardware 4x96@12.0\ninfer-stack catalog import inkling-2-mini-maya-long --dry-run',.55,5.25,12.23,1.14,16);
 text(s,'Import is not a fit guarantee; actual acquisition still enforces placement requirements.',.57,6.54,12.17,.28,{fontSize:14,color:C.muted});notes(s,'The browser includes registered suggestions with fit/backend warnings. The CLI also exposes explicit imports and offline hardware simulation. Standalone bundled presets can be imported by CLI. A mismatched profile may be tuned, but importing it does not bypass placement. No claim that every recipe works on every GPU.',src('docs/model-profiles.md','docs/deepseek-v4-flash-sm120.md','infer_stack/leasing/discovery.py','infer_stack/templates/suggestion-pool.yaml'));
}
// 9 - lifecycle
{
 const s=slide('Lifecycle','Acquire resources only for the demand you have','A real generation through the front door is the readiness gate.');
 const px=[.57,3.03,5.49,7.95,10.41],pw=2.35;
 [['Acquire','Resolve & admit'],['Realize','Reuse or start'],['Ready','Probe generation'],['Use','Client requests'],['Release','End this lease']].forEach((a,i)=>labelBox(s,a[0],a[1],px[i],2.43,pw,1.34));
 for(let i=0;i<4;i++)line(s,px[i]+pw+.02,3.10,px[i+1]-.02,3.10,C.teal,1.6,true);
 const yy=4.22;block(s,'Other leases remain','Shared deployment stays available.',.57,yy,3.95,1.62,C.blue);block(s,'No remaining demand','Stop or keep warm, according to policy.',4.68,yy,3.95,1.62,C.teal);block(s,'Disk cache persists','Weights and compiler caches may be reused later.',8.79,yy,3.97,1.62,C.orange);
 text(s,'For automated jobs: infer-stack run --endpoint NAME -- COMMAND',.57,6.19,12.20,.49,{fontSize:20,bold:true,color:C.teal});notes(s,'The lease covers the consumer’s demand. Compatible deployments can coalesce; otherwise the backend realizes new desired state. A release does not stop workloads owned by other leases. Suggested endpoints normally stop on final release; keep-warm is a separate policy. Disk caches avoid downloads and some compilation, but a restarted process still needs to load weights into VRAM. TTLs provide cleanup backstops. The run command releases the acquired lease when its child exits.',src('docs/operations.md','docs/persistent-caches-and-warm-restarts.md','docs/backend-parity.md','README.md'));
}
// 10 - TUI
{
 const s=slide('Operations','One interface for everyday operations','Use the TUI interactively; use CLI verbs in scripts and evaluation pipelines.');
 const a=[['Browse & configure','catalog endpoint list\ncatalog import NAME','Inspect fit, preview YAML, edit endpoint'],['Acquire & release','acquire NAME\nrelease --env-file lease.env','Manage demand and deployment status'],['Observe & diagnose','ps\nlogs --engine -f','Monitor devices, instances and engine logs']];
 a.forEach((p,i)=>{let y=2.14+i*1.47;rect(s,.55,y,12.23,1.29,C.white,C.soft);text(s,p[0],.77,y+.14,3.25,.46,{fontSize:21,bold:true});text(s,p[2],.77,y+.70,3.80,.47,{fontSize:14,color:C.muted,valign:'top'});text(s,p[1],4.91,y+.13,7.64,1.00,{fontFace:M,fontSize:17,color:C.teal,valign:'mid'});});
 text(s,'Launch with infer-stack tui. Engine-only logs exclude the gateway and WebUI.',.57,6.62,12.2,.28,{fontSize:14,color:C.muted});notes(s,'The rows are a conceptual workflow, not a screenshot. The TUI can browse/import, edit endpoints, acquire/release, and view logs and instances. The CLI flags shown reflect the recent engine-only logs overlay. Each command in the middle column is prefixed with infer-stack.',src('README.md','docs/operations.md','docs/model-profiles.md','infer_stack/tui.py','infer_stack/cli/commands_runtime.py'));
}
// 11 - backends
{
 const s=slide('Backends','Same lease interface, different realization layers','Move from a workstation to a Kubernetes cluster without changing the normal client contract.');
 const x1=.55,x2=6.79,y=2.18,w=5.99;
 block(s,'Docker Compose','Local Docker daemon and GPU planner.\n\nSupports vLLM, Ollama and compatible custom launchers.',x1,y,w,3.12,C.teal);block(s,'KubeAI','Kubernetes Model resources and cluster scheduler.\n\nGPU resource profiles, stock vLLM and cluster operations.',x2,y,w,3.12,C.blue);
 rect(s,.55,5.56,12.23,.95,C.dark);text(s,'Shared: catalog  ·  ledger  ·  controller  ·  endpoint aliases  ·  lease environment',.75,5.68,11.83,.66,{fontSize:19,bold:true,color:C.white,align:'center'});
 text(s,'Backend parity is not complete. Custom launcher commands/mounts and Ollama are Compose-only.',.57,6.64,12.17,.23,{fontSize:12,color:C.muted});notes(s,'KubeAI delegates scheduling to Kubernetes and uses GPU resource profiles instead of host index pins. Cluster setup is an additional operation. The project also has a remote control service with token or identity credentials, TLS off-host, and a separate inference credential; see kubeai-backend.md. The two backends share the core control-plane contract, not every feature.',src('docs/backend-parity.md','docs/kubeai-backend.md','docs/cluster-setup.md'));
}
// 12 - recipes
{
 const s=slide('Model integrations','Specialized runtimes fit the same endpoint contract','Three examples show how explicit profiles capture hardware and launcher differences.');
 const cards=[['HyperQwen','Qwen3.8 27B','Single RTX 3090 profiles; different images and context configurations.','Measured',C.blue],['Maya','Inkling 2 Mini','Native CUDA server on one RTX 3090; managed image build and persistent model files.','Measured',C.teal],['DeepSeek V4 Flash','SM120 profile','TP4 / 128K recipe passed generation readiness on four RTX PRO 6000s; TP2 is experimental.','Readiness only',C.orange]];
 cards.forEach((a,i)=>{let x=.55+i*4.16;rect(s,x,2.19,3.91,4.21,C.white,C.soft);rect(s,x,2.19,3.91,.08,a[4]);text(s,a[0],x+.18,2.47,3.55,.54,{fontSize:24,bold:true});text(s,a[1],x+.18,3.09,3.55,.37,{fontSize:15,bold:true,color:a[4]});text(s,a[2],x+.18,3.68,3.55,1.76,{fontSize:17,color:C.muted,valign:'top'});tag(s,a[3].toUpperCase(),x+.18,5.73,3.55,a[4]===C.orange?C.orangeLight:a[4]===C.blue?C.blueLight:C.tealLight,a[4]);});
 notes(s,'HyperQwen and Maya are the endpoint families in the provided benchmark. The two HyperQwen images differed in the test. Maya is a custom native server managed under the generic launcher contract (not vLLM internally). DeepSeek TP4 readiness was recorded in project docs on Nov 20, 2026; there is no throughput result and no single-3090 claim. TP2 is an unverified starting profile for two suitable SM120 cards. The project’s docs distinguish readiness, performance, and quality.',[...src('docs/model-profiles.md','docs/maya-inkling-mini.md','docs/deepseek-v4-flash-sm120.md'),...benchmark]);
}
// 13 - benchmark methods
{
 const s=slide('Measurement','A benchmark should characterize the workload','The runner measures the actual serving path, not a kernel in isolation.');
 code(s,"infer-stack benchmark run 'qwen3.8-*' 'inkling-2-mini-maya*'",.55,2.10,12.23,.68,16);
 const arr=[['Fresh','Unique prompt prefix'],['Exact prefix','Identical prompt'],['Extended prefix','Same prefix + additional code']];arr.forEach((a,i)=>labelBox(s,a[0],a[1],.55+i*4.16,3.08,3.91,1.31,C.white,i===0?C.teal:C.blue));
 text(s,'Sweep',.57,4.71,2.0,.40,{fontSize:21,bold:true});text(s,'1K → 8K → 32K → 64K → 128K → 160K target tokens; up to 512 output tokens.',2.62,4.66,10.12,.56,{fontSize:18});
 text(s,'Record',.57,5.40,2.0,.40,{fontSize:21,bold:true});text(s,'TTFT, streamed t/s, end-to-end t/s, reported cached tokens, GPU identity, PCIe, image & provenance.',2.62,5.33,10.12,.75,{fontSize:18});
 text(s,'Live terminal metrics + HTML dashboard; append-only JSONL ledger for offline comparisons.',.57,6.55,12.2,.28,{fontSize:15,color:C.teal,bold:true});notes(s,'Each endpoint is acquired, measured, and released before the next. Fresh means a new user prefix, not a cold process or globally flushed cache. The benchmark attempts reuse; the backend may not report or implement it. TTFT includes transport, queueing and processing. Stream rate estimates first-to-last observed output, not a pure engine decode counter. Runs use the normal gateway; the current benchmark targets local Compose with LiteLLM. Live snapshots are derived from the ledger.',src('docs/benchmark.md','infer_stack/benchmark/runner.py','infer_stack/benchmark/dashboard.py'));
}
// 14 - chart
{
 const s=slide('Case study · RTX 3090','Maya maintains throughput as context grows','Six endpoint configurations measured on the same physical 3090 (observed Gen3 ×2).');
 const x=1.12,y=2.66,w=8.33,h=3.61;const max=200;const series=[{id:ep.q,name:'HyperQwen',c:C.grey},{id:ep.ql,name:'HyperQwen Long',c:C.blue},{id:ep.qh,name:'HyperQwen Huge',c:C.orange},{id:ep.ml,name:'Maya Long',c:C.teal}];
 text(s,'Fresh stream t/s',.56,2.10,5,.32,{fontSize:16,bold:true});
 for(let v=0;v<=200;v+=50){let yy=y+h-h*v/max;line(s,x,yy,x+w,yy,C.soft,.7);text(s,String(v),.56,yy-.13,.46,.26,{fontSize:11,color:C.muted,align:'right'});}
 contexts.forEach((v,i)=>{let xx=x+w*i/5;text(s,labels[i],xx-.32,y+h+.14,.64,.28,{fontSize:12,color:C.muted,align:'center'});});
 for(let ser of series){let last=null;contexts.forEach((v,i)=>{let value=data(ser.id,v,'fresh','stream_tps');if(value==null){last=null;return}let xx=x+w*i/5,yy=y+h-h*value/max;if(last)line(s,last[0],last[1],xx,yy,ser.c,2.5);circle(s,xx-.055,yy-.055,.11,ser.c);last=[xx,yy];});}
 let lx=9.86;series.forEach((ser,i)=>{let yy=2.18+i*.41;line(s,lx,yy+.15,lx+.36,yy+.15,ser.c,2.7);circle(s,lx+.13,yy+.09,.12,ser.c);text(s,ser.name,lx+.47,yy,2.38,.30,{fontSize:13,color:C.ink});});
 rect(s,9.79,4.06,2.99,2.19,C.tealLight);text(s,'4.22×',9.98,4.28,2.60,.77,{fontSize:39,bold:true,color:C.teal});text(s,'Maya Long vs. HyperQwen Huge at 160K: 92.5 vs 21.9 t/s',9.98,5.16,2.60,.87,{fontSize:16,color:C.ink,valign:'top'});
 text(s,'Target context (Ki tokens)  ·  Estimated stream rate; one trial per point; skipped targets omitted.',.57,6.76,12.18,.18,{fontSize:11,color:C.muted});notes(s,'Data: fresh phase, API-reported completion tokens over observed stream interval. 96 successful requests and four skipped context targets across six configurations. Each point is one trial. The chart shows the three HyperQwen variants and Maya Long; the other two Maya variants also sustained 86.6–92.6 t/s at 160 Ki. The GPU was the same card for every trial, observed PCIe Gen3 x2; this is not a PCIe-causality experiment. Different images, tokenizers, model quality and speculative decoding settings confound a pure engine comparison.',benchmark);
}
// 15 - prefix reuse
{
 const s=slide('Case study · RTX 3090','The ranking reverses on short follow-up requests','At 160K target tokens, HyperQwen Huge showed substantial prefix reuse; Maya Long did not.');
 const targets=[{title:'Time to first token',unit:'seconds',key:'ttft_s',max:220,smaller:true},{title:'End-to-end output rate',unit:'tokens/s',key:'e2e_tps',max:24,smaller:false}];
 const series=[{id:ep.qh,n:'HyperQwen Huge',c:C.orange},{id:ep.ml,n:'Maya Long',c:C.teal}];
 targets.forEach((a,i)=>{let x=.55+i*6.22;rect(s,x,2.15,6.01,3.83,C.white,C.soft);text(s,a.title,x+.2,2.34,5.61,.42,{fontSize:21,bold:true});text(s,a.unit+'  ·  '+(a.smaller?'lower is better':'higher is better'),x+.2,2.80,5.61,.25,{fontSize:12,color:C.muted});
  ['fresh','extended-prefix'].forEach((ph,j)=>{let yy=3.17+j*1.32;text(s,ph==='fresh'?'Fresh':'Extended prefix',x+.2,yy,5.61,.26,{fontSize:13,bold:true,color:C.muted});series.forEach((ss,k)=>{let v=data(ss.id,163840,ph,a.key),by=yy+.35+k*.35;let bar=4.52*v/a.max;rect(s,x+.20,by,bar,.23,ss.c);text(s,v.toFixed(v>=10?1:2),x+.25+bar,by-.015,.95,.26,{fontSize:13,bold:true,color:ss.c});});});
 });
 line(s,.75,6.26,1.13,6.26,C.orange,4);text(s,'HyperQwen Huge',1.26,6.10,2.60,.34,{fontSize:15});line(s,4.05,6.26,4.43,6.26,C.teal,4);text(s,'Maya Long',4.56,6.10,2.25,.34,{fontSize:15});text(s,'512-token budget, one repeat',8.28,6.10,4.5,.34,{fontSize:15,color:C.muted,align:'right'});
 text(s,'Reported extended-prefix cached tokens: 163,968 (HyperQwen Huge) vs. 0 (Maya Long).',.57,6.63,12.18,.28,{fontSize:14,color:C.muted});notes(s,'Fresh TTFT 208 s vs 98.6 s; extended-prefix TTFT 2.13 s vs 99.4 s. Extended-prefix end-to-end output rate 21.5 vs 4.87 t/s. Fresh E2E 2.21 vs 4.92 t/s. All displayed rates are measured single-trial estimates. A reported zero cached tokens and almost unchanged TTFT are evidence of no observed benefit here, not proof that the engine can never reuse prefixes.',benchmark);
}
// 16 - implications
{
 const s=slide('Case study · interpretation','Choose for the request pattern, not peak tokens/s','Long-context reasoning and frequent tool calls stress different parts of the serving path.');
 block(s,'Long generation / compaction','Fresh prompt ingestion and sustained stream rate matter. Maya Long is a strong performance candidate in this sweep.',.55,2.15,5.96,2.91,C.teal);block(s,'Short iterative requests','Prefix reuse can dominate total latency. HyperQwen Huge performed much better on repeated or extended prompts at 160K.',6.81,2.15,5.97,2.91,C.orange);
 rect(s,.55,5.32,12.23,1.24,C.white,C.soft);text(s,'~3,130 output tokens',.75,5.47,4.44,.74,{fontSize:29,bold:true,color:C.ink});text(s,'Estimated crossover at 160K, using measured extended-prefix TTFT + output tokens / stream rate.',5.38,5.49,7.14,.72,{fontSize:17,color:C.muted});
 text(s,'Extrapolation from 512-token trials; not a measurement of longer generation or solution quality.',.57,6.67,12.18,.24,{fontSize:12,color:C.muted});notes(s,'Crossover estimate: (99.3702 − 2.1282) / (1/23.5707 − 1/88.3704) ≈ 3,126 output tokens, rounded to ~3,130. Assumes constant streamed rate and TTFT; actual throughput may change as context grows or speculative acceptance changes. Do not equate speed with task success. For a compaction-heavy agent, profile actual prompts, tool-call intervals, cache persistence, and successful completion time.',benchmark);
}
// 17 - limitations and next steps
{
 const s=slide('Boundaries & next steps','Make the next measurements more discriminating','Current capabilities are useful; the project remains planning-stage software.');
 block(s,'Operating boundaries','Linux and trusted operators. One control plane per host/namespace. Protect credentials and avoid exposing unauthenticated WebUI.',.55,2.13,5.96,2.10,C.orange);block(s,'Benchmark boundaries','Single-request serving performance. One repetition, one physical 3090, one synthetic workload; not a model-quality evaluation.',6.81,2.13,5.97,2.10,C.blue);
 text(s,'Next experiments',.57,4.53,12.18,.43,{fontSize:22,bold:true,color:C.teal});bullets(s,['Repeat the sweep on the better-connected 3090 with matched configurations and multiple trials.','Measure actual agent sessions, including compaction, tool calls and time to a correct solution.','Investigate Maya prefix reuse and HyperQwen Huge’s throughput drop with engine telemetry.'],.57,5.08,12.19,1.77,17);notes(s,'The next experiments are recommendations, not implemented features. Benchmark data does not isolate hardware bandwidth, software revision, model, or tokenizer effects. The project’s security docs specify trusted operator boundaries and private config/data roots, and do not claim multi-tenant isolation. Source maturity is listed as planning in pyproject.toml.',[...src('docs/planning/known-limitations.md','docs/benchmark.md','pyproject.toml'),...benchmark]);
}
// 18 - Close
{
 const s=pptx.addSlide();s.background={color:C.dark};rect(s,0,0,.16,H,C.cyan);text(s,'GET STARTED',.66,.50,10,.30,{fontSize:12,color:C.cyan,bold:true,charSpacing:2});text(s,'Run the model you need.\nMeasure the workload you care about.',.64,1.29,12,1.55,{fontSize:38,bold:true,color:C.white,valign:'top'});
 const arr=[['Explore','infer-stack tui'],['Automate','infer-stack run --endpoint NAME -- COMMAND'],['Compare','infer-stack benchmark run NAME1 NAME2']];arr.forEach((a,i)=>{let y=3.14+i*.87;rect(s,.65,y,12.02,.72,C.dark2);text(s,a[0],.87,y+.12,2.16,.49,{fontSize:19,bold:true,color:C.cyan});text(s,a[1],3.12,y+.12,9.29,.49,{fontSize:18,fontFace:M,color:C.white});});
 text(s,'github.com/AIQ-Kitware/infer_stack',.68,6.20,11.9,.44,{fontSize:19,color:C.cyan,hyperlink:{url:repo}});footer(s,pptx._slides.length,'get started',true);notes(s,'Close with three entry points: TUI for exploration, run for scoped automated work, benchmark for comparison. Link to project README and task-oriented documentation index. The artifact is a technical introduction and not an official release announcement.',src('README.md','docs/index.md','docs/benchmark.md','pyproject.toml'));
}
// Optional development-only layout diagnostics. The deck does not require
// these helpers to build, and the helper implementation is not vendored.
// Chart elements, accent bars and label overlays intentionally overlap.
if (process.env.INFER_STACK_SLIDE_QA_HELPERS) {
    const { warnIfSlideHasOverlaps, warnIfSlideElementsOutOfBounds } =
        require(path.resolve(process.env.INFER_STACK_SLIDE_QA_HELPERS));
    for (const slide of pptx._slides) {
        warnIfSlideHasOverlaps(slide, pptx);
        warnIfSlideElementsOutOfBounds(slide, pptx);
    }
}

pptx.writeFile({ fileName: outputFile })
    .then(() => console.log(`Wrote ${outputFile}`))
    .catch(error => {
        console.error(error);
        process.exitCode = 1;
    });

const {make,C,src,repo}=require('../layout.cjs');
module.exports=async function(){
const {slide,cover,note,card,strip,flow,txt,rect,line,write}=make('what','What is infer-stack?',C.blue);
function block(s,x,y,w,h,title,body,fill=C.white,accent=C.blue){
    rect(s,x,y,w,h,fill,C.soft);
    txt(s,title,x+.18,y+.14,w-.36,.44,{fontSize:22,bold:true,color:accent});
    txt(s,body,x+.18,y+.66,w-.36,h-.79,{fontSize:17,color:C.muted,valign:'top'});
}
{
const s=cover('Conceptual introduction / 8–10 minutes','What is infer-stack?','A control layer for running and managing model inference on your own infrastructure.','Named endpoints · resource leases · a consistent API');
note(s,'The goal of this deck is a mental model: what infer-stack is, what it owns, and what it delegates. No prior familiarity with its commands is necessary. The why deck is the separate adoption argument; the how deck is the tutorial.',src('README.md','docs/catalog.md','docs/operations.md'));
}
{
const s=slide('It sits between applications and inference engines.','Two paths: manage serving resources, and send inference requests.');
// Separate control and data paths; all shapes are editable.
block(s,.55,2.25,3.35,1.35,'Operator / job','CLI or terminal UI');
block(s,4.98,2.25,3.35,1.35,'Infer-stack','Catalog, leases, placement',C.blueLight,C.blue);
block(s,9.43,2.25,3.35,1.35,'Backend','Compose or KubeAI');
line(s,3.94,2.92,4.93,2.92,C.blue,2,true);line(s,8.38,2.92,9.38,2.92,C.blue,2,true);
line(s,.55,3.86,12.78,3.86,C.soft,1);
block(s,.55,4.10,3.35,1.35,'Application','Agent, script, or WebUI');
block(s,4.98,4.10,3.35,1.35,'Gateway','LiteLLM / endpoint alias',C.tealLight,C.teal);
block(s,9.43,4.10,3.35,1.35,'Serving engine','Generates tokens',C.white,C.teal);
line(s,3.94,4.77,4.93,4.77,C.teal,2,true);line(s,8.38,4.77,9.38,4.77,C.teal,2,true);
line(s,11.11,3.63,11.11,4.06,C.blue,1.5,true);
strip(s,'Infer-stack manages the serving lifecycle; the engine performs inference.');
note(s,'Logical diagram of the normal gateway-enabled managed serving path. Blue indicates lifecycle/control; teal indicates inference requests. The backend realizes desired serving resources; client requests do not pass through the leasing controller. Compose can also be configured without LiteLLM; external routes can target an already-running compatible server. Open WebUI is optional. KubeAI backend support differs from Compose.',src('README.md','docs/catalog.md','docs/operations.md','docs/kubeai-backend.md','infer_stack/leasing/controller.py'));
}
{
const s=slide('Four terms explain the whole system.','Separate what is configured, what is running, and who needs it.');
const rows=[['Model','Weights / model identity','The artifact that the engine loads.'],['Endpoint','Named serving recipe','Engine, model, runtime, placement and policy.'],['Deployment','Realized serving resources','A running or managed instance of a compatible recipe.'],['Lease','A consumer’s demand','A claim on a managed endpoint; not necessarily exclusive.']];
rows.forEach((r,i)=>{const y=2.19+i*1.00;rect(s,.55,y,12.23,.94,C.white,C.soft);txt(s,r[0],.76,y+.11,2.22,.72,{fontSize:24,bold:true,color:C.blue});txt(s,r[1],3.02,y+.11,3.55,.72,{fontSize:19,bold:true});txt(s,r[2],6.76,y+.11,5.79,.72,{fontSize:17,color:C.muted});});
strip(s,'The catalog defines endpoints. Active leases determine serving demand.',6.25,C.blueLight,C.blue);
note(s,'Model identities can include weight revision and quantization. Endpoints name the API-facing configuration. Deployments are the reconciled serving resources; compatible endpoints and leases may share one deployment. Leases express demand, not a guarantee of exclusive GPU allocation. External endpoints are a special case with route access rather than a local serving lease.',src('docs/catalog.md','docs/operations.md','infer_stack/leasing/catalog.py','infer_stack/leasing/controller.py'));
}
{
const s=slide('One model can have multiple endpoints.','The serving recipe is distinct from the model weights.');
block(s,.55,2.39,4.18,2.76,'Model: smol135','SmolLM2-135M-Instruct\n\nOne declared weight source.',C.white,C.blue);
block(s,6.65,2.19,6.13,1.49,'smol135-short','vLLM · context limit 2,048 tokens',C.blueLight,C.blue);
block(s,6.65,4.00,6.13,1.49,'smol135-long','vLLM · context limit 8,192 tokens',C.tealLight,C.teal);
line(s,4.75,3.77,5.54,3.77,C.blue,1.5);line(s,5.54,2.94,5.54,4.74,C.blue,1.5);line(s,5.54,2.94,6.61,2.94,C.blue,1.5,true);line(s,5.54,4.74,6.61,4.74,C.blue,1.5,true);
strip(s,'Endpoint names select a recipe, not just a checkpoint.');
note(s,'Illustrative endpoint names, not necessarily built-in catalog names. The 2,048 and 8,192 settings illustrate distinct runtime context limits for one model. The 8,192 setting appears in the catalog guide; neither profile is claimed to have been benchmarked here. Actual context capacity depends on model and runtime support and GPU memory. A recipe also includes runtime image, flags, placement and reclaim policy.',src('docs/catalog.md','docs/model-profiles.md','infer_stack/leasing/catalog.py'));
}
{
const s=slide('A lease is a claim, not a server.','Compatible demand can share a running deployment.');
const stages=[['Alice acquires','1 active lease','A'],['Bob acquires','2 active leases','A + B'],['Alice releases','1 active lease','B']];
stages.forEach((a,i)=>{const x=.55+i*4.16;rect(s,x,2.25,3.91,3.18,C.white,C.soft);txt(s,a[0],x+.17,2.44,3.57,.46,{fontSize:22,bold:true});txt(s,a[1],x+.17,3.0,3.57,.34,{fontSize:16,color:C.muted});rect(s,x+.17,3.53,3.57,.53,C.blueLight);txt(s,a[2],x+.27,3.60,3.37,.37,{fontSize:18,bold:true,color:C.blue,align:'center'});line(s,x+1.955,4.08,x+1.955,4.35,C.blue,1.5,true);rect(s,x+.17,4.39,3.57,.79,C.tealLight);txt(s,'Same deployment',x+.27,4.53,3.37,.48,{fontSize:18,bold:true,color:C.teal,align:'center'});if(i<2)line(s,x+3.94,3.79,x+4.13,3.79,C.grey,1.5,true);});
strip(s,'After the final release, reclaim policy determines whether the engine stops or stays warm.');
note(s,'A and B are independent leases, not user-authentication identities. The example assumes the same compatible managed endpoint and available resources. Sharing depends on configuration. Releasing A does not terminate B’s active demand; when the last lease ends the reclaim policy controls residency. Persistent model caches on disk are independent of running VRAM residency.',src('docs/operations.md','docs/catalog.md','docs/persistent-caches-and-warm-restarts.md','infer_stack/leasing/controller.py'));
}
{
const s=slide('What does infer-stack actually manage?','A catalog and lifecycle manager with operational and measurement tools.');
card(s,.55,2.23,5.97,1.76,'Discover and configure','Suggest compatible recipes; browse and edit other known profiles.',C.blue);
card(s,6.81,2.23,5.97,1.76,'Place and reconcile','Use GPU constraints, start or reuse deployments, verify readiness.',C.teal);
card(s,.55,4.24,5.97,1.76,'Connect and observe','Publish aliases and connection details; inspect status and logs.',C.blue);
card(s,6.81,4.24,5.97,1.76,'Measure and compare','Benchmark context and prefix reuse; retain configuration identity.',C.orange);
strip(s,'CLI and TUI are two interfaces to the same catalog and managed lifecycle.',6.25);
note(s,'Discovery proposals are not fit guarantees. Browse exposes the registered suggestion pool and warns on hardware or backend mismatch; import is not a placement override. Benchmarking currently targets the local Compose/LiteLLM route and is not a model-quality evaluation. Status and logs read backend state. See the how, evidence, and demo decks for operational detail.',src('README.md','docs/model-profiles.md','docs/operations.md','docs/benchmark.md','docs/catalog.md'));
}
{
const s=slide('Where can it run, and what can it serve?','The backend chooses how managed resources are realized.');
card(s,.55,2.23,5.97,2.65,'Docker Compose','Linux workstation or server.\nvLLM, Ollama and compatible custom engines. Local GPU placement.',C.teal);
card(s,6.81,2.23,5.97,2.65,'KubeAI','Kubernetes cluster.\nStock vLLM via Model resources and GPU resource profiles.',C.blue);
rect(s,.55,5.13,12.23,.84,C.white,C.soft);txt(s,'Already-running endpoints',.76,5.28,4.15,.51,{fontSize:22,bold:true});txt(s,'Route an external OpenAI-compatible API through the gateway.',5.03,5.28,7.53,.51,{fontSize:17,color:C.muted});
strip(s,'Backend capabilities differ; not every engine or custom profile works everywhere.',6.25,C.orangeLight,'9E551A');
note(s,'Compose supports custom launcher commands and managed image recipes; KubeAI does not have full parity. External endpoints have no local serving lease and use access/run through the configured gateway. The CPU simulator can exercise the control path without real model intelligence. Supported configuration assumes one trusted control plane per Docker host or Kubernetes namespace.',src('docs/catalog.md','docs/kubeai-backend.md','docs/backend-parity.md','docs/mock-endpoints.md','docs/planning/known-limitations.md'));
}
{
const s=slide('The boundary is the key to understanding it.','Infer-stack manages inference infrastructure, not the model’s intelligence.');
card(s,.55,2.21,5.97,3.36,'It is','A declarative catalog, resource lifecycle manager, and consistent serving interface. Includes discovery, operations and benchmarking.',C.blue);
card(s,6.81,2.21,5.97,3.36,'It is not','A new inference kernel, a training framework, a guarantee of model quality or fit, or a security boundary for mutually untrusted users.',C.orange);
txt(s,'Next: Why infer-stack?  ·  Using infer-stack  ·  Measuring inference  ·  Infer-stack live',.57,5.86,12.16,.52,{fontSize:18,bold:true,color:C.teal});
txt(s,'github.com/AIQ-Kitware/infer_stack',.57,6.48,12.15,.3,{fontSize:16,color:C.blue,hyperlink:{url:repo}});
note(s,'End with a mental model, not a performance pitch. Operators remain responsible for suitable engines, hardware, credentials and trust boundaries. The existing technical overview is for architecture depth; the other decks stand alone. Do not claim speedups just because infer-stack is present.',src('README.md','docs/planning/known-limitations.md','docs/catalog.md','docs/benchmark.md'));
}
await write();
};

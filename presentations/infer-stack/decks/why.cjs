const {make,C,src,repo,bench}=require('../layout.cjs');
const {ep,v,fmt}=require('../metrics.cjs');
module.exports=async function(){
const {slide,cover,note,card,code,strip,flow,step,txt,rect,line,write}=make('why','Why infer-stack?');
{
const s=cover('Adoption brief / 5–8 minutes','Your GPU should serve\nthe work you are doing.','One inference control layer for local research, agents and evaluations.','Declare an endpoint. Acquire it when needed. Release it when done.');
note(s,'Open with the pain of changing models or workloads throughout a day. This is not a new inference engine, a promise of faster kernels, or a hosted service. The audience is someone deciding whether to adopt the project.',src('README.md','docs/operations.md'));
}
{
const s=slide('A model download is not a serving workflow.','One GPU, three jobs. Every change has an operational cost.');
const tasks=[['09:00','Coding assistant','Small, fast endpoint'],['13:00','Long reasoning','Large context profile'],['17:00','Evaluation','Different model']];
tasks.forEach((a,i)=>{const x=.55+i*4.16;txt(s,a[0],x,2.27,3.91,.45,{fontSize:18,color:C.muted,bold:true});rect(s,x,2.84,3.91,1.15,C.white,C.soft);txt(s,a[1],x+.17,3,3.57,.46,{fontSize:24,bold:true});txt(s,a[2],x+.17,3.49,3.57,.32,{fontSize:15,color:C.muted});});
line(s,.55,4.24,12.78,4.24,C.orange,1.5);
const actions=['Find image + flags','Stop / restart / verify','Reconnect client + record'];
actions.forEach((a,i)=>{const x=.55+i*4.16;txt(s,a,x,4.52,3.91,.67,{fontSize:19,bold:true,color:'9E551A',align:'center'});});
strip(s,'The goal is to encode this work once, then let each job declare its demand.');
note(s,'These are representative manual tasks, not measured time savings. Ask the audience which steps are most painful. Infer-stack captures the serving configuration and reconciles actual resources.',src('README.md','docs/catalog.md','docs/operations.md'));
}
{
const s=slide('Use a model like a resource, not a pet server.','The endpoint name is stable; the deployment is managed behind it.');
flow(s,[['Discover','Choose a known profile'],['Acquire','Start or reuse'],['Use','One API contract'],['Release','End your demand']],2.55);
rect(s,.55,4.54,12.23,1.15,C.dark);txt(s,'infer-stack run --endpoint NAME -- your-command',.79,4.76,11.76,.68,{fontFace:'Liberation Mono',fontSize:23,color:C.white});
strip(s,'The command receives connection details; the wrapper releases its own lease.');
note(s,'A lease represents demand, not necessarily one exclusive GPU or process. Compatible deployments may be shared. Normal command completion and graceful cancellation clean up; hard kills and crashes need TTL/manual recovery. The two-hour soft TTL is the default for run.',src('docs/operations.md','infer_stack/leasing/controller.py'));
}
{
const s=slide('One interface, three reasons to adopt it.','Start from the workload, then pick a serving configuration.');
card(s,.55,2.19,3.9,3.6,'Local development','Use a small model today, a long-context model tomorrow. Browse and edit known configurations.',C.teal);
card(s,4.72,2.19,3.9,3.6,'Evaluation','Run each model under a scoped lease. Retain serving identity and compare results across runs.',C.blue);
card(s,8.88,2.19,3.9,3.6,'Shared GPUs','Reconcile compatible demand and reclaim idle deployments instead of manually juggling containers.',C.orange);
strip(s,'Same endpoint contract for scripts, CLI, terminal UI and supported front doors.');
note(s,'These are use cases, not performance guarantees. The supported operating model is one trusted control plane per Docker host or Kubernetes namespace. It is not isolation for mutually untrusted users.',src('README.md','docs/catalog.md','docs/operations.md','docs/planning/known-limitations.md'));
}
{
const s=slide('A real example: one 3090, two different winners.','Measured at a target of 160 Ki input tokens; one 512-token request per phase.');
const metrics=[['Fresh generation','Maya Long',v(ep.ml,163840,'fresh','stream_tps'),'HyperQwen Huge',v(ep.qh,163840,'fresh','stream_tps'),C.teal,'stream tokens/s'],['Extended-prefix request','HyperQwen Huge',v(ep.qh,163840,'extended-prefix','e2e_tps'),'Maya Long',v(ep.ml,163840,'extended-prefix','e2e_tps'),C.orange,'end-to-end tokens/s']];
metrics.forEach((a,i)=>{const x=.55+i*6.26;rect(s,x,2.22,5.97,3.65,C.white,C.soft);txt(s,a[0],x+.22,2.43,5.53,.46,{fontSize:23,bold:true});txt(s,fmt(a[2]),x+.22,3.04,3.1,1.0,{fontSize:53,bold:true,color:a[5]});txt(s,a[6],x+.24,4.05,5.45,.33,{fontSize:16,color:C.muted});txt(s,a[1],x+.24,4.46,5.47,.46,{fontSize:22,bold:true,color:a[5]});txt(s,`vs. ${fmt(a[4],i===0?1:2)} for ${a[3]}`,x+.24,5.03,5.47,.5,{fontSize:18,color:C.muted});});
strip(s,'A serving choice should match the request pattern, not a single tokens/s number.');
note(s,'All six tested endpoint configurations used the same RTX 3090 with observed PCIe Gen3 x2. This compares endpoint configurations, not identical model quality or an isolated engine effect. The full evidence deck explains the methodology and caveats.',bench);
}
{
const s=slide('Start locally. Keep the control contract.','Compose and KubeAI share the catalog and lease concepts.');
card(s,.55,2.25,5.97,3.26,'Workstation / Compose','GPU-aware placement on a single Docker host. vLLM, Ollama and compatible custom launchers.',C.teal);
card(s,6.81,2.25,5.97,3.26,'Cluster / KubeAI','Kubernetes scheduling and stock vLLM resource profiles. Shared catalog and lease workflow.',C.blue);
strip(s,'Backend parity is not complete; custom launchers and Ollama are Compose-only.',6.13,C.orangeLight,'9E551A');
note(s,'KubeAI requires a configured cluster; there is no transparent migration of a running process. The project is Linux-oriented, planning-stage software. Do not promise multitenant security or high availability.',src('docs/backend-parity.md','docs/cluster-setup.md','docs/kubeai-backend.md','docs/planning/known-limitations.md'));
}
{
const s=slide('Try the workflow before committing to it.','Begin with the terminal UI or a no-GPU simulator.');
code(s,"umask 077\npip install 'infer-stack[tui]'\ninfer-stack config init --backend compose\ninfer-stack config set ui false\ninfer-stack catalog suggest --simulator --apply\ninfer-stack tui",.55,2.2,12.23,2.65,19);
txt(s,'Next: the hands-on deck walks through acquisition, API calls, scoped jobs and cleanup.',.57,5.12,12.18,.68,{fontSize:22,bold:true});
txt(s,'github.com/AIQ-Kitware/infer_stack',.57,6.18,12.15,.42,{fontSize:18,color:C.teal,hyperlink:{url:repo}});
note(s,'Prerequisites: Linux, Python 3.10+, Docker Compose, trusted host, private config/data directories and restrictive umask. Use a fresh or deliberately prepared control plane. Disabling UI avoids default unauthenticated WebUI; protect published gateway ports and keys. Simulator exercises serving lifecycle only. The tutorial includes the complete workflow.',src('README.md','docs/mock-endpoints.md','docs/planning/known-limitations.md'));
}
await write();
};

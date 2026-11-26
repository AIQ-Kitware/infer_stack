const {make,C,src,repo}=require('../layout.cjs');
module.exports=async function(){
const {slide,cover,note,card,code,strip,flow,step,txt,rect,line,write}=make('how','Using infer-stack',C.blue);
{
const s=cover('Hands-on tutorial / 15–20 minutes','From zero to a\nworking endpoint.','Follow one model from setup to API request to release.','Linux + Docker Compose • real GPU or no-GPU simulator');
note(s,'Audience: someone who wants to use infer-stack. This deck is a self-contained tutorial for a trusted local Compose host, not Kubernetes installation. Run from the presentations/infer-stack directory if using the supplied demo_client.py.',src('README.md','docs/operations.md','docs/mock-endpoints.md'));
}
{
const s=slide('01 / Prepare a trusted local host','Python 3.10+, Docker and Compose. GPU mode also needs NVIDIA drivers and Container Toolkit.','Setup');
code(s,"umask 077\npip install 'infer-stack[tui]'\ninfer-stack config init --backend compose\ninfer-stack config set ui false\ninfer-stack paths",.55,2.24,12.23,2.92,20);
strip(s,'Use one control plane per Docker host. Keep generated credentials private.',6.12,C.orangeLight,'9E551A');
txt(s,"For a source checkout: pip install -e '.[tui]'",.57,5.42,12.15,.45,{fontSize:16,color:C.muted});
note(s,'Use a virtual environment. The source-checkout install command is pip install -e \'.[tui]\'. On existing installations do not reset settings or change config while workloads are live. Default WebUI is unauthenticated and Compose publishes host ports; this tutorial disables WebUI and assumes a trusted, firewalled host. Verify existing file permissions.',src('README.md','docs/compose.md','docs/planning/known-limitations.md'));
}
{
const s=slide('02 / Discover what can run','Choose one branch. Both produce a catalog endpoint you can acquire.','Discovery');
txt(s,'GPU: small real model',.57,2.24,5.95,.44,{fontSize:23,bold:true,color:C.teal});
code(s,'infer-stack catalog suggest --apply\ninfer-stack catalog endpoint list\nENDPOINT=smollm2-135m',.55,2.83,5.97,1.78,15);
txt(s,'CPU: exercise the workflow',6.83,2.24,5.95,.44,{fontSize:23,bold:true,color:C.blue});
code(s,'infer-stack catalog suggest \\\n  --simulator --apply\ninfer-stack catalog endpoint list\nENDPOINT=mock-smol',6.81,2.83,5.97,1.78,15);
txt(s,'Use the exact endpoint name from your catalog. If the small GPU model is absent, choose another compatible chat endpoint.',.57,4.95,12.1,.66,{fontSize:19,color:C.muted});
strip(s,'Simulator output is mock text, not model intelligence or real performance.');
note(s,'Suggestions are hardware-aware and additive; they are not fit guarantees. The no-GPU simulator needs Docker and its image, not GPU drivers or model weights. Do not interpret the simulator response as a real model result.',src('README.md','docs/model-profiles.md','docs/mock-endpoints.md'));
}
{
const s=slide('03 / Acquire and verify','Acquire waits for a generation probe, not merely a running container.','Lifecycle');
code(s,'infer-stack acquire "$ENDPOINT" \\\n  --ttl 2h --env-file lease.env\ninfer-stack test "$ENDPOINT"\ninfer-stack leases\ninfer-stack ps',.55,2.22,7.07,3.02,19);
step(s,'A','First start','May pull or build an image and download weights.',7.91,2.24,4.85,1.52);
step(s,'B','Ready','The gateway can serve the endpoint after verification.',7.91,3.87,4.85,1.52);
strip(s,'Review deployment changes when prompted. Never commit lease.env.');
note(s,'Acquire creates a named lease and writes a trusted local shell environment file. Choose a longer timeout if necessary. The two-hour TTL here is explicit. The gateway is configured in the default Compose path. If a command fails, inspect engine logs before trying changes.',src('README.md','docs/operations.md','infer_stack/cli/commands_leasing.py'));
}
{
const s=slide('04 / Send an ordinary API request','No engine-specific client code is required for this chat request.','Client');
code(s,'(\n  source lease.env\n  python demo_client.py "Say hello."\n)',.55,2.22,7.05,2.25,22);
card(s,7.9,2.22,4.88,2.25,'Connection','OPENAI_BASE_URL\nOPENAI_API_KEY\nINFER_STACK_MODELS',C.blue);
txt(s,'The supplied Python script uses the standard library to POST /chat/completions. A single endpoint name becomes the model ID.',.57,4.84,12.17,.76,{fontSize:21,color:C.muted});
strip(s,'Run from presentations/infer-stack/; the script is included with the slides.');
note(s,'The script uses only the standard library and requires exactly one selected chat endpoint. It does not print credentials. The parentheses prevent exported connection variables from leaking into the parent shell. For a different working directory use the absolute script path. This is a real request for a real endpoint and synthetic text for a simulator.',src('README.md','infer_stack/leasing/envfile.py','presentations/infer-stack/demo_client.py'));
}
{
const s=slide('05 / End the lease, not the whole stack','Only release the demand you created.','Cleanup');
code(s,'infer-stack release --env-file lease.env &&\n  rm -- lease.env\ninfer-stack leases',.55,2.25,12.23,1.85,22);
flow(s,[['Lease ends','Your claim is removed'],['Reconcile','Other demand may remain'],['Reclaim','Stop or keep warm by policy']],4.47);
strip(s,'Model weights remain cached on disk; the gateway can serve the next endpoint.');
note(s,'Remove the env file only after successful release. The slide commands are sequential for the presenter; if release fails, retain the file and diagnose. Suggested endpoints stop at last release; keep-warm policy can change this. Do not use release --all as routine cleanup.',src('README.md','docs/operations.md','docs/persistent-caches-and-warm-restarts.md'));
}
{
const s=slide('06 / Make the lease a job dependency','The child process receives the same API environment automatically.','Automation');
code(s,'infer-stack run --endpoint "$ENDPOINT" \\\n  --ttl 2h -- \\\n  python demo_client.py "Explain recursion."',.55,2.28,12.23,1.97,22);
flow(s,[['Acquire','Wait for readiness'],['Execute','Pass environment to child'],['Release','Clean up this job’s lease']],4.45);
strip(s,'Ctrl-C / SIGTERM cleanup; TTL and explicit release for hard kills or crashes.');
note(s,'Use run for evaluations and agents rather than keeping manual leases open. Run has a default soft TTL of two hours; select an appropriate TTL for long jobs. The child can also read INFER_STACK_ENDPOINT_DESCRIPTOR for configured serving provenance. The command after -- is forwarded unchanged.',src('docs/operations.md','infer_stack/cli/commands_leasing.py','presentations/infer-stack/demo_client.py'));
}
{
const s=slide('07 / Explore and troubleshoot interactively','The TUI and CLI operate on the same catalog and lifecycle.','Day two');
card(s,.55,2.19,5.97,3.39,'Discover / edit','Open infer-stack tui. Browse models (i), inspect fit warnings, import & edit, acquire.',C.blue);
card(s,6.81,2.19,5.97,3.39,'Inspect / diagnose','See active endpoints and GPU placement. Follow only the engine logs, without gateway or UI noise.',C.teal);
code(s,'infer-stack tui\ninfer-stack logs --engine -f',.55,5.87,12.23,.87,17);
note(s,'Show the actual running TUI rather than a fabricated screenshot. Browse is broader than hardware-fit suggestions; import does not bypass placement constraints. Engine-only follow works with dynamically appearing instances. The TUI requires the tui extra. See the model profiles and operations guides.',src('docs/model-profiles.md','docs/operations.md','infer_stack/tui.py'));
}
{
const s=slide('You now have a reusable serving workflow.','Replace the endpoint and keep the application code.','Next steps');
step(s,'01','Change models','List catalog endpoints; import a known profile or tune one in the TUI.',.56,2.25,12.1,1.15);
step(s,'02','Automate','Use run for one evaluation command or agent session.',.56,3.52,12.1,1.15);
step(s,'03','Compare','Use benchmark run NAME1 NAME2 and inspect the live report.',.56,4.8,12.1,1.15);
txt(s,'Documentation: github.com/AIQ-Kitware/infer_stack',.57,6.32,12.1,.39,{fontSize:18,color:C.blue,hyperlink:{url:repo}});
note(s,'The benchmark currently targets local Compose with the normal LiteLLM gateway. Do not run a model performance benchmark on simulator outputs. For a cluster, consult the separate KubeAI setup instructions. The demo deck goes beyond this tutorial.',src('README.md','docs/index.md','docs/benchmark.md','docs/kubeai-backend.md'));
}
await write();
};

const {make,C,src,bench,repo}=require('../layout.cjs');
const {ep,v,fmt}=require('../metrics.cjs');
module.exports=async function(){
const {slide,cover,note,card,code,strip,flow,step,txt,rect,line,write}=make('demo','Infer-stack live',C.teal);
{
const s=cover('Live demo / 15–25 minutes','What can you do\nwith inference on demand?','A sequence of small demonstrations, not a walkthrough of every command.','Browse → share → automate → place → measure');
note(s,'Use the companion DEMO.md runbook. Prepare a trusted local Compose installation; keep a CPU simulator available, pre-pull or pre-warm real models, and open the saved benchmark report as a fallback. This is a demo of control and observation, not a speed contest.',src('README.md','docs/operations.md','docs/benchmark.md','presentations/infer-stack/DEMO.md'));
}
{
const s=slide('Demo 1 / Browse models beyond your hardware','Show the difference between a recommended recipe and an experimental starting point.','Discover');
code(s,'infer-stack catalog suggest\ninfer-stack catalog suggest \\\n  --simulate-hardware 4x96@12.0\ninfer-stack tui',.55,2.22,7.05,2.6,19);
card(s,7.9,2.22,4.88,3.39,'What to show','Browse models (i) → filter → preview YAML → Import & edit. Look for fit warnings and placement fields.',C.teal);
txt(s,'Offline simulation lets you plan for a different GPU class without pretending to own those GPUs.',.57,5.13,7.0,.83,{fontSize:20,color:C.muted,valign:'top'});
strip(s,'Import does not bypass actual acquisition constraints.');
note(s,'Do not acquire an incompatible DeepSeek profile. Simulated hardware is only planning. In Browse, select a known profile and inspect context and GPU requirements. If demonstrating edit, choose a profile not already in use and avoid redefining live endpoints. Show actual UI; no mock screenshot.',src('docs/model-profiles.md','docs/deepseek-v4-flash-sm120.md','infer_stack/tui.py'));
}
{
const s=slide('Demo 2 / Two clients, one deployment','Take two independent leases for the same shareable endpoint.','Share');
code(s,'infer-stack acquire "$ENDPOINT" \\\n  --ttl 30m --env-file a.env\ninfer-stack acquire "$ENDPOINT" \\\n  --ttl 30m --env-file b.env\ninfer-stack leases\ninfer-stack ps',.55,2.21,7.02,3.08,18);
rect(s,7.89,2.23,2.18,.86,C.blueLight);txt(s,'Lease A',8.06,2.43,1.84,.4,{fontSize:21,bold:true,color:C.blue,align:'center'});rect(s,10.6,2.23,2.18,.86,C.tealLight);txt(s,'Lease B',10.77,2.43,1.84,.4,{fontSize:21,bold:true,color:C.teal,align:'center'});line(s,8.98,3.12,9.95,3.83,C.blue,2,true);line(s,11.69,3.12,10.72,3.83,C.teal,2,true);rect(s,8.2,3.91,4.3,1.15,C.dark);txt(s,'Compatible deployment',8.4,4.23,3.9,.5,{fontSize:19,bold:true,color:C.white,align:'center'});
strip(s,'Release A; B should remain usable. Then release B.');
note(s,'Use a default shareable managed endpoint, such as mock-smol, on a clean or preflighted demo host. Check actual leases and deployment IDs; compatible sharing depends on policy and configuration. Release A with infer-stack release --env-file a.env, then run (source b.env; python demo_client.py "Still there?"). Release B and inspect ps. Remove files only after successful release. The graphic illustrates the expected result, not a recorded screenshot.',src('docs/operations.md','infer_stack/leasing/controller.py','presentations/infer-stack/DEMO.md'));
}
{
const s=slide('Demo 3 / The workload controls the lease','Use the same Python client with a simulator or a real model.','Automate');
code(s,'infer-stack run --endpoint mock-smol -- \\\n  python demo_client.py "What is recursion?"\n\ninfer-stack run --endpoint smollm2-135m -- \\\n  python demo_client.py "What is recursion?"',.55,2.2,12.23,2.6,19);
flow(s,[['Same client','No API rewiring'],['Different runtime','CPU mock or GPU'],['Automatic release','Own lease only']],5.02);
note(s,'Both endpoints must be in the catalog; choose another compatible real chat endpoint if smollm2-135m is unavailable. This compares the lifecycle, not model answers or performance. The first command can work on a host with no GPU; the second needs compatible hardware and weights. Show ps/leases before and after. Run defaults to a two-hour TTL backstop; select a longer TTL for long jobs.',src('docs/mock-endpoints.md','docs/operations.md','presentations/infer-stack/demo_client.py'));
}
{
const s=slide('Demo 4 / Leave another GPU free','Constrain placement per acquisition, or pin it durably in the catalog.','Placement');
code(s,'infer-stack acquire "$REAL_ENDPOINT" \\\n  --allowed-gpus 1 --ttl 30m \\\n  --env-file gpu.env\ninfer-stack ps\ninfer-stack logs --engine -f',.55,2.24,7.05,2.78,18);
rect(s,7.89,2.25,4.89,1.30,C.white,C.soft);txt(s,'GPU 0',8.1,2.45,2,.45,{fontSize:24,bold:true});txt(s,'Not eligible for this call',10.14,2.49,2.38,.68,{fontSize:15,color:C.muted,align:'right'});rect(s,7.89,3.73,4.89,1.30,C.tealLight);txt(s,'GPU 1',8.1,3.96,2,.45,{fontSize:24,bold:true,color:C.teal});txt(s,'Eligible placement',10.14,4.01,2.38,.57,{fontSize:15,color:C.teal,align:'right'});
txt(s,'Validate actual GPU indices and available memory first. Use a one-GPU endpoint.',.57,5.36,12.13,.6,{fontSize:19,color:C.muted});
strip(s,'Scope the call, not every deployment on the host.');
note(s,'Optional hardware demo: use a compatible one-GPU endpoint and a host with multiple eligible GPUs. Ensure no incompatible existing deployment/lease changes the intended example. This option filters the inventory for the call only; it is not an exclusive reservation of the other GPU. Show nvidia-smi and ps, follow logs in a separate terminal, then release gpu.env. Do not disturb unrelated workloads.',src('docs/catalog.md','docs/operations.md','docs/benchmark.md'));
}
{
const s=slide('Demo 5 / Validate failure handling without a GPU','A simulator is useful for testing the system around the model.','Test infrastructure');
card(s,.55,2.2,5.97,3.35,'API fidelity','mock-echo returns the input. mock-flaky injects provider errors to exercise retries.',C.blue);
card(s,6.81,2.2,5.97,3.35,'Evaluation plumbing','An oracle mock uses an answer-key fixture and configured ability to test metric calculations.',C.orange);
code(s,'infer-stack run --catalog dev/e2e_tests/catalog-mock.yaml \\\n  --backend compose --endpoint mock-echo -- \\\n  python presentations/infer-stack/demo_client.py "Echo this."',.55,5.65,12.23,1.08,14);
note(s,'Optional source-checkout demo, run from repo root with the supplied client path. Use the same configured control plane and an otherwise quiescent demo stack; explicit catalog can alter published configuration. Pre-pull the llm-d simulator. The oracle mock requires its own image build and catalog. Its answers are synthetic, not a real evaluation result. The runbook has exact echo commands and a safe discussion-only fallback for flaky/oracle.',src('docs/mock-endpoints.md','dev/e2e_tests/catalog-mock.yaml','dev/e2e_tests/catalog-mock-oracle.yaml'));
}
{
const s=slide('Demo 6 / Watch measurements arrive live','Start a short sweep and open the generated HTML report.','Measure');
code(s,'infer-stack benchmark run "$REAL_ENDPOINT" \\\n  --contexts 1024,8192 \\\n  --generation-tokens 128',.55,2.2,12.23,1.66,21);
flow(s,[['Acquire','Ready probe'],['Measure','Fresh + reuse'],['Report','Update each result'],['Release','Move to next']],4.22);
strip(s,'No need to wait for the full sweep to inspect the first results.');
note(s,'Use a real endpoint with sufficient configured context, not the simulator. Pick two compatible endpoints for a serial comparison if time allows. Open the report directory printed at startup. Use a prepared report from a prior run if time or GPU availability is limited; explain it is historical. Startup and download can take substantial time. The benchmark skips already-shared deployments and preserves other leases.',src('docs/benchmark.md','infer_stack/benchmark/dashboard.py'));
}
{
const s=slide('Turn the demo into a measurable improvement.','The same infrastructure makes it possible to compare profiles and iterate on them.','Close');
card(s,.55,2.23,5.97,3.35,'Long reasoning',`Maya Long: ${fmt(v(ep.ml,163840,'fresh','stream_tps'))} fresh stream tokens/s at 160K on the measured RTX 3090.`,C.teal);
card(s,6.81,2.23,5.97,3.35,'Frequent follow-ups',`HyperQwen Huge: ${fmt(v(ep.qh,163840,'extended-prefix','ttft_s'),2)} s extended-prefix TTFT at the same target.`,C.orange);
strip(s,'Next demo: tune a profile, rerun the same benchmark, and compare the ledger.');
note(s,'Historical single-trial evidence, same physical GPU (observed Gen3 x2). The two headline values are deliberately different metrics, not a direct speed comparison. See the separate evidence deck for comparable numbers and caveats. This final slide connects the practical demos to the decision loop; do not claim task quality from throughput.',bench);
}
await write();
};

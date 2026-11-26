// Historical benchmark snapshot shared by audience decks.
const rows=require('./data/benchmark-2026-11-25.json');
const contexts=[1024,8192,32768,65536,131072,163840];
const labels=['1K','8K','32K','64K','128K','160K'];
const ep={q:'qwen3.8-27b-dbirks-hyperqwen',ql:'qwen3.8-27b-dbirks-hyperqwen-long',qh:'qwen3.8-27b-dbirks-hyperqwen-huge',ml:'inkling-2-mini-maya-long'};
function v(id,n,phase,key){return rows.find(r=>r.endpoint===id&&r.target===n&&r.phase===phase&&r.ok===1)?.[key]??null;}
const fmt=(n,d=1)=>n==null?'—':n.toFixed(d);
module.exports={rows,contexts,labels,ep,v,fmt};

// Shared presentation design. No dependencies beyond the local PptxGenJS.
const PptxGenJS = require('pptxgenjs');
const fs = require('node:fs');
const path = require('node:path');
const W = 40 / 3, H = 7.5;
const C = { bg:'F7F8F6', white:'FFFFFF', ink:'172B38', muted:'536675', soft:'DCE5E5', teal:'087E80', tealLight:'D7EFEB', blue:'3569A7', blueLight:'E4ECF8', orange:'D17A29', orangeLight:'F7EBD9', dark:'102834', dark2:'193C49', cyan:'80D9CB', grey:'8C9BA5', light:'D2E4E8' };
const F='Aptos', M='Liberation Mono';
const repo='https://github.com/AIQ-Kitware/infer_stack';
const src=(...paths)=>paths.map(p=>`infer-stack repository / ${p} (Nov 25, 2026 supplied snapshot and subsequent overlays; ${repo})`);
const bench=['User-provided Nov 25, 2026 summary.csv, report.md and configurations.json; sanitized snapshot in presentations/infer-stack/data/benchmark-2026-11-25.json.'];
function make(slug, label, accent=C.teal) {
    const p=new PptxGenJS(); p.layout='LAYOUT_WIDE'; p.author='Infer-stack'; p.title=`infer-stack | ${label}`; p.subject=label; p.lang='en-US';
    const S=p.ShapeType;
    function txt(s,t,x,y,w,h,o={}) { s.addText(t,{x,y,w,h,fontFace:F,fontSize:20,color:C.ink,margin:0,valign:'mid',breakLine:false,...o}); }
    function rect(s,x,y,w,h,fill=C.white,border=null) { s.addShape(S.rect,{x,y,w,h,fill:{color:fill},line:{color:border||fill,transparency:border?0:100,width:.8}}); }
    function line(s,x1,y1,x2,y2,color=C.soft,width=1,arrow=false) { s.addShape(S.line,{x:x1,y:y1,w:x2-x1,h:y2-y1,line:{color,width,endArrowType:arrow?'triangle':undefined}}); }
    function dot(s,x,y,d,color) { s.addShape(S.ellipse,{x,y,w:d,h:d,fill:{color},line:{color,transparency:100}}); }
    function footer(s,dark=false) { const color=dark?C.light:C.muted; line(s,.55,7.03,12.78,7.03,dark?'3C5A65':C.soft,.7); txt(s,`infer-stack  /  ${label}`,.55,7.12,11.1,.19,{fontSize:10,color}); txt(s,String(p._slides.length).padStart(2,'0'),12.05,7.12,.73,.19,{fontSize:10,color,align:'right'}); }
    function slide(title,sub='',k=label) { const s=p.addSlide(); s.background={color:C.bg}; txt(s,k.toUpperCase(),.55,.29,12.2,.3,{fontSize:11,bold:true,color:accent,charSpacing:1.5}); txt(s,title,.55,.75,12.25,.75,{fontSize:32,bold:true}); if(sub)txt(s,sub,.57,1.56,12.2,.56,{fontSize:17,color:C.muted,valign:'top'}); footer(s); return s; }
    function cover(k,title,sub,callout) { const s=p.addSlide(); s.background={color:C.dark}; rect(s,0,0,.16,H,C.cyan); txt(s,`INFER-STACK  /  ${k.toUpperCase()}`,.65,.54,12,.31,{fontSize:12,bold:true,color:C.cyan,charSpacing:2}); txt(s,title,.64,1.33,11.95,2.12,{fontSize:49,bold:true,color:C.white,valign:'mid'}); txt(s,sub,.67,3.67,11.75,1.05,{fontSize:25,color:C.light,valign:'top'}); rect(s,.66,5.3,11.98,.92,C.dark2); txt(s,callout,.9,5.44,11.5,.64,{fontSize:20,bold:true,color:C.cyan}); footer(s,true); return s; }
    function note(s,body,sources) { s.addNotes(`${body}\n\n[Sources]\n${sources.map(x=>'- '+x).join('\n')}\n[/Sources]`); }
    function card(s,x,y,w,h,title,body,color=accent) { rect(s,x,y,w,h,C.white,C.soft); rect(s,x,y,.065,h,color); txt(s,title,x+.23,y+.20,w-.46,.64,{fontSize:24,bold:true,valign:'top'}); txt(s,body,x+.23,y+.95,w-.46,h-1.12,{fontSize:19,color:C.muted,valign:'top'}); }
    function code(s,t,x,y,w,h,fs=18) { rect(s,x,y,w,h,C.dark); txt(s,t,x+.22,y+.18,w-.44,h-.36,{fontFace:M,fontSize:fs,color:'E2F4F0',valign:'top'}); }
    function strip(s,t,y=6.13,fill=C.tealLight,fg=C.teal) { rect(s,.55,y,12.23,.66,fill); txt(s,t,.77,y+.06,11.78,.54,{fontSize:17,bold:true,color:fg}); }
    function flow(s,items,y=2.6) { const gap=.22,w=(12.23-gap*(items.length-1))/items.length; items.forEach(([t,b],i)=>{const x=.55+i*(w+gap); rect(s,x,y,w,1.55,C.white,C.soft); txt(s,t,x+.12,y+.17,w-.24,.43,{fontSize:21,bold:true,color:accent,align:'center'}); txt(s,b,x+.13,y+.67,w-.26,.71,{fontSize:15,color:C.muted,align:'center',valign:'top'}); if(i<items.length-1)line(s,x+w+.015,y+.78,x+w+gap-.015,y+.78,accent,1.5,true); }); }
    function step(s,num,title,body,x,y,w,h) { txt(s,num,x,y,.55,.5,{fontSize:25,bold:true,color:accent}); txt(s,title,x+.68,y,w-.68,.5,{fontSize:23,bold:true}); txt(s,body,x+.68,y+.57,w-.68,h-.62,{fontSize:18,color:C.muted,valign:'top'}); }
    async function write() { const output=path.join(__dirname,'build',`infer-stack-${slug}.pptx`); fs.mkdirSync(path.dirname(output),{recursive:true}); if(process.env.INFER_STACK_SLIDE_QA_HELPERS){const q=require(path.resolve(process.env.INFER_STACK_SLIDE_QA_HELPERS)); for(const s of p._slides){q.warnIfSlideHasOverlaps(s,p);q.warnIfSlideElementsOutOfBounds(s,p);}} await p.writeFile({fileName:output}); console.log(`Wrote ${output} (${p._slides.length} slides)`); }
    return {p,txt,rect,line,dot,slide,cover,note,card,code,strip,flow,step,write};
}
module.exports={make,C,W,H,F,M,repo,src,bench};

/* Headless numerical/UI-handler smoke check; Canvas layout is inspected separately.
   Run: node examples/check_secondary_visualization.js */
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const dir=path.join(__dirname,'../results/secondary_visualization');
const html=fs.readFileSync(path.join(dir,'index.html'),'utf8');
const samples=JSON.parse(fs.readFileSync(path.join(dir,'field_samples.json'),'utf8'));
const data=html.match(/<script id="mode-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes={};
function context(){
  return {createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),
          putImageData(){},clearRect(){},drawImage(){},strokeRect(){},fillText(){}};
}
function element(id){
  return nodes[id]||(nodes[id]={value:'',checked:false,textContent:'',width:1000,height:200,
    listeners:{},addEventListener(k,fn){this.listeners[k]=fn;},getContext:context});
}
element('mode-data').textContent=data;
element('amplitude').value='0.06';element('time').value='0';element('height').value='3';
element('comoving').checked=true;element('growth').checked=true;
const queue=[];
const sandbox={document:{getElementById:element,createElement:()=>element('raster'+Object.keys(nodes).length)},
  window:{},requestAnimationFrame:fn=>queue.push(fn),performance:{now:()=>100},console};
vm.createContext(sandbox);vm.runInContext(script,sandbox);
const app=sandbox.window.secondaryVisualization;
let error=0;
for(const c of samples){
  element('height').value=String(c.z);element('time').value=String(c.t);element('amplitude').value=String(c.a);
  element('comoving').checked=c.moving;element('growth').checked=c.growth;
  app.render();
  for(const v of [app.primary,app.secondary,app.combined])assert(Array.from(v).every(Number.isFinite));
  c.indices.forEach((k,i)=>{
    error=Math.max(error,Math.abs(app.primary[k]-c.primary[i]),Math.abs(app.secondary[k]-c.secondary[i]));
    assert(Math.abs(app.combined[k]-app.primary[k]-app.secondary[k])<1e-12);
  });
}
assert(error<1e-11,'Python/JavaScript reconstruction mismatch: '+error);
element('primary-only').listeners.click();
assert(Array.from(app.secondary).every(v=>v===0));
assert(Array.from(app.combined).every((v,i)=>v===app.primary[i]));
element('illustrate').listeners.click();assert(element('amplitude').value==='0.25');
element('reset').listeners.click();assert(element('time').value==='0');
element('play').listeners.click();assert(queue.length);
queue.shift()(200);assert(Number(element('time').value)>0);
element('play').listeners.click();assert(element('play').textContent==='播放发展过程');
assert(!/__MODE_DATA__|__PHASE_FIGURE__/.test(html));
assert(!/<(?:script|img)[^>]+src=["']https?:/.test(html));
const result={python_javascript_max_abs_error:error,point_comparisons:samples.length*samples[0].indices.length*2,
              primary_only_exact:true,finite_fields:true,controls_and_playback_handlers:true,offline_assets:true,
              scope:'Numerical and event-handler checks using a DOM/Canvas shim, not a real browser layout test'};
fs.writeFileSync(path.join(dir,'visualization_checks.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));

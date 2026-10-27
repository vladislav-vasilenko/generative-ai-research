/** Strict KaTeX validation of authored VAE notebook formulas; optional external tool dependency. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const course=path.resolve(__dirname,'..');
const option=process.argv.find(a=>a.startsWith('--katex-path='));
let katex;
try{katex=require(option?path.resolve(option.slice('--katex-path='.length)):'katex');}
catch(e){console.error('Provide --katex-path=/ABS/node_modules/katex (optional renderer, not a Python lab dependency).');process.exit(2);}
const report={renderer:'KaTeX '+katex.version,strict:true,scope:'Formula parse/render only; no GitHub/Colab browser parity claim',files:[],errors:[]};
for(const name of fs.readdirSync(path.join(course,'notebooks')).filter(n=>n.endsWith('.ipynb')).sort()){
 const data=fs.readFileSync(path.join(course,'notebooks',name));const nb=JSON.parse(data);let formulas=0;
 nb.cells.forEach((cell,index)=>{
  if(cell.cell_type!=='markdown')return;
  const source=Array.isArray(cell.source)?cell.source.join(''):cell.source;
  for(const m of source.matchAll(/\$\$([\s\S]*?)\$\$/g)){
   formulas++;try{katex.renderToString(m[1].trim(),{displayMode:true,throwOnError:true,strict:'error'});}
   catch(e){report.errors.push({file:name,cell:index,error:e.message});}
  }
 });
 report.files.push({notebook:name,source_sha256:crypto.createHash('sha256').update(data).digest('hex'),formulas});
}
report.total_formulas=report.files.reduce((n,f)=>n+f.formulas,0);report.passed=!report.errors.length;
fs.writeFileSync(path.join(course,'reports/math-rendering.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:report.passed,total_formulas:report.total_formulas,errors:report.errors},null,2));
if(!report.passed)process.exit(1);

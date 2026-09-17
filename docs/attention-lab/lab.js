'use strict';
const $ = id => document.getElementById(id);
const slides = [...document.querySelectorAll('.slide')];
let current = 0, reading = false;
document.body.classList.add('js');
slides.forEach((slide, i) => {
  const option = document.createElement('option');
  option.value = i;
  option.textContent = `${i + 1}. ${slide.querySelector('h2').textContent}`;
  $('slide-picker').append(option);
});
function showSlide(index) {
  current = Math.min(slides.length - 1, Math.max(0, index));
  slides.forEach((s, i) => { s.hidden = !reading && i !== current; });
  $('slide-picker').value = current;
  $('previous').disabled = current === 0;
  $('next').disabled = current === slides.length - 1;
  $('slide-count').textContent = `${current + 1} / ${slides.length}`;
  $('slide-status').textContent = reading ? `All ${slides.length} slides are visible.` : `${current + 1} / ${slides.length} · ${slides[current].querySelector('h2').textContent}`;
}
function setReading(value) {
  reading = value;
  document.body.classList.toggle('read-all', reading);
  $('read-all').setAttribute('aria-pressed', reading);
  $('read-all').textContent = reading ? 'One slide' : 'Read all';
  showSlide(current);
}
function present(value) {
  if (value) setReading(false);
  document.body.classList.toggle('presentation', value);
  $('present').setAttribute('aria-pressed', value);
  $('present').textContent = value ? 'Exit' : 'Present';
  $('deck').scrollIntoView();
}
$('previous').onclick = () => showSlide(current - 1);
$('next').onclick = () => showSlide(current + 1);
$('slide-picker').onchange = e => { showSlide(Number(e.target.value)); if (reading) slides[current].scrollIntoView(); };
$('read-all').onclick = () => setReading(!reading);
$('present').onclick = () => present(!document.body.classList.contains('presentation'));
$('print').onclick = () => window.print();
let openDetails = [];
window.addEventListener('beforeprint', () => {
  openDetails = slides.map(s => s.querySelector('details').open);
  slides.forEach(s => { s.querySelector('details').open = true; });
});
window.addEventListener('afterprint', () => slides.forEach((s,i) => { s.querySelector('details').open = openDetails[i] || false; }));
document.addEventListener('keydown', e => {
  if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
  if (e.key === 'Escape' && document.body.classList.contains('presentation')) present(false);
  const bounds = $('deck').getBoundingClientRect();
  if (!reading && bounds.top < innerHeight * .6 && bounds.bottom > innerHeight * .4) {
    if (e.key === 'ArrowRight') { e.preventDefault(); showSlide(current + 1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); showSlide(current - 1); }
  }
});
function handleHash() {
  const found = location.hash.match(/^#slide-(\d+)$/);
  if (found) { showSlide(Number(found[1])-1); slides[current].scrollIntoView(); }
}
showSlide(0); handleHash(); window.addEventListener('hashchange', handleHash);
const NS='http://www.w3.org/2000/svg';
function svgEl(tag,attrs,text) {
  const el=document.createElementNS(NS,tag);
  for(const [key,value] of Object.entries(attrs)) el.setAttribute(key,value);
  if(text !== undefined) el.textContent=text;
  return el;
}
function drawMask() {
  const mode=$('mask-mode').value, svg=$('mask-svg'); svg.replaceChildren();
  const n=mode==='square'?64:12, size=300/n, x0=135,y0=40;
  let count=0;
  const doc=i=>i<4?0:i<7?1:2;
  for(let q=0;q<n;q++) for(let k=0;k<n;k++) {
    let allowed;
    if(mode==='square') allowed=true;
    else if(mode==='state') allowed=k===q;
    else if(mode==='sparse') allowed=k<=q && (q-k<3 || k%4===0);
    else allowed=doc(q)===doc(k) && k<=q && (mode!=='window'||q-k<3);
    if(allowed) count++;
    const r=svgEl('rect',{x:x0+k*size+.4,y:y0+q*size+.4,width:size-.8,height:size-.8,class:allowed?'allowed':'masked'});
    r.append(svgEl('title',{},`Query ${q}, key ${k}: ${allowed?'allowed':'masked'}`));svg.append(r);
  }
  svg.append(svgEl('text',{x:285,y:22,'text-anchor':'middle','font-size':15},mode==='square'?'Key square (0–63)':'Key position →'));
  svg.append(svgEl('text',{x:26,y:194,'font-size':14},'Query ↓'));
  if(mode!=='square') for(let i=0;i<n;i++) {
    svg.append(svgEl('text',{x:x0+(i+.5)*size,y:360,'font-size':12,'text-anchor':'middle'},i));
    svg.append(svgEl('text',{x:125,y:y0+(i+.7)*size,'font-size':12,'text-anchor':'end'},i));
  }
  if(['causal','window','state'].includes(mode)) {
    for(const [start,len,label] of [[0,4,'A'],[4,3,'B'],[7,5,'C']]) {
      svg.append(svgEl('rect',{x:x0+start*size,y:y0+start*size,width:len*size,height:len*size,class:'boundary'}));
      svg.append(svgEl('text',{x:455,y:y0+(start+len/2)*size,'font-size':14},`Game ${label}`));
    }
  }
  const captions={
    causal:'Only the lower triangle within each game is visible. No game can use another game’s tokens.',
    window:'Each query can see itself and up to two previous positions in its game. Across layers, information can travel further through cached hidden states.',
    state:'The diagonal remains: every position only sees itself. Its attention weight is 1. This requires training for the changed input contract.',
    square:'All square pairs communicate within one board, including distant files and diagonals. There is no causal order among squares.',
    sparse:'One toy 12-token sequence: each query reads the last three tokens plus causally visible entries 0, 4 and 8. No learned indexer or compression is simulated.'
  };
  $('mask-caption').textContent=`${count.toLocaleString()} / ${(n*n).toLocaleString()} permitted pairs. ${captions[mode]}`;
}
$('mask-mode').onchange=drawMask;drawMask();
function scoreUpdate() {
  const match=Number($('match').value), bias=Number($('bias').value), self=$('only-self').checked;
  $('match-value').textContent=match.toFixed(1); $('bias-value').textContent=bias.toFixed(1);
  const logits=[match,.3,-.8,.5+bias], vals=[2,-1,.5,1], names=['Older position','Middle position','Previous position','Current position'];
  const exps=self?[0,0,0,1]:logits.map(x=>Math.exp(x-Math.max(...logits)));
  const total=exps.reduce((a,b)=>a+b,0), weights=exps.map(x=>x/total);
  $('score-bars').replaceChildren();
  names.forEach((name,i)=>{
    const row=document.createElement('div');row.className='bar-row';
    row.innerHTML=`<div class="bar-label"><span>${name}</span><strong>${(100*weights[i]).toFixed(1)}%</strong></div><div class="bar-track"><div class="bar-fill" style="width:${100*weights[i]}%"></div></div>`;
    $('score-bars').append(row);
  });
  const value=weights.reduce((s,w,i)=>s+w*vals[i],0);
  $('score-result').textContent=`Weighted scalar feature: ${value.toFixed(3)}. Values: [2, −1, 0.5, 1]. ${self?'One key: Q/K scores cannot change the mixture.':'A score change redistributes probability across all allowed keys.'}`;
}
['match','bias','only-self'].forEach(id=>$(id).addEventListener('input',scoreUpdate));
$('reset-scores').onclick=()=>{$('match').value=1;$('bias').value=0;$('only-self').checked=false;scoreUpdate();};scoreUpdate();
function bytesLabel(bytes) {
  if(bytes===0) return '0 bytes';
  const units=['bytes','KiB','MiB','GiB','TiB'];let power=0;
  while(bytes>=1024 && power<units.length-1){bytes/=1024;power++;}
  return `${bytes.toLocaleString(undefined,{maximumFractionDigits:2})} ${units[power]}`;
}
function memoryUpdate() {
  let valid=true;
  for(const id of ['history','games','nodes','packed']) {
    const input=$(id),n=Number(input.value),ok=input.value!==''&&Number.isInteger(n)&&n>=Number(input.min)&&n<=Number(input.max);
    input.setAttribute('aria-invalid',!ok);valid=valid&&ok;
  }
  $('memory-error').textContent=valid?'':'Enter whole numbers within the indicated bounds: prefix 1–512, games 1–256, entries 0–10,000, packed tokens 1–40,960.';
  if(!valid){$('memory-results').textContent='Estimates paused until the inputs are valid.';return;}
  const H=+$('model-size').value,b=+$('precision').value,T=+$('history').value,G=+$('games').value,R=+$('nodes').value,S=+$('packed').value,L=8,d=64;
  const C=L*H*(d+d)*b, entries=G*(T+R),base=entries*C,dense=H*S*S*b;
  const rows=[
    ['One token’s K/V across 8 layers',C,'Raw cache, both K and V'],
    ['Shared prefixes across games',G*T*C,`${G} games × ${T} positions`],
    ['Unique retained branch entries',G*R*C,`${G} games × ${R} entries`],
    ['Prefix + branch cache baseline',base,'No padding or workspace duplication'],
    ['Alternative: 4 KV heads (GQA)',entries*L*4*128*b,`${(H/4).toFixed(1)}× smaller raw cache`],
    ['Alternative: 256-channel joint latent',entries*L*256*b,'Hypothetical; no separate positional latent'],
    ['Alternative: 4 layer cache owners',base/2,'Hypothetical sharing; not current semantics'],
    ['Alternative: FP4 main-cache-style storage',entries*L*H*128*.5625,'4.5 bits/channel including scale; illustrative'],
    ['One layer’s dense additive mask',dense,dense>256*1024*1024?'Exceeds the 256 MiB guard; current dense path refuses this.':'Excludes boolean mask, indexing temporaries and other activations.']
  ];
  $('memory-results').innerHTML=rows.map(([name,value,note])=>`<div class="memory-row"><span>${name}</span><strong>${bytesLabel(value)}</strong><em>${note}</em></div>`).join('');
}
$('memory-form').onsubmit=e=>e.preventDefault();
$('memory-form').addEventListener('input',memoryUpdate);
$('reset-memory').onclick=()=>{$('memory-form').reset();memoryUpdate();};memoryUpdate();
function amdahlUpdate(){const f=+$('attn-share').value/100,s=+$('kernel-speed').value,remaining=1-f+f/s;$('share-value').textContent=`${Math.round(f*100)}%`;$('amdahl').textContent=`Whole-program speedup: ${(1/remaining).toFixed(2)}×. Runtime falls by ${((1-remaining)*100).toFixed(1)}%.`;}
['attn-share','kernel-speed'].forEach(id=>$(id).addEventListener('input',amdahlUpdate));amdahlUpdate();
const pieces=['♜♞♝♛♚♝♞♜','♟♟♟♟♟♟♟♟','        ','        ','        ','        ','♙♙♙♙♙♙♙♙','♖♘♗♕♔♗♘♖'];
pieces.forEach((rank,r)=>[...rank].forEach((piece,c)=>{const square=document.createElement('span');square.className=`square ${(r+c)%2?'dark':''} ${r>=6?'white-piece':''}`;square.textContent=piece;square.setAttribute('aria-hidden','true');$('board').append(square);}));
function stateUpdate(){const a=$('history-case').value==='A';$('moves').textContent=a?'1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8':'1. Nf3 Nf6 2. Ng5 Ng4 3. Nf3 Nf6 4. Ng1 Ng8';$('repetition-result').textContent=a?'Initial position occurred 3 times. A threefold repetition draw can be claimed.':'Initial position occurred 2 times. No threefold repetition draw can be claimed here.';}
$('history-case').onchange=stateUpdate;stateUpdate();

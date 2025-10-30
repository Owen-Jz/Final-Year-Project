const form = document.getElementById('analyze-form');
const btn = document.getElementById('analyze-btn');
const statusEl = document.getElementById('status');
const dlBtn = document.getElementById('download-json');
const overlay = document.getElementById('overlay');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('file');
  if (!fileInput.files[0]) return;
  btn.disabled = true; dlBtn.disabled = true; overlay.style.display = 'flex';
  statusEl.textContent = 'Analyzing...';
  const privacy = document.getElementById('privacy').checked;
  const deviceId = document.getElementById('device_id').value.trim();
  const data = new FormData();
  data.append('file', fileInput.files[0]);
  data.append('privacy_mode', privacy ? 'true' : 'false');
  if (deviceId) data.append('device_id', deviceId);
  try {
    const res = await fetch('/analyze', { method: 'POST', body: data });
    if (!res.ok) {
      statusEl.textContent = 'Error ' + res.status + ': ' + (await res.text());
      btn.disabled = false; return;
    }
    const json = await res.json();
    renderReport(json);
    statusEl.textContent = 'Done';
    dlBtn.disabled = false;
    dlBtn.onclick = () => downloadJson(json);
  } catch (err) {
    statusEl.textContent = 'Error: ' + err;
  } finally {
    btn.disabled = false; overlay.style.display = 'none';
  }
});

function renderReport(j) {
  document.getElementById('report').textContent = JSON.stringify(j, null, 2);
  // Top stats
  const dec = document.getElementById('decision');
  const ens = document.getElementById('ens');
  const ml = document.getElementById('ml');
  dec.textContent = j?.ensemble?.decision ?? '—';
  ens.textContent = (j?.ensemble?.weighted_score ?? 0).toFixed(2);
  ml.textContent = (j?.ml?.score ?? 0).toFixed(2);
  const conf = Math.max(0, Math.min(1, j?.ensemble?.weighted_score ?? 0));
  document.getElementById('confbar').style.width = (conf*100).toFixed(0) + '%';
  dec.className = '';
  if (j?.ensemble?.decision === 'LIKELY_MANIPULATED') dec.classList.add('badge','bad');
  else if (j?.ensemble?.decision === 'SUSPECT') dec.classList.add('badge','warn');
  else dec.classList.add('badge','ok');

  // Badges for metadata flags
  const mb = document.getElementById('meta-badges');
  mb.innerHTML = '';
  (j?.metadata?.flags || []).forEach(f => {
    const span = document.createElement('span');
    span.className = 'badge warn';
    span.style.marginRight = '6px';
    span.textContent = f;
    mb.appendChild(span);
  });

  // Images
  const heat = j?.prnu?.heatmap_image;
  if (heat) document.getElementById('heatmap').src = heat;
  const resi = (j?.prnu?.residual_images || [])[0];
  if (resi) document.getElementById('residual').src = resi;
  const f1 = (j?.evidence?.frames || [])[0];
  if (f1) document.getElementById('frame1').src = f1;

  // Provider chip
  const provider = document.getElementById('provider');
  provider.textContent = 'ML: ' + (j?.ml?.provider || 'unknown');

  // Heatmap grid (up to 3)
  const hg = document.getElementById('heatmap-grid');
  hg.innerHTML = '';
  (j?.prnu?.heatmap_images || []).forEach(src => {
    const d = document.createElement('div'); d.className = 'imgbox';
    const i = document.createElement('img'); i.src = src; d.appendChild(i); hg.appendChild(d);
  });

  // Faces table (top 5 by score)
  const faces = (j?.prnu?.face_region_scores || []).slice().sort((a,b)=>b.score-a.score).slice(0,5);
  const faceText = faces.map(f => `frame ${f.frame_index} bbox ${f.bbox} score ${(f.score||0).toFixed(3)}`).join('\n');
  document.getElementById('faces').textContent = faceText || '—';
}
// Health indicator
(async function(){
  try{
    const r = await fetch('/health');
    const j = await r.json();
    document.getElementById('health').textContent = 'health: ' + (j?.status || 'unknown');
  }catch{ document.getElementById('health').textContent = 'health: error'; }
})();

function downloadJson(j) {
  const blob = new Blob([JSON.stringify(j, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `report_${j?.task_id || 'analysis'}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}



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
  // Top stats
  const dec = document.getElementById('decision');
  const ens = document.getElementById('ens');
  const ml = document.getElementById('ml');
  dec.textContent = j?.ensemble?.decision ?? '—';
  ens.textContent = (j?.ensemble?.weighted_score ?? 0).toFixed(2);
  ml.textContent = (j?.ml?.score ?? 0).toFixed(2);
  const conf = Math.max(0, Math.min(1, j?.ensemble?.weighted_score ?? 0));
  document.getElementById('confbar').style.width = (conf*100).toFixed(0) + '%';

  // Badges for metadata flags
  const mb = document.getElementById('meta-badges');
  mb.innerHTML = '';
  const metaFlags = j?.metadata?.flags || [];
  metaFlags.forEach(f => {
    const span = document.createElement('span');
    span.className = 'badge badge--warn';
    span.textContent = f;
    mb.appendChild(span);
  });
  if (metaFlags.length === 0) {
    const span = document.createElement('span');
    span.className = 'badge badge--neutral';
    span.textContent = 'No metadata flags detected';
    mb.appendChild(span);
  }

  // Images
  const heat = j?.prnu?.heatmap_image;
  document.getElementById('heatmap').src = heat || '';
  const resi = (j?.prnu?.residual_images || [])[0];
  document.getElementById('residual').src = resi || '';
  const f1 = (j?.evidence?.frames || [])[0];
  document.getElementById('frame1').src = f1 || '';

  // Provider chip
  const provider = document.getElementById('provider');
  provider.textContent = 'ML: ' + (j?.ml?.provider || 'unknown');

  // AI Explanation (llava:7b frame-by-frame analysis)
  const rawResp = j?.ml?.raw_response || {};
  const verdict = rawResp.verdict || '';
  const explanation = rawResp.explanation || rawResp.rationale || '';
  
  // Verdict badge
  const verdictDiv = document.getElementById('verdict');
  if (verdict) {
    verdictDiv.textContent = `Verdict: ${verdict}`;
    if (verdict.includes('AUTHENTIC')) {
      verdictDiv.style.setProperty('color', 'var(--color-positive)');
    } else if (verdict.includes('MANIPULATED')) {
      verdictDiv.style.setProperty('color', 'var(--color-negative)');
    } else {
      verdictDiv.style.setProperty('color', 'var(--color-warning)');
    }
  } else {
    verdictDiv.textContent = '';
  }
  
  const explanationText = explanation || rawResp.note || 'No explanation available (using stub mode or model did not provide explanation).';
  document.getElementById('explanation').textContent = explanationText;
  
  // Frame-by-frame analysis
  const frameAnalysisDiv = document.getElementById('frame-analysis');
  frameAnalysisDiv.innerHTML = '';
  frameAnalysisDiv.classList.remove('empty');
  const frameAnalysis = rawResp.frame_analysis || [];
  if (frameAnalysis && Array.isArray(frameAnalysis) && frameAnalysis.length > 0) {
    frameAnalysis.forEach((fa, idx) => {
      const frameCard = document.createElement('article');
      frameCard.className = 'frame-card';

      const header = document.createElement('div');
      header.className = 'frame-card__header';

      const frameTitle = document.createElement('div');
      frameTitle.className = 'frame-card__title';
      frameTitle.textContent = `${fa.frame_label || `Frame ${fa.frame_index || idx}`}`;
      header.appendChild(frameTitle);
      
      const authBadge = document.createElement('span');
      authBadge.className = fa.authentic ? 'badge badge--ok' : 'badge badge--alert';
      authBadge.textContent = fa.authentic ? 'AUTHENTIC' : 'MANIPULATED';
      header.appendChild(authBadge);
      
      frameCard.appendChild(header);
      
      if (fa.assessment) {
        const assessment = document.createElement('div');
        assessment.className = 'frame-card__assessment';
        assessment.textContent = fa.assessment;
        frameCard.appendChild(assessment);
      }
      
      if (fa.visual_artifacts && Array.isArray(fa.visual_artifacts) && fa.visual_artifacts.length > 0) {
        const artifactsDiv = document.createElement('div');
        artifactsDiv.className = 'frame-card__artifacts';
        artifactsDiv.innerHTML = '<strong>Artifacts:</strong> ' + fa.visual_artifacts.join(', ');
        frameCard.appendChild(artifactsDiv);
      }
      
      frameAnalysisDiv.appendChild(frameCard);
    });
  } else {
    frameAnalysisDiv.textContent = 'No frame-level insights available.';
    frameAnalysisDiv.classList.add('empty');
  }
  
  // Key findings
  const kfDiv = document.getElementById('key-findings');
  kfDiv.innerHTML = '';
  const keyFindings = rawResp.key_findings;
  if (keyFindings && Array.isArray(keyFindings) && keyFindings.length > 0) {
    const ul = document.createElement('ul');
    keyFindings.forEach(f => {
      const li = document.createElement('li');
      li.textContent = f;
      ul.appendChild(li);
    });
    kfDiv.appendChild(ul);
  } else {
    kfDiv.textContent = 'No key findings were highlighted.';
  }
  
  // ML Confidence
  const mlConf = rawResp.confidence || '';
  const confDiv = document.getElementById('ml-confidence');
  if (mlConf) {
    confDiv.textContent = `Model Confidence: ${mlConf.toUpperCase()}`;
  } else {
    confDiv.textContent = '';
  }

  // Heatmap grid (up to 3)
  const hg = document.getElementById('heatmap-grid');
  hg.innerHTML = '';
  const heatmaps = (j?.prnu?.heatmap_images || []).slice(0, 3);
  if (heatmaps.length > 0) {
    hg.classList.remove('hidden');
    heatmaps.forEach((src, index) => {
      const tile = document.createElement('article');
      tile.className = 'image-tile image-tile--compact';
      const header = document.createElement('header');
      header.textContent = `Heatmap ${index + 1}`;
      const img = document.createElement('img');
      img.src = src;
      img.alt = `Heatmap ${index + 1}`;
      tile.appendChild(header);
      tile.appendChild(img);
      hg.appendChild(tile);
    });
  } else {
    hg.classList.add('hidden');
  }

  // Faces table (top 5 by score)
  const faces = (j?.prnu?.face_region_scores || []).slice().sort((a,b)=>b.score-a.score).slice(0,5);
  const faceText = faces.map(f => `frame ${f.frame_index}  score ${(f.score||0).toFixed(3)}  bbox [${(f.bbox||[]).join(', ')}]`).join('\n');
  document.getElementById('faces').textContent = faceText || '—';
}

function downloadJson(j) {
  const blob = new Blob([JSON.stringify(j, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `report_${j?.task_id || 'analysis'}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

// Health indicator
(async function(){
  try{
    const r = await fetch('/health');
    const j = await r.json();
    document.getElementById('health').textContent = 'health: ' + (j?.status || 'unknown');
  }catch{ document.getElementById('health').textContent = 'health: error'; }
})();



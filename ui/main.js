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
  (j?.metadata?.flags || []).forEach(f => {
    const span = document.createElement('span');
    span.className = 'badge warn';
    span.textContent = f;
    mb.appendChild(span);
  });

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
      verdictDiv.style.color = '#19c37d';
    } else if (verdict.includes('MANIPULATED')) {
      verdictDiv.style.color = '#ff5470';
    } else {
      verdictDiv.style.color = '#ffb020';
    }
  } else {
    verdictDiv.textContent = '';
  }
  
  const explanationText = explanation || rawResp.note || 'No explanation available (using stub mode or model did not provide explanation).';
  document.getElementById('explanation').textContent = explanationText;
  
  // Frame-by-frame analysis
  const frameAnalysisDiv = document.getElementById('frame-analysis');
  frameAnalysisDiv.innerHTML = '';
  const frameAnalysis = rawResp.frame_analysis || [];
  if (frameAnalysis && Array.isArray(frameAnalysis) && frameAnalysis.length > 0) {
    frameAnalysis.forEach((fa, idx) => {
      const frameCard = document.createElement('div');
      frameCard.style.padding = '10px'; frameCard.style.background = '#0a0f1a'; frameCard.style.border = '1px solid #1f2a44'; frameCard.style.borderRadius = '8px';
      
      const header = document.createElement('div');
      header.style.display = 'flex'; header.style.justifyContent = 'space-between'; header.style.alignItems = 'center'; header.style.marginBottom = '6px';
      
      const frameTitle = document.createElement('div');
      frameTitle.style.fontWeight = '600'; frameTitle.textContent = `${fa.frame_label || `Frame ${fa.frame_index || idx}`}`;
      header.appendChild(frameTitle);
      
      const authBadge = document.createElement('span');
      authBadge.className = fa.authentic ? 'badge ok' : 'badge bad';
      authBadge.textContent = fa.authentic ? 'AUTHENTIC' : 'MANIPULATED';
      header.appendChild(authBadge);
      
      frameCard.appendChild(header);
      
      if (fa.assessment) {
        const assessment = document.createElement('div');
        assessment.style.color = '#e6edf3'; assessment.style.fontSize = '13px'; assessment.style.marginBottom = '6px';
        assessment.textContent = fa.assessment;
        frameCard.appendChild(assessment);
      }
      
      if (fa.visual_artifacts && Array.isArray(fa.visual_artifacts) && fa.visual_artifacts.length > 0) {
        const artifactsDiv = document.createElement('div');
        artifactsDiv.style.fontSize = '12px'; artifactsDiv.style.color = 'var(--muted)';
        artifactsDiv.innerHTML = '<strong>Artifacts:</strong> ' + fa.visual_artifacts.join(', ');
        frameCard.appendChild(artifactsDiv);
      }
      
      frameAnalysisDiv.appendChild(frameCard);
    });
  } else {
    frameAnalysisDiv.textContent = '—';
    frameAnalysisDiv.style.color = 'var(--muted)';
  }
  
  // Key findings
  const kfDiv = document.getElementById('key-findings');
  kfDiv.innerHTML = '';
  const keyFindings = rawResp.key_findings;
  if (keyFindings && Array.isArray(keyFindings) && keyFindings.length > 0) {
    const ul = document.createElement('ul');
    ul.style.margin = '8px 0 0 0'; ul.style.paddingLeft = '20px'; ul.style.color = '#e6edf3'; ul.style.fontSize = '13px';
    keyFindings.forEach(f => {
      const li = document.createElement('li');
      li.textContent = f;
      ul.appendChild(li);
    });
    kfDiv.appendChild(ul);
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
  (j?.prnu?.heatmap_images || []).slice(0,3).forEach(src => {
    const d = document.createElement('div'); d.className = 'imgbox';
    const i = document.createElement('img'); i.src = src; d.appendChild(i); hg.appendChild(d);
  });

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



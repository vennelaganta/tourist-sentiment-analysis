let probChart = null;
const box = $('review');

box.addEventListener('input', () => { $('count').textContent = `${box.value.length} / 5000`; });
api('/api/health').then(h => {
  Object.entries(h.supported_languages || {}).forEach(([code, name]) => {
    const o = document.createElement('option'); o.value = code; o.textContent = `${name} (${code})`; $('lang').appendChild(o);
  });
}).catch(() => {});
$('exampleSel').addEventListener('change', e => {
  const opt = e.target.selectedOptions[0];
  if (!opt || !opt.value) return;
  box.value = opt.value; box.dispatchEvent(new Event('input'));
  $('lang').value = 'auto'; e.target.value = ''; box.focus();   // language is auto-detected unless you pick one
});
document.querySelectorAll('[data-example]').forEach(b => b.addEventListener('click', () => {
  box.value = b.dataset.example; box.dispatchEvent(new Event('input')); box.focus();
}));

function reset() {
  box.value = ''; box.dispatchEvent(new Event('input')); $('lang').value = 'auto';
  showAlert($('alert'), ''); show($('result'), false); show($('placeholder'), true);
  if (probChart) { probChart.destroy(); probChart = null; }
  box.focus();
}
$('clearBtn').addEventListener('click', reset);
$('againBtn').addEventListener('click', () => { reset(); window.scrollTo({top: 0, behavior: 'smooth'}); });

function drawChart(probs) {
  if (probChart) probChart.destroy();
  const labels = Object.keys(probs);
  probChart = new Chart($('probChart'), {
    type: 'bar',
    data: {labels, datasets: [{data: labels.map(l => +(probs[l] * 100).toFixed(2)),
      backgroundColor: labels.map(l => COLORS[l] || '#13a89e'), borderRadius: 8}]},
    options: {indexAxis: 'y', maintainAspectRatio: false, plugins: {legend: {display: false},
      tooltip: {callbacks: {label: c => ` ${c.parsed.x.toFixed(1)}%`}}},
      scales: {x: {min: 0, max: 100, ticks: {callback: v => v + '%'}}}}
  });
}

async function analyze() {
  const text = box.value.trim();
  showAlert($('alert'), '');
  if (!text) { showAlert($('alert'), 'Please enter a review before analyzing.', 'warn'); box.focus(); return; }
  const btn = $('analyzeBtn'); btn.disabled = true;
  btn.querySelector('span').textContent = 'Analyzing…';
  try {
    const r = await api('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                          body: JSON.stringify({review: text, language: $('lang').value})});
    const lab = $('resLabel'); lab.replaceChildren(badge(r.sentiment)); lab.firstChild.style.fontSize = '1.4rem';
    $('resConf').textContent = r.confidence != null ? `Confidence: ${pct(r.confidence)}` : '';
    const how = {detected: 'detected automatically', user: 'chosen by you', unknown: 'could not be determined'}[r.language_source];
    $('resLang').textContent = `Language: ${r.language ? r.language_name + ' (' + r.language + ')' : 'Unknown'} — ${how}` +
      (r.language_supported === false ? ' · not a fine-tuning language' : '');
    const wl = $('warnList'); wl.replaceChildren();
    (r.warnings || []).forEach(w => { const li = document.createElement('li'); li.textContent = w; wl.appendChild(li); });
    show(wl, (r.warnings || []).length > 0);
    $('resReview').textContent = r.review;       // textContent: user text is never parsed as HTML
    $('resExplain').textContent = r.explanation;
    show($('chartWrap'), !!r.probabilities); show($('noProb'), !r.probabilities);
    show($('placeholder'), false); show($('result'), true);
    if (r.probabilities) drawChart(r.probabilities);
    icons();
  } catch (e) {
    showAlert($('alert'), e.message, 'error');
  } finally {
    btn.disabled = false; btn.querySelector('span').textContent = 'Analyze Sentiment';
  }
}
$('analyzeBtn').addEventListener('click', analyze);
box.addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze(); });

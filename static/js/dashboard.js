const charts = {};
function chart(id, config) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), config); }
function cell(tr, content, tag = 'td', cls = '') {
  const c = document.createElement(tag); if (cls) c.className = cls;
  if (content instanceof Node) c.appendChild(content); else c.textContent = content;
  tr.appendChild(c); return c;
}
function fill(table, header, rows) {
  table.replaceChildren();
  const hr = document.createElement('tr'); header.forEach(h => cell(hr, h, 'th'));
  const thead = document.createElement('thead'); thead.appendChild(hr);
  const tbody = document.createElement('tbody');
  rows.forEach(r => { const tr = document.createElement('tr'); r.forEach(v => cell(tr, v)); tbody.appendChild(tr); });
  table.append(thead, tbody);
}

function fillLangFilter(d) {
  const sel = $('langFilter'), keep = sel.value;
  sel.replaceChildren(new Option('All languages', ''));
  Object.keys(d.languages).forEach(c => sel.appendChild(new Option(d.language_names[c] || c, c)));
  sel.value = keep;
}

async function loadStats() {
  const d = await api('/api/dashboard');
  $('sTotal').textContent = d.total;
  fillLangFilter(d);
  $('sPos').textContent = d.counts.Positive; $('sNeg').textContent = d.counts.Negative; $('sNeu').textContent = d.counts.Neutral;
  const has = d.total > 0;
  show($('emptyState'), !has); show($('charts'), has);
  if (!has) return;
  const vals = LABELS.map(l => d.counts[l]), cols = LABELS.map(l => COLORS[l]);
  chart('doughnut', {type: 'doughnut', data: {labels: LABELS, datasets: [{data: vals, backgroundColor: cols, borderWidth: 3}]},
    options: {maintainAspectRatio: false, cutout: '62%', plugins: {legend: {position: 'bottom'}}}});
  const codes = Object.keys(d.languages);
  chart('langChart', {type: 'bar', data: {labels: codes.map(c => d.language_names[c] || c), datasets: [{label: 'Reviews', data: codes.map(c => d.languages[c]), backgroundColor: '#13a89e', borderRadius: 8}]},
    options: {maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {y: {beginAtZero: true, ticks: {precision: 0}}}}});
  chart('bars', {type: 'bar', data: {labels: LABELS, datasets: [{label: 'Reviews', data: vals, backgroundColor: cols, borderRadius: 10}]},
    options: {maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {y: {beginAtZero: true, ticks: {precision: 0}}}}});
}

async function loadHistory() {
  const params = new URLSearchParams({limit: 25, q: $('q').value, sentiment: $('filter').value, language: $('langFilter').value});
  const d = await api('/api/history?' + params);
  const body = $('rows'); body.replaceChildren();
  if (!d.items.length) {
    const tr = document.createElement('tr'); const td = cell(tr, 'No saved predictions match.'); td.colSpan = 5; td.className = 'empty'; body.appendChild(tr);
  }
  d.items.forEach(r => {
    const tr = document.createElement('tr');
    cell(tr, r.timestamp.replace('T', ' ').replace(/\+00:00$/, ''));
    cell(tr, r.review.length > 220 ? r.review.slice(0, 220) + '…' : r.review, 'td', 'review');
    cell(tr, r.language ? r.language.toUpperCase() : '–');
    cell(tr, badge(r.sentiment));
    cell(tr, r.confidence ? pct(parseFloat(r.confidence)) : '–');
    body.appendChild(tr);
  });
  $('rowInfo').textContent = `Showing ${d.items.length} of ${d.matched} matching predictions.`;
}

let runs = [], activeModel = null;
function renderRun(r) {
  const o = r.overall, smp = r.sample;
  $('metricsMeta').textContent = `Model ${r.model_id} · ${r.split} split · ${r.scope}${r.balanced_classes ? ' · class-balanced' : ''} · ${smp.n} reviews sampled from ${smp.scanned} streamed rows · ` +
    `${r.runtime.reviews_per_second} reviews/s on ${r.runtime.device} · run on ${r.created_at.slice(0, 10)}`;
  $('mAcc').textContent = pct(o.accuracy); $('mMacro').textContent = pct(o.macro.f1);
  $('mWeighted').textContent = pct(o.weighted.f1); $('mBal').textContent = pct(o.balanced_accuracy);
  const warns = [];
  if (activeModel && r.model_id !== activeModel) warns.push(`This run evaluated ${r.model_id}, which is NOT the model the website is currently serving (${activeModel}).`);
  warns.push(`This is a ${r.split}-split SAMPLE of ${smp.n} reviews (up to ${smp.per_language_quota} per language), not the full split. ` +
    'Because languages are sampled equally, overall scores are language-balanced and differ from scores on the natural, English-heavy distribution.');
  if (r.split !== 'test') warns.push('This is a validation run used for model selection; the held-out test split is not used here.');
  const low = Object.entries(r.per_language).filter(([, v]) => !v.reliable).map(([k, v]) => `${k} (n=${v.n})`);
  if (low.length) warns.push('Too few examples for reliable per-language estimates: ' + low.join(', ') + '.');
  if (smp.missing_expected_languages.length) warns.push('No examples collected for: ' + smp.missing_expected_languages.join(', ') + '.');
  if (smp.truncated_reviews) warns.push(`${smp.truncated_reviews} reviews were longer than the model limit and were truncated.`);
  $('evalWarn').textContent = warns.join(' ');
  fill($('classTable'), ['Class', 'Precision', 'Recall', 'F1', 'Support'],
    Object.entries(o.per_class).map(([c, v]) => [c, pct(v.precision), pct(v.recall), pct(v.f1), v.support]));
  const cm = o.confusion_matrix;
  fill($('cmTable'), ['True \\ Pred', ...cm.labels], cm.labels.map((l, i) => [l, ...cm.matrix[i]]));
  const langs = Object.entries(r.per_language);
  fill($('langTable'), ['Language', 'Reviews (n)', 'Accuracy', 'Macro F1', 'Weighted F1', 'Reliability'],
    langs.map(([k, v]) => [k, v.n, pct(v.accuracy), pct(v.macro_f1), pct(v.weighted_f1), v.reliable ? 'ok' : 'low n']));
  chart('langF1', {type: 'bar', data: {labels: langs.map(([k, v]) => `${k} (n=${v.n})`), datasets: [{label: 'Macro F1', data: langs.map(([, v]) => +(v.macro_f1 * 100).toFixed(1)),
      backgroundColor: langs.map(([, v]) => v.reliable ? '#13a89e' : '#c9d3df'), borderRadius: 6}]},
    options: {maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {y: {min: 0, max: 100, ticks: {callback: v => v + '%'}}}}});
  const doms = Object.entries(r.per_domain).sort((a, b) => b[1].n - a[1].n);
  fill($('domTable'), ['Domain', 'Reviews (n)', 'Accuracy', 'Macro F1', 'Reliability'],
    doms.map(([k, v]) => [k || 'unknown', v.n, pct(v.accuracy), pct(v.macro_f1), v.reliable ? 'ok' : 'low n']));
  $('sampleInfo').textContent = 'Languages: ' + Object.entries(smp.language_counts).map(([k, v]) => `${k} ${v}`).join(', ') +
    ' · Classes: ' + Object.entries(smp.class_counts).map(([k, v]) => `${k} ${v}`).join(', ') +
    ` · duplicates skipped: ${smp.duplicates_skipped}` +
    (smp.overlap_check ? ` · train-overlap check: ${smp.overlap_check.eval_texts_also_in_scanned_train} eval texts also found in the first ${smp.overlap_check.train_rows_scanned} train rows` : ' · train-overlap check: not run');
}

async function loadMetrics() {
  let m;
  try { m = await api('/api/metrics'); }
  catch (e) { show($('metricsEmpty'), true); show($('metricsBody'), false); return; }
  runs = m.runs; activeModel = m.active_model; show($('metricsEmpty'), false); show($('metricsBody'), true);
  const sel = $('runSel'); sel.replaceChildren();
  runs.forEach((r, i) => sel.appendChild(new Option(`${r.split} · ${r.model_id.split('/').pop()} · ${r.scope} · n=${r.sample.n}${r.balanced_classes ? ' · balanced' : ''}`, i)));
  renderRun(runs[0]);
}
$('runSel').addEventListener('change', e => renderRun(runs[+e.target.value]));

async function refresh() {
  const btn = $('refreshBtn'); btn.disabled = true; showAlert($('dashAlert'), '');
  try { await Promise.all([loadStats(), loadHistory(), loadMetrics()]); }
  catch (e) { showAlert($('dashAlert'), e.message, 'error'); }
  finally { btn.disabled = false; }
}
let timer;
$('q').addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => loadHistory().catch(e => showAlert($('dashAlert'), e.message)), 300); });
$('langFilter').addEventListener('change', () => loadHistory().catch(e => showAlert($('dashAlert'), e.message)));
$('filter').addEventListener('change', () => loadHistory().catch(e => showAlert($('dashAlert'), e.message)));
$('refreshBtn').addEventListener('click', refresh);
refresh();

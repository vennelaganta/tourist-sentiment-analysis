let current = null;
const fileInput = $('file'), drop = $('drop');

function td(tr, text, tag = 'td') { const c = document.createElement(tag); c.textContent = text; tr.appendChild(c); }
function table(el, header, rows) {
  el.replaceChildren();
  const th = document.createElement('tr'); header.forEach(h => td(th, h, 'th'));
  const head = document.createElement('thead'); head.appendChild(th);
  const body = document.createElement('tbody');
  rows.forEach(r => { const tr = document.createElement('tr'); r.forEach(v => td(tr, v)); body.appendChild(tr); });
  el.append(head, body);
}
function options(select, names, allowNone) {
  select.replaceChildren();
  if (allowNone) { const o = document.createElement('option'); o.value = ''; o.textContent = '— none (unlabelled data) —'; select.appendChild(o); }
  names.forEach(n => { const o = document.createElement('option'); o.value = o.textContent = n; select.appendChild(o); });
}
function quote(s) { return '"' + s.replace(/"/g, '') + '"'; }
function updateCommand() {
  if (!current) return;
  let c = `python classical_baseline\\train_model.py --data data\\${current.filename} --text-col ${quote($('textCol').value)}`;
  if ($('labelCol').value) c += ` --label-col ${quote($('labelCol').value)}`;
  $('cmd').textContent = c; show($('cmdBox'), !!$('labelCol').value);
}

async function upload(file) {
  showAlert($('alert'), ''); show($('out'), false);
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.csv')) { showAlert($('alert'), 'Only .csv files are accepted.'); return; }
  $('dropText').textContent = 'Uploading ' + file.name + '…';
  const fd = new FormData(); fd.append('file', file);
  try {
    const d = await api('/api/upload-dataset', {method: 'POST', body: fd});
    current = d;
    $('nRows').textContent = d.rows.toLocaleString(); $('nCols').textContent = d.columns; $('fname').textContent = d.filename;
    const names = d.column_info.map(c => c.name);
    table($('colTable'), ['Column', 'Missing values'], d.column_info.map(c => [c.name, c.missing]));
    options($('textCol'), names, false); options($('labelCol'), names, true);
    if (d.suggested_text_column) $('textCol').value = d.suggested_text_column;
    if (d.suggested_label_column) $('labelCol').value = d.suggested_label_column;
    table($('prevTable'), names, d.preview.map(r => names.map(n => r[n])));
    show($('summary'), false); updateCommand(); show($('out'), true); icons();
    $('dropText').textContent = 'Loaded ' + d.filename + ' — click to choose a different file';
  } catch (e) {
    showAlert($('alert'), e.message); $('dropText').textContent = 'Click to choose a .csv file or drag it here';
  }
}

async function validate() {
  const box = $('summary'); showAlert($('alert'), '');
  try {
    const s = await api('/api/dataset-summary', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({upload_id: current.upload_id, text_column: $('textCol').value, label_column: $('labelCol').value || null})});
    const lines = [`Usable reviews: ${s.usable_reviews} · missing text: ${s.missing_text} · duplicate reviews: ${s.duplicate_reviews} · average length: ${s.avg_words} words`];
    if (s.label_column) {
      lines.push('Recognised labels: ' + (Object.entries(s.label_distribution).map(([k, v]) => `${k} ${v}`).join(', ') || 'none'));
      if (s.missing_labels) lines.push(`Missing labels: ${s.missing_labels}`);
      if (s.unrecognised_labels.length) lines.push('Unrecognised label values: ' + s.unrecognised_labels.join(', ') + '  → pass --label-map when training (see README).');
    }
    box.replaceChildren(...lines.map(t => { const p = document.createElement('p'); p.className = 'small'; p.style.marginBottom = '.4rem'; p.textContent = t; return p; }));
    box.className = 'alert ok'; box.style.display = 'block'; show(box, true); updateCommand();
  } catch (e) { showAlert($('alert'), e.message); }
}

fileInput.addEventListener('change', () => upload(fileInput.files[0]));
['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => upload(e.dataTransfer.files[0]));
$('checkBtn').addEventListener('click', validate);
$('textCol').addEventListener('change', updateCommand); $('labelCol').addEventListener('change', updateCommand);

/* ---- Hugging Face dataset profile (real output of inspect_dataset.py) ---- */
(async function loadProfile() {
  try {
    const p = await api('/api/dataset-profile');
    show($('profileBody'), true); show($('profileEmpty'), false);
    const sp = p.split_sizes || {};
    $('pSplits').textContent = Object.keys(sp).length ? Object.entries(sp).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(' · ') : 'n/a';
    $('pCols').textContent = Object.keys(p.features || {}).join(', ') || 'n/a';
    $('pSample').textContent = p.sample.rows.toLocaleString();
    const mk = (id, obj, color) => new Chart($(id), {type: 'bar', data: {labels: Object.keys(obj), datasets: [{data: Object.values(obj), backgroundColor: color, borderRadius: 6}]},
      options: {maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {y: {beginAtZero: true}}}});
    mk('pLabels', p.sample.label_counts, '#13315c'); mk('pLangs', p.sample.language_counts, '#13a89e');
    $('pNote').textContent = p.sample.note + ` Median review length: ${p.sample.text_length_chars.median} characters (95th percentile ${p.sample.text_length_chars.p95}).`;
  } catch (e) { show($('profileEmpty'), true); }
})();

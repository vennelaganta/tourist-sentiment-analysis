/* Shared helpers: API wrapper, nav toggle, health pill, reveal animation */
const COLORS = {Positive: '#12a150', Negative: '#e03e3e', Neutral: '#e8912d'};
const LABELS = ['Positive', 'Negative', 'Neutral'];

async function api(path, options = {}) {
  let res;
  try { res = await fetch(path, options); }
  catch (e) { throw new Error('Cannot reach the Flask server. Is it running (python app.py)?'); }
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON body */ }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
  return data;
}
const $ = (id) => document.getElementById(id);
function show(el, on = true) { el.classList.toggle('hidden', !on); }
function showAlert(el, msg, type = 'error') {
  el.className = `alert ${type}`; el.textContent = msg; show(el, !!msg);
}
function icons() { if (window.lucide) lucide.createIcons(); }
function pct(x) { return (x * 100).toFixed(1) + '%'; }
function badge(label) {
  const b = document.createElement('span');
  b.className = `badge ${label.toLowerCase()}`; b.textContent = label; return b;
}

document.addEventListener('DOMContentLoaded', async () => {
  icons();
  const btn = $('menuBtn'), menu = $('menu');
  btn.addEventListener('click', () => {
    const open = menu.classList.toggle('open'); btn.setAttribute('aria-expanded', open);
  });
  const io = new IntersectionObserver((es) => es.forEach(e => e.isIntersecting && e.target.classList.add('in')), {threshold: .12});
  document.querySelectorAll('.reveal').forEach(el => io.observe(el));
  pollHealth();
});

async function pollHealth() {
  try {
    const h = await api('/api/health');
    const name = (h.model && h.model.model_id ? h.model.model_id.split('/').pop() : 'model');
    const states = {ready: ['ok', `Ready · ${name}`], loading: ['', 'Loading multilingual model…'],
                    not_loaded: ['', 'Model idle'], error: ['bad', 'Model error – see Analyze page']};
    const [cls, text] = states[h.model_status] || ['bad', 'Unknown'];
    $('healthDot').className = 'dot ' + cls; $('healthText').textContent = text;
    $('healthText').title = h.model_error || '';
    if (h.model_status === 'loading') setTimeout(pollHealth, 3000);
  } catch (e) {
    $('healthDot').className = 'dot bad'; $('healthText').textContent = 'Backend offline';
  }
}

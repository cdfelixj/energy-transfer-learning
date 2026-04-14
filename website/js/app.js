'use strict';

// =====================================================================
// Constants
// =====================================================================

const EXPERIMENTS = {
  rat_education:      'Rat / Education (Exp 1)',
  rat_education_new:  'Rat / Education — Replication (Exp 2)',
  eagle_education:    'Eagle / Education (Exp 3)',
  lamb_education:     'Lamb / Education (Exp 4)',
  office_any:         'Office — Hog (Exp 5)',
  lodging_any:        'Lodging — Robin (Exp 6)',
};

const STRATEGIES = {
  pretransfer: { label: 'Scratch',          color: '#6b7280' },
  transfer:    { label: 'Full Fine-Tuning', color: '#2563eb' },
  frozen:      { label: 'Frozen Backbone',  color: '#059669' },
};

const WEEKS = [1, 2, 4, 8, 16, 32, 64, 104];
const DATA_ROOT = '../results/experiments/';

// inference constants
const INF_WINDOW  = 336;  // rolling display window (2 weeks)
const TICK_MS     = 80;   // ms per animation frame
const SPEED_MAP   = { '1x': 24, '5x': 120, '20x': 480 };

// =====================================================================
// Helpers
// =====================================================================

Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
Chart.defaults.color = '#64748b';
Chart.defaults.font.size = 11;

function parseCSV(url) {
  return new Promise((resolve, reject) => {
    Papa.parse(url, {
      download: true, header: true, dynamicTyping: true, skipEmptyLines: true,
      complete: r => resolve(r.data),
      error:    e => reject(e),
    });
  });
}

function getMetricLabel(m) {
  return { mae: 'MAE (kWh)', rmse: 'RMSE (kWh)', r2: 'R²' }[m] || m.toUpperCase();
}

function baseOptions(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: true,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#0f172a',
        titleColor: '#f8fafc',
        bodyColor: '#94a3b8',
        padding: 10,
        borderColor: '#1e293b',
        borderWidth: 1,
        cornerRadius: 8,
        titleFont: { size: 12, weight: '600' },
        bodyFont: { size: 11 },
      },
    },
    scales: {
      x: { grid: { color: 'rgba(15,23,42,.04)' }, ticks: { font: { size: 11 }, maxRotation: 0 } },
      y: {
        grid: { color: 'rgba(15,23,42,.04)' },
        ticks: { font: { size: 11 } },
        title: { display: !!yLabel, text: yLabel, font: { size: 11 } },
      },
    },
  };
}

function buildLegend(id, items) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = items.map(i =>
    `<span class="legend-item"><span class="legend-dot" style="background:${i.color}"></span>${i.label}</span>`
  ).join('');
}

// =====================================================================
// LIVE INFERENCE
// =====================================================================

let infData   = [];    // all CSV rows
let infIdx    = 0;     // current position
let infTimer  = null;
let infSpeed  = 24;    // points per tick (1x default)
let infChart  = null;
let infPlaying = false;

async function initLiveInference() {
  const wrap = document.getElementById('inference-loading');
  if (wrap) wrap.style.display = 'block';

  infData = await parseCSV('data/live_inference.csv');

  if (wrap) wrap.style.display = 'none';

  const scrubber = document.getElementById('inference-scrubber');
  scrubber.max = infData.length - 1;
  scrubber.value = 0;

  // Create chart with empty data
  const ctx = document.getElementById('inference-chart').getContext('2d');
  infChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Actual',         data: [], borderColor: '#0f172a', backgroundColor: '#0f172a', borderWidth: 2,   pointRadius: 0, tension: 0.2 },
        { label: 'Blended Pred.',  data: [], borderColor: '#2563eb', backgroundColor: '#2563eb', borderWidth: 2,   pointRadius: 0, tension: 0.2 },
        { label: 'Transfer Only',  data: [], borderColor: '#0891b2', backgroundColor: '#0891b2', borderWidth: 1.5, pointRadius: 0, tension: 0.2, borderDash: [5,3] },
        { label: 'Pre-Transfer',   data: [], borderColor: '#7c3aed', backgroundColor: '#7c3aed', borderWidth: 1.5, pointRadius: 0, tension: 0.2, borderDash: [3,3] },
      ],
    },
    options: {
      ...baseOptions('Energy (kWh)'),
      animation: false,
      scales: {
        x: {
          grid: { color: 'rgba(15,23,42,.04)' },
          ticks: {
            font: { size: 11 },
            maxTicksLimit: 8,
            maxRotation: 0,
          },
        },
        y: {
          grid: { color: 'rgba(15,23,42,.04)' },
          ticks: { font: { size: 11 } },
          title: { display: true, text: 'Energy (kWh)', font: { size: 11 } },
        },
      },
    },
  });

  buildLegend('inference-legend', [
    { label: 'Actual',        color: '#111827' },
    { label: 'Blended',       color: '#2563eb' },
    { label: 'Transfer Only', color: '#0891b2' },
    { label: 'Pre-Transfer',  color: '#7c3aed' },
  ]);

  updateInferenceStats(infData[0]);

  // Controls
  document.getElementById('play-btn').addEventListener('click', togglePlay);
  document.getElementById('reset-btn').addEventListener('click', resetInference);

  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      infSpeed = SPEED_MAP[btn.dataset.speed] || 24;
    });
  });

  scrubber.addEventListener('input', () => {
    infIdx = parseInt(scrubber.value, 10);
    renderInferenceWindow();
    updateInferenceStats(infData[infIdx]);
  });
}

function togglePlay() {
  infPlaying ? pauseInference() : playInference();
}

function playInference() {
  if (infPlaying) return;
  if (infIdx >= infData.length - 1) infIdx = 0; // auto-restart if at end
  infPlaying = true;
  document.getElementById('play-btn').textContent = '⏸ Pause';
  document.getElementById('play-btn').classList.add('primary');
  infTimer = setInterval(inferenceStep, TICK_MS);
}

function pauseInference() {
  clearInterval(infTimer);
  infTimer = null;
  infPlaying = false;
  document.getElementById('play-btn').textContent = '▶ Play';
  document.getElementById('play-btn').classList.remove('primary');
}

function resetInference() {
  pauseInference();
  infIdx = 0;
  document.getElementById('inference-scrubber').value = 0;
  // Clear chart
  infChart.data.labels = [];
  infChart.data.datasets.forEach(d => d.data = []);
  infChart.update('none');
  updateInferenceStats(infData[0]);
}

function inferenceStep() {
  infIdx = Math.min(infIdx + infSpeed, infData.length - 1);
  document.getElementById('inference-scrubber').value = infIdx;
  renderInferenceWindow();
  updateInferenceStats(infData[infIdx]);
  if (infIdx >= infData.length - 1) pauseInference();
}

function renderInferenceWindow() {
  const start = Math.max(0, infIdx - INF_WINDOW);
  const slice = infData.slice(start, infIdx + 1);

  // Build compact x labels: show date for every ~24th point
  const labels = slice.map((r, i) => {
    const globalIdx = start + i;
    if (globalIdx % 24 === 0) {
      // Format: "Mar 27" from ISO timestamp
      const d = new Date(r.timestamp);
      return d.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' });
    }
    return '';
  });

  infChart.data.labels = labels;
  infChart.data.datasets[0].data = slice.map(r => r.actual_kwh);
  infChart.data.datasets[1].data = slice.map(r => r.prediction_kwh);
  infChart.data.datasets[2].data = slice.map(r => r.transfer_pred_kwh);
  infChart.data.datasets[3].data = slice.map(r => r.pretransfer_pred_kwh);
  infChart.update('none');
}

function updateInferenceStats(row) {
  if (!row) return;

  // Timestamp
  const ts = document.getElementById('inference-timestamp');
  if (ts) ts.textContent = row.timestamp ? `${row.timestamp}` : '';

  // Stat cards
  setLiveStat('ls-actual',     row.actual_kwh?.toFixed(1),         'kWh');
  setLiveStat('ls-prediction', row.prediction_kwh?.toFixed(1),     'kWh blended');
  setLiveStat('ls-weight',     ((row.blend_weight_transfer ?? 0.5) * 100).toFixed(1) + '%', 'weight on transfer');
  setLiveStat('ls-model',      row.active_model ?? '—',            '');

  // MAE cards
  const maeTransfer = document.getElementById('mae-transfer');
  const maePre      = document.getElementById('mae-pretransfer');
  if (maeTransfer) maeTransfer.textContent = row.rolling_mae_transfer?.toFixed(1) + ' kWh';
  if (maePre)      maePre.textContent      = row.rolling_mae_pretransfer?.toFixed(1) + ' kWh';
}

function setLiveStat(id, value, sub) {
  const el = document.getElementById(id);
  if (!el) return;
  const valEl = el.querySelector('.ls-value');
  const subEl = el.querySelector('.ls-sub');
  if (valEl) valEl.textContent = value ?? '—';
  if (subEl) subEl.textContent = sub;
}

// =====================================================================
// DATA EFFICIENCY RACE
// =====================================================================

let raceData   = null;  // shared with effData if available
let raceIdx    = 0;     // 0..7 — index into WEEKS
let raceTimer  = null;
let raceChart  = null;
const RACE_TICK_MS = 900;

async function initEfficiencyRace() {
  if (!raceData) raceData = await parseCSV('data/all_data_efficiency.csv');
  renderRaceChart();

  document.getElementById('race-play-btn').addEventListener('click', toggleRace);
  document.getElementById('race-reset-btn').addEventListener('click', resetRace);
  document.getElementById('race-experiment').addEventListener('change', resetRace);
  document.getElementById('race-metric').addEventListener('change', () => {
    renderRaceChart();
    // update sub labels on cards
    const metric = document.getElementById('race-metric').value;
    const sub = metric === 'r2' ? 'R² (higher = better)' : `kWh ${metric.toUpperCase()}`;
    ['rs-pretransfer', 'rs-transfer', 'rs-frozen'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.querySelector('.ls-sub').textContent = sub;
    });
  });
}

function toggleRace() {
  raceTimer ? pauseRace() : playRace();
}

function playRace() {
  if (raceTimer) return;
  if (raceIdx >= WEEKS.length - 1) raceIdx = 0;
  document.getElementById('race-play-btn').textContent = '⏸ Pause';
  document.getElementById('race-play-btn').classList.add('primary');
  raceTimer = setInterval(raceStep, RACE_TICK_MS);
}

function pauseRace() {
  clearInterval(raceTimer);
  raceTimer = null;
  document.getElementById('race-play-btn').textContent = '▶ Play';
  document.getElementById('race-play-btn').classList.remove('primary');
}

function resetRace() {
  pauseRace();
  raceIdx = 0;
  renderRaceChart();
}

function raceStep() {
  if (raceIdx >= WEEKS.length - 1) { pauseRace(); return; }
  raceIdx++;
  renderRaceChart();
}

function renderRaceChart() {
  const exp    = document.getElementById('race-experiment').value;
  const metric = document.getElementById('race-metric').value;
  const visWeeks = WEEKS.slice(0, raceIdx + 1);
  const labels   = visWeeks.map(String);
  const isR2     = metric === 'r2';

  // Update week badge
  const wkBadge = document.getElementById('race-week-display');
  if (wkBadge) wkBadge.textContent = `${WEEKS[raceIdx]} week${WEEKS[raceIdx] > 1 ? 's' : ''}`;

  const strats = ['pretransfer', 'transfer', 'frozen'];
  const datasets = strats.map(s => {
    const rows = raceData.filter(r => r.experiment === exp && r.strategy === s);
    const data = visWeeks.map(w => {
      const row = rows.find(r => r.weeks === w);
      return row ? +row[metric] : null;
    });
    return {
      label: STRATEGIES[s].label,
      data,
      borderColor: STRATEGIES[s].color,
      backgroundColor: STRATEGIES[s].color,
      tension: 0.3, pointRadius: 5, pointHoverRadius: 7, borderWidth: 2.5,
    };
  });

  // Stat cards — current week values
  const currentWeek = WEEKS[raceIdx];
  const vals = {};
  strats.forEach(s => {
    const rows = raceData.filter(r => r.experiment === exp && r.strategy === s);
    const row = rows.find(r => r.weeks === currentWeek);
    vals[s] = row ? +row[metric] : null;
    const el = document.getElementById(`rs-${s}`);
    if (!el) return;
    const vEl = el.querySelector('.ls-value');
    if (vEl) vEl.textContent = vals[s] !== null ? vals[s].toFixed(isR2 ? 3 : 1) : '—';
    el.classList.remove('race-leader');
  });

  // Determine leader
  const validStrats = strats.filter(s => vals[s] !== null);
  if (validStrats.length) {
    const leader = isR2
      ? validStrats.reduce((a, b) => vals[a] > vals[b] ? a : b)
      : validStrats.reduce((a, b) => vals[a] < vals[b] ? a : b);
    const leaderEl = document.getElementById(`rs-${leader}`);
    if (leaderEl) leaderEl.classList.add('race-leader');
    const ldrCard = document.getElementById('rs-leader');
    if (ldrCard) {
      ldrCard.querySelector('.ls-value').textContent = STRATEGIES[leader].label;
      ldrCard.querySelector('.ls-value').style.color = STRATEGIES[leader].color;
    }
  }

  if (raceChart) {
    raceChart.data.labels = labels;
    raceChart.data.datasets = datasets;
    raceChart.options.scales.y.title.text = getMetricLabel(metric);
    raceChart.update();
  } else {
    raceChart = new Chart(document.getElementById('race-chart').getContext('2d'), {
      type: 'line',
      data: { labels, datasets },
      options: { ...baseOptions(getMetricLabel(metric)), scales: {
        x: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:'Training data (weeks)',font:{size:11}} },
        y: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:getMetricLabel(metric),font:{size:11}} },
      }},
    });
  }
  buildLegend('race-legend', strats.map(s => ({ label: STRATEGIES[s].label, color: STRATEGIES[s].color })));
}

// =====================================================================
// TAB 1 — Data Efficiency
// =====================================================================

let effData  = null;
let effChart = null;

async function initEfficiency() {
  effData = await parseCSV('data/all_data_efficiency.csv');
  renderEfficiency();
  document.getElementById('eff-experiment').addEventListener('change', renderEfficiency);
  document.getElementById('eff-metric').addEventListener('change', renderEfficiency);
  document.querySelectorAll('.eff-strat-cb').forEach(cb => cb.addEventListener('change', renderEfficiency));
}

function renderEfficiency() {
  const exp    = document.getElementById('eff-experiment').value;
  const metric = document.getElementById('eff-metric').value;
  const active = [...document.querySelectorAll('.eff-strat-cb:checked')].map(e => e.value);

  const datasets = active.map(s => {
    const rows = effData.filter(r => r.experiment === exp && r.strategy === s).sort((a, b) => a.weeks - b.weeks);
    return {
      label: STRATEGIES[s]?.label || s,
      data: rows.map(r => r[metric] !== undefined ? +r[metric] : null),
      borderColor: STRATEGIES[s]?.color || '#888',
      backgroundColor: STRATEGIES[s]?.color || '#888',
      tension: 0.3, pointRadius: 4, pointHoverRadius: 6, borderWidth: 2,
    };
  });

  if (effChart) {
    effChart.data.datasets = datasets;
    effChart.options.scales.y.title.text = getMetricLabel(metric);
    effChart.update();
  } else {
    effChart = new Chart(document.getElementById('efficiency-chart').getContext('2d'), {
      type: 'line',
      data: { labels: WEEKS.map(String), datasets },
      options: { ...baseOptions(getMetricLabel(metric)), scales: {
        x: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:'Training data (weeks)',font:{size:11}} },
        y: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:getMetricLabel(metric),font:{size:11}} },
      }},
    });
  }
  buildLegend('eff-legend', active.map(s => ({ label: STRATEGIES[s]?.label || s, color: STRATEGIES[s]?.color || '#888' })));
}

// =====================================================================
// TAB 2 — 8-Week Snapshot
// =====================================================================

let snapChart = null;

async function initSnapshot() {
  const data = await parseCSV('data/all_data_efficiency.csv');
  renderSnapshot(data, 'mae');
  document.getElementById('snap-metric').addEventListener('change', () =>
    renderSnapshot(data, document.getElementById('snap-metric').value)
  );
}

function renderSnapshot(data, metric) {
  const strats  = ['pretransfer', 'transfer', 'frozen'];
  const expKeys = Object.keys(EXPERIMENTS);
  const labels  = expKeys.map(k => EXPERIMENTS[k].replace(/ \(Exp \d+\)/, '').replace(' — Replication', ' (R)'));

  const datasets = strats.map(s => ({
    label: STRATEGIES[s].label,
    data: expKeys.map(exp => {
      const r = data.find(d => d.experiment === exp && d.strategy === s && d.weeks === 8);
      return r ? +r[metric] : null;
    }),
    backgroundColor: STRATEGIES[s].color + 'cc',
    borderColor: STRATEGIES[s].color,
    borderWidth: 1.5,
    borderRadius: 3,
  }));

  if (snapChart) {
    snapChart.data.datasets = datasets;
    snapChart.options.scales.y.title.text = getMetricLabel(metric);
    snapChart.update();
  } else {
    snapChart = new Chart(document.getElementById('snapshot-chart').getContext('2d'), {
      type: 'bar',
      data: { labels, datasets },
      options: { ...baseOptions(getMetricLabel(metric)), scales: {
        x: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}} },
        y: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:getMetricLabel(metric),font:{size:11}} },
      }},
    });
  }
  buildLegend('snap-legend', strats.map(s => ({ label: STRATEGIES[s].label, color: STRATEGIES[s].color })));
}

// =====================================================================
// TAB 3 — N-Source Ablation
// =====================================================================

let ablChart = null;

async function initAblation() {
  const ns = [1, 2, 3, 4, 5, 10, 15];
  const all = await Promise.all(
    ns.map(n => parseCSV(`${DATA_ROOT}multitransfer_ablation/data_efficiency_multitransfer_n${n}.csv`).then(d => ({ n, d })))
  );
  renderAblation(all, 'mae');
  document.getElementById('abl-metric').addEventListener('change', () =>
    renderAblation(all, document.getElementById('abl-metric').value)
  );
  document.querySelectorAll('.abl-week-cb').forEach(cb =>
    cb.addEventListener('change', () => renderAblation(all, document.getElementById('abl-metric').value))
  );
}

function renderAblation(all, metric) {
  const ns = [1, 2, 3, 4, 5, 10, 15];
  const activeWks = [...document.querySelectorAll('.abl-week-cb:checked')].map(e => +e.value);
  const wkColors  = { 4: '#0891b2', 8: '#d97706', 16: '#7c3aed', 32: '#059669' };

  const datasets = activeWks.map(wk => ({
    label: `${wk} wk`,
    data: ns.map(n => {
      const obj = all.find(a => a.n === n);
      if (!obj) return null;
      const row = obj.d.find(r => r.weeks === wk);
      return row ? +row[metric] : null;
    }),
    borderColor: wkColors[wk] || '#888', backgroundColor: wkColors[wk] || '#888',
    tension: 0.3, pointRadius: 4, pointHoverRadius: 6, borderWidth: 2,
  }));

  if (ablChart) {
    ablChart.data.datasets = datasets;
    ablChart.options.scales.y.title.text = getMetricLabel(metric);
    ablChart.update();
  } else {
    ablChart = new Chart(document.getElementById('ablation-chart').getContext('2d'), {
      type: 'line',
      data: { labels: ns.map(String), datasets },
      options: { ...baseOptions(getMetricLabel(metric)), scales: {
        x: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:'Number of source buildings (N)',font:{size:11}} },
        y: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:getMetricLabel(metric),font:{size:11}} },
      }},
    });
  }
  buildLegend('abl-legend', activeWks.map(wk => ({ label: `${wk} weeks`, color: wkColors[wk] || '#888' })));
}

// =====================================================================
// TAB 4 — Advanced: PRIME + Cross-Type
// =====================================================================

let primeChart = null;
let ctChart    = null;

async function initAdvanced() {
  const [primeData, ssData, stData, ctData] = await Promise.all([
    parseCSV('data/prime_comparison.csv'),
    parseCSV(`${DATA_ROOT}cross_type_transfer/data_efficiency_transfer_samesite.csv`),
    parseCSV(`${DATA_ROOT}cross_type_transfer/data_efficiency_transfer_sametype.csv`),
    parseCSV(`${DATA_ROOT}cross_type_transfer/data_efficiency_transfer_crosstype.csv`),
  ]);

  renderPrime(primeData, 'mae');
  renderCrossType({ ssData, stData, ctData }, 'mae');

  document.getElementById('prime-metric').addEventListener('change', () =>
    renderPrime(primeData, document.getElementById('prime-metric').value)
  );
  document.getElementById('ct-metric').addEventListener('change', () =>
    renderCrossType({ ssData, stData, ctData }, document.getElementById('ct-metric').value)
  );

  document.querySelectorAll('.sub-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.sub-tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.target).classList.add('active');
    });
  });
}

function renderPrime(data, metric) {
  const kP  = metric === 'mae' ? 'prime_mae'            : 'prime_rmse';
  const kS  = metric === 'mae' ? 'pretransfer_mae'      : 'pretransfer_rmse';
  const kSt = metric === 'mae' ? 'prime_streaming_mae'  : 'prime_streaming_rmse';
  const sorted = [...data].sort((a, b) => a.weeks - b.weeks);
  const labels = sorted.map(r => String(r.weeks));

  const streamingData = sorted.map(r => (r[kSt] !== undefined && r[kSt] !== '' && r[kSt] !== null) ? +r[kSt] : null);
  const hasStreaming = streamingData.some(v => v !== null);

  const datasets = [
    { label: 'PRIME (fine-tuned)', data: sorted.map(r => +r[kP]), borderColor: '#7c3aed', backgroundColor: '#7c3aed', tension: 0.3, pointRadius: 4, pointHoverRadius: 6, borderWidth: 2 },
    { label: 'Scratch',            data: sorted.map(r => +r[kS]), borderColor: '#6b7280', backgroundColor: '#6b7280', tension: 0.3, pointRadius: 4, pointHoverRadius: 6, borderWidth: 2, borderDash: [5,4] },
  ];
  if (hasStreaming) {
    datasets.push({ label: 'PRIME (streaming)', data: streamingData, borderColor: '#0891b2', backgroundColor: '#0891b2', tension: 0.3, pointRadius: 4, pointHoverRadius: 6, borderWidth: 1.8, borderDash: [3,2] });
  }

  if (primeChart) {
    primeChart.data.labels = labels; primeChart.data.datasets = datasets;
    primeChart.options.scales.y.title.text = getMetricLabel(metric); primeChart.update();
  } else {
    primeChart = new Chart(document.getElementById('prime-chart').getContext('2d'), {
      type: 'line', data: { labels, datasets },
      options: { ...baseOptions(getMetricLabel(metric)), scales: {
        x: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:'Training data (weeks)',font:{size:11}} },
        y: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:getMetricLabel(metric),font:{size:11}} },
      }},
    });
  }
  const legend = [{ label: 'PRIME (fine-tuned)', color: '#7c3aed' }, { label: 'Scratch', color: '#6b7280' }];
  if (hasStreaming) legend.push({ label: 'PRIME (streaming)', color: '#0891b2' });
  buildLegend('prime-legend', legend);
}

function renderCrossType({ ssData, stData, ctData }, metric) {
  const sort = arr => [...arr].sort((a, b) => a.weeks - b.weeks);
  const labels = sort(ssData).map(r => String(r.weeks));
  const series = [
    { label: 'Same-Site',  color: '#0891b2', data: sort(ssData) },
    { label: 'Same-Type',  color: '#d97706', data: sort(stData) },
    { label: 'Cross-Type', color: '#dc2626', data: sort(ctData) },
  ];
  const datasets = series.map(s => ({
    label: s.label, data: s.data.map(r => r[metric] !== undefined ? +r[metric] : null),
    borderColor: s.color, backgroundColor: s.color, tension: 0.3, pointRadius: 4, pointHoverRadius: 6, borderWidth: 2,
  }));

  if (ctChart) {
    ctChart.data.labels = labels; ctChart.data.datasets = datasets;
    ctChart.options.scales.y.title.text = getMetricLabel(metric); ctChart.update();
  } else {
    ctChart = new Chart(document.getElementById('crosstype-chart').getContext('2d'), {
      type: 'line', data: { labels, datasets },
      options: { ...baseOptions(getMetricLabel(metric)), scales: {
        x: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:'Training data (weeks)',font:{size:11}} },
        y: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:getMetricLabel(metric),font:{size:11}} },
      }},
    });
  }
  buildLegend('ct-legend', series.map(s => ({ label: s.label, color: s.color })));
}

// =====================================================================
// TAB 5 — Switch Policy
// =====================================================================

let switchChart = null;

async function initSwitch() {
  const data = await parseCSV('data/switch_results.csv');
  const sorted = [...data].sort((a, b) => a.weeks - b.weeks);
  const labels  = sorted.map(r => String(r.weeks));
  const oracle  = sorted.map(r => Math.min(+r.pretransfer_mae, +r.transfer_mae));

  const datasets = [
    { label: 'Scratch',       data: sorted.map(r => +r.pretransfer_mae), borderColor: '#6b7280', backgroundColor: '#6b7280', tension: 0.3, pointRadius: 4, borderWidth: 2, borderDash: [4,3] },
    { label: 'Full FT',       data: sorted.map(r => +r.transfer_mae),    borderColor: '#2563eb', backgroundColor: '#2563eb', tension: 0.3, pointRadius: 4, borderWidth: 2, borderDash: [4,3] },
    { label: 'Auto-Switch',   data: oracle,                               borderColor: '#059669', backgroundColor: '#059669', tension: 0.3, pointRadius: 6, borderWidth: 2.5 },
  ];

  switchChart = new Chart(document.getElementById('switch-chart').getContext('2d'), {
    type: 'line', data: { labels, datasets },
    options: { ...baseOptions('MAE (kWh)'), scales: {
      x: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:'Training data (weeks)',font:{size:11}} },
      y: { grid:{color:'rgba(15,23,42,.04)'}, ticks:{font:{size:11}}, title:{display:true,text:'MAE (kWh)',font:{size:11}} },
    }},
  });
  buildLegend('switch-legend', [
    { label: 'Scratch', color: '#6b7280' },
    { label: 'Full Fine-Tuning', color: '#2563eb' },
    { label: 'Auto-Switch (Oracle)', color: '#059669' },
  ]);
}

// =====================================================================
// Tab Navigation
// =====================================================================

const TAB_INITS = {
  'tab-efficiency': initEfficiency,
  'tab-snapshot':   initSnapshot,
  'tab-ablation':   initAblation,
  'tab-advanced':   initAdvanced,
  'tab-switch':     initSwitch,
  'tab-methodology': () => { console.log('Methodology tab loaded'); },
};
const tabLoaded = {};

function switchTab(id) {
  document.querySelectorAll('.tab-btn').forEach(b  => b.classList.toggle('active',  b.dataset.tab === id));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id          === id));
  if (!tabLoaded[id] && TAB_INITS[id]) { tabLoaded[id] = true; TAB_INITS[id](); }
}

// =====================================================================
// Gallery Lightbox
// =====================================================================

function initGallery() {
  const lb    = document.getElementById('lightbox');
  const lbImg = document.getElementById('lightbox-img');
  const lbCap = document.getElementById('lightbox-caption');

  document.querySelectorAll('.gallery-item').forEach(item => {
    item.addEventListener('click', () => {
      lbImg.src = item.dataset.src;
      lbCap.textContent = item.dataset.caption;
      lb.classList.add('open');
      document.body.style.overflow = 'hidden';
    });
  });

  function closeLb() { lb.classList.remove('open'); document.body.style.overflow = ''; lbImg.src = ''; }
  document.getElementById('lightbox-close').addEventListener('click', closeLb);
  lb.addEventListener('click', e => { if (e.target === lb) closeLb(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLb(); });
}

// =====================================================================
// Boot
// =====================================================================

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-btn').forEach(btn =>
    btn.addEventListener('click', () => switchTab(btn.dataset.tab))
  );
  switchTab('tab-efficiency');
  initGallery();
  initLiveInference();
  initEfficiencyRace();
});

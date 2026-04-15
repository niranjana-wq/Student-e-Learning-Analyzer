/**
 * app.js – Frontend logic for AttentionAI Dashboard
 *
 * Connects to:
 *  GET /data_stream  → Server-Sent Events (JSON metrics per frame)
 *  GET /video_feed   → MJPEG stream
 *  POST /control     → { action: "start"|"stop" }
 */

// ── DOM refs ───────────────────────────────────────────────────
const el = id => document.getElementById(id);

const livePill     = el('livePill');
const livePillText = el('livePillText');
const startBtn     = el('startBtn');
const stopBtn      = el('stopBtn');

const videoFeed    = el('videoFeed');
const videoOffline = el('videoOffline');
const camIndicator = el('camIndicator');
const camDot       = camIndicator.querySelector('.cam-dot');
const camLabel     = el('camLabel');

const statusCard   = el('statusCard');
const ringFill     = el('ringFill');
const statusEmoji  = el('statusEmoji');
const statusLabel  = el('statusLabel');
const confPct      = el('confPct');

const chipGaze  = el('chipGaze');
const chipBlink = el('chipBlink');
const chipYawn  = el('chipYawn');
const chipMove  = el('chipMove');
const gazeDir   = el('gazeDir');
const blinkStat = el('blinkStat');
const yawnStat  = el('yawnStat');
const moveStat  = el('moveStat');

const timerFill = el('timerFill');
const timerNums = el('timerNums');

const sessionTime  = el('sessionTime');
const attentivePct = el('attentivePct');
const alertCount   = el('alertCount');
const frameCount   = el('frameCount');

const earVal    = el('earVal');    const earBar    = el('earBar');    const earStatus = el('earStatus');
const marVal    = el('marVal');    const marBar    = el('marBar');    const marStatus = el('marStatus');
const gazeVal   = el('gazeVal');   const gazeBar   = el('gazeBar');   const gazeStatus = el('gazeStatus');
const ebVal     = el('ebVal');     const ebBar     = el('ebBar');     const ebStatus = el('ebStatus');
const yawVal    = el('yawVal');
const pitchVal  = el('pitchVal');
const rollVal   = el('rollVal');

// ── Ring math ─────────────────────────────────────────────────
const RING_CIRCUMFERENCE = 2 * Math.PI * 50; // r=50 → ~314.16

function setRing(pct, colorClass) {
  const offset = RING_CIRCUMFERENCE * (1 - pct / 100);
  ringFill.style.strokeDashoffset = offset;
  ringFill.className = 'ring-fill ' + (colorClass || '');
}

// ── Chart setup ───────────────────────────────────────────────
const MAX_CHART_POINTS = 120;  // 120 × 0.5s ≈ 60 s

const chartCtx = el('attentionChart').getContext('2d');
const chartData = {
  labels: [],
  datasets: [{
    label: 'Confidence %',
    data: [],
    borderColor: 'hsl(232,90%,68%)',
    backgroundColor: 'hsla(232,90%,68%,.08)',
    borderWidth: 2,
    pointRadius: 0,
    fill: true,
    tension: 0.4,
  }]
};

const attentionChart = new Chart(chartCtx, {
  type: 'line',
  data: chartData,
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => `${ctx.parsed.y.toFixed(1)} %`
        }
      }
    },
    scales: {
      x: {
        display: false,
        grid: { display: false }
      },
      y: {
        min: 0, max: 100,
        grid: { color: 'hsla(240,10%,20%,.6)' },
        ticks: {
          color: 'hsl(240,8%,42%)',
          font: { size: 11, family: 'Inter' },
          callback: v => v + '%',
          maxTicksLimit: 5
        },
        border: { display: false }
      }
    }
  }
});

function pushChartPoint(confidence, statusStr) {
  const now = new Date().toLocaleTimeString('en', { hour12: false });
  chartData.labels.push(now);

  // Color the data point based on status
  const color = statusStr === 'ATTENTIVE'   ? 'hsl(142,72%,50%)' :
                statusStr === 'INATTENTIVE' ? 'hsl(4,85%,60%)'   :
                'hsl(232,90%,68%)';
  chartData.datasets[0].borderColor = color;
  chartData.datasets[0].backgroundColor = color.replace(')', ',.08)').replace('hsl', 'hsla');
  chartData.datasets[0].data.push(confidence);

  if (chartData.labels.length > MAX_CHART_POINTS) {
    chartData.labels.shift();
    chartData.datasets[0].data.shift();
  }
  attentionChart.update('none');
}

// ── Session timer ─────────────────────────────────────────────
function formatTime(secs) {
  const m = Math.floor(secs / 60).toString().padStart(2, '0');
  const s = Math.floor(secs % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

// ── Clamp helper ──────────────────────────────────────────────
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ── Apply metrics to UI ────────────────────────────────────────
let prevStatus = '';

function applyMetrics(m) {
  const status = m.status || '---';
  const conf   = m.confidence || 0;

  // ── Status card ──
  const isAtt  = status === 'ATTENTIVE';
  const isInat = status === 'INATTENTIVE';
  const isNF   = status === 'NO FACE';

  statusCard.classList.toggle('is-attentive',   isAtt);
  statusCard.classList.toggle('is-inattentive',  isInat || isNF);

  let ringColor = '';
  let labelColor = '';
  let emoji = '😴';

  if (isAtt)       { ringColor = 'green';  labelColor = 'green';  emoji = '🎯'; }
  else if (isInat) { ringColor = 'red';    labelColor = 'red';    emoji = '😴'; }
  else if (isNF)   { ringColor = 'red';    labelColor = 'red';    emoji = '👤'; }
  else             { ringColor = 'yellow'; labelColor = 'yellow'; emoji = '⏳'; }

  setRing(conf, ringColor);
  statusEmoji.textContent = emoji;
  statusLabel.textContent = status;
  statusLabel.className   = 'status-label ' + labelColor;
  confPct.textContent     = isAtt || isInat ? conf.toFixed(1) + ' %' : '— %';

  // ── Alert timer bar ──
  const ic  = m.inattentive_counter  || 0;
  const ith = m.inattentive_threshold || 45;
  const pct = clamp((ic / ith) * 100, 0, 100);
  timerFill.style.width = pct + '%';
  timerNums.textContent = `${ic} / ${ith}`;

  // ── Detail chips ──
  setChip(chipGaze,  gazeDir,  m.gaze_dir  || '—', m.gaze_dir  === 'CENTER', m.gaze_dir !== 'CENTER' && m.gaze_dir !== '---');
  setChip(chipBlink, blinkStat, m.blink    || '—', m.blink === 'Open',       m.blink === 'BLINKING');
  setChip(chipYawn,  yawnStat,  m.yawn     === 'YES' ? 'Yawning' : m.yawn === 'NO' ? 'No Yawn' : '—', m.yawn === 'NO', m.yawn === 'YES');
  setChip(chipMove,  moveStat,  m.head_move || '—', m.head_move === '---', false);

  // ── Metric cards ──
  // EAR  (higher = more open, threshold 0.22)
  const earPct = clamp(m.ear * 200, 0, 100);
  earVal.textContent   = m.ear.toFixed(3);
  earBar.style.width   = earPct + '%';
  earStatus.textContent = m.blink === 'BLINKING' ? '⚠ Eyes Closed / Blinking' :
                           m.ear > 0 ? 'Eyes Open' : '—';
  earBar.style.background = m.blink === 'BLINKING' ? 'var(--red)' : 'var(--accent)';

  // MAR (threshold 0.45)
  const marPct = clamp(m.mar * 150, 0, 100);
  marVal.textContent   = m.mar.toFixed(3);
  marBar.style.width   = marPct + '%';
  marStatus.textContent = m.yawn === 'YES' ? '⚠ Yawning Detected' :
                           m.mar > 0 ? 'Mouth Closed' : '—';
  marBar.style.background = m.yawn === 'YES' ? 'var(--red)' : 'var(--purple)';

  // Gaze (center ~1.0)
  const gazeOff = Math.abs(m.gaze_ratio - 1.0);
  const gazePct = clamp(100 - gazeOff * 80, 0, 100);
  gazeVal.textContent  = m.gaze_ratio.toFixed(3);
  gazeBar.style.width  = gazePct + '%';
  gazeStatus.textContent = m.gaze_dir && m.gaze_dir !== '---' ? `Looking ${m.gaze_dir}` : '—';
  gazeBar.style.background = m.gaze_dir === 'CENTER' ? 'var(--cyan)' : 'var(--red)';

  // Eyebrow (threshold 0.11)
  const ebPct = clamp(m.eb_ratio ? m.eb_ratio * 500 : 0, 0, 100);
  ebVal.textContent  = (m.eb_ratio || 0).toFixed(3);
  ebBar.style.width  = ebPct + '%';
  ebStatus.textContent = m.eyebrow === 'RAISED' ? '⚠ Eyebrows Raised' : m.eyebrow === 'Normal' ? 'Normal' : '—';

  // Head pose
  yawVal.textContent   = `${m.yaw !== undefined ? m.yaw.toFixed(1) : '—'}°`;
  pitchVal.textContent = `${m.pitch !== undefined ? m.pitch.toFixed(1) : '—'}°`;
  rollVal.textContent  = `${m.roll !== undefined ? m.roll.toFixed(1) : '—'}°`;

  // Pose colors
  const yawBad = Math.abs(m.yaw || 0) > 20;
  const pitchBad = (m.pitch || 0) < -20;
  yawVal.style.color   = yawBad   ? 'var(--red)' : 'var(--text-1)';
  pitchVal.style.color = pitchBad ? 'var(--red)' : 'var(--text-1)';

  // ── Session stats ──
  sessionTime.textContent  = formatTime(m.session_elapsed || 0);
  attentivePct.textContent = m.attentive_pct != null ? m.attentive_pct.toFixed(1) + ' %' : '—';
  alertCount.textContent   = m.alert_count != null ? m.alert_count : '0';
  frameCount.textContent   = m.total_frames != null ? m.total_frames.toLocaleString() : '—';

  // ── Chart ──
  pushChartPoint(conf, status);
}

function setChip(chipEl, textEl, text, isGood, isWarn) {
  textEl.textContent = text;
  chipEl.classList.toggle('active', !!isGood);
  chipEl.classList.toggle('warn',   !!isWarn);
}

// ── SSE connection ─────────────────────────────────────────────
let evtSource = null;

function connectSSE() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/data_stream');

  evtSource.onmessage = e => {
    try {
      const m = JSON.parse(e.data);
      applyMetrics(m);
    } catch (_) {}
  };

  evtSource.onerror = () => {
    // SSE will auto-reconnect; don't show error unless session is stopped
  };
}

function disconnectSSE() {
  if (evtSource) { evtSource.close(); evtSource = null; }
}

// ── Session control ────────────────────────────────────────────
let running = false;

async function controlAnalyzer(action) {
  try {
    const res = await fetch('/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action })
    });
    const data = await res.json();
    if (data.ok) {
      running = action === 'start';
      updateUIState(running);
    }
  } catch(e) {
    console.error('Control error:', e);
  }
}

function updateUIState(isRunning) {
  running = isRunning;

  startBtn.disabled = isRunning;
  stopBtn.disabled  = !isRunning;

  if (isRunning) {
    // Live pill
    livePill.classList.add('live');
    livePillText.textContent = 'LIVE';

    // Camera
    camDot.classList.add('active');
    camLabel.textContent = 'Camera On';

    // Video
    videoOffline.classList.add('hidden');
    videoFeed.classList.remove('hidden');
    videoFeed.src = '/video_feed?' + Date.now();

    // SSE
    connectSSE();

    // Clear old chart
    chartData.labels = [];
    chartData.datasets[0].data = [];
    attentionChart.update('none');

  } else {
    livePill.classList.remove('live');
    livePillText.textContent = 'OFFLINE';

    camDot.classList.remove('active');
    camLabel.textContent = 'Camera Off';

    videoFeed.src = '';
    videoFeed.classList.add('hidden');
    videoOffline.classList.remove('hidden');

    disconnectSSE();

    // Reset ring
    setRing(0, '');
    statusLabel.textContent = 'OFFLINE';
    statusLabel.className   = 'status-label';
    statusEmoji.textContent = '😴';
    confPct.textContent     = '— %';
    statusCard.classList.remove('is-attentive', 'is-inattentive');

    timerFill.style.width = '0%';
    timerNums.textContent = '0 / 45';
  }
}

// ── On load ───────────────────────────────────────────────────
updateUIState(false);

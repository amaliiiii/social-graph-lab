"use strict";
/* Лаборатория графов взаимодействий: клиентская часть без внешних библиотек.
   Раскладка вершин: силовая схема Фрухтермана и Рейнгольда, считается в браузере. */

const $ = (s) => document.querySelector(s);
const fmtNum = (v) => {
  if (v === null || v === undefined) return "не вычислено";
  return Number.isInteger(v) ? String(v) : v.toLocaleString("ru-RU", { maximumFractionDigits: 4 });
};
const C = { blue: "#4c79a6", bluePale: "#a9bdd0", ink: "#16202a", signal: "#c21e6b", signalSoft: "#f0b3d0" };

const state = {
  source: { dataset: "demo-social" },
  data: null,           // ответ /api/analyze
  pos: [],              // положения вершин в условных координатах 0..1
  metric: "pagerank",
  topK: 5,
  seeds: new Set(),     // метки начальных вершин
  steps: null,          // ответ /api/cascade
  stepOf: new Map(),    // метка -> номер шага активации (для показанных вершин)
  stepAt: 0,
  run: 0,
  timer: null,
  hover: -1,
  quiz: null,
  view: { x: 0, y: 0, w: 1, h: 1, dpr: 1 },
};

/* ---------------------------------------------------------------- сеть */
async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  let json = {};
  try { json = await res.json(); } catch (e) { /* пустой ответ */ }
  if (!res.ok) throw new Error(typeof json.detail === "string" ? json.detail : `Ошибка ${res.status}`);
  return json;
}
const setStatus = (t) => { $("#status").textContent = t; };

/* ------------------------------------------------------------ раскладка */
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function computeLayout(n, edges) {
  const rnd = mulberry32(42);
  const W = 1000, H = 700;
  const p = Array.from({ length: n }, () => ({ x: W / 2 + (rnd() - 0.5) * W * 0.8, y: H / 2 + (rnd() - 0.5) * H * 0.8, dx: 0, dy: 0 }));
  if (n < 2) return p.map(() => ({ x: 0.5, y: 0.5 }));
  const k = 0.9 * Math.sqrt((W * H) / n);
  let t = W / 8;
  const iters = n > 200 ? 180 : 300;
  for (let it = 0; it < iters; it++) {
    for (const q of p) { q.dx = 0; q.dy = 0; }
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = p[i].x - p[j].x, dy = p[i].y - p[j].y;
        let d = Math.hypot(dx, dy);
        if (d < 0.01) { dx = rnd() - 0.5; dy = rnd() - 0.5; d = 0.5; }
        const f = (k * k) / d / d;
        p[i].dx += dx * f; p[i].dy += dy * f;
        p[j].dx -= dx * f; p[j].dy -= dy * f;
      }
    }
    for (const [a, b] of edges) {
      const dx = p[a].x - p[b].x, dy = p[a].y - p[b].y;
      const d = Math.max(Math.hypot(dx, dy), 0.01);
      const f = d / k;
      p[a].dx -= dx * f; p[a].dy -= dy * f;
      p[b].dx += dx * f; p[b].dy += dy * f;
    }
    for (const q of p) {
      q.dx += (W / 2 - q.x) * 0.02; q.dy += (H / 2 - q.y) * 0.02;   // слабое притяжение к центру
      const d = Math.max(Math.hypot(q.dx, q.dy), 0.01);
      const m = Math.min(d, t);
      q.x += (q.dx / d) * m; q.y += (q.dy / d) * m;
    }
    t *= 0.985;
  }
  const xs = p.map((q) => q.x), ys = p.map((q) => q.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  return p.map((q) => ({ x: (q.x - x0) / Math.max(x1 - x0, 1), y: (q.y - y0) / Math.max(y1 - y0, 1) }));
}

/* ------------------------------------------------------------ вспомогательное */
function metricValues() {
  return state.data.nodes.map((nd) => nd.m[state.metric]);
}
function rankedIndices() {
  const vals = metricValues();
  return vals.map((v, i) => i)
    .filter((i) => vals[i] !== null)
    .sort((a, b) => vals[b] - vals[a] || a - b);
}
function topSet() {
  return new Set(rankedIndices().slice(0, state.topK));
}
function radii() {
  const vals = metricValues().filter((v) => v !== null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  return metricValues().map((v) => {
    if (v === null) return 6;
    const u = hi > lo ? (v - lo) / (hi - lo) : 0.4;
    return 4.5 + 15 * Math.sqrt(u);
  });
}
const labelIndex = () => new Map(state.data.nodes.map((nd, i) => [nd.label, i]));

/* ------------------------------------------------------------ рисование */
function resizeCanvas() {
  const cv = $("#canvas");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  state.view = { x: 0, y: 0, w, h, dpr };
}
function toScreen(q) {
  const { w, h } = state.view, m = 46;
  return { x: m + q.x * (w - 2 * m), y: m + q.y * (h - 2 * m) };
}

function nodeStyle(i, label) {
  const active = state.steps !== null && state.stepOf.has(label) && state.stepOf.get(label) <= state.stepAt;
  if (state.steps !== null) {
    if (active) {
      const s = state.stepOf.get(label);
      if (s === 0) return { fill: C.signal, ring: C.ink };
      return { fill: s === state.stepAt ? C.signal : C.signalSoft, ring: s === state.stepAt ? C.ink : C.signal };
    }
    return { fill: C.bluePale, ring: null };
  }
  if (state.seeds.has(label)) return { fill: C.signal, ring: C.ink };
  return { fill: C.blue, ring: null };
}

function draw() {
  const cv = $("#canvas"), ctx = cv.getContext("2d");
  const { w, h, dpr } = state.view;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!state.data) return;
  const nodes = state.data.nodes;
  const sp = state.pos.map(toScreen);
  const r = radii();
  const top = topSet();
  const directed = state.data.directed;

  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(22,32,42,0.26)";
  ctx.fillStyle = "rgba(22,32,42,0.4)";
  for (const [a, b, wt] of state.data.edges) {
    const A = sp[a], B = sp[b];
    const dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy) || 1;
    const ux = dx / d, uy = dy / d;
    const ex = B.x - ux * (r[b] + 1), ey = B.y - uy * (r[b] + 1);
    ctx.lineWidth = Math.min(1 + Math.log(Math.max(wt, 1)), 3);
    ctx.beginPath(); ctx.moveTo(A.x + ux * r[a], A.y + uy * r[a]); ctx.lineTo(ex, ey); ctx.stroke();
    if (directed && nodes.length <= 400) {
      const L = 7, wd = 3;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - ux * L - uy * wd, ey - uy * L + ux * wd);
      ctx.lineTo(ex - ux * L + uy * wd, ey - uy * L - ux * wd);
      ctx.closePath(); ctx.fill();
    }
  }
  nodes.forEach((nd, i) => {
    const st = nodeStyle(i, nd.label);
    ctx.beginPath(); ctx.arc(sp[i].x, sp[i].y, r[i], 0, Math.PI * 2);
    ctx.fillStyle = st.fill; ctx.fill();
    if (top.has(i)) {                       // лучшие по мере: тёмное кольцо снаружи
      ctx.beginPath(); ctx.arc(sp[i].x, sp[i].y, r[i] + 3.5, 0, Math.PI * 2);
      ctx.lineWidth = 2.5; ctx.strokeStyle = C.ink; ctx.stroke();
    }
    if (st.ring) { ctx.lineWidth = 1.5; ctx.strokeStyle = st.ring; ctx.beginPath(); ctx.arc(sp[i].x, sp[i].y, r[i], 0, Math.PI * 2); ctx.stroke(); }
    if (i === state.hover) { ctx.lineWidth = 2; ctx.strokeStyle = "#f2a900"; ctx.beginPath(); ctx.arc(sp[i].x, sp[i].y, r[i] + 6, 0, Math.PI * 2); ctx.stroke(); }
  });
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
  nodes.forEach((nd, i) => {
    if (nodes.length <= 40 || top.has(i) || state.seeds.has(nd.label)) {
      ctx.lineWidth = 3; ctx.strokeStyle = "#f7f8f5"; ctx.strokeText(nd.label, sp[i].x, sp[i].y - r[i] - 6);
      ctx.fillStyle = C.ink; ctx.fillText(nd.label, sp[i].x, sp[i].y - r[i] - 6);
    }
  });
}

/* ---------------------------------------------------------- интерфейс */
function renderSeeds() {
  const ul = $("#seedList");
  ul.innerHTML = "";
  for (const lab of state.seeds) {
    const li = document.createElement("li");
    li.append(lab);
    const b = document.createElement("button");
    b.type = "button"; b.textContent = "×"; b.setAttribute("aria-label", `Убрать ${lab} из начальных вершин`);
    b.addEventListener("click", () => toggleSeed(lab));
    li.append(b); ul.append(li);
  }
}
function renderTable() {
  const body = $("#tableBody");
  body.innerHTML = "";
  const meta = state.data.metrics.find((m) => m.id === state.metric);
  const mode = state.data.modes[state.metric];
  $("#tableCaption").textContent = `Лучшие по мере «${meta.label}»` + (mode === "sampled" ? " (оценка по выборке)" : "");
  const vals = metricValues();
  const ranked = rankedIndices().slice(0, 10);
  if (!ranked.length) {
    body.innerHTML = '<tr><td colspan="4">Эта мера для данного графа не вычислялась: граф слишком велик.</td></tr>';
    return;
  }
  ranked.forEach((i, pos) => {
    const lab = state.data.nodes[i].label;
    const tr = document.createElement("tr");
    if (state.seeds.has(lab)) tr.className = "seed";
    tr.innerHTML = `<td>${pos + 1}</td><td></td><td class="num">${fmtNum(vals[i])}</td><td></td>`;
    tr.children[1].textContent = lab;
    const btn = document.createElement("button");
    btn.type = "button"; btn.className = "quiet";
    btn.textContent = state.seeds.has(lab) ? "убрать" : "добавить";
    btn.addEventListener("click", () => toggleSeed(lab));
    tr.children[3].append(btn);
    body.append(tr);
  });
}
function renderMetricHint() {
  const m = state.data.metrics.find((x) => x.id === state.metric);
  $("#metricHint").textContent = m.hint;
}
function describeGraph() {
  const d = state.data;
  let t = `${d.n} вершин, ${d.m} рёбер, ${d.directed ? "ориентированный" : "неориентированный"} граф.`;
  if (d.shown < d.n) t += ` На рисунке ${d.shown} вершин с наибольшим PageRank; меры и каскады считаются по всему графу.`;
  if (d.modes.betweenness === "sampled") t += " Близость и посредничество оценены по выборке начальных вершин.";
  if (d.modes.betweenness === "skipped") t += " Близость и посредничество для такого графа не вычисляются.";
  $("#graphInfo").textContent = t;
  $("#canvas").setAttribute("aria-label", `Граф взаимодействий: ${d.n} вершин, ${d.m} рёбер. Подробности в таблице лучших вершин.`);
}
function refresh() {
  if (!state.data) return;
  renderSeeds(); renderTable(); draw();
}

function toggleSeed(label) {
  if (state.seeds.has(label)) state.seeds.delete(label); else state.seeds.add(label);
  resetCascade();
  refresh();
}
function resetCascade() {
  stopPlay();
  state.steps = null; state.stepOf = new Map(); state.stepAt = 0;
  $("#stepSlider").disabled = true; $("#stepSlider").max = 0; $("#stepSlider").value = 0; $("#stepOut").textContent = "0";
  $("#playCascade").disabled = true;
  $("#cascadeInfo").textContent = "";
}

/* -------------------------------------------------------------- загрузка */
async function loadGraph() {
  setStatus("Загрузка и расчёт мер…");
  try {
    const data = await api("/api/analyze", { source: state.source, max_nodes: 150 });
    state.data = data;
    state.seeds = new Set();
    resetCascade();
    const ms = $("#metric");
    if (!ms.options.length) {
      data.metrics.forEach((m) => ms.add(new Option(m.label, m.id)));
      ms.value = state.metric;
    }
    if (data.modes.betweenness === "skipped" && ["closeness", "betweenness"].includes(state.metric)) {
      state.metric = "pagerank"; ms.value = "pagerank";
    }
    state.pos = computeLayout(data.nodes.length, data.edges.map((e) => [e[0], e[1]]));
    $("#topK").max = Math.min(20, data.nodes.length);
    state.topK = Math.min(state.topK, data.nodes.length);
    $("#topK").value = state.topK; $("#topKOut").textContent = state.topK;
    describeGraph(); renderMetricHint(); resizeCanvas(); refresh(); newQuestion();
    $("#compareInfo").textContent = ""; $("#quizRes").textContent = "";
    setStatus("Граф загружен.");
  } catch (e) {
    setStatus("Ошибка: " + e.message);
  }
}

/* -------------------------------------------------------------- каскад */
async function runCascade() {
  if (!state.seeds.size) { $("#cascadeInfo").textContent = "Сначала выберите начальные вершины."; return; }
  stopPlay();
  try {
    const res = await api("/api/cascade", {
      source: state.source, seeds: [...state.seeds], model: $("#model").value,
      p: parseFloat($("#prob").value), run: state.run++, max_nodes: 150,
    });
    state.steps = res;
    state.stepOf = new Map();
    res.steps.forEach((st, t) => st.nodes.forEach((lab) => state.stepOf.set(lab, t)));
    state.stepAt = res.steps.length - 1;
    const sl = $("#stepSlider");
    sl.max = res.steps.length - 1; sl.value = state.stepAt; sl.disabled = res.steps.length < 2;
    $("#playCascade").disabled = res.steps.length < 2;
    updateCascadeInfo(); draw();
  } catch (e) {
    $("#cascadeInfo").textContent = "Ошибка: " + e.message;
  }
}
function updateCascadeInfo() {
  const s = state.steps;
  if (!s) return;
  let active = 0;
  for (let t = 0; t <= state.stepAt; t++) active += s.steps[t].new;
  $("#stepOut").textContent = String(state.stepAt);
  $("#cascadeInfo").textContent =
    `Шаг ${state.stepAt} из ${s.steps.length - 1}: узнали ${active} из ${s.n}. Итог этого запуска: ${s.total}. ` +
    "Повторный запуск берёт новый случайный мир.";
}
function setStep(t) {
  state.stepAt = t; updateCascadeInfo(); draw();
}
function stopPlay() {
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  $("#playCascade").textContent = "Воспроизвести";
}
function togglePlay() {
  if (state.timer) { stopPlay(); return; }
  if (!state.steps) return;
  const last = state.steps.steps.length - 1;
  if (state.stepAt >= last) { state.stepAt = 0; }
  $("#playCascade").textContent = "Пауза";
  $("#stepSlider").value = state.stepAt; setStep(state.stepAt);
  state.timer = setInterval(() => {
    if (state.stepAt >= last) { stopPlay(); return; }
    $("#stepSlider").value = state.stepAt + 1; setStep(state.stepAt + 1);
  }, 750);
}

async function runCompare() {
  const btn = $("#compare");
  btn.disabled = true; $("#compareInfo").textContent = "Считаю…";
  try {
    const r = await api("/api/compare", {
      source: state.source, metric: state.metric, k: state.topK,
      model: $("#model").value, p: parseFloat($("#prob").value), runs: 100,
    });
    const f = (x) => x.toLocaleString("ru-RU", { maximumFractionDigits: 1 });
    const label = state.data.metrics.find((m) => m.id === state.metric).label;
    $("#compareInfo").textContent =
      `Лучшие ${r.k} по мере «${label}»: в среднем ${f(r.top.mean)} ± ${f(r.top.ci)}. ` +
      `Случайные ${r.k}: ${f(r.random.mean)} ± ${f(r.random.ci)}. ` +
      `Разность ${f(r.gain.mean)} ± ${f(r.gain.ci)}` + (r.ratio ? `, отношение ${f(r.ratio)}.` : ".") +
      ` Запусков: ${r.runs}; интервал 95 %.`;
  } catch (e) {
    $("#compareInfo").textContent = "Ошибка: " + e.message;
  } finally { btn.disabled = false; }
}

/* ------------------------------------------------------------ тренажёр */
const QUIZ_METRICS = ["in_degree", "out_degree", "closeness", "betweenness", "pagerank"];
function newQuestion() {
  if (!state.data) return;
  const d = state.data;
  const ok = QUIZ_METRICS.filter((m) => d.nodes[0].m[m] !== null);
  if (d.n > 400 || !ok.length) { $("#quizQ").textContent = "Тренажёр работает на графах до 400 вершин."; state.quiz = null; return; }
  const node = d.nodes[Math.floor(Math.random() * d.nodes.length)].label;
  const metric = ok[Math.floor(Math.random() * ok.length)];
  state.quiz = { node, metric };
  const label = d.metrics.find((m) => m.id === metric).label;
  $("#quizQ").textContent = `Чему равна мера «${label}» у вершины ${node}? Подсказка: наведите курсор на вершину.`;
  $("#quizAns").value = ""; $("#quizRes").textContent = "";
}
async function checkQuiz() {
  if (!state.quiz) return;
  const guess = parseFloat($("#quizAns").value);
  if (Number.isNaN(guess)) { $("#quizRes").textContent = "Введите число."; return; }
  try {
    const r = await api("/api/quiz", { source: state.source, node: state.quiz.node, metric: state.quiz.metric, guess });
    $("#quizRes").textContent = r.correct ? `Верно: ${fmtNum(r.value)}.` : `Неверно. Правильное значение: ${fmtNum(r.value)}.`;
  } catch (e) { $("#quizRes").textContent = "Ошибка: " + e.message; }
}

/* ------------------------------------------------------- мышь и подсказка */
function pick(ev) {
  const rect = $("#canvas").getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const r = radii();
  let best = -1, bd = 1e9;
  state.pos.forEach((q, i) => {
    const s = toScreen(q), d = Math.hypot(s.x - x, s.y - y);
    if (d < r[i] + 5 && d < bd) { bd = d; best = i; }
  });
  return { i: best, x, y };
}
function onMove(ev) {
  if (!state.data) return;
  const { i, x, y } = pick(ev);
  const tip = $("#tip");
  if (i !== state.hover) { state.hover = i; draw(); }
  if (i < 0) { tip.hidden = true; return; }
  const nd = state.data.nodes[i];
  tip.innerHTML = "";
  const b = document.createElement("b"); b.textContent = nd.label; tip.append(b);
  state.data.metrics.forEach((m) => {
    tip.append(document.createElement("br"), `${m.label}: ${fmtNum(nd.m[m.id])}`);
  });
  tip.hidden = false;
  const wrap = $(".canvasWrap").getBoundingClientRect();
  tip.style.left = Math.min(x + 14, wrap.width - 270) + "px";
  tip.style.top = Math.max(y - 10, 4) + "px";
}
function onClick(ev) {
  if (!state.data) return;
  const { i } = pick(ev);
  if (i >= 0) toggleSeed(state.data.nodes[i].label);
}

/* ----------------------------------------------------------------- старт */
async function init() {
  try {
    const src = await api("/api/sources");
    const sel = $("#dataset");
    src.datasets.forEach((d) => sel.add(new Option(d.label, d.id)));
    sel.value = state.source.dataset;
  } catch (e) { setStatus("Не удалось получить список наборов: " + e.message); }

  $("#dataset").addEventListener("change", (e) => { state.source = { dataset: e.target.value }; loadGraph(); });
  $("#metric").addEventListener("change", (e) => { state.metric = e.target.value; renderMetricHint(); refresh(); });
  $("#topK").addEventListener("input", (e) => { state.topK = +e.target.value; $("#topKOut").textContent = e.target.value; refresh(); });
  $("#prob").addEventListener("input", (e) => { $("#probOut").textContent = (+e.target.value).toLocaleString("ru-RU", { minimumFractionDigits: 2 }); });
  $("#model").addEventListener("change", (e) => { $("#prob").disabled = e.target.value !== "ic"; });
  $("#seedTop").addEventListener("click", () => {
    state.seeds = new Set(rankedIndices().slice(0, state.topK).map((i) => state.data.nodes[i].label));
    resetCascade(); refresh();
  });
  $("#seedClear").addEventListener("click", () => { state.seeds = new Set(); resetCascade(); refresh(); });
  $("#runCascade").addEventListener("click", runCascade);
  $("#playCascade").addEventListener("click", togglePlay);
  $("#stepSlider").addEventListener("input", (e) => { stopPlay(); setStep(+e.target.value); });
  $("#compare").addEventListener("click", runCompare);
  $("#quizCheck").addEventListener("click", checkQuiz);
  $("#quizNext").addEventListener("click", newQuestion);
  $("#quizAns").addEventListener("keydown", (e) => { if (e.key === "Enter") checkQuiz(); });
  $("#loadText").addEventListener("click", () => {
    const text = $("#edgesText").value;
    if (!text.trim()) { setStatus("Вставьте список рёбер или выберите файл."); return; }
    state.source = { edges_text: text, directed: $("#directed").checked };
    loadGraph();
  });
  $("#edgesFile").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => { $("#edgesText").value = String(rd.result); };
    rd.readAsText(f);
  });
  const cv = $("#canvas");
  cv.addEventListener("mousemove", onMove);
  cv.addEventListener("mouseleave", () => { state.hover = -1; $("#tip").hidden = true; draw(); });
  cv.addEventListener("click", onClick);
  new ResizeObserver(() => { resizeCanvas(); draw(); }).observe(cv);
  await loadGraph();
}
init();

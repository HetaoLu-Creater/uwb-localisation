'use strict';

const el = id => document.getElementById(id);
const canvas = el('field');
const ctx = canvas.getContext('2d');
const canvasCentre = [350, 350];
const fieldRadiusPixels = 210;
const anchorLabelGapPixels = 26;

let config;
let trace;
let original = [];
let clock = 0;
let playing = false;
let liveTrail = [];
let previousTick = performance.now();

const number = value => Number.isFinite(value) ? (value * 100).toFixed(2) : '—';
const anchors = () => Object.values(config.anchors).map(anchor => [anchor.x, anchor.y]);

function visible(row, time) {
  if (!row || !row.displayed_position_m) return null;
  if (row.source_time_s === null || row.source_time_s === undefined) {
    return row.displayed_position_m;
  }
  return time - row.source_time_s <= config.maximum_age_s + 1e-10
    ? row.displayed_position_m
    : null;
}

function at(rows, time) {
  let low = 0;
  let high = rows.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (rows[middle].time_s <= time) low = middle + 1;
    else high = middle;
  }
  return low ? rows[low - 1] : null;
}

function fieldTransform(points) {
  const field = config.field;
  const centre = field.center || [
    points.reduce((sum, point) => sum + point[0], 0) / points.length,
    points.reduce((sum, point) => sum + point[1], 0) / points.length,
  ];
  const radius = field.radius_m || Math.max(
    ...points.map(point => Math.hypot(point[0] - centre[0], point[1] - centre[1])),
  );
  const scale = fieldRadiusPixels / radius;
  return point => [
    canvasCentre[0] + (point[0] - centre[0]) * scale,
    canvasCentre[1] - (point[1] - centre[1]) * scale,
  ];
}

function drawDot(position, colour, radius, toCanvas) {
  if (!position) return;
  const point = toCanvas(position);
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.arc(point[0], point[1], radius, 0, 2 * Math.PI);
  ctx.fill();
}

function drawAnchor(position, index, toCanvas) {
  const point = toCanvas(position);
  const delta = [point[0] - canvasCentre[0], point[1] - canvasCentre[1]];
  const length = Math.hypot(...delta) || 1;
  const direction = [delta[0] / length, delta[1] / length];
  const label = [
    point[0] + direction[0] * anchorLabelGapPixels,
    point[1] + direction[1] * anchorLabelGapPixels,
  ];
  const vertical = Math.abs(direction[0]) < 0.35;
  const above = vertical && direction[1] < 0;
  ctx.save();
  drawDot(position, '#111111', 7, toCanvas);
  ctx.fillStyle = '#111111';
  ctx.textAlign = vertical ? 'center' : direction[0] < 0 ? 'right' : 'left';
  ctx.textBaseline = above ? 'bottom' : 'top';
  ctx.font = 'bold 20px Arial, Helvetica, sans-serif';
  ctx.fillText(`A${index}`, label[0], above ? label[1] - 22 : label[1]);
  ctx.font = '16px Arial, Helvetica, sans-serif';
  ctx.fillText(
    `(${number(position[0])}, ${number(position[1])}) cm`,
    label[0],
    above ? label[1] : label[1] + 24,
  );
  ctx.restore();
}

function drawGrid() {
  if (!el('grid').checked) return;
  ctx.save();
  ctx.beginPath();
  ctx.arc(...canvasCentre, fieldRadiusPixels, 0, 2 * Math.PI);
  ctx.clip();
  ctx.strokeStyle = '#e0e7ed';
  ctx.lineWidth = 1;
  for (let index = 0; index <= 6; index++) {
    const coordinate = 140 + index * 70;
    ctx.beginPath();
    ctx.moveTo(coordinate, 140);
    ctx.lineTo(coordinate, 560);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(140, coordinate);
    ctx.lineTo(560, coordinate);
    ctx.stroke();
  }
  ctx.restore();
}

function drawTrajectory(time, toCanvas) {
  if (!el('trail').checked) return;
  ctx.strokeStyle = '#b01818';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let connected = false;
  let lastTime = null;
  for (const sample of trace ? trace.rows : liveTrail) {
    if (sample.time_s > time) break;
    const position = sample.displayed_position_m;
    if (!position || (lastTime !== null && sample.time_s - lastTime > config.maximum_age_s)) {
      connected = false;
    }
    if (position) {
      const point = toCanvas(position);
      if (connected) ctx.lineTo(...point);
      else ctx.moveTo(...point);
      connected = true;
    }
    lastTime = sample.time_s;
  }
  ctx.stroke();
}

function tableRow(values) {
  const row = document.createElement('tr');
  for (const value of values) {
    const cell = document.createElement('td');
    cell.textContent = value;
    row.appendChild(cell);
  }
  return row;
}

function draw(row, time) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const points = anchors();
  const toCanvas = fieldTransform(points);
  drawGrid();
  if (config.field.center && config.field.radius_m) {
    ctx.beginPath();
    ctx.strokeStyle = '#8194a9';
    ctx.lineWidth = 2;
    ctx.arc(...canvasCentre, fieldRadiusPixels, 0, 2 * Math.PI);
    ctx.stroke();
  }
  points.forEach((point, index) => drawAnchor(point, index, toCanvas));

  const tracked = row
    && time - (row.tracked_source_time_s ?? row.time_s) <= config.prediction_age_s + 1e-10
    ? row.tracked_position_m
    : null;
  const display = visible(row, time);
  const saved = at(original, time);
  const showOriginal = el('originalToggle').checked;
  const originalPosition = showOriginal && saved && time - saved.time_s <= config.prediction_age_s
    ? saved.tracked_position_m
    : null;

  drawTrajectory(time, toCanvas);
  drawDot(tracked, '#2f6f9f', 7, toCanvas);
  drawDot(originalPosition, '#777777', 11, toCanvas);
  drawDot(display, '#b01818', 9, toCanvas);
  el('originalLegend').hidden = !showOriginal;
  el('status').textContent = display ? 'Position displayed' : 'No displayed position';

  el('positions').replaceChildren();
  const positions = [['Displayed', display], ['Tracked', tracked]];
  if (original.length && showOriginal) positions.push(['Original recorded', originalPosition]);
  for (const [label, position] of positions) {
    el('positions').appendChild(tableRow([label, number(position?.[0]), number(position?.[1])]));
  }

  el('time').textContent = `${time.toFixed(2)} s`;
  el('error').textContent = row?.error || '';
  el('ranges').replaceChildren();
  for (let index = 0; index < 3; index++) {
    el('ranges').appendChild(tableRow([
      `A${index}`,
      number(row?.raw_ranges_m?.[index]),
      number(row?.filtered_ranges_m?.[index]),
    ]));
  }
}

async function poll() {
  try {
    const response = await fetch('/api/state');
    if (!response.ok) throw new Error('The local program is unavailable');
    const row = await response.json();
    el('mode').textContent = row.running === false ? 'Live · Stopped' : 'Live · Recording';
    liveTrail.push(row);
    if (liveTrail.length > 10000) liveTrail.shift();
    draw(row, row.time_s || 0);
  } catch (error) {
    el('error').textContent = error.message;
  }
  setTimeout(poll, 50);
}

function animate(now) {
  if (playing) {
    clock = Math.min(trace.duration_s, clock + (now - previousTick) / 1000 * Number(el('speed').value));
    if (clock >= trace.duration_s) {
      playing = false;
      el('play').textContent = 'Play';
    }
  }
  previousTick = now;
  el('seek').value = clock;
  draw(at(trace.rows, clock), clock);
  requestAnimationFrame(animate);
}

async function start() {
  config = await (await fetch('/api/config')).json();
  if (config.replay) {
    trace = await (await fetch('/api/trace')).json();
    trace.rows = trace.rows.filter(row => row.admitted);
    original = trace.original_rows || [];
    el('mode').textContent = 'Replay';
    el('originalControl').hidden = !original.length;
    el('playback').hidden = false;
    el('seek').max = trace.duration_s;
    el('play').onclick = () => {
      playing = !playing;
      el('play').textContent = playing ? 'Pause' : 'Play';
    };
    el('restart').onclick = () => { clock = 0; };
    el('seek').oninput = event => { clock = Number(event.target.value); };
    requestAnimationFrame(animate);
  } else {
    el('mode').textContent = 'Live · Recording';
    poll();
  }
}

start().catch(error => { el('error').textContent = error.message; });

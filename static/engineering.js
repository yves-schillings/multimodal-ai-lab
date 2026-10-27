'use strict';

// One optional view over the existing application's identity and case controls.
const el = id => document.getElementById(id);
let actor = 'officer-a';
let authenticated = false;
let csrf = sessionStorage.getItem('lab-csrf') || '';
let status = null;
const fmt = (value, digits = 3) => value !== null && value !== undefined && Number.isFinite(Number(value))
  ? Number(value).toFixed(digits) : 'Unavailable';

function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
}

function message(text) { el('message').textContent = text || ''; }

async function api(path, method = 'GET', body) {
  const options = { method, headers: {} };
  if (csrf) options.headers['X-Lab-CSRF'] = csrf;
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path + (path.includes('?') ? '&' : '?') + 'actor=' + encodeURIComponent(actor), options);
  let data;
  try { data = await response.json(); }
  catch { throw Error('The local service did not return a result.'); }
  if (!response.ok) throw Error(data.detail || 'The request failed.');
  return data;
}

function details(target, label, data) {
  const panel = node('details');
  panel.append(node('summary', label), node('pre', JSON.stringify(data, null, 2)));
  target.append(panel);
}

function table(target, headers, rows) {
  const result = node('table'), head = node('thead'), heading = node('tr');
  headers.forEach(text => heading.append(node('th', text)));
  head.append(heading);
  result.append(head);
  const body = node('tbody');
  rows.forEach(values => {
    const row = node('tr');
    values.forEach(value => row.append(node('td', String(value))));
    body.append(row);
  });
  result.append(body);
  target.append(result);
}

async function guarded(button, action) {
  message('');
  if (button) button.disabled = true;
  try { await action(); }
  catch (error) { message(error.message); }
  finally { if (button) button.disabled = false; }
}

async function refresh() {
  status = await api('/api/engineering/status');
  const select = el('case'), previous = select.value;
  const cases = (await api('/api/cases')).cases;
  select.replaceChildren();
  cases.forEach(item => {
    const option = node('option', item.title);
    option.value = item.id;
    select.append(option);
  });
  if (cases.some(item => item.id === previous)) select.value = previous;
  el('agent-status').textContent = status.agentic_search.configured
    ? 'Pinned local model profile. Up to 3 retrieval attempts and 90 seconds. Model digests are checked at request time.'
    : 'The optional agentic model profile is not configured.';
  el('search-button').disabled = !status.agentic_search.configured;
  el('vision').disabled = !status.vision.available;

  const fields = el('invoice-fields');
  if (!fields.children.length) {
    const defaults = {amount_ratio: 2.4, purchase_order_match: 0, supplier_history_days: 90,
      duplicate_reference: 1, delivery_confirmed: 0, recent_bank_change: 1};
    for (const field of status.explainability.fields) {
      const label = node('label', field.label), input = node('input');
      Object.assign(input, {name: field.key, type: 'number', min: field.minimum, max: field.maximum,
        step: field.binary ? '1' : 'any', value: defaults[field.key], required: true});
      label.append(input);
      fields.append(label);
    }
  }
  const canExplain = status.explainability.available && status.can_run_explainability;
  el('explain-default').disabled = !canExplain;
  el('explain-input').disabled = !canExplain;
  el('identity-status').textContent = authenticated
    ? 'Signed in as ' + actor + ' with a local password session.'
    : 'Caller-selected actors are simulated identities in this loopback-only profile.';
}

// Session handling reuses the base application's cookie and CSRF token.
el('actor').onchange = () => guarded(null, async () => {
  actor = el('actor').value;
  await refresh();
  el('answer').replaceChildren();
});

el('login').onsubmit = event => {
  event.preventDefault();
  guarded(null, async () => {
    const response = await fetch('/api/auth/login', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username: el('username').value, password: el('password').value})});
    el('password').value = '';
    const data = await response.json();
    if (!response.ok) throw Error(data.detail || 'Sign-in failed.');
    csrf = data.csrf;
    sessionStorage.setItem('lab-csrf', csrf);
    actor = data.principal.id;
    el('login').hidden = true;
    el('logout').hidden = false;
    await refresh();
  });
};

el('logout').onclick = () => guarded(null, async () => {
  await api('/api/auth/logout', 'POST');
  sessionStorage.removeItem('lab-csrf');
  location.reload();
});

el('sample').onclick = () => guarded(el('sample'), async () => {
  const data = await api('/api/engineering/example', 'POST', {});
  await refresh();
  el('case').value = data.case.id;
  message(data.notice);
});

el('search').onsubmit = event => {
  event.preventDefault();
  guarded(el('search-button'), async () => {
    if (!el('case').value) throw Error('Choose an authorized case or load the fictional example.');
    const out = el('answer');
    out.replaceChildren(node('p', 'The local agent is running. It may return an explicit limitation.'));
    const data = await api('/api/cases/' + encodeURIComponent(el('case').value) + '/agent-question', 'POST',
      {question: el('question').value});
    out.replaceChildren(node('h3', data.abstained ? 'Answer withheld' : 'Source-grounded response'));
    out.append(node('p', data.answer));
    out.append(node('p', 'Stop reason: ' + data.stop_reason + ' · ' + data.attempt_count + ' attempt(s)', 'small'));
    for (const source of data.sources || []) {
      const panel = node('div', undefined, 'source');
      panel.append(node('strong', source.title || source.id), node('p', source.text));
      out.append(panel);
    }
    details(out, 'Structured actions and source identifiers', data.attempts || []);
    details(out, 'Complete structured result', data);
  });
};

el('vision').onclick = () => guarded(el('vision'), async () => {
  const report = await api('/api/engineering/vision'), out = el('vision-results');
  out.replaceChildren(node('p', report.scope));
  table(out, ['Device', 'Batch', 'Model p95 ms', 'Pipeline p95 ms'], (report.benchmarks || []).map(item =>
    [item.device, item.batch_size_images, fmt(item.model_call.p95_ms), fmt(item.full_pipeline.p95_ms)]));
  if (report.performance_probe) {
    out.append(node('h3', 'Larger ResNet-18 performance probe: 100 requests/s offered for 10 seconds'));
    table(out, ['Device / precision', 'Instances', 'Success / offered', 'HTTP p50 / p95 / p99 ms'],
      report.performance_probe.cases.map(item => [item.device + ' / ' + item.precision, item.model_instances,
        item.load.successful_requests + ' / ' + item.load.offered_requests,
        ['p50_ms', 'p95_ms', 'p99_ms'].map(key => fmt(item.load.http_end_to_end_latency[key], 2)).join(' / ')]));
  }
  out.append(node('p', 'Retained measurements only. Random weights test execution performance, not detection quality.', 'warning'));
  details(out, 'Measurement protocol, failures and limitations', report);
});

function shapBars(out, shap) {
  out.append(node('p', 'Contributions add in log-odds, not percentage points. Baseline ' +
    fmt(shap.base_value_log_odds) + ' + contributions = ' + fmt(shap.reconstructed_margin_log_odds) + '.'));
  const bars = node('div', undefined, 'bars');
  const maximum = Math.max(...shap.contributions.map(item => Math.abs(item.contribution_log_odds)), 1e-12);
  for (const item of shap.contributions) {
    const row = node('div', undefined, 'bar-row'), track = node('div', undefined, 'bar-track');
    const bar = node('div', undefined, 'bar' + (item.contribution_log_odds > 0 ? ' positive' : ''));
    bar.style.width = (100 * Math.abs(item.contribution_log_odds) / maximum) + '%';
    track.append(bar);
    row.append(node('span', item.feature), track, node('span', fmt(item.contribution_log_odds)));
    bars.append(row);
  }
  out.append(bars);
}

function renderExplanation(comparison) {
  const linear = comparison.linear_baseline, tree = comparison.tree_model, out = el('explanation');
  out.replaceChildren(node('h3', 'Two models score the same fictional invoice'));
  table(out, ['Model', 'Review-suggested probability'], [
    ['Logistic regression', fmt(100 * linear.case.probability_review_suggested, 2) + '%'],
    ['Histogram gradient boosting', fmt(100 * tree.case.probability_review_suggested, 2) + '%']]);
  out.append(node('p', comparison.case.ground_truth_note, 'warning'));
  out.append(node('h3', 'Tree SHAP explains the boosted-tree result'));
  shapBars(out, tree.shap);
  details(out, 'Readable baseline: global logistic coefficients', comparison.linear_coefficients);
  details(out, 'Linear SHAP: the baseline for the same inputs', linear.shap);
  out.append(node('h3', 'LIME approximates the logistic baseline near this case'));
  const probability = linear.lime.surrogate_probability_at_case, error = linear.lime.case_absolute_error;
  const passes = Number.isFinite(probability) && probability >= 0 && probability <= 1 && error <= 0.05;
  out.append(node('p', 'Raw surrogate output ' + fmt(probability, 4) + ' · absolute case error ' + fmt(error, 4)));
  out.append(node('p', passes
    ? 'This small numerical check passes. Usefulness still requires domain review.'
    : 'This approximation fails the illustrative check: output in [0,1] and absolute case error at most 0.05. Its raw output is preserved and is not a reliable probability.',
    passes ? 'warning' : 'failure'));
  table(out, ['Condition', 'Local weight'], linear.lime.contributions.map(item => [item.condition, fmt(item.weight)]));
  details(out, 'Held-out comparison and its limitations', comparison.model_comparison);
  details(out, 'Complete explanations and checks', comparison);
}

async function explain(useInputs, button) {
  await guarded(button, async () => {
    let invoice = null;
    if (useInputs) {
      invoice = {};
      el('invoice-fields').querySelectorAll('input').forEach(input => { invoice[input.name] = Number(input.value); });
    }
    el('explanation').replaceChildren(node('p', 'Fitting two small synthetic models and computing their explanations…'));
    renderExplanation(await api('/api/engineering/explain', 'POST', {invoice}));
  });
}

el('explain-default').onclick = () => explain(false, el('explain-default'));
el('explain').onsubmit = event => { event.preventDefault(); explain(true, el('explain-input')); };

el('acceptance').onclick = () => guarded(el('acceptance'), async () => {
  const report = await api('/api/engineering/acceptance'), out = el('acceptance-results');
  out.replaceChildren(node('h3', 'Technical decision: ' + report.decision));
  out.append(node('p', 'Human review remains required. Production approval: false.', 'warning'));
  table(out, ['Check', 'Status', 'Reason'], report.checks.map(item => [item.id, item.status, item.detail]));
  details(out, 'Evidence hashes and scope', report);
});

async function init() {
  const response = await fetch('/api/status');
  if (!response.ok) throw Error('Open this view through the local Python service.');
  authenticated = !(await response.json()).simulated_identity;
  if (authenticated) {
    const me = await fetch('/api/auth/me');
    if (!me.ok || !csrf) {
      el('login').hidden = false;
      el('identity-status').textContent = 'Sign in in this tab to obtain its local session and CSRF token.';
      return;
    }
    actor = (await me.json()).principal.id;
    el('logout').hidden = false;
  } else el('simulated-controls').hidden = false;
  await refresh();
}

init().catch(error => message(error.message));

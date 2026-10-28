'use strict';

/**
 * @file Optional browser view for the synthetic AI engineering exercises.
 * Reuses the base application's session cookie, per-tab CSRF token and actor
 * checks. Rendering uses textContent, not HTML supplied by a model. The page
 * displays retained measurements and teaching diagnostics; it cannot approve
 * production release. See docs/engineering/portal-reference.md for all handlers.
 */

/** Look up a fixed page control. @param {string} id DOM ID. @returns {HTMLElement|null} */
const el = id => document.getElementById(id);
let actor = 'officer-a';
let authenticated = false;
let csrf = sessionStorage.getItem('lab-csrf') || '';
let status = null;
/**
 * Format a finite numeric value for the study view, or return "Unavailable".
 * @param {*} value Number or coercible scalar; null/undefined are unavailable.
 * @param {number} [digits=3] Decimal places passed to Number.toFixed.
 * @returns {string} Display text. Invalid digits propagate RangeError.
 */
const fmt = (value, digits = 3) => value !== null && value !== undefined && Number.isFinite(Number(value))
  ? Number(value).toFixed(digits) : 'Unavailable';

/**
 * Build an unattached DOM node with safely escaped text.
 * @param {string} tag HTML element name.
 * @param {*} [text] Value assigned through textContent when supplied.
 * @param {string} [className] Optional fixed CSS class name.
 * @returns {HTMLElement} New element; does not insert it into the document.
 */
function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
}

/** Show or clear the message area. @param {string} text User-visible notice. @returns {void} */
function message(text) { el('message').textContent = text || ''; }

/**
 * Call a same-origin lab route using the current actor and per-tab CSRF token.
 * @param {string} path Application-owned route; must not come from model output.
 * @param {string} [method='GET'] HTTP method.
 * @param {*} [body] Optional JSON-serializable payload.
 * @returns {Promise<Object>} Parsed response after an HTTP-success check.
 * @throws {Error} Non-JSON response, non-success status or fetch/serialization error.
 * Actor query parameters support simulated profiles; they do not authenticate a
 * request. The server remains authoritative for identity, case rights and CSRF.
 */
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

/**
 * Append a collapsible structured-evidence panel using escaped JSON text.
 * @param {HTMLElement} target Parent that receives the panel.
 * @param {string} label Summary shown before expansion.
 * @param {*} data JSON-serializable report, trace or limitation values.
 * @returns {void} Mutates target. JSON serialization errors propagate.
 */
function details(target, label, data) {
  const panel = node('details');
  panel.append(node('summary', label), node('pre', JSON.stringify(data, null, 2)));
  target.append(panel);
}

/**
 * Append a table of escaped scalar cells to the supplied result area.
 * @param {HTMLElement} target Parent that receives the complete table.
 * @param {string[]} headers Column labels.
 * @param {Array<Array<*>>} rows Ordered rows; each cell is converted with String.
 * @returns {void} Builds DOM elements only; no data is evaluated as HTML.
 */
function table(target, headers, rows) {
  const result = node('table'), head = node('thead'), heading = node('tr');
  // Header callback: text is one label; append its escaped TH to the heading row.
  headers.forEach(text => heading.append(node('th', text)));
  head.append(heading);
  result.append(head);
  const body = node('tbody');
  // Row callback: values is one ordered cell array; append its completed TR.
  rows.forEach(values => {
    const row = node('tr');
    // Cell callback: value is a scalar converted to escaped table text.
    values.forEach(value => row.append(node('td', String(value))));
    body.append(row);
  });
  result.append(body);
  target.append(result);
}

/**
 * Run a UI operation with an optional disabled button and visible error message.
 * @param {HTMLButtonElement|null} button Control disabled for this invocation.
 * @param {function(): Promise<*>} action Operation called once and awaited.
 * @returns {Promise<void>} Errors are shown in the page and are not rethrown.
 * The button is re-enabled in finally. This is not server-side concurrency or
 * authorization control; simultaneous calls can still reach the server lock.
 */
async function guarded(button, action) {
  message('');
  if (button) button.disabled = true;
  try { await action(); }
  catch (error) { message(error.message); }
  finally { if (button) button.disabled = false; }
}

/**
 * Refresh capabilities, authorized case choices and role-dependent controls.
 * @returns {Promise<void>} Updates global status and the page's DOM controls.
 * @throws {Error} A status/case API call fails or a required DOM control is absent.
 * Preserves the selected case when still present. Initializes invoice fields
 * once from server-owned bounds; does not run inference or fitting.
 */
async function refresh() {
  status = await api('/api/engineering/status');
  const select = el('case'), previous = select.value;
  const cases = (await api('/api/cases')).cases;
  select.replaceChildren();
  // Case callback: item is one server-returned case; append its selectable ID/title.
  cases.forEach(item => {
    const option = node('option', item.title);
    option.value = item.id;
    select.append(option);
  });
  // Predicate callback: retain the old selection only if a returned case has its ID.
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

/**
 * Actor-change handler for the simulated identity selector.
 * @returns {Promise<void>} Refreshes cases, then clears the previous answer.
 * The guarded action callback takes no arguments and mutates actor/page state.
 * A local-session server will reject actor impersonation despite this selector.
 */
el('actor').onchange = () => guarded(null, async () => {
  actor = el('actor').value;
  await refresh();
  el('answer').replaceChildren();
});

/**
 * Login form handler; posts fixed login fields and clears the password control.
 * @param {SubmitEvent} event Browser submission cancelled by preventDefault.
 * @returns {void} Starts a guarded, zero-argument asynchronous login action.
 * That action stores the returned CSRF token in this tab's sessionStorage,
 * selects the authenticated principal and refreshes controls. Errors are shown.
 */
el('login').onsubmit = event => {
  event.preventDefault();
  // Login action callback: request the local session and install its tab-scoped CSRF token.
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

/**
 * Logout handler and zero-argument guarded action.
 * @returns {Promise<void>} Revokes the local session, removes its tab token and reloads.
 * Errors remain visible through guarded; the token is removed only after success.
 */
el('logout').onclick = () => guarded(null, async () => {
  await api('/api/auth/logout', 'POST');
  sessionStorage.removeItem('lab-csrf');
  location.reload();
});

/**
 * Fixture-load handler and zero-argument guarded action.
 * @returns {Promise<void>} Creates the fixed fictional case, selects it and shows its notice.
 * This explicit action writes a synthetic case on the server; it is not a real review.
 */
el('sample').onclick = () => guarded(el('sample'), async () => {
  const data = await api('/api/engineering/example', 'POST', {});
  await refresh();
  el('case').value = data.case.id;
  message(data.notice);
});

/**
 * Question form handler for the selected authorized case.
 * @param {SubmitEvent} event Browser submission cancelled before the API call.
 * @returns {void} Starts a guarded, zero-argument agent-question action.
 * Displays the result, exact sources and structured action trace. Source-grounded
 * is a UI label, not proof of semantic correctness: the known wrong-object case
 * can carry valid quotations and still requires human validation.
 */
el('search').onsubmit = event => {
  event.preventDefault();
  // Question action callback: call the fixed case endpoint and replace the answer panels.
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

/**
 * Vision-evidence handler and zero-argument guarded rendering action.
 * @returns {Promise<void>} Reads the retained report and renders timing tables.
 * It never sends an image or starts live GPU inference. Callback maps below turn
 * benchmark records into display rows; random weights have no detection quality.
 */
el('vision').onclick = () => guarded(el('vision'), async () => {
  const report = await api('/api/engineering/vision'), out = el('vision-results');
  out.replaceChildren(node('p', report.scope));
  // Benchmark callback: item becomes a device/batch/model/pipeline p95 display row.
  table(out, ['Device', 'Batch', 'Model p95 ms', 'Pipeline p95 ms'], (report.benchmarks || []).map(item =>
    [item.device, item.batch_size_images, fmt(item.model_call.p95_ms), fmt(item.full_pipeline.p95_ms)]));
  if (report.performance_probe) {
    out.append(node('h3', 'Larger ResNet-18 performance probe: 100 requests/s offered for 10 seconds'));
    table(out, ['Device / precision', 'Instances', 'Success / offered', 'HTTP p50 / p95 / p99 ms'],
      // Probe callback: item supplies one configuration's counts and HTTP percentiles.
      report.performance_probe.cases.map(item => [item.device + ' / ' + item.precision, item.model_instances,
        item.load.successful_requests + ' / ' + item.load.offered_requests,
        // Percentile callback: key selects an existing metric; format without recomputation.
        ['p50_ms', 'p95_ms', 'p99_ms'].map(key => fmt(item.load.http_end_to_end_latency[key], 2)).join(' / ')]));
  }
  out.append(node('p', 'Retained measurements only. Random weights test execution performance, not detection quality.', 'warning'));
  details(out, 'Measurement protocol, failures and limitations', report);
});

/**
 * Draw signed SHAP contribution bars in the explainer's log-odds output scale.
 * @param {HTMLElement} out Parent receiving the summary and bar chart.
 * @param {Object} shap Valid report with baseline, reconstructed margin and contributions.
 * @returns {void} Uses absolute magnitudes for bar widths and sign for styling.
 * This renderer does not compute SHAP, test additivity or convert shares to
 * percentage points. The API supplies those calculations and their limitations.
 */
function shapBars(out, shap) {
  out.append(node('p', 'Contributions add in log-odds, not percentage points. Baseline ' +
    fmt(shap.base_value_log_odds) + ' + contributions = ' + fmt(shap.reconstructed_margin_log_odds) + '.'));
  const bars = node('div', undefined, 'bars');
  // Magnitude callback: item contributes an absolute log-odds size for chart normalization.
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

/**
 * Render the same-invoice linear/tree comparison and raw explanation checks.
 * @param {Object} comparison run_explainability_comparison result from the local API.
 * @returns {void} Replaces the explanation area; all values are escaped DOM text.
 * The UI teaching diagnostic requires LIME's case output in [0,1] and absolute
 * error <=0.05. It is not the complete service gate or a business fidelity test.
 * Tree SHAP and LIME describe different models; their pass/fail is not a direct
 * comparison of methods. Raw failed outputs remain visible as study evidence.
 */
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
  // LIME row callback: display one surrogate condition and its local fitted weight.
  table(out, ['Condition', 'Local weight'], linear.lime.contributions.map(item => [item.condition, fmt(item.weight)]));
  details(out, 'Held-out comparison and its limitations', comparison.model_comparison);
  details(out, 'Complete explanations and checks', comparison);
}

/**
 * Request the default or edited synthetic invoice explanation under UI guarding.
 * @param {boolean} useInputs True reads numeric inputs; false sends invoice=null.
 * @param {HTMLButtonElement} button Control disabled during fitting/explanation.
 * @returns {Promise<void>} Renders the comparison or displays the API error.
 * Optional fitting may take time and is permitted only by server-side reviewer
 * controls. Browser numeric conversion does not replace server validation.
 */
async function explain(useInputs, button) {
  // Explanation action callback: build bounded inputs, call the fixed route and render.
  await guarded(button, async () => {
    let invoice = null;
    if (useInputs) {
      invoice = {};
      // Input callback: input supplies its server-defined name and numeric browser value.
      el('invoice-fields').querySelectorAll('input').forEach(input => { invoice[input.name] = Number(input.value); });
    }
    el('explanation').replaceChildren(node('p', 'Fitting two small synthetic models and computing their explanations…'));
    renderExplanation(await api('/api/engineering/explain', 'POST', {invoice}));
  });
}

/** Default-explanation click handler. @returns {Promise<void>} Uses the fixed teaching invoice. */
el('explain-default').onclick = () => explain(false, el('explain-default'));
/** Edited-invoice submit handler. @param {SubmitEvent} event Cancelled submission. @returns {void} */
el('explain').onsubmit = event => { event.preventDefault(); explain(true, el('explain-input')); };

/**
 * Evidence-gate handler and zero-argument guarded rendering action.
 * @returns {Promise<void>} Displays retained check statuses and content hashes.
 * The API's configured evidence filenames determine which run is evaluated;
 * this button does not automatically choose the new BGE comparison or approve release.
 */
el('acceptance').onclick = () => guarded(el('acceptance'), async () => {
  const report = await api('/api/engineering/acceptance'), out = el('acceptance-results');
  out.replaceChildren(node('h3', 'Technical decision: ' + report.decision));
  out.append(node('p', 'Human review remains required. Production approval: false.', 'warning'));
  // Check-row callback: item supplies one recorded gate ID, status and reason.
  table(out, ['Check', 'Status', 'Reason'], report.checks.map(item => [item.id, item.status, item.detail]));
  details(out, 'Evidence hashes and scope', report);
});

/**
 * Discover identity mode and initialize the optional view against the local service.
 * @returns {Promise<void>} Shows login or simulated controls, then refreshes capabilities.
 * @throws {Error} The base status route fails, or a later fetch/refresh fails.
 * A password-session profile requires both a current server session and a CSRF
 * token in this tab. Existing cookies alone do not bypass that requirement.
 */
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

// Initialization rejection callback: error supplies a user-visible startup diagnostic.
init().catch(error => message(error.message));

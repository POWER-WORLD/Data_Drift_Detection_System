/**
 * Data Drift Detection Platform — Enterprise Frontend Controller.
 * Manages Dark/White theme switching, SPA tab navigation, Plotly.js telemetry rendering,
 * and asynchronous REST API interactions.
 */

// ── Application State ──────────────────────────────────────────────────────
const state = {
    theme: localStorage.getItem('ddd_theme') || 'dark',
    activeView: 'view-overview',
    activeBatch: 'batch_01_no_drift.csv',
    analysisData: null,
    alerts: [],
    selectedFeature: 'MonthlyCharges',
    thresholds: {
        ks: 0.05,
        psiMod: 0.10,
        psiSev: 0.25,
        chi2: 0.05,
        js: 0.10,
        risk: 0.20
    }
};

// ── Theme Engine ───────────────────────────────────────────────────────────
function setTheme(theme) {
    state.theme = theme;
    localStorage.setItem('ddd_theme', theme);
    document.documentElement.setAttribute('data-theme', theme);

    // Update switcher pills
    const darkBtn = document.getElementById('btn-theme-dark');
    const lightBtn = document.getElementById('btn-theme-light');
    if (darkBtn && lightBtn) {
        if (theme === 'dark') {
            darkBtn.classList.add('active');
            lightBtn.classList.remove('active');
        } else {
            lightBtn.classList.add('active');
            darkBtn.classList.remove('active');
        }
    }

    // Refresh charts with updated theme palette
    if (state.analysisData) {
        renderCharts(state.analysisData);
    }
}

function getChartColors() {
    const isDark = (state.theme === 'dark');
    return {
        paperBg: 'rgba(0,0,0,0)',
        plotBg: 'rgba(0,0,0,0)',
        textColor: isDark ? '#cbd5e1' : '#0f172a',
        gridColor: isDark ? 'rgba(255, 255, 255, 0.07)' : 'rgba(15, 23, 42, 0.08)',
        blue: isDark ? '#3b82f6' : '#2563eb',
        pink: isDark ? '#ec4899' : '#db2777',
        green: isDark ? '#10b981' : '#059669',
        yellow: isDark ? '#f59e0b' : '#d97706',
        red: isDark ? '#ef4444' : '#dc2626'
    };
}

// ── SPA View Navigation ────────────────────────────────────────────────────
const VIEW_METADATA = {
    'view-overview': {
        title: 'Dashboard Overview',
        sub: 'Real-time drift monitoring & model degradation telemetry'
    },
    'view-drift-analysis': {
        title: 'Feature Drift Analysis',
        sub: 'Deep-dive statistical audit into individual feature empirical distributions'
    },
    'view-model-health': {
        title: 'Model Health & Reliability',
        sub: 'Classification performance degradation, confusion matrices & confidence analysis'
    },
    'view-alerts': {
        title: 'Alert Intelligence Center',
        sub: 'Automated drift warnings, degradation notices & remediation recommendations'
    },
    'view-upload': {
        title: 'Upload & Ingest Batches',
        sub: 'Ingest production batches, configure test sensitivities & run analysis'
    },
    'view-settings': {
        title: 'Settings & Architecture',
        sub: 'Statistical sensitivity calibration, system diagnostics & author specs'
    }
};

function switchView(viewId) {
    state.activeView = viewId;

    // Toggle view panels
    document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
    const targetPanel = document.getElementById(viewId);
    if (targetPanel) {
        targetPanel.classList.add('active');
    }

    // Toggle navigation buttons
    document.querySelectorAll('.sidebar-nav .nav-item').forEach(b => {
        b.classList.toggle('active', b.getAttribute('data-view') === viewId);
    });

    // Update top header crumbs
    const meta = VIEW_METADATA[viewId] || { title: 'Dashboard', sub: '' };
    document.getElementById('page-crumb-title').textContent = meta.title;
    document.getElementById('page-crumb-sub').textContent = meta.sub;

    // Refresh view specific charts
    if (state.analysisData) {
        if (viewId === 'view-overview') {
            renderRiskGauge(state.analysisData.dataset_drift);
            renderFeaturePSIChart(state.analysisData.feature_drift);
        } else if (viewId === 'view-drift-analysis') {
            renderFeatureDistribution();
        } else if (viewId === 'view-model-health') {
            renderModelHealthCharts(state.analysisData.performance);
        }
    }

    // Trigger chart resize if needed
    window.dispatchEvent(new Event('resize'));
}

// ── API Interactions ───────────────────────────────────────────────────────
async function fetchBatches() {
    try {
        const res = await fetch('/api/batches');
        const data = await res.json();
        const select = document.getElementById('select-preset-batch');
        if (select && data.batches && data.batches.length > 0) {
            select.innerHTML = '';
            data.batches.forEach(b => {
                const opt = document.createElement('option');
                opt.value = b.filename;
                opt.textContent = `${b.filename} (${b.size_kb} KB)`;
                select.appendChild(opt);
            });
        }
    } catch (err) {
        console.warn('Could not fetch batch list:', err);
    }
}

async function executeBatchAnalysis() {
    const select = document.getElementById('select-preset-batch');
    const batchName = select ? select.value : state.activeBatch;
    const btn = document.getElementById('btn-run-analysis');

    if (btn) {
        btn.disabled = true;
        btn.textContent = '⏳ Analyzing 8 Statistical Tests...';
    }

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                batch_name: batchName,
                thresholds: {
                    ks_pvalue: state.thresholds.ks,
                    psi_moderate: state.thresholds.psiMod,
                    psi_severe: state.thresholds.psiSev,
                    chi2_pvalue: state.thresholds.chi2,
                    js_moderate: state.thresholds.js
                }
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Analysis request failed');
        }

        const data = await response.json();
        state.analysisData = data;
        state.activeBatch = batchName;

        // Update UI
        updateSidebarStatus(batchName, data.dataset_drift);
        updateOverviewKPIs(data);
        renderCharts(data);
        populateOverviewTable(data);
        populateFeatureDropdown(data);
        renderFeatureDistribution();
        fetchAlerts();

        // Switch to overview tab to show results
        switchView('view-overview');
    } catch (err) {
        alert('Analysis Error: ' + err.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = '🔍 Run Full Drift Detection & Telemetry Analysis';
        }
    }
}

function updateSidebarStatus(batchName, datasetDrift) {
    const batchEl = document.getElementById('sidebar-active-batch');
    const badgeEl = document.getElementById('sidebar-risk-badge');
    if (batchEl) batchEl.textContent = batchName;

    if (badgeEl && datasetDrift) {
        const risk = (datasetDrift.risk_level || 'LOW').toUpperCase();
        const colors = { LOW: 'green', MEDIUM: 'yellow', HIGH: 'red' };
        const dotColor = colors[risk] || 'green';
        badgeEl.innerHTML = `<span class="badge-dot ${dotColor}"></span> ${risk} Risk`;
    }
}

function updateOverviewKPIs(data) {
    const ds = data.dataset_drift || {};
    const risk = (ds.risk_level || 'LOW').toUpperCase();
    const driftedCount = ds.drifted_count || 0;
    const totalCount = ds.total_features || 20;
    const driftedPct = (ds.drifted_pct || 0).toFixed(1);

    const riskEl = document.getElementById('kpi-overall-risk');
    const riskDeltaEl = document.getElementById('kpi-risk-delta');
    if (riskEl) {
        riskEl.textContent = risk;
        riskEl.className = `kpi-value ${risk === 'LOW' ? 'text-green' : (risk === 'MEDIUM' ? 'text-yellow' : 'text-red')}`;
    }
    if (riskDeltaEl) {
        riskDeltaEl.textContent = risk === 'LOW' ? 'Dataset Healthy' : `${risk} Risk Detected`;
    }

    const countEl = document.getElementById('kpi-drifted-count');
    const pctEl = document.getElementById('kpi-drifted-pct');
    if (countEl) countEl.textContent = `${driftedCount} / ${totalCount}`;
    if (pctEl) {
        pctEl.textContent = `${driftedPct}% shifted`;
        pctEl.className = `kpi-delta ${driftedCount === 0 ? 'positive' : 'negative'}`;
    }

    // Model health
    const perf = data.performance || {};
    const health = perf.overall_health || 'HEALTHY';
    const healthEl = document.getElementById('kpi-model-health');
    const healthDeltaEl = document.getElementById('kpi-health-delta');
    if (healthEl) {
        healthEl.textContent = health;
        healthEl.className = `kpi-value ${health === 'HEALTHY' ? 'text-green' : (health === 'WARNING' ? 'text-yellow' : 'text-red')}`;
    }
    if (healthDeltaEl) {
        if (perf.f1_score && perf.f1_score.delta != null) {
            const delta = perf.f1_score.delta;
            healthDeltaEl.textContent = `F1 Δ: ${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`;
        } else {
            healthDeltaEl.textContent = 'Telemetry Active';
        }
    }

    // Confidence
    const conf = data.confidence_stats || {};
    const confEl = document.getElementById('kpi-avg-confidence');
    if (confEl) {
        if (conf.mean_confidence != null) {
            confEl.textContent = `${(conf.mean_confidence * 100).toFixed(1)}%`;
        } else {
            confEl.textContent = 'N/A';
        }
    }

    // Model health tab KPIs
    const bF1El = document.getElementById('kpi-baseline-f1');
    const curF1El = document.getElementById('kpi-batch-f1');
    const deltaF1El = document.getElementById('kpi-f1-delta');
    const rocAucEl = document.getElementById('kpi-roc-auc');
    const lowConfEl = document.getElementById('kpi-low-conf-rate');

    if (bF1El) bF1El.textContent = (perf.f1_score && perf.f1_score.baseline != null) ? perf.f1_score.baseline.toFixed(3) : 'N/A';
    if (curF1El) curF1El.textContent = (perf.f1_score && perf.f1_score.current != null) ? perf.f1_score.current.toFixed(3) : 'N/A';
    if (deltaF1El) {
        if (perf.f1_score && perf.f1_score.delta != null) {
            const delta = perf.f1_score.delta;
            deltaF1El.textContent = `Δ ${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`;
        } else {
            deltaF1El.textContent = 'N/A';
        }
    }
    if (rocAucEl) rocAucEl.textContent = (perf.roc_auc && perf.roc_auc.current != null) ? perf.roc_auc.current.toFixed(3) : 'N/A';
    if (lowConfEl) {
        lowConfEl.textContent = (conf.low_confidence_pct != null) ? `${conf.low_confidence_pct.toFixed(1)}%` : 'N/A';
    }
}

// ── Interactive Plotly Telemetry Renderers ─────────────────────────────────
function renderCharts(data) {
    renderRiskGauge(data.dataset_drift);
    renderFeaturePSIChart(data.feature_drift);
    renderModelHealthCharts(data.performance);
}

function renderRiskGauge(ds) {
    const chartDiv = document.getElementById('chart-risk-gauge');
    if (!chartDiv || !window.Plotly) return;

    const wrapper = chartDiv.closest('.chart-scroll-wrapper');
    const containerWidth = wrapper ? (wrapper.clientWidth || 340) : 340;
    const computedWidth = Math.max(containerWidth, 320);
    chartDiv.style.width = computedWidth + 'px';
    chartDiv.style.minWidth = '320px';

    const colors = getChartColors();
    const driftPct = ds ? (ds.drifted_pct || 0) : 0;
    const risk = ds ? ds.risk_level : 'LOW';
    const barColor = risk === 'LOW' ? colors.green : (risk === 'MEDIUM' ? colors.yellow : colors.red);

    const gaugeData = [{
        type: 'indicator',
        mode: 'gauge+number',
        value: driftPct,
        number: { suffix: '%', font: { size: 36, color: colors.textColor } },
        gauge: {
            axis: { range: [0, 100], tickcolor: colors.textColor, tickfont: { color: colors.textColor } },
            bar: { color: barColor, thickness: 0.65 },
            bgcolor: colors.gridColor,
            borderwidth: 0,
            steps: [
                { range: [0, 20], color: 'rgba(16,185,129,0.1)' },
                { range: [20, 50], color: 'rgba(245,158,11,0.1)' },
                { range: [50, 100], color: 'rgba(239,68,68,0.1)' }
            ],
            threshold: { line: { color: colors.red, width: 3 }, thickness: 0.8, value: 50 }
        }
    }];

    const layout = {
        paper_bgcolor: colors.paperBg,
        plot_bgcolor: colors.plotBg,
        margin: { t: 25, b: 20, l: 30, r: 30 },
        height: 290,
        width: computedWidth,
        autosize: false
    };

    Plotly.newPlot(chartDiv, gaugeData, layout, { responsive: true, displayModeBar: false });
}

function renderFeaturePSIChart(featureDrift) {
    const chartDiv = document.getElementById('chart-feature-psi');
    if (!chartDiv || !window.Plotly || !featureDrift) return;

    const colors = getChartColors();
    const numTests = featureDrift.numerical || {};
    const catTests = featureDrift.categorical || {};

    const psiList = [];
    for (const [colName, tests] of Object.entries(numTests)) {
        const psiVal = tests.psi ? tests.psi.value : 0;
        if (psiVal !== null && !isNaN(psiVal)) {
            psiList.push({
                name: colName,
                psi: parseFloat(psiVal),
                isCategorical: false,
                status: tests.psi ? tests.psi.severity : 'STABLE'
            });
        }
    }

    // Fallback to categorical Jensen-Shannon divergence if no numerical features exist
    if (psiList.length === 0) {
        for (const [colName, tests] of Object.entries(catTests)) {
            const jsVal = tests.jensen_shannon ? tests.jensen_shannon.value : 0;
            if (jsVal !== null && !isNaN(jsVal)) {
                psiList.push({
                    name: colName,
                    psi: parseFloat(jsVal),
                    isCategorical: true,
                    status: (tests.jensen_shannon && tests.jensen_shannon.drift_detected) ? 'DRIFT' : 'STABLE'
                });
            }
        }
    }

    if (psiList.length === 0) {
        chartDiv.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-muted)">No features available for drift comparison.</div>';
        return;
    }

    // Sort descending by drift score so highest-drift features appear first on the left
    psiList.sort((a, b) => b.psi - a.psi);

    // Compute dynamic width so horizontal scrolling accommodates all features comfortably
    const wrapper = chartDiv.closest('.chart-scroll-wrapper');
    const containerWidth = wrapper ? (wrapper.clientWidth || 550) : 550;
    const barSpacingWidth = Math.max(psiList.length * 80, 540);
    const computedWidth = Math.max(containerWidth, barSpacingWidth);

    chartDiv.style.width = computedWidth + 'px';
    chartDiv.style.minWidth = computedWidth + 'px';

    const barColors = psiList.map(item => {
        if (item.psi >= 0.25) return colors.red;
        if (item.psi >= 0.10) return colors.yellow;
        return colors.green;
    });

    const barData = [{
        type: 'bar',
        x: psiList.map(i => i.name),
        y: psiList.map(i => i.psi),
        orientation: 'v',
        marker: {
            color: barColors,
            line: {
                width: 1,
                color: isDarkTheme() ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)'
            }
        },
        text: psiList.map(i => i.psi.toFixed(3)),
        textposition: 'outside',
        textfont: {
            color: colors.textColor,
            size: 11,
            family: 'Inter, sans-serif'
        },
        hovertemplate: '<b>%{x}</b><br>Drift Score: %{y:.4f}<extra></extra>'
    }];

    const maxVal = Math.max(...psiList.map(i => i.psi), 0.30);
    const isCatOnly = psiList[0]?.isCategorical;

    const layout = {
        paper_bgcolor: colors.paperBg,
        plot_bgcolor: colors.plotBg,
        font: { color: colors.textColor, family: 'Inter, sans-serif' },
        margin: { t: 30, b: 75, l: 55, r: 40 },
        width: computedWidth,
        height: 290,
        autosize: false,
        xaxis: {
            title: { text: 'Feature Name', font: { size: 12, color: colors.textColor } },
            tickangle: -25,
            gridcolor: colors.gridColor,
            tickfont: { color: colors.textColor, size: 11 },
            automargin: true
        },
        yaxis: {
            title: { text: isCatOnly ? 'JS Divergence' : 'PSI Divergence', font: { size: 12, color: colors.textColor } },
            gridcolor: colors.gridColor,
            tickfont: { color: colors.textColor, size: 11 },
            zerolinecolor: colors.gridColor,
            range: [0, maxVal * 1.25]
        },
        shapes: [
            {
                type: 'line',
                xref: 'paper',
                x0: 0,
                x1: 1,
                yref: 'y',
                y0: 0.10,
                y1: 0.10,
                line: { color: colors.yellow, width: 1.5, dash: 'dash' }
            },
            {
                type: 'line',
                xref: 'paper',
                x0: 0,
                x1: 1,
                yref: 'y',
                y0: 0.25,
                y1: 0.25,
                line: { color: colors.red, width: 1.5, dash: 'dash' }
            }
        ],
        annotations: [
            {
                xref: 'paper',
                x: 0.99,
                yref: 'y',
                y: 0.10,
                text: 'Moderate (0.10)',
                showarrow: false,
                font: { size: 10, color: colors.yellow },
                xanchor: 'right',
                yanchor: 'bottom'
            },
            {
                xref: 'paper',
                x: 0.99,
                yref: 'y',
                y: 0.25,
                text: 'Severe (0.25)',
                showarrow: false,
                font: { size: 10, color: colors.red },
                xanchor: 'right',
                yanchor: 'bottom'
            }
        ]
    };

    Plotly.newPlot(chartDiv, barData, layout, { responsive: true, displayModeBar: false });
}

async function renderFeatureDistribution() {
    const select = document.getElementById('select-drift-feature');
    const feature = select ? select.value : state.selectedFeature;
    const chartDiv = document.getElementById('chart-distribution-overlay');
    const stripEl = document.getElementById('feature-stats-strip');
    const tableBody = document.getElementById('table-feature-tests-body');

    if (!chartDiv || !window.Plotly) return;

    try {
        const res = await fetch(`/api/feature-distribution/${encodeURIComponent(feature)}`);
        if (!res.ok) return;
        const data = await res.json();
        const colors = getChartColors();

        const wrapper = chartDiv.closest('.chart-scroll-wrapper');
        const containerWidth = wrapper ? (wrapper.clientWidth || 600) : 600;

        if (data.type === 'numerical') {
            const computedWidth = Math.max(containerWidth, 540);
            chartDiv.style.width = computedWidth + 'px';
            chartDiv.style.minWidth = '540px';

            const traceBaseline = {
                type: 'histogram',
                x: data.baseline,
                name: 'Baseline Training',
                opacity: 0.65,
                marker: { color: colors.blue },
                histnorm: 'probability density',
                nbinsx: 35
            };
            const traceProd = {
                type: 'histogram',
                x: data.production,
                name: 'Production Batch',
                opacity: 0.65,
                marker: { color: colors.pink },
                histnorm: 'probability density',
                nbinsx: 35
            };

            const layout = {
                barmode: 'overlay',
                paper_bgcolor: colors.paperBg,
                plot_bgcolor: colors.plotBg,
                font: { color: colors.textColor, family: 'Inter, sans-serif' },
                xaxis: { title: feature, gridcolor: colors.gridColor },
                yaxis: { title: 'Density', gridcolor: colors.gridColor },
                margin: { t: 30, b: 40, l: 50, r: 20 },
                height: 360,
                width: computedWidth,
                autosize: false
            };

            Plotly.newPlot(chartDiv, [traceBaseline, traceProd], layout, { responsive: true, displayModeBar: false });

            // Summary stats strip
            if (stripEl) {
                stripEl.innerHTML = `
                    <div class="stat-pill"><div class="stat-pill-label">Baseline Mean</div><div class="stat-pill-val">${data.baseline_mean.toFixed(2)}</div></div>
                    <div class="stat-pill"><div class="stat-pill-label">Production Mean</div><div class="stat-pill-val">${data.production_mean.toFixed(2)}</div></div>
                    <div class="stat-pill"><div class="stat-pill-label">Baseline Std</div><div class="stat-pill-val">±${data.baseline_std.toFixed(2)}</div></div>
                    <div class="stat-pill"><div class="stat-pill-label">Production Std</div><div class="stat-pill-val">±${data.production_std.toFixed(2)}</div></div>
                `;
            }
        } else {
            // Categorical Feature
            const numCats = (data.categories || []).length;
            const computedWidth = Math.max(containerWidth, Math.max(540, numCats * 65));
            chartDiv.style.width = computedWidth + 'px';
            chartDiv.style.minWidth = computedWidth + 'px';

            const traceBaseline = {
                type: 'bar',
                x: data.categories,
                y: data.baseline_proportions,
                name: 'Baseline Proportion',
                marker: { color: colors.blue }
            };
            const traceProd = {
                type: 'bar',
                x: data.categories,
                y: data.production_proportions,
                name: 'Production Proportion',
                marker: { color: colors.pink }
            };

            const layout = {
                barmode: 'group',
                paper_bgcolor: colors.paperBg,
                plot_bgcolor: colors.plotBg,
                font: { color: colors.textColor, family: 'Inter, sans-serif' },
                xaxis: { title: feature, gridcolor: colors.gridColor, tickangle: numCats > 4 ? -25 : 0, automargin: true },
                yaxis: { title: 'Relative Proportion', gridcolor: colors.gridColor, range: [0, 1] },
                margin: { t: 30, b: numCats > 4 ? 60 : 40, l: 50, r: 20 },
                height: 360,
                width: computedWidth,
                autosize: false
            };

            Plotly.newPlot(chartDiv, [traceBaseline, traceProd], layout, { responsive: true, displayModeBar: false });

            if (stripEl) {
                stripEl.innerHTML = `
                    <div class="stat-pill"><div class="stat-pill-label">Total Categories</div><div class="stat-pill-val">${data.categories.length}</div></div>
                    <div class="stat-pill"><div class="stat-pill-label">Feature Type</div><div class="stat-pill-val">Categorical</div></div>
                `;
            }
        }

        // Fill test breakdown table
        if (tableBody && state.analysisData) {
            const numData = (state.analysisData.feature_drift.numerical || {})[feature];
            const catData = (state.analysisData.feature_drift.categorical || {})[feature];

            if (numData) {
                tableBody.innerHTML = `
                    <tr>
                        <td><strong>Kolmogorov-Smirnov (KS)</strong></td>
                        <td>Empirical CDF Shift</td>
                        <td>p = ${parseFloat(numData.ks_test.p_value).toFixed(4)}</td>
                        <td>α = 0.05</td>
                        <td><span class="badge-status ${numData.ks_test.drift_detected ? 'drift' : 'stable'}">${numData.ks_test.drift_detected ? '🔴 Drift' : '🟢 Stable'}</span></td>
                    </tr>
                    <tr>
                        <td><strong>Population Stability Index (PSI)</strong></td>
                        <td>Quantile Binned Divergence</td>
                        <td>PSI = ${parseFloat(numData.psi.value).toFixed(4)}</td>
                        <td>< 0.10 (Moderate: 0.10, Severe: 0.25)</td>
                        <td><span class="badge-status ${numData.psi.drift_detected ? 'drift' : 'stable'}">${numData.psi.severity}</span></td>
                    </tr>
                    <tr>
                        <td><strong>Mean Shift Test</strong></td>
                        <td>First Moment Delta</td>
                        <td>Δ = ${(parseFloat(numData.mean_shift.pct_change) * 100).toFixed(2)}%</td>
                        <td>Threshold: 15%</td>
                        <td><span class="badge-status ${numData.mean_shift.drift_detected ? 'drift' : 'stable'}">${numData.mean_shift.drift_detected ? '🔴 Shifted' : '🟢 Stable'}</span></td>
                    </tr>
                    <tr>
                        <td><strong>Jensen-Shannon Distance</strong></td>
                        <td>Symmetric KL Divergence</td>
                        <td>JS = ${parseFloat(numData.jensen_shannon.value).toFixed(4)}</td>
                        <td>Threshold: 0.10</td>
                        <td><span class="badge-status ${numData.jensen_shannon.drift_detected ? 'drift' : 'stable'}">${numData.jensen_shannon.drift_detected ? '🔴 Drift' : '🟢 Stable'}</span></td>
                    </tr>
                `;
            } else if (catData) {
                tableBody.innerHTML = `
                    <tr>
                        <td><strong>Chi-Square (χ²) Goodness-of-Fit</strong></td>
                        <td>Contingency Proportions</td>
                        <td>p = ${parseFloat(catData.chi_square.p_value).toFixed(4)}</td>
                        <td>α = 0.05</td>
                        <td><span class="badge-status ${catData.chi_square.drift_detected ? 'drift' : 'stable'}">${catData.chi_square.drift_detected ? '🔴 Drift' : '🟢 Stable'}</span></td>
                    </tr>
                    <tr>
                        <td><strong>Jensen-Shannon Distance</strong></td>
                        <td>Discrete Divergence</td>
                        <td>JS = ${parseFloat(catData.jensen_shannon.value).toFixed(4)}</td>
                        <td>Threshold: 0.10</td>
                        <td><span class="badge-status ${catData.jensen_shannon.drift_detected ? 'drift' : 'stable'}">${catData.jensen_shannon.drift_detected ? '🔴 Drift' : '🟢 Stable'}</span></td>
                    </tr>
                `;
            }
        }
    } catch (err) {
        console.error('Error rendering distribution:', err);
    }
}

function renderModelHealthCharts(perf) {
    const cmpDiv = document.getElementById('chart-metrics-comparison');
    const cmDiv = document.getElementById('chart-confusion-matrix');
    if (!window.Plotly) return;

    const colors = getChartColors();

    if (cmpDiv && perf) {
        const wrapper = cmpDiv.closest('.chart-scroll-wrapper');
        const containerWidth = wrapper ? (wrapper.clientWidth || 500) : 500;
        const computedWidth = Math.max(containerWidth, 500);
        cmpDiv.style.width = computedWidth + 'px';
        cmpDiv.style.minWidth = '500px';

        const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'];
        const labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'];
        const bVals = metrics.map(m => perf[m] ? perf[m].baseline : 0);
        const cVals = metrics.map(m => perf[m] ? perf[m].current : 0);

        const data = [
            { type: 'bar', name: 'Baseline', x: labels, y: bVals, marker: { color: colors.blue } },
            { type: 'bar', name: 'Production', x: labels, y: cVals, marker: { color: colors.pink } }
        ];
        const layout = {
            barmode: 'group',
            paper_bgcolor: colors.paperBg,
            plot_bgcolor: colors.plotBg,
            font: { color: colors.textColor, family: 'Inter, sans-serif' },
            yaxis: { range: [0, 1.1], gridcolor: colors.gridColor },
            xaxis: { gridcolor: colors.gridColor },
            margin: { t: 20, b: 35, l: 45, r: 20 },
            height: 290,
            width: computedWidth,
            autosize: false
        };
        Plotly.newPlot(cmpDiv, data, layout, { responsive: true, displayModeBar: false });
    }

    if (cmDiv) {
        const wrapper = cmDiv.closest('.chart-scroll-wrapper');
        const containerWidth = wrapper ? (wrapper.clientWidth || 360) : 360;
        const computedWidth = Math.max(containerWidth, 340);
        cmDiv.style.width = computedWidth + 'px';
        cmDiv.style.minWidth = '340px';

        if (perf && perf.confusion_matrix && Array.isArray(perf.confusion_matrix) && perf.confusion_matrix.length === 2) {
            const cmData = [{
                z: perf.confusion_matrix,
                x: ['Predicted Negative (0)', 'Predicted Positive (1)'],
                y: ['Actual Negative (0)', 'Actual Positive (1)'],
                type: 'heatmap',
                colorscale: [
                    [0, isDarkTheme() ? '#121a2d' : '#f1f5f9'],
                    [1, colors.blue]
                ],
                showscale: false
            }];
            const cmLayout = {
                paper_bgcolor: colors.paperBg,
                plot_bgcolor: colors.plotBg,
                font: { color: colors.textColor, family: 'Inter, sans-serif' },
                margin: { t: 30, b: 40, l: 110, r: 20 },
                height: 290,
                width: computedWidth,
                autosize: false
            };
            Plotly.newPlot(cmDiv, cmData, cmLayout, { responsive: true, displayModeBar: false });
        } else {
            cmDiv.innerHTML = '<div style="padding:60px 15px;text-align:center;color:var(--text-muted);font-size:0.88rem;">Confusion matrix telemetry is active for supervised batches with target labels.<br><span style="font-size:0.8rem;color:var(--text-dim);margin-top:6px;display:inline-block;">Real-time statistical drift algorithms are actively running across all features.</span></div>';
        }
    }
}

function isDarkTheme() {
    return state.theme === 'dark';
}

function populateOverviewTable(data) {
    const tableBody = document.querySelector('#table-overview-features tbody');
    if (!tableBody || !data.feature_drift) return;

    tableBody.innerHTML = '';
    const numTests = data.feature_drift.numerical || {};
    const catTests = data.feature_drift.categorical || {};

    for (const [colName, tests] of Object.entries(numTests)) {
        const isDrift = tests.is_drifted || false;
        const ksP = tests.ks_test ? tests.ks_test.p_value.toFixed(4) : '—';
        const psiVal = tests.psi ? tests.psi.value.toFixed(3) : '—';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${colName}</strong></td>
            <td><span class="badge-version">Numerical</span></td>
            <td>KS & PSI</td>
            <td>PSI: ${psiVal}</td>
            <td>p = ${ksP}</td>
            <td><span class="badge-status ${isDrift ? 'drift' : 'stable'}">${isDrift ? '🔴 Drift' : '🟢 Stable'}</span></td>
        `;
        tableBody.appendChild(row);
    }

    for (const [colName, tests] of Object.entries(catTests)) {
        const isDrift = tests.is_drifted || false;
        const chiP = tests.chi_square ? tests.chi_square.p_value.toFixed(4) : '—';
        const jsVal = tests.jensen_shannon ? tests.jensen_shannon.value.toFixed(3) : '—';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${colName}</strong></td>
            <td><span class="badge-version">Categorical</span></td>
            <td>Chi-Square & JS</td>
            <td>JS: ${jsVal}</td>
            <td>p = ${chiP}</td>
            <td><span class="badge-status ${isDrift ? 'drift' : 'stable'}">${isDrift ? '🔴 Drift' : '🟢 Stable'}</span></td>
        `;
        tableBody.appendChild(row);
    }
}

function populateFeatureDropdown(data) {
    const select = document.getElementById('select-drift-feature');
    if (!select || !data.feature_drift) return;

    select.innerHTML = '';
    const numCols = Object.keys(data.feature_drift.numerical || {});
    const catCols = Object.keys(data.feature_drift.categorical || {});
    const allCols = [...numCols, ...catCols];

    allCols.forEach((col, idx) => {
        const opt = document.createElement('option');
        opt.value = col;
        const typeBadge = numCols.includes(col) ? '🔢' : '🔤';
        opt.textContent = `${typeBadge} ${col}`;
        if (idx === 0) {
            opt.selected = true;
            state.selectedFeature = col;
        }
        select.appendChild(opt);
    });

    if (allCols.length > 0 && !allCols.includes(state.selectedFeature)) {
        state.selectedFeature = allCols[0];
    }
}

// ── Alerts Subsystem ───────────────────────────────────────────────────────
async function fetchAlerts(severityFilter = 'ALL') {
    try {
        const url = severityFilter === 'ALL' ? '/api/alerts' : `/api/alerts?severity=${severityFilter}`;
        const res = await fetch(url);
        const data = await res.json();
        state.alerts = data.alerts || [];

        renderAlerts(state.alerts);
        updateAlertKPIs(state.alerts);
    } catch (err) {
        console.warn('Could not fetch alerts:', err);
    }
}

function updateAlertKPIs(alerts) {
    const total = alerts.length;
    const crit = alerts.filter(a => (a.severity || '').toUpperCase() === 'CRITICAL').length;
    const warn = alerts.filter(a => (a.severity || '').toUpperCase() === 'WARNING').length;
    const info = alerts.filter(a => (a.severity || '').toUpperCase() === 'INFO').length;

    const tEl = document.getElementById('kpi-total-alerts');
    const cEl = document.getElementById('kpi-crit-alerts');
    const wEl = document.getElementById('kpi-warn-alerts');
    const iEl = document.getElementById('kpi-info-alerts');
    const badgeEl = document.getElementById('sidebar-alert-badge');

    if (tEl) tEl.textContent = total;
    if (cEl) cEl.textContent = crit;
    if (wEl) wEl.textContent = warn;
    if (iEl) iEl.textContent = info;
    if (badgeEl) badgeEl.textContent = total;
}

function renderAlerts(alerts) {
    const container = document.getElementById('alert-stream-container');
    if (!container) return;

    if (!alerts || alerts.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">🔔</span>
                <div class="empty-title">Zero Active Alerts</div>
                <div class="empty-desc">The system logs alerts when distribution drift or degradation breaches thresholds.</div>
            </div>
        `;
        return;
    }

    container.innerHTML = alerts.map(a => {
        const sev = (a.severity || 'INFO').toLowerCase();
        const icon = sev === 'critical' ? '🚨' : (sev === 'warning' ? '⚠️' : 'ℹ️');
        return `
            <div class="alert-card-ui ${sev}">
                <div class="alert-card-header">
                    <div class="alert-card-title">${icon} [${a.severity.toUpperCase()}] ${a.batch ? '• Batch: ' + a.batch : ''}</div>
                    <div class="alert-card-time">${a.timestamp || ''}</div>
                </div>
                <div class="alert-card-msg">${a.message || ''}</div>
                <div class="alert-card-action">↳ Action: ${a.action || 'Monitor distribution'}</div>
            </div>
        `;
    }).join('');
}

function filterAlerts(severity) {
    document.querySelectorAll('.filter-pill').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.includes(severity) || (severity === 'ALL' && btn.textContent === 'All'));
    });
    fetchAlerts(severity);
}

async function clearAllAlerts() {
    if (confirm('Clear all historical alerts?')) {
        await fetch('/api/alerts/clear', { method: 'POST' });
        fetchAlerts();
    }
}

// ── File Upload ────────────────────────────────────────────────────────────
async function handleFileSelect(event) {
    const file = event.target.files[0];
    const statusMsg = document.getElementById('upload-status-msg');
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
        statusMsg.innerHTML = '<span style="color:var(--accent-red)">❌ Only CSV files (.csv) are accepted.</span>';
        return;
    }

    statusMsg.innerHTML = '<span>⏳ Ingesting and analyzing dataset schema...</span>';
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
            const numCount = (data.numerical_features || []).length;
            const catCount = (data.categorical_features || []).length;
            const schemaTag = data.is_telco_schema ? 'Standard Churn Schema' : `Custom Schema (${data.columns} cols: ${numCount} num, ${catCount} cat)`;
            statusMsg.innerHTML = `
                <div style="background:var(--bg-glass-hover);padding:10px 14px;border-radius:8px;border:1px solid var(--border-accent);margin-top:8px;">
                    <div style="color:var(--accent-green);font-weight:600;">✅ Successfully Ingested: ${data.filename}</div>
                    <div style="color:var(--text-secondary);font-size:0.85rem;margin-top:4px;">
                        ${data.rows} rows • ${data.columns} columns (${numCount} numerical, ${catCount} categorical) • ${data.size_kb} KB
                    </div>
                    <div style="color:var(--text-accent);font-size:0.80rem;margin-top:2px;">
                        📌 ${data.message}
                    </div>
                </div>
            `;
            await fetchBatches();
            const select = document.getElementById('select-preset-batch');
            if (select) select.value = data.filename;
        } else {
            statusMsg.innerHTML = `<span style="color:var(--accent-red)">❌ Upload failed: ${data.detail || 'Validation error'}</span>`;
        }
    } catch (err) {
        statusMsg.innerHTML = `<span style="color:var(--accent-red)">❌ Upload failed: ${err.message}</span>`;
    }
}

// ── Settings ───────────────────────────────────────────────────────────────
function updateSliderVal(id) {
    if (id === 'ks') {
        const val = document.getElementById('slider-cfg-ks').value;
        document.getElementById('val-cfg-ks').textContent = parseFloat(val).toFixed(2);
        state.thresholds.ks = parseFloat(val);
    } else if (id === 'psi-mod') {
        const val = document.getElementById('slider-cfg-psi-mod').value;
        document.getElementById('val-cfg-psi-mod').textContent = parseFloat(val).toFixed(2);
        state.thresholds.psiMod = parseFloat(val);
    } else if (id === 'psi-sev') {
        const val = document.getElementById('slider-cfg-psi-sev').value;
        document.getElementById('val-cfg-psi-sev').textContent = parseFloat(val).toFixed(2);
        state.thresholds.psiSev = parseFloat(val);
    } else if (id === 'chi2') {
        const val = document.getElementById('slider-cfg-chi2').value;
        document.getElementById('val-cfg-chi2').textContent = parseFloat(val).toFixed(2);
        state.thresholds.chi2 = parseFloat(val);
    } else if (id === 'js') {
        const val = document.getElementById('slider-cfg-js').value;
        document.getElementById('val-cfg-js').textContent = parseFloat(val).toFixed(2);
        state.thresholds.js = parseFloat(val);
    } else if (id === 'risk') {
        const val = document.getElementById('slider-cfg-risk').value;
        document.getElementById('val-cfg-risk').textContent = parseFloat(val).toFixed(2);
        state.thresholds.risk = parseFloat(val);
    }
}

async function saveSettings() {
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                thresholds: {
                    ks_pvalue: state.thresholds.ks,
                    psi_moderate: state.thresholds.psiMod,
                    psi_severe: state.thresholds.psiSev,
                    chi2_pvalue: state.thresholds.chi2,
                    js_moderate: state.thresholds.js,
                    drifted_features_pct_moderate: state.thresholds.risk
                }
            })
        });
        alert('✅ Runtime statistical test thresholds updated successfully.');
    } catch (err) {
        alert('Failed to save settings: ' + err.message);
    }
}

// ── Application Initialization ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize Theme
    setTheme(state.theme);

    // 2. Fetch Batches & Initial Data
    await fetchBatches();
    await fetchAlerts();

    // 3. Execute default initial analysis on batch_01
    executeBatchAnalysis();
});

// Debounced resize listener to keep charts responsive
let resizeTimer = null;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        if (state.analysisData) {
            if (state.activeView === 'view-overview') {
                renderRiskGauge(state.analysisData.dataset_drift);
                renderFeaturePSIChart(state.analysisData.feature_drift);
            } else if (state.activeView === 'view-drift-analysis') {
                renderFeatureDistribution();
            } else if (state.activeView === 'view-model-health') {
                renderModelHealthCharts(state.analysisData.performance);
            }
        }
    }, 150);
});


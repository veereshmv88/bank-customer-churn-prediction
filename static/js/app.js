/**
 * Bank Intelligence — Customer Risk Platform
 * Swiss International Typographic Style × Premium Fintech Controller
 * Vanilla JavaScript Frontend
 */

// Global state and chart registry
const AppState = {
  activeView: "overview",
  metadata: null,
  analytics: null,
  metrics: null,
  sampleCustomers: [],
  charts: {}
};

// Safe Chart Destructor to prevent canvas reuse errors
function renderChart(canvasId, config) {
  if (AppState.charts[canvasId]) {
    AppState.charts[canvasId].destroy();
  }
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  AppState.charts[canvasId] = new Chart(ctx, config);
  return AppState.charts[canvasId];
}

// Theme Manager & Chart Palette Adapter
function getTheme() {
  return document.documentElement.getAttribute("data-theme") || "light";
}

function updateChartThemeDefaults(theme) {
  const isDark = theme === "dark";
  Chart.defaults.color = isDark ? "#A5AFBC" : "#666D76";
  Chart.defaults.borderColor = isDark ? "#28313D" : "#DDE1E5";
  Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.tooltip.backgroundColor = isDark ? "#1B212A" : "#171A1F";
  Chart.defaults.plugins.tooltip.titleColor = isDark ? "#F3F5F7" : "#FFFFFF";
  Chart.defaults.plugins.tooltip.bodyColor = isDark ? "#A5AFBC" : "#DDE1E5";
  Chart.defaults.plugins.tooltip.borderColor = isDark ? "#28313D" : "#2C3038";
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 4;
  Chart.defaults.plugins.tooltip.titleFont = { weight: 700, family: "'Inter', sans-serif" };
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.pointStyle = "circle";
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.padding = 14;
  Chart.defaults.plugins.legend.labels.color = isDark ? "#A5AFBC" : "#666D76";
}

// Initial default styling based on active theme
updateChartThemeDefaults(getTheme());

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("bank_theme", theme);
  updateChartThemeDefaults(theme);
  // Re-render currently visible charts with matching palette
  loadCurrentView();
}

function toggleTheme() {
  const current = getTheme();
  const next = current === "dark" ? "light" : "dark";
  setTheme(next);
}

// View Name Lookup for Breadcrumbs
const VIEW_TITLES = {
  "overview": "Overview",
  "calculator": "Risk Assessment",
  "risk-analytics": "Risk Analytics",
  "model-performance": "Model Performance",
  "explainability": "Explainability",
  "what-if": "Scenario Analysis",
  "customer-analytics": "Customer Insights",
  "methodology": "Methodology"
};

// ==================== VIEW NAVIGATION ====================

function initNavigation() {
  const navLinks = document.querySelectorAll(".nav-link");
  navLinks.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetView = link.getAttribute("data-view");
      switchView(targetView);
    });
  });

  // Topbar theme toggle button
  const themeToggleBtn = document.getElementById("btn-theme-toggle");
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      toggleTheme();
    });
  }

  // Topbar refresh button
  const refreshBtn = document.getElementById("btn-topbar-refresh");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", async () => {
      refreshBtn.style.transform = "rotate(180deg)";
      AppState.analytics = null;
      AppState.metrics = null;
      await loadCurrentView();
      setTimeout(() => { refreshBtn.style.transform = "rotate(0deg)"; }, 300);
    });
  }
}

function switchView(viewName) {
  AppState.activeView = viewName;

  // Update Breadcrumb
  const breadcrumb = document.getElementById("topbar-view-name");
  if (breadcrumb) {
    breadcrumb.textContent = VIEW_TITLES[viewName] || "Dashboard";
  }

  // Update Nav Links
  document.querySelectorAll(".nav-link").forEach(link => {
    if (link.getAttribute("data-view") === viewName) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });

  // Update View Sections
  document.querySelectorAll(".view-section").forEach(sec => {
    if (sec.id === `view-${viewName}`) {
      sec.classList.add("active");
    } else {
      sec.classList.remove("active");
    }
  });

  loadCurrentView();
}

async function loadCurrentView() {
  const viewName = AppState.activeView;
  if (viewName === "overview") await loadOverview();
  if (viewName === "risk-analytics") await loadRiskAnalytics();
  if (viewName === "model-performance") await loadModelPerformance();
  if (viewName === "explainability") await loadExplainability();
  if (viewName === "customer-analytics") await loadCustomerAnalytics();
}

// ==================== API HELPERS ====================

async function fetchJSON(url, options = {}) {
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP error ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.error(`API Fetch Error (${url}):`, err);
    throw err;
  }
}

// ==================== PAGE 1: OVERVIEW ====================

async function loadOverview() {
  try {
    if (!AppState.analytics) {
      AppState.analytics = await fetchJSON("/api/analytics");
    }
    const data = AppState.analytics;
    const kpis = data.kpis;

    document.getElementById("kpi-total-cust").textContent = kpis.total_customers.toLocaleString();
    document.getElementById("kpi-churn-rate").textContent = `${kpis.churn_rate}%`;
    const churnedEl = document.getElementById("kpi-churned");
    if (churnedEl) churnedEl.textContent = kpis.churned_customers.toLocaleString();
    const retainedEl = document.getElementById("kpi-retained");
    if (retainedEl) retainedEl.textContent = kpis.retained_customers.toLocaleString();
    document.getElementById("kpi-high-risk").textContent = kpis.predicted_high_risk.toLocaleString();

    // 1. Churn vs Retention Donut
    renderChart("chart-overview-donut", {
      type: "doughnut",
      data: {
        labels: ["Retained Accounts", "Churned Accounts"],
        datasets: [{
          data: [kpis.retained_customers, kpis.churned_customers],
          backgroundColor: ["#62D6A1", "#FF7777"],
          borderWidth: 2,
          borderColor: "#151920"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" }
        },
        cutout: "70%"
      }
    });

    // 2. Churn by Geography
    const geoLabels = Object.keys(data.geography);
    const geoRates = geoLabels.map(g => data.geography[g].churn_rate);
    renderChart("chart-overview-geo", {
      type: "bar",
      data: {
        labels: geoLabels,
        datasets: [{
          label: "Churn Rate (%)",
          data: geoRates,
          backgroundColor: "#78D6A3",
          borderRadius: 2,
          maxBarThickness: 42
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, max: 40, ticks: { callback: v => v + '%' } }
        }
      }
    });

    // 3. Churn by Age Groups
    const ageLabels = Object.keys(data.age_groups);
    const ageRates = ageLabels.map(a => data.age_groups[a].churn_rate);
    renderChart("chart-overview-age", {
      type: "line",
      data: {
        labels: ageLabels.map(l => l + " yrs"),
        datasets: [{
          label: "Attrition Rate (%)",
          data: ageRates,
          borderColor: "#FF7777",
          backgroundColor: "rgba(255, 119, 119, 0.12)",
          fill: true,
          tension: 0.2,
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: "#FF7777"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, max: 65, ticks: { callback: v => v + '%' } }
        }
      }
    });

    // 4. Active Status Impact (From Reference Design)
    if (data.activity && document.getElementById("chart-overview-active")) {
      const actKeys = ["Active Member", "Inactive Member"];
      const actRates = actKeys.map(k => data.activity[k] ? data.activity[k].churn_rate : 0);
      renderChart("chart-overview-active", {
        type: "bar",
        data: {
          labels: actKeys,
          datasets: [{
            label: "Attrition Rate (%)",
            data: actRates,
            backgroundColor: ["#62D6A1", "#FF7777"],
            borderRadius: 2,
            maxBarThickness: 46
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false } },
            y: { beginAtZero: true, max: 35, ticks: { callback: v => v + '%' } }
          }
        }
      });
    }

  } catch (err) {
    console.error("Overview error:", err);
  }
}

// ==================== PAGE 2: RISK ASSESSMENT ====================

function initCalculator() {
  const calcForm = document.getElementById("form-calculator");
  if (!calcForm) return;

  calcForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    await handleCalculateRisk();
  });

  // Preset buttons
  document.querySelectorAll(".calc-preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.getAttribute("data-preset-idx"));
      if (AppState.sampleCustomers && AppState.sampleCustomers[idx]) {
        populateCalculatorForm(AppState.sampleCustomers[idx]);
      }
    });
  });
}

function getCalculatorFormData() {
  return {
    CreditScore: parseFloat(document.getElementById("calc-credit").value),
    Geography: document.getElementById("calc-geo").value,
    Gender: document.getElementById("calc-gender").value,
    Age: parseFloat(document.getElementById("calc-age").value),
    Tenure: parseFloat(document.getElementById("calc-tenure").value),
    Balance: parseFloat(document.getElementById("calc-balance").value),
    NumOfProducts: parseInt(document.getElementById("calc-products").value),
    HasCrCard: document.getElementById("calc-crcard").checked ? 1 : 0,
    IsActiveMember: document.getElementById("calc-active").checked ? 1 : 0,
    EstimatedSalary: parseFloat(document.getElementById("calc-salary").value)
  };
}

function populateCalculatorForm(profile) {
  document.getElementById("calc-credit").value = profile.CreditScore;
  document.getElementById("calc-geo").value = profile.Geography;
  document.getElementById("calc-gender").value = profile.Gender;
  document.getElementById("calc-age").value = profile.Age;
  document.getElementById("calc-tenure").value = profile.Tenure;
  document.getElementById("calc-balance").value = profile.Balance;
  document.getElementById("calc-products").value = profile.NumOfProducts;
  document.getElementById("calc-salary").value = profile.EstimatedSalary;
  document.getElementById("calc-crcard").checked = profile.HasCrCard === 1;
  document.getElementById("calc-active").checked = profile.IsActiveMember === 1;
}

async function handleCalculateRisk() {
  const btn = document.getElementById("btn-calc-submit");
  const errBox = document.getElementById("calc-error-box");
  errBox.style.display = "none";
  btn.disabled = true;
  btn.textContent = "Computing Risk Estimate...";

  try {
    const profile = getCalculatorFormData();
    const res = await fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile)
    });

    renderCalculatorResults(res);
  } catch (err) {
    errBox.textContent = err.message || "Failed to score customer profile.";
    errBox.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Assess Customer Risk &rarr;";
  }
}

function renderCalculatorResults(res) {
  // Hide empty state and show result container
  document.getElementById("calc-empty-state").style.display = "none";
  const resultContent = document.getElementById("calc-result-content");
  resultContent.style.display = "block";
  resultContent.classList.add("visible");

  const probVal = document.getElementById("res-probability");
  const scoreVal = document.getElementById("res-score");
  const badgeVal = document.getElementById("res-category-badge");
  const classVal = document.getElementById("res-predicted-class");
  const modelVal = document.getElementById("res-model-used");
  const factorsList = document.getElementById("res-factors-list");
  const progressFill = document.getElementById("res-progress-fill");

  probVal.textContent = `${res.churn_percentage}%`;
  scoreVal.textContent = `${res.risk_score} / 100`;
  badgeVal.textContent = res.risk_category.toUpperCase();

  // Color mapping
  const catKey = res.risk_category.toLowerCase().split(' ')[0];
  badgeVal.className = `badge-risk ${catKey}`;

  // Progress track fill
  if (progressFill) {
    progressFill.style.width = `${Math.min(100, Math.max(3, res.churn_percentage))}%`;
    progressFill.style.backgroundColor = catKey === "high" ? "#C83F49" : (catKey === "medium" ? "#B66A12" : "#147D55");
  }

  classVal.textContent = res.predicted_class === 1 ? "At-Risk Account (Class 1)" : "Retained Account (Class 0)";
  modelVal.textContent = `Model: ${res.model_name} (Optimal Decision Threshold: ${res.classification_threshold})`;

  // Render top factors
  factorsList.innerHTML = "";
  if (res.top_factors && res.top_factors.length > 0) {
    res.top_factors.forEach(f => {
      const row = document.createElement("div");
      row.className = "factor-row";
      const isPos = f.impact > 0;
      const methodLabel = f.attribution_method ? ` [${f.attribution_method}]` : "";
      row.innerHTML = `
        <span style="color: var(--text-primary); font-weight: 500;">${f.feature}${methodLabel}</span>
        <span class="factor-tag ${isPos ? 'inc' : 'dec'}">
          ${isPos ? '▲ +' : '▼ '}${f.impact} log-odds (${f.direction})
        </span>
      `;
      factorsList.appendChild(row);
    });
  } else {
    factorsList.innerHTML = "<span style='font-size:12px; color:var(--text-muted);'>Attribution completed.</span>";
  }
}

// ==================== PAGE 3: RISK ANALYTICS ====================

async function loadRiskAnalytics() {
  try {
    if (!AppState.analytics) {
      AppState.analytics = await fetchJSON("/api/analytics");
    }
    const data = AppState.analytics;
    const kpis = data.kpis;

    document.getElementById("ra-low-risk").textContent = kpis.predicted_low_risk.toLocaleString();
    document.getElementById("ra-med-risk").textContent = kpis.predicted_medium_risk.toLocaleString();
    document.getElementById("ra-high-risk").textContent = kpis.predicted_high_risk.toLocaleString();
    document.getElementById("ra-avg-prob").textContent = `${kpis.avg_predicted_probability}%`;

    // 1. Risk Tier Donut
    renderChart("chart-ra-donut", {
      type: "doughnut",
      data: {
        labels: ["Low Risk (<25%)", "Medium Risk (25–50%)", "High Risk (≥50%)"],
        datasets: [{
          data: [kpis.predicted_low_risk, kpis.predicted_medium_risk, kpis.predicted_high_risk],
          backgroundColor: ["#62D6A1", "#F3BE68", "#FF7777"],
          borderWidth: 2,
          borderColor: "#151920"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "bottom" } },
        cutout: "68%"
      }
    });

    // 2. Probability Distribution Histogram
    const bands = data.risk_bands;
    renderChart("chart-ra-histogram", {
      type: "bar",
      data: {
        labels: Object.keys(bands),
        datasets: [{
          label: "Number of Customers",
          data: Object.values(bands),
          backgroundColor: "#86B7FF",
          borderRadius: 2,
          maxBarThickness: 40
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true }
        }
      }
    });

    // 3. Populate probability band ledger table
    const tbody = document.getElementById("table-risk-bands-body");
    if (tbody && bands) {
      tbody.innerHTML = "";
      const total = kpis.total_customers || 10000;
      Object.entries(bands).forEach(([interval, count]) => {
        const pct = ((count / total) * 100).toFixed(2);
        let tierClass = "low";
        let tierLabel = "LOW RISK";
        if (interval.includes("25") || interval.includes("50")) {
          tierClass = "medium";
          tierLabel = "MEDIUM RISK";
        }
        if (interval.includes("50") || interval.includes("75") || interval.includes("100")) {
          tierClass = "high";
          tierLabel = interval.includes("75") ? "CRITICAL RISK" : "HIGH RISK";
        }
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${interval}</strong></td>
          <td class="num-tabular">${count.toLocaleString()}</td>
          <td class="num-tabular">${pct}%</td>
          <td><span class="badge-risk ${tierClass}">${tierLabel}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

  } catch (err) {
    console.error("Risk analytics error:", err);
  }
}

// ==================== PAGE 4: MODEL PERFORMANCE & TABS ====================

function initModelPerformanceTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(btn.getAttribute("data-tab"));
      if (targetPanel) targetPanel.classList.add("active");
    });
  });
}

async function loadModelPerformance() {
  try {
    if (!AppState.metrics) {
      AppState.metrics = await fetchJSON("/api/metrics");
    }
    const data = AppState.metrics;

    // Populate comparison table
    const tbody = document.getElementById("table-model-comp-body");
    tbody.innerHTML = "";
    data.cv_comparison.forEach(m => {
      const tr = document.createElement("tr");
      if (m.Model === data.selected_model) tr.className = "highlight";
      tr.innerHTML = `
        <td><strong>${m.Model}</strong> ${m.Model === data.selected_model ? '<span class="badge-risk low" style="padding:1px 6px; font-size:10px; margin-left:6px;">SELECTED</span>' : ''}</td>
        <td class="num-tabular">${(m.Accuracy * 100).toFixed(2)}%</td>
        <td class="num-tabular">${(m.Precision * 100).toFixed(2)}%</td>
        <td class="num-tabular">${(m.Recall * 100).toFixed(2)}%</td>
        <td class="num-tabular">${(m.F1_Score * 100).toFixed(2)}%</td>
        <td class="num-tabular">${m.ROC_AUC.toFixed(4)}</td>
        <td class="num-tabular">${m.PR_AUC.toFixed(4)}</td>
      `;
      tbody.appendChild(tr);
    });

    // Confusion Matrix at optimal operating threshold (0.28)
    const optM = data.test_metrics_optimal_threshold || data.test_metrics_default_threshold;
    document.getElementById("cm-tn").textContent = optM.true_negatives.toLocaleString();
    document.getElementById("cm-fp").textContent = optM.false_positives.toLocaleString();
    document.getElementById("cm-fn").textContent = optM.false_negatives.toLocaleString();
    document.getElementById("cm-tp").textContent = optM.true_positives.toLocaleString();

    // ROC Curve
    const rocPoints = data.curve_data.roc_curve;
    renderChart("chart-perf-roc", {
      type: "line",
      data: {
        datasets: [
          {
            label: `${data.selected_model} (AUC = ${optM.roc_auc})`,
            data: rocPoints,
            borderColor: "#78D6A3",
            backgroundColor: "rgba(120, 214, 163, 0.08)",
            fill: true,
            pointRadius: 0,
            borderWidth: 2
          },
          {
            label: "Random Reference",
            data: [{x:0, y:0}, {x:1, y:1}],
            borderColor: "#788493",
            borderDash: [4, 4],
            pointRadius: 0,
            borderWidth: 1.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate" } },
          y: { min: 0, max: 1, title: { display: true, text: "True Positive Rate (Recall)" } }
        }
      }
    });

    // PR Curve
    const prPoints = data.curve_data.pr_curve;
    renderChart("chart-perf-pr", {
      type: "line",
      data: {
        datasets: [{
          label: `PR Curve (AP = ${optM.pr_auc})`,
          data: prPoints,
          borderColor: "#86B7FF",
          backgroundColor: "rgba(134, 183, 255, 0.08)",
          fill: true,
          pointRadius: 0,
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Recall" } },
          y: { min: 0, max: 1, title: { display: true, text: "Precision" } }
        }
      }
    });

    // Calibration Curve
    const calibPoints = data.curve_data.calibration_curve;
    renderChart("chart-perf-calib", {
      type: "line",
      data: {
        datasets: [
          {
            label: `Calibration (Brier = ${optM.brier_score})`,
            data: calibPoints,
            borderColor: "#62D6A1",
            pointRadius: 4,
            pointBackgroundColor: "#62D6A1",
            borderWidth: 2
          },
          {
            label: "Perfect Calibration",
            data: [{x:0, y:0}, {x:1, y:1}],
            borderColor: "#788493",
            borderDash: [4, 4],
            pointRadius: 0,
            borderWidth: 1.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "Mean Predicted Probability" } },
          y: { min: 0, max: 1, title: { display: true, text: "Observed Positive Fraction" } }
        }
      }
    });

  } catch (err) {
    console.error("Model performance load error:", err);
  }
}

// ==================== PAGE 5: EXPLAINABILITY ====================

async function loadExplainability() {
  try {
    const data = await fetchJSON("/api/explain");
    const topFeats = data.global_importance.slice(0, 10);

    renderChart("chart-explain-importance", {
      type: "bar",
      data: {
        labels: topFeats.map(f => f.feature),
        datasets: [{
          label: "Relative Importance",
          data: topFeats.map(f => f.importance),
          backgroundColor: "#78D6A3",
          borderRadius: 2,
          maxBarThickness: 18
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { beginAtZero: true, grid: { color: "#28313D" } },
          y: { grid: { display: false } }
        }
      }
    });
  } catch (err) {
    console.error("Explainability load error:", err);
  }
}

// ==================== PAGE 6: SCENARIO ANALYSIS ====================

function initWhatIfSimulator() {
  const syncBtn = document.getElementById("btn-whatif-sync");
  const simBtn = document.getElementById("btn-whatif-run");

  if (syncBtn) {
    syncBtn.addEventListener("click", () => {
      // Copy current form to scenario form
      const curr = getWhatIfProfile("curr");
      setWhatIfProfile("scen", curr);
      // Run comparison automatically
      runWhatIfSimulation();
    });
  }

  if (simBtn) {
    simBtn.addEventListener("click", async () => {
      await runWhatIfSimulation();
    });
  }
}

function getWhatIfProfile(prefix) {
  return {
    CreditScore: parseFloat(document.getElementById(`${prefix}-credit`).value),
    Geography: document.getElementById(`${prefix}-geo`).value,
    Gender: document.getElementById(`${prefix}-gender`).value,
    Age: parseFloat(document.getElementById(`${prefix}-age`).value),
    Tenure: parseFloat(document.getElementById(`${prefix}-tenure`).value),
    Balance: parseFloat(document.getElementById(`${prefix}-balance`).value),
    NumOfProducts: parseInt(document.getElementById(`${prefix}-products`).value),
    HasCrCard: document.getElementById(`${prefix}-crcard`).checked ? 1 : 0,
    IsActiveMember: document.getElementById(`${prefix}-active`).checked ? 1 : 0,
    EstimatedSalary: parseFloat(document.getElementById(`${prefix}-salary`).value)
  };
}

function setWhatIfProfile(prefix, profile) {
  document.getElementById(`${prefix}-credit`).value = profile.CreditScore;
  document.getElementById(`${prefix}-geo`).value = profile.Geography;
  document.getElementById(`${prefix}-gender`).value = profile.Gender;
  document.getElementById(`${prefix}-age`).value = profile.Age;
  document.getElementById(`${prefix}-tenure`).value = profile.Tenure;
  document.getElementById(`${prefix}-balance`).value = profile.Balance;
  document.getElementById(`${prefix}-products`).value = profile.NumOfProducts;
  document.getElementById(`${prefix}-salary`).value = profile.EstimatedSalary;
  document.getElementById(`${prefix}-crcard`).checked = profile.HasCrCard === 1;
  document.getElementById(`${prefix}-active`).checked = profile.IsActiveMember === 1;
}

async function runWhatIfSimulation() {
  const btn = document.getElementById("btn-whatif-run");
  const errBox = document.getElementById("whatif-error-box");
  errBox.style.display = "none";
  btn.disabled = true;
  btn.textContent = "Comparing Profiles...";

  try {
    const currentProfile = getWhatIfProfile("curr");
    const scenarioProfile = getWhatIfProfile("scen");

    const res = await fetchJSON("/api/what-if", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ current: currentProfile, scenario: scenarioProfile })
    });

    renderWhatIfResults(res);
  } catch (err) {
    errBox.textContent = err.message || "Simulation failed.";
    errBox.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Run Scenario Comparison &rarr;";
  }
}

function renderWhatIfResults(res) {
  document.getElementById("whatif-curr-prob").textContent = `${res.current.churn_percentage}%`;
  document.getElementById("whatif-curr-score").textContent = `${res.current.risk_score} / 100`;
  document.getElementById("whatif-curr-cat").textContent = res.current.risk_category.toUpperCase();
  document.getElementById("whatif-curr-cat").className = `badge-risk ${res.current.risk_category.toLowerCase().split(' ')[0]}`;

  document.getElementById("whatif-scen-prob").textContent = `${res.scenario.churn_percentage}%`;
  document.getElementById("whatif-scen-score").textContent = `${res.scenario.risk_score} / 100`;
  document.getElementById("whatif-scen-cat").textContent = res.scenario.risk_category.toUpperCase();
  document.getElementById("whatif-scen-cat").className = `badge-risk ${res.scenario.risk_category.toLowerCase().split(' ')[0]}`;

  const deltaEl = document.getElementById("whatif-delta-val");
  const deltaText = document.getElementById("whatif-summary-text");
  const isIncrease = res.probability_difference_pp > 0;
  deltaEl.textContent = `${isIncrease ? '+' : ''}${res.probability_difference_pp}% pts`;
  deltaEl.className = `delta-number ${isIncrease ? 'inc' : 'dec'}`;
  deltaText.textContent = res.summary;

  // Comparison Bar Chart
  renderChart("chart-whatif-compare", {
    type: "bar",
    data: {
      labels: ["Churn Probability (%)", "Risk Score (/100)"],
      datasets: [
        {
          label: "Current Profile",
          data: [res.current.churn_percentage, res.current.risk_score],
          backgroundColor: "#86B7FF",
          borderRadius: 2,
          maxBarThickness: 34
        },
        {
          label: "Scenario Profile",
          data: [res.scenario.churn_percentage, res.scenario.risk_score],
          backgroundColor: isIncrease ? "#FF7777" : "#62D6A1",
          borderRadius: 2,
          maxBarThickness: 34
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { beginAtZero: true, max: 100 }
      }
    }
  });
}

// ==================== PAGE 7: CUSTOMER INSIGHTS ====================

async function loadCustomerAnalytics() {
  try {
    if (!AppState.analytics) {
      AppState.analytics = await fetchJSON("/api/analytics");
    }
    const data = AppState.analytics;

    // 1. Products vs Churn
    const prodKeys = Object.keys(data.products);
    renderChart("chart-ca-products", {
      type: "bar",
      data: {
        labels: prodKeys.map(k => `${k} Product(s)`),
        datasets: [{
          label: "Churn Rate (%)",
          data: prodKeys.map(k => data.products[k].churn_rate),
          backgroundColor: "#FF7777",
          borderRadius: 2,
          maxBarThickness: 38
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' } }
        }
      }
    });

    // 2. Credit Card vs Churn
    const ccKeys = Object.keys(data.credit_card);
    renderChart("chart-ca-creditcard", {
      type: "bar",
      data: {
        labels: ccKeys,
        datasets: [{
          label: "Churn Rate (%)",
          data: ccKeys.map(k => data.credit_card[k].churn_rate),
          backgroundColor: "#78D6A3",
          borderRadius: 2,
          maxBarThickness: 38
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, max: 30, ticks: { callback: v => v + '%' } }
        }
      }
    });

    // 3. Gender vs Churn
    const genKeys = Object.keys(data.gender);
    renderChart("chart-ca-gender", {
      type: "bar",
      data: {
        labels: genKeys,
        datasets: [{
          label: "Churn Rate (%)",
          data: genKeys.map(k => data.gender[k].churn_rate),
          backgroundColor: ["#86B7FF", "#78D6A3"],
          borderRadius: 2,
          maxBarThickness: 38
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, max: 35, ticks: { callback: v => v + '%' } }
        }
      }
    });

  } catch (err) {
    console.error("Customer insights error:", err);
  }
}

// ==================== APP INITIALIZATION ====================

async function initApp() {
  initNavigation();
  initCalculator();
  initModelPerformanceTabs();
  initWhatIfSimulator();

  try {
    // Health check & metadata
    const health = await fetchJSON("/api/health");
    if (health.model_loaded) {
      document.getElementById("system-status-text").textContent = "Production Online";
      const modelBadge = document.getElementById("topbar-model-badge");
      if (modelBadge) {
        modelBadge.textContent = `${health.model_name} (Threshold: ${health.threshold})`;
      }
      const sidebarModelInfo = document.getElementById("sidebar-model-info");
      if (sidebarModelInfo) {
        sidebarModelInfo.textContent = `${health.model_name} · T=${health.threshold}`;
      }
    }

    // Sample presets
    const sampleResp = await fetchJSON("/api/customers");
    AppState.sampleCustomers = sampleResp.customers || [];

    // Trigger initial overview load
    await loadOverview();
  } catch (err) {
    console.error("Initialization failed:", err);
    document.getElementById("system-status-text").textContent = "Service Offline";
  }
}

document.addEventListener("DOMContentLoaded", initApp);
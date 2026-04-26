const form = document.getElementById('predictForm');
const bodyEl = document.getElementById('body');
const compareToggle = document.getElementById('compareToggle');
const predictBtn = document.getElementById('predictBtn');
const formError = document.getElementById('formError');
const resultsEl = document.getElementById('results');
const resultSummary = document.getElementById('resultSummary');
const healthStatus = document.getElementById('healthStatus');
const modelInfo = document.getElementById('modelInfo');
const refreshInfoBtn = document.getElementById('refreshInfo');

function labelClass(label) {
  if (label === 'Positive') return 'Positive';
  if (label === 'Negative') return 'Negative';
  return 'Neutral';
}

function displayModelName(result) {
  return result.model_display_name || result.model_key;
}

function renderResultCard(result) {
  const probRows = Object.entries(result.probabilities || {})
    .map(([label, value]) => {
      const percent = Math.round(Number(value) * 100);
      return `
        <div class="prob-row">
          <span>${label}</span>
          <div class="bar"><span style="width:${percent}%"></span></div>
          <span>${percent}%</span>
        </div>
      `;
    }).join('');

  return `
    <article class="prediction-box">
      <div class="prediction-head">
        <div>
          <div class="model-name">${displayModelName(result)}</div>
          <div class="model-version">${result.model_version || '-'}</div>
        </div>
        <span class="badge ${labelClass(result.label)}">${result.label}</span>
      </div>

      <div class="meta-grid">
        <div class="meta-item">
          <strong>Confidence</strong>
          ${(Number(result.confidence) * 100).toFixed(1)}%
        </div>
        <div class="meta-item">
          <strong>Latency</strong>
          ${result.latency_ms} ms
        </div>
      </div>

      <div class="prob-bars">${probRows}</div>
    </article>
  `;
}

async function checkHealth() {
  if (!healthStatus) return;

  try {
    const res = await fetch('/health');
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'API Error');
    }

    healthStatus.textContent = `API OK · ${data.models_loaded.length} models`;
    healthStatus.classList.remove('is-error');
    healthStatus.classList.add('is-ok');
  } catch (err) {
    healthStatus.textContent = 'API Error';
    healthStatus.classList.remove('is-ok');
    healthStatus.classList.add('is-error');
  }
}

async function loadDatasetSample(label) {
  formError.textContent = '';

  try {
    const res = await fetch(`/examples?label=${encodeURIComponent(label)}&n=1`);
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'โหลดตัวอย่างไม่ได้');
    }

    if (!data.items || data.items.length === 0) {
      throw new Error('ไม่พบตัวอย่างใน dataset');
    }

    bodyEl.value = data.items[0].text;
    bodyEl.focus();
  } catch (err) {
    formError.textContent = err.message || 'โหลดตัวอย่างจาก dataset ไม่ได้';
  }
}

if (form) {
  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    formError.textContent = '';
    resultsEl.innerHTML = '';
    resultSummary.textContent = 'กำลังประมวลผล...';
    predictBtn.disabled = true;

    try {
      const payload = {
        body: bodyEl.value,
        compare: compareToggle.checked,
        model_key: 'auto'
      };

      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'เกิดข้อผิดพลาดในการประมวลผล');
      }

      resultSummary.innerHTML = `
        Primary result:
        <strong>${data.primary_result.label}</strong>
        · Confidence ${(Number(data.primary_result.confidence) * 100).toFixed(1)}%
        · Total latency ${data.latency_ms} ms
      `;

      resultsEl.innerHTML = data.results
        .map(result => renderResultCard(result))
        .join('');
    } catch (err) {
      resultSummary.textContent = 'ยังไม่มีผลลัพธ์';
      formError.textContent = err.message || 'เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง';
    } finally {
      predictBtn.disabled = false;
    }
  });
}

document.querySelectorAll('[data-label]').forEach((btn) => {
  btn.addEventListener('click', () => {
    loadDatasetSample(btn.dataset.label);
  });
});

if (refreshInfoBtn) {
  refreshInfoBtn.addEventListener('click', async () => {
    modelInfo.textContent = 'Loading model info...';

    try {
      const res = await fetch('/model/info');
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'โหลดข้อมูลโมเดลไม่ได้');
      }

      const lines = data.available_models.map((item) => {
        const name = item.model_display_name || item.model_key;
        const accuracy = item.metrics?.accuracy ?? '-';
        const macroF1 = item.metrics?.macro_f1 ?? '-';
        return `${name}: ${item.model_version || '-'} · Acc ${accuracy} · Macro-F1 ${macroF1}`;
      });

      modelInfo.innerHTML = `
        <strong>Best model:</strong> ${data.best_model_key}<br>
        ${lines.join('<br>')}
      `;
    } catch (err) {
      modelInfo.textContent = err.message || 'โหลดข้อมูลโมเดลไม่ได้';
    }
  });
}

checkHealth();

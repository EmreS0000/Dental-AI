/**
 * Dental AI v2 - Frontend Application
 * =====================================
 * - Röntgen yükleme ve analiz
 * - Sonuç görselleştirme
 * - İnteraktif FDI diş haritası
 * - Klinik bulgu ve tedavi paneli
 */

const API_BASE = window.location.origin;

// Hastalık renkleri (config.py ile uyumlu)
const DISEASE_COLORS = {
    caries: '#FFA500',
    deep_caries: '#FF0000',
    periapical_lesion: '#FF00FF',
    impacted_tooth: '#00FFFF',
    non_odontogenic_lesion: '#00FF00',
    pericoronal_lesion: '#4444FF',
};

const DISEASE_NAMES_TR = {
    caries: 'Çürük',
    deep_caries: 'Derin Çürük',
    periapical_lesion: 'Periapikal Lezyon',
    impacted_tooth: 'Gömülü Diş',
    non_odontogenic_lesion: 'Non-odontogenic Lezyon',
    pericoronal_lesion: 'Perikoronal Lezyon',
};

// ═══════════════════════════════════════════════════════════════
// DOM Elements
// ═══════════════════════════════════════════════════════════════
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const uploadBtn = document.getElementById('upload-btn');
const uploadPreview = document.getElementById('upload-preview');
const previewImage = document.getElementById('preview-image');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const resultsSection = document.getElementById('results-section');
const visImage = document.getElementById('vis-image');
const findingsList = document.getElementById('findings-list');
const modelStatus = document.getElementById('model-status');

let selectedFile = null;

// ═══════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initToothMap();
    buildLegend();
    checkHealth();
});

// ═══════════════════════════════════════════════════════════════
// Health Check
// ═══════════════════════════════════════════════════════════════
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        const dot = modelStatus.querySelector('.status-dot');
        const text = modelStatus.querySelector('.status-text');
        if (data.model_status === 'loaded') {
            dot.classList.add('online');
            text.textContent = 'Model Hazır';
        } else {
            text.textContent = 'Model Yükleniyor...';
        }
    } catch (e) {
        modelStatus.querySelector('.status-text').textContent = 'API Bağlantısı Yok';
    }
}

// ═══════════════════════════════════════════════════════════════
// Upload Handling
// ═══════════════════════════════════════════════════════════════
function initUpload() {
    uploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    // Drag & drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    analyzeBtn.addEventListener('click', () => runAnalysis());
    clearBtn.addEventListener('click', () => clearUpload());
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Lütfen bir görsel dosyası seçin (JPEG, PNG)');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        uploadPreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadArea.style.display = '';
    uploadPreview.style.display = 'none';
    resultsSection.style.display = 'none';
    resetToothMap();
}

// ═══════════════════════════════════════════════════════════════
// Analysis
// ═══════════════════════════════════════════════════════════════
async function runAnalysis() {
    if (!selectedFile) return;

    loadingOverlay.style.display = 'flex';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('xray_image', selectedFile);

    try {
        const res = await fetch(`${API_BASE}/api/v2/analyze`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Analiz başarısız');
        }

        const data = await res.json();
        displayResults(data);
    } catch (e) {
        alert('Analiz hatası: ' + e.message);
        console.error(e);
    } finally {
        loadingOverlay.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// ═══════════════════════════════════════════════════════════════
// Display Results
// ═══════════════════════════════════════════════════════════════
function displayResults(data) {
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    const report = data.report;

    // Summary
    document.getElementById('stat-findings').textContent = report.summary.total_findings;
    document.getElementById('stat-teeth').textContent = report.summary.affected_teeth;
    document.getElementById('stat-priority').textContent = report.summary.high_priority_count;

    // Visualization image
    if (data.visualization) {
        visImage.src = data.visualization;
    }

    // Update tooth map
    resetToothMap();
    updateToothMap(report.all_detections || []);

    // Findings list
    renderFindings(report.teeth || []);
}

// ═══════════════════════════════════════════════════════════════
// Tooth Map
// ═══════════════════════════════════════════════════════════════
function initToothMap() {
    document.querySelectorAll('.tooth').forEach(el => {
        el.addEventListener('mouseenter', (e) => showToothTooltip(e, el));
        el.addEventListener('mouseleave', () => hideToothTooltip(el));
    });
}

function showToothTooltip(e, el) {
    const fdi = el.dataset.fdi;
    const existing = el.querySelector('.tooth-tooltip');
    if (existing) return;

    const diseases = el._diseases || [];
    let text = `Diş #${fdi}`;
    if (diseases.length > 0) {
        text += ': ' + diseases.map(d => DISEASE_NAMES_TR[d] || d).join(', ');
    }

    const tip = document.createElement('div');
    tip.className = 'tooth-tooltip';
    tip.textContent = text;
    el.appendChild(tip);
}

function hideToothTooltip(el) {
    const tip = el.querySelector('.tooth-tooltip');
    if (tip) tip.remove();
}

function resetToothMap() {
    document.querySelectorAll('.tooth').forEach(el => {
        el.className = 'tooth';
        el._diseases = [];
    });
}

function updateToothMap(detections) {
    const toothDiseases = {};

    detections.forEach(det => {
        const fdi = det.tooth_number;
        const disease = det.class || det.disease;
        if (!toothDiseases[fdi]) toothDiseases[fdi] = [];
        if (!toothDiseases[fdi].includes(disease)) {
            toothDiseases[fdi].push(disease);
        }
    });

    Object.entries(toothDiseases).forEach(([fdi, diseases]) => {
        const el = document.getElementById(`tooth-${fdi}`);
        if (!el) return;

        el.classList.add('affected');
        el._diseases = diseases;

        // En ciddi hastalığın rengini kullan
        const priorityOrder = ['deep_caries', 'periapical_lesion', 'non_odontogenic_lesion',
                                'pericoronal_lesion', 'impacted_tooth', 'caries'];
        let primaryDisease = diseases[0];
        for (const p of priorityOrder) {
            if (diseases.includes(p)) { primaryDisease = p; break; }
        }
        el.classList.add(`disease-${primaryDisease}`);
    });
}

function buildLegend() {
    const legend = document.getElementById('legend');
    Object.entries(DISEASE_NAMES_TR).forEach(([key, name]) => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `<span class="legend-color" style="background:${DISEASE_COLORS[key]}"></span>${name}`;
        legend.appendChild(item);
    });
}

// ═══════════════════════════════════════════════════════════════
// Render Findings
// ═══════════════════════════════════════════════════════════════
function renderFindings(teeth) {
    findingsList.innerHTML = '';

    if (!teeth || teeth.length === 0) {
        findingsList.innerHTML = `
            <div class="no-findings">
                <div class="nf-icon">✅</div>
                <h4>Patolojik Bulgu Saptanmadı</h4>
                <p>Görsel analizde belirgin hastalık tespit edilmemiştir.</p>
            </div>
        `;
        return;
    }

    teeth.forEach((tooth, idx) => {
        tooth.findings.forEach((finding, fIdx) => {
            const card = document.createElement('div');
            card.className = 'finding-card';
            card.style.animationDelay = `${(idx * 2 + fIdx) * 0.08}s`;

            const disease = finding.disease || finding.class;
            const diseaseTr = finding.disease_tr || DISEASE_NAMES_TR[disease] || disease;
            const color = DISEASE_COLORS[disease] || '#888';
            const conf = finding.confidence || 0;
            const treatment = finding.treatment || {};
            const priority = treatment.priority || 'orta';

            const borderColor = color;
            card.style.borderLeftColor = borderColor;

            const badgeClass = priority === 'yüksek' ? 'badge-high' : 'badge-medium';
            const badgeText = priority === 'yüksek' ? 'Yüksek Öncelik' : 'Orta Öncelik';

            card.innerHTML = `
                <div class="finding-header">
                    <span class="finding-tooth">
                        <span class="fdi-num">#${tooth.tooth_number}</span> ${tooth.tooth_name}
                    </span>
                    <span class="finding-badge ${badgeClass}">${badgeText}</span>
                </div>
                <div class="finding-disease">
                    <span class="disease-dot" style="background:${color}"></span>
                    <span class="disease-name">${diseaseTr}</span>
                </div>
                <div class="finding-conf">Güven Skoru: %${(conf * 100).toFixed(1)}</div>
                ${treatment.tr ? `
                <div class="finding-treatment">
                    <span class="treatment-label">💊 Önerilen Tedavi</span>
                    ${treatment.tr}
                </div>
                ` : ''}
            `;

            // Karta tıklayınca ilgili dişi vurgula
            card.addEventListener('click', () => {
                const toothEl = document.getElementById(`tooth-${tooth.tooth_number}`);
                if (toothEl) {
                    toothEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    toothEl.style.transform = 'scale(1.4)';
                    setTimeout(() => { toothEl.style.transform = ''; }, 600);
                }
            });

            findingsList.appendChild(card);
        });
    });
}

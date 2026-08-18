/**
 * DarwixAI — Enterprise RAG & Q1 Voice Agent Interactive Dashboard
 * Client Controller & Real-Time Orchestrator
 */

document.addEventListener('DOMContentLoaded', () => {
  // ==========================================
  // 1. Navigation & Tab Switching
  // ==========================================
  const navItems = document.querySelectorAll('.nav-item');
  const tabPanes = document.querySelectorAll('.tab-pane');

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      navItems.forEach(n => n.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      item.classList.add('active');
      const tabId = `pane-${item.dataset.tab}`;
      const targetPane = document.getElementById(tabId);
      if (targetPane) targetPane.classList.add('active');

      // Refresh specific tab views when selected
      if (item.dataset.tab === 'knowledge') loadDocuments();
      if (item.dataset.tab === 'vectorstore') loadChunks();
    });
  });

  // ==========================================
  // 2. System Telemetry & Status
  // ==========================================
  const headerEmbModel = document.getElementById('header-emb-model');
  const headerLlmModel = document.getElementById('header-llm-model');
  const headerChunkCount = document.getElementById('header-chunk-count');
  const navDocCount = document.getElementById('nav-doc-count');
  const docTableCount = document.getElementById('doc-table-count');

  async function fetchTelemetry() {
    try {
      const res = await fetch('/api/v1/stats');
      if (res.ok) {
        const stats = await res.json();
        if (headerEmbModel) headerEmbModel.textContent = stats.embedding_model.split('/').pop() || stats.embedding_model;
        if (headerLlmModel) headerLlmModel.textContent = stats.llm_model.split('/').pop() || stats.llm_provider;
        if (headerChunkCount) headerChunkCount.textContent = stats.total_indexed_chunks;
        if (navDocCount) navDocCount.textContent = stats.total_raw_documents;
        if (docTableCount) docTableCount.textContent = stats.total_raw_documents;
      }
    } catch (err) {
      console.warn('Telemetry load failed:', err);
    }
  }

  fetchTelemetry();

  // Quick Ingest Button in Header
  const btnQuickIngest = document.getElementById('btn-quick-ingest');
  if (btnQuickIngest) {
    btnQuickIngest.addEventListener('click', async () => {
      btnQuickIngest.disabled = true;
      btnQuickIngest.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Syncing...';
      try {
        const res = await fetch('/api/v1/ingest', { method: 'POST' });
        const data = await res.json();
        alert(`Ingestion Complete: ${data.message}`);
        fetchTelemetry();
        loadDocuments();
      } catch (e) {
        alert('Ingestion error: ' + e);
      } finally {
        btnQuickIngest.disabled = false;
        btnQuickIngest.innerHTML = '<i class="fa-solid fa-rotate"></i> <span>Sync Base</span>';
      }
    });
  }

  // ==========================================
  // 3. RAG Studio & Query Execution
  // ==========================================
  const queryInput = document.getElementById('query-input');
  const btnSubmitQuery = document.getElementById('btn-submit-query');
  const btnClearQuery = document.getElementById('btn-clear-query');
  const answerBody = document.getElementById('answer-body');
  const availabilityBadge = document.getElementById('availability-badge');
  const citationsWrapper = document.getElementById('citations-wrapper');
  const citationsList = document.getElementById('citations-list');
  const citationCount = document.getElementById('citation-count');
  const debugContentBody = document.getElementById('debug-content-body');
  const voiceSpeechBox = document.getElementById('voice-speech-box');
  const btnPlayVoice = document.getElementById('btn-play-voice');
  const btnStopVoice = document.getElementById('btn-stop-voice');
  const waveformBox = document.getElementById('waveform-box');
  const voiceRateSelect = document.getElementById('voice-rate-select');

  // Retrieval Tuning Controls
  const btnToggleTuning = document.getElementById('btn-toggle-tuning');
  const tuningDrawer = document.getElementById('tuning-drawer');
  const sliderTopK = document.getElementById('slider-top-k');
  const sliderTopN = document.getElementById('slider-top-n');
  const valTopK = document.getElementById('val-top-k');
  const valTopN = document.getElementById('val-top-n');

  if (btnToggleTuning && tuningDrawer) {
    btnToggleTuning.addEventListener('click', () => {
      tuningDrawer.classList.toggle('open');
    });
  }

  if (sliderTopK && valTopK) {
    sliderTopK.addEventListener('input', (e) => valTopK.textContent = e.target.value);
  }
  if (sliderTopN && valTopN) {
    sliderTopN.addEventListener('input', (e) => valTopN.textContent = e.target.value);
  }

  // Quick Prompt Chips
  document.querySelectorAll('.prompt-chips .chip').forEach(chip => {
    chip.addEventListener('click', () => {
      if (queryInput) {
        queryInput.value = chip.dataset.query;
        executeQuery();
      }
    });
  });

  if (btnClearQuery) {
    btnClearQuery.addEventListener('click', () => {
      if (queryInput) queryInput.value = '';
    });
  }

  if (btnSubmitQuery) {
    btnSubmitQuery.addEventListener('click', executeQuery);
  }

  if (queryInput) {
    queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        executeQuery();
      }
    });
  }

  let latestSpeechResponse = '';

  async function executeQuery() {
    const q = queryInput.value.trim();
    if (!q) return;

    btnSubmitQuery.disabled = true;
    btnSubmitQuery.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Retrieving...';
    availabilityBadge.className = 'badge-status badge-neutral';
    availabilityBadge.textContent = 'Processing...';
    answerBody.innerHTML = '<div class="empty-state"><i class="fa-solid fa-spinner fa-spin"></i><p>Querying ChromaDB vector index and generating grounded response...</p></div>';

    const topK = parseInt(sliderTopK ? sliderTopK.value : 10);
    const topN = parseInt(sliderTopN ? sliderTopN.value : 3);

    try {
      const res = await fetch('/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: topK, top_n: topN })
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      // Render Answer
      answerBody.innerHTML = `<p>${escapeHtml(data.answer).replace(/\n/g, '<br>')}</p>`;

      // Status Badge
      if (data.is_available) {
        availabilityBadge.className = 'badge-status badge-grounded';
        availabilityBadge.textContent = 'Grounded Context';
      } else {
        availabilityBadge.className = 'badge-status badge-refusal';
        availabilityBadge.textContent = 'Information Refusal';
      }

      // Render Citations
      if (data.citations && data.citations.length > 0) {
        citationsWrapper.style.display = 'block';
        citationCount.textContent = data.citations.length;
        citationsList.innerHTML = data.citations.map((c, idx) => `
          <div class="citation-card">
            <div class="citation-header">
              <span class="citation-source"><i class="fa-solid fa-file-lines text-indigo"></i> ${escapeHtml(c.source || 'Doc')}</span>
              <span class="citation-score">Relevance: ${c.score ? c.score.toFixed(3) : 'Top Match'}</span>
            </div>
            <div class="citation-excerpt">"${escapeHtml(c.excerpt || '')}"</div>
          </div>
        `).join('');
      } else {
        citationsWrapper.style.display = 'none';
      }

      // Update Q1 Voice Agent response
      latestSpeechResponse = data.speech_response || data.answer;
      voiceSpeechBox.innerHTML = `<strong>Q1 Spoken Response:</strong><br>"${escapeHtml(latestSpeechResponse)}"`;
      btnPlayVoice.disabled = false;
      btnStopVoice.disabled = false;

      // Auto-play voice if user desires or keep ready
      // playSpeech(latestSpeechResponse);

      // Context Debugger
      debugContentBody.textContent = JSON.stringify(data, null, 2);

    } catch (err) {
      answerBody.innerHTML = `<div class="empty-state text-rose"><i class="fa-solid fa-triangle-exclamation"></i><p>Query Error: ${escapeHtml(err.message)}</p></div>`;
      availabilityBadge.className = 'badge-status badge-refusal';
      availabilityBadge.textContent = 'Error';
    } finally {
      btnSubmitQuery.disabled = false;
      btnSubmitQuery.innerHTML = '<i class="fa-solid fa-paper-plane"></i> <span>Execute RAG</span>';
    }
  }

  // ==========================================
  // 4. Q1 Voice Agent: Web Speech Synthesizer
  // ==========================================
  let currentUtterance = null;

  function playSpeech(text) {
    if (!('speechSynthesis' in window)) {
      alert('Your browser does not support Web Speech API synthesis.');
      return;
    }

    window.speechSynthesis.cancel();
    const rate = parseFloat(voiceRateSelect ? voiceRateSelect.value : 1.0);
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = rate;
    utter.pitch = 1.0;

    // Pick English Voice
    const voices = window.speechSynthesis.getVoices();
    const chosen = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha'))) || voices[0];
    if (chosen) utter.voice = chosen;

    utter.onstart = () => {
      waveformBox.classList.add('playing');
      btnPlayVoice.innerHTML = '<i class="fa-solid fa-volume-high"></i> <span>Speaking...</span>';
    };

    utter.onend = () => {
      waveformBox.classList.remove('playing');
      btnPlayVoice.innerHTML = '<i class="fa-solid fa-play"></i> <span>Play Speech</span>';
    };

    utter.onerror = () => {
      waveformBox.classList.remove('playing');
      btnPlayVoice.innerHTML = '<i class="fa-solid fa-play"></i> <span>Play Speech</span>';
    };

    currentUtterance = utter;
    window.speechSynthesis.speak(utter);
  }

  if (btnPlayVoice) {
    btnPlayVoice.addEventListener('click', () => {
      if (latestSpeechResponse) {
        playSpeech(latestSpeechResponse);
      }
    });
  }

  if (btnStopVoice) {
    btnStopVoice.addEventListener('click', () => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      waveformBox.classList.remove('playing');
      btnPlayVoice.innerHTML = '<i class="fa-solid fa-play"></i> <span>Play Speech</span>';
    });
  }

  // ==========================================
  // 5. Speech-to-Text Microphone Input
  // ==========================================
  const btnMic = document.getElementById('btn-mic');
  const micStatusLabel = document.getElementById('mic-status-label');
  let recognition = null;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      btnMic.classList.add('recording');
      if (micStatusLabel) micStatusLabel.textContent = 'Listening...';
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      if (queryInput) {
        queryInput.value = transcript;
        executeQuery();
      }
    };

    recognition.onerror = () => {
      btnMic.classList.remove('recording');
      if (micStatusLabel) micStatusLabel.textContent = '';
    };

    recognition.onend = () => {
      btnMic.classList.remove('recording');
      if (micStatusLabel) micStatusLabel.textContent = '';
    };
  }

  if (btnMic) {
    btnMic.addEventListener('click', () => {
      if (!recognition) {
        alert('Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.');
        return;
      }
      if (btnMic.classList.contains('recording')) {
        recognition.stop();
      } else {
        recognition.start();
      }
    });
  }

  // ==========================================
  // 6. Knowledge Base & Ingestion Manager
  // ==========================================
  const uploadDropzone = document.getElementById('upload-dropzone');
  const fileUploadInput = document.getElementById('file-upload-input');
  const btnRefreshDocs = document.getElementById('btn-refresh-docs');
  const btnRunFullIngest = document.getElementById('btn-run-full-ingest');
  const ingestStatusBox = document.getElementById('ingest-status-box');
  const ingestStatusText = document.getElementById('ingest-status-text');
  const docsTbody = document.getElementById('docs-tbody');

  async function loadDocuments() {
    if (!docsTbody) return;
    try {
      const res = await fetch('/api/v1/documents');
      const data = await res.json();

      if (data.documents && data.documents.length > 0) {
        docsTbody.innerHTML = data.documents.map(doc => `
          <tr>
            <td><strong><i class="fa-regular fa-file-code text-cyan"></i> ${escapeHtml(doc.filename)}</strong></td>
            <td><span class="badge-tag">${escapeHtml(doc.format)}</span></td>
            <td>${doc.size_human}</td>
            <td>${doc.modified_time}</td>
            <td>
              <button class="btn btn-outline-sm btn-view-doc" data-filename="${escapeHtml(doc.filename)}">
                <i class="fa-solid fa-eye"></i> View
              </button>
            </td>
          </tr>
        `).join('');

        document.querySelectorAll('.btn-view-doc').forEach(btn => {
          btn.addEventListener('click', () => openDocModal(btn.dataset.filename));
        });
      } else {
        docsTbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No raw documents found in data/raw/</td></tr>';
      }
    } catch (err) {
      docsTbody.innerHTML = `<tr><td colspan="5" class="text-center text-rose">Error loading documents: ${err}</td></tr>`;
    }
  }

  if (uploadDropzone && fileUploadInput) {
    uploadDropzone.addEventListener('click', () => fileUploadInput.click());
    uploadDropzone.addEventListener('dragover', (e) => { e.preventDefault(); uploadDropzone.classList.add('dragover'); });
    uploadDropzone.addEventListener('dragleave', () => uploadDropzone.classList.remove('dragover'));
    uploadDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadDropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
    });

    fileUploadInput.addEventListener('change', () => {
      if (fileUploadInput.files.length) uploadFiles(fileUploadInput.files);
    });
  }

  async function uploadFiles(fileList) {
    for (const file of fileList) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch('/api/v1/upload?auto_ingest=true', { method: 'POST', body: formData });
        const result = await res.json();
        showIngestStatus(result.message);
      } catch (err) {
        showIngestStatus('Upload failed: ' + err, true);
      }
    }
    fetchTelemetry();
    loadDocuments();
  }

  function showIngestStatus(msg, isError = false) {
    if (!ingestStatusBox || !ingestStatusText) return;
    ingestStatusBox.style.display = 'block';
    ingestStatusBox.style.borderColor = isError ? 'var(--accent-rose)' : 'var(--accent-emerald)';
    ingestStatusBox.style.color = isError ? 'var(--accent-rose)' : 'var(--accent-emerald)';
    ingestStatusText.innerHTML = (isError ? '<i class="fa-solid fa-circle-xmark"></i> ' : '<i class="fa-solid fa-circle-check"></i> ') + msg;
  }

  if (btnRunFullIngest) {
    btnRunFullIngest.addEventListener('click', async () => {
      btnRunFullIngest.disabled = true;
      btnRunFullIngest.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Indexing Pipeline...';
      try {
        const res = await fetch('/api/v1/ingest', { method: 'POST' });
        const data = await res.json();
        showIngestStatus(data.message);
        fetchTelemetry();
      } catch (err) {
        showIngestStatus('Ingestion failed: ' + err, true);
      } finally {
        btnRunFullIngest.disabled = false;
        btnRunFullIngest.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> <span>Run Pipeline Ingestion</span>';
      }
    });
  }

  if (btnRefreshDocs) btnRefreshDocs.addEventListener('click', loadDocuments);

  // Document Preview Modal
  const docModal = document.getElementById('doc-modal');
  const modalFilename = document.getElementById('modal-filename');
  const modalContent = document.getElementById('modal-content');
  const btnCloseModal = document.getElementById('btn-close-modal');

  async function openDocModal(filename) {
    if (!docModal) return;
    modalFilename.textContent = filename;
    modalContent.textContent = 'Loading content...';
    docModal.style.display = 'flex';

    try {
      const res = await fetch(`/api/v1/document-content?filename=${encodeURIComponent(filename)}`);
      const data = await res.json();
      modalContent.textContent = data.content || '(Empty file)';
    } catch (err) {
      modalContent.textContent = 'Failed to load content: ' + err;
    }
  }

  if (btnCloseModal && docModal) {
    btnCloseModal.addEventListener('click', () => docModal.style.display = 'none');
    docModal.addEventListener('click', (e) => {
      if (e.target === docModal) docModal.style.display = 'none';
    });
  }

  // ==========================================
  // 7. PII Redaction Live Sandbox
  // ==========================================
  const piiInputText = document.getElementById('pii-input-text');
  const piiOutputBox = document.getElementById('pii-output-box');
  const piiRedactedBadge = document.getElementById('pii-redacted-badge');
  const btnSanitizeNow = document.getElementById('btn-sanitize-now');

  const btnSamplePii1 = document.getElementById('btn-sample-pii-1');
  const btnSamplePii2 = document.getElementById('btn-sample-pii-2');
  const btnSamplePii3 = document.getElementById('btn-sample-pii-3');

  if (btnSamplePii1) {
    btnSamplePii1.addEventListener('click', () => {
      piiInputText.value = "Patient: John Doe, SSN: 123-45-6789. Admitted on 2026-03-12 for cardiac evaluation.\nContact spouse at 555-019-2834 or email john.doe@healthcorp.org.";
      runSanitizationPreview();
    });
  }

  if (btnSamplePii2) {
    btnSamplePii2.addEventListener('click', () => {
      piiInputText.value = "Case Manager: Alice Johnson\nDepartment: Claims & Appeals\nDirect Line: +1 (800) 555-0149\nEscalations Inbox: alice.claims@darwixai-insurance.com\nInternal ID: 99482";
      runSanitizationPreview();
    });
  }

  if (btnSamplePii3) {
    btnSamplePii3.addEventListener('click', () => {
      piiInputText.value = "Cardholder: Robert Smith\nPayment Card: 4532-8921-9034-1298\nExp Date: 12/28\nCVV: 492\nBilling Email: rsmith.payments@finmail.net";
      runSanitizationPreview();
    });
  }

  if (btnSanitizeNow) {
    btnSanitizeNow.addEventListener('click', runSanitizationPreview);
  }

  async function runSanitizationPreview() {
    const raw = piiInputText.value.trim();
    if (!raw) return;

    try {
      const res = await fetch('/api/v1/sanitize-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: raw })
      });
      const data = await res.json();

      let highlighted = escapeHtml(data.sanitized);
      // Highlight Redaction tokens
      highlighted = highlighted.replace(/\[([A-Z_]+_REDACTED)\]/g, '<span class="highlight-redacted">[$1]</span>');

      piiOutputBox.innerHTML = highlighted;
      piiRedactedBadge.textContent = `${data.masked_count} Masked Elements`;
    } catch (e) {
      piiOutputBox.innerHTML = `<span class="text-rose">Sanitization Error: ${e}</span>`;
    }
  }

  // ==========================================
  // 8. Vector Store Explorer
  // ==========================================
  const chunksContainer = document.getElementById('chunks-container');
  const chunkSearchInput = document.getElementById('chunk-search-input');
  const btnRefreshChunks = document.getElementById('btn-refresh-chunks');

  let cachedChunks = [];

  async function loadChunks() {
    if (!chunksContainer) return;
    chunksContainer.innerHTML = '<div class="text-center" style="grid-column: 1/-1;">Loading indexed ChromaDB chunks...</div>';
    try {
      const res = await fetch('/api/v1/chunks?limit=50');
      const data = await res.json();
      cachedChunks = data.chunks || [];
      renderChunks(cachedChunks);
    } catch (e) {
      chunksContainer.innerHTML = `<div class="text-center text-rose" style="grid-column: 1/-1;">Error loading chunks: ${e}</div>`;
    }
  }

  function renderChunks(list) {
    if (!list || list.length === 0) {
      chunksContainer.innerHTML = '<div class="text-center text-muted" style="grid-column: 1/-1;">No chunks indexed in ChromaDB yet. Run pipeline ingestion first.</div>';
      return;
    }

    chunksContainer.innerHTML = list.map(c => `
      <div class="chunk-card">
        <div class="chunk-top">
          <span class="chunk-title">${escapeHtml(c.chunk_id)}</span>
          <span class="badge-tag">${escapeHtml(c.category)}</span>
        </div>
        <div class="chunk-text">${escapeHtml(c.content.length > 220 ? c.content.substring(0, 220) + '...' : c.content)}</div>
        <div class="chunk-meta">
          <span><i class="fa-regular fa-file"></i> ${escapeHtml(c.source)}</span>
          <span>${c.char_count} chars</span>
        </div>
      </div>
    `).join('');
  }

  if (chunkSearchInput) {
    chunkSearchInput.addEventListener('input', (e) => {
      const term = e.target.value.toLowerCase();
      const filtered = cachedChunks.filter(c => 
        c.content.toLowerCase().includes(term) || 
        c.chunk_id.toLowerCase().includes(term) ||
        c.source.toLowerCase().includes(term)
      );
      renderChunks(filtered);
    });
  }

  if (btnRefreshChunks) btnRefreshChunks.addEventListener('click', loadChunks);

  // ==========================================
  // 9. Golden Benchmark Evaluation Runner
  // ==========================================
  const btnRunBenchmarks = document.getElementById('btn-run-benchmarks');
  const benchAccuracy = document.getElementById('bench-accuracy');
  const benchTotal = document.getElementById('bench-total');
  const benchPassed = document.getElementById('bench-passed');
  const benchLatency = document.getElementById('bench-latency');
  const benchmarksTbody = document.getElementById('benchmarks-tbody');

  if (btnRunBenchmarks) {
    btnRunBenchmarks.addEventListener('click', async () => {
      btnRunBenchmarks.disabled = true;
      btnRunBenchmarks.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Suite...';
      benchmarksTbody.innerHTML = '<tr><td colspan="7" class="text-center"><i class="fa-solid fa-spinner fa-spin"></i> Executing golden test cases against knowledge base...</td></tr>';

      try {
        const res = await fetch('/api/v1/benchmarks');
        const data = await res.json();

        benchAccuracy.textContent = `${data.accuracy_percent}%`;
        benchTotal.textContent = data.total_tests;
        benchPassed.textContent = data.passed_tests;

        const totalLatency = data.results.reduce((acc, r) => acc + r.latency_ms, 0);
        const avgLatency = (totalLatency / (data.results.length || 1)).toFixed(1);
        benchLatency.textContent = `${avgLatency} ms`;

        benchmarksTbody.innerHTML = data.results.map(r => `
          <tr>
            <td>
              <span class="badge-status ${r.passed ? 'badge-grounded' : 'badge-refusal'}">
                ${r.passed ? 'PASSED' : 'FAILED'}
              </span>
            </td>
            <td><code>${escapeHtml(r.id)}</code></td>
            <td><strong>${escapeHtml(r.query)}</strong></td>
            <td><span class="badge-tag">${r.expected_available ? 'Available' : 'Refusal Expected'}</span></td>
            <td><span class="text-secondary">"${escapeHtml(r.answer.length > 70 ? r.answer.substring(0, 70) + '...' : r.answer)}"</span></td>
            <td><code>${r.latency_ms} ms</code></td>
            <td><span class="badge-tag">${r.citations.length} docs</span></td>
          </tr>
        `).join('');

      } catch (err) {
        benchmarksTbody.innerHTML = `<tr><td colspan="7" class="text-center text-rose">Benchmark execution error: ${err}</td></tr>`;
      } finally {
        btnRunBenchmarks.disabled = false;
        btnRunBenchmarks.innerHTML = '<i class="fa-solid fa-play"></i> <span>Run Benchmark Suite</span>';
      }
    });
  }

  // Helper utility
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});

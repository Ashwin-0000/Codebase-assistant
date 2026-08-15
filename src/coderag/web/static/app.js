/* ═══════════════════════════════════════════════════════════════════════════
   app.js — CodeRAG Web UI
   Vanilla ES2022 — no build step, no framework.

   API surface used:
     GET  /api/status
     POST /api/index          → { job_id }
     GET  /api/index/progress/:id
     POST /api/ask            → AskResponse
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── API constants ─────────────────────────────────────────────────────────
const API = {
  status:   '/api/status',
  index:    '/api/index',
  progress: (id) => `/api/index/progress/${id}`,
  ask:      '/api/ask',
};

// ── DOM helpers ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const el = {
  messages:      $('messages'),
  questionInput: $('question-input'),
  askBtn:        $('ask-btn'),
  inputWrapper:  $('input-wrapper'),
  indexForm:     $('index-form'),
  repoInput:     $('repo-input'),
  forceReindex:  $('force-reindex'),
  indexBtn:      $('index-btn'),
  progressBox:   $('index-progress'),
  progressFill:  $('progress-fill'),
  progressText:  $('progress-text'),
  statusWidget:  $('status-widget'),
  refreshStatus: $('refresh-status'),
  topK:          $('top-k'),
  modelChip:     $('model-chip'),
  sidebarToggle: $('sidebar-toggle'),
  sidebar:       $('sidebar'),
};

// ── App state ─────────────────────────────────────────────────────────────
let _isAsking   = false;
let _isIndexed  = false;
let _progressId = null;   // setInterval handle for progress polling

// ═════════════════════════════════════════════════════════════════════════
//  STATUS
// ═════════════════════════════════════════════════════════════════════════

async function fetchStatus() {
  try {
    const res  = await fetch(API.status);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    _renderStatus(data);
    _isIndexed = data.indexed;
    _updateSendBtn();
    el.modelChip.textContent = `${data.embedding_provider} · ${data.llm_provider}`;
  } catch (err) {
    el.statusWidget.innerHTML =
      `<p class="error-bubble" style="font-size:.75rem">⚠ ${_esc(err.message)}</p>`;
  }
}

function _renderStatus(s) {
  const statusBadge = s.indexed
    ? `<span class="badge badge-green"><span class="status-dot pulse"></span>Indexed</span>`
    : `<span class="badge badge-muted"><span class="status-dot"></span>Not indexed</span>`;

  el.statusWidget.innerHTML = `
    <div class="status-row">
      <span class="status-label">Status</span>
      ${statusBadge}
    </div>
    ${s.indexed ? `
    <div class="status-row">
      <span class="status-label">Files</span>
      <span class="status-value">${s.files.toLocaleString()}</span>
    </div>
    <div class="status-row">
      <span class="status-label">Chunks</span>
      <span class="status-value">${s.chunks.toLocaleString()}</span>
    </div>
    ${s.last_indexed ? `<div class="status-row">
      <span class="status-label">Last indexed</span>
      <span class="status-value" style="font-size:.7rem">${_esc(s.last_indexed)}</span>
    </div>` : ''}
    <div class="status-row">
      <span class="status-label">Graph</span>
      <span class="status-value">${s.graph_present ? '✓' : '–'}</span>
    </div>` : `<p style="font-size:.77rem;color:var(--text-dim);margin-top:4px">
      Index a repository to get started.
    </p>`}
  `;
}

el.refreshStatus.addEventListener('click', () => {
  el.statusWidget.innerHTML =
    `<div class="skeleton-block">
       <div class="skeleton-line"></div>
       <div class="skeleton-line short"></div>
     </div>`;
  fetchStatus();
});

// ═════════════════════════════════════════════════════════════════════════
//  INDEXING
// ═════════════════════════════════════════════════════════════════════════

el.indexForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const repo  = el.repoInput.value.trim();
  const force = el.forceReindex.checked;
  if (!repo) { el.repoInput.focus(); return; }

  el.indexBtn.disabled = true;
  el.indexBtn.textContent = 'Starting…';
  el.progressBox.classList.remove('hidden');
  el.progressFill.className = 'progress-fill';
  el.progressText.textContent = 'Resolving repository…';

  try {
    const res = await fetch(API.index, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo, force }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(err.detail || 'Indexing request failed');
    }
    const { job_id } = await res.json();
    _startProgressPolling(job_id);
  } catch (err) {
    el.progressFill.className = 'progress-fill errored';
    el.progressText.textContent = `Error: ${err.message}`;
    _resetIndexBtn();
  }
});

function _startProgressPolling(jobId) {
  if (_progressId) clearInterval(_progressId);
  _progressId = setInterval(async () => {
    try {
      const res  = await fetch(API.progress(jobId));
      const data = await res.json();

      if (data.status === 'running') {
        el.progressText.textContent =
          `Indexing… ${data.files_processed} files · ${data.chunks_indexed} chunks`;
        return;
      }

      clearInterval(_progressId);
      _progressId = null;

      if (data.status === 'done') {
        el.progressFill.className = 'progress-fill done';
        el.progressText.textContent =
          `✓ Done — ${data.files_processed} files, ` +
          `${data.chunks_indexed} chunks (${data.elapsed_seconds}s)`;
        fetchStatus();
      } else {
        el.progressFill.className = 'progress-fill errored';
        el.progressText.textContent = `Error: ${data.error || 'Unknown error'}`;
      }
      _resetIndexBtn();
    } catch {
      clearInterval(_progressId);
      _progressId = null;
      _resetIndexBtn();
    }
  }, 1000);
}

function _resetIndexBtn() {
  el.indexBtn.disabled = false;
  el.indexBtn.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
      <polyline points="21 15 21 21 3 21 3 15"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
    Index Repository`;
}

// ═════════════════════════════════════════════════════════════════════════
//  ASK
// ═════════════════════════════════════════════════════════════════════════

el.askBtn.addEventListener('click', _submitQuestion);

el.questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    _submitQuestion();
  }
});

el.questionInput.addEventListener('input', () => {
  _autoResize(el.questionInput);
  _updateSendBtn();
});

function _updateSendBtn() {
  const hasText = el.questionInput.value.trim().length > 0;
  el.askBtn.disabled = !hasText || _isAsking;
}

async function _submitQuestion() {
  const question = el.questionInput.value.trim();
  if (!question || _isAsking) return;

  _isAsking = true;
  _updateSendBtn();
  el.questionInput.value = '';
  _autoResize(el.questionInput);

  _appendUserMessage(question);
  const typingId = _appendTyping();

  try {
    const res = await fetch(API.ask, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        top_k: Math.max(1, parseInt(el.topK.value) || 8),
      }),
    });

    _removeTyping(typingId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Request failed' }));
      _appendError(err.detail || 'Request failed');
    } else {
      const data = await res.json();
      _appendAssistantMessage(data);
    }
  } catch (err) {
    _removeTyping(typingId);
    _appendError(err.message);
  } finally {
    _isAsking = false;
    _updateSendBtn();
    el.questionInput.focus();
  }
}

// ═════════════════════════════════════════════════════════════════════════
//  MESSAGE RENDERING
// ═════════════════════════════════════════════════════════════════════════

function _appendUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'message user';
  div.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">U</div>
    <div class="msg-body">
      <div class="msg-bubble">${_fmt(text)}</div>
    </div>`;
  el.messages.appendChild(div);
  _scrollBottom();
}

function _appendAssistantMessage(data) {
  const citationsHtml = data.citations.length
    ? `<div class="citations" aria-label="Source citations">
        ${data.citations.map((c, i) =>
          `<span class="cite-chip" title="${_esc(c)}">
             <span class="cite-num" aria-hidden="true">${i + 1}</span>
             ${_esc(_shortenCitation(c))}
           </span>`
        ).join('')}
       </div>`
    : '';

  const offlineNotice = data.offline_mode
    ? `<div class="offline-notice" role="status">
         <span class="offline-icon" aria-hidden="true">⚡</span>
         <span><strong>Offline Mode</strong> — No API key configured. Answers are generated locally without an LLM.
         Add an <code>OPENAI_API_KEY</code> or <code>ANTHROPIC_API_KEY</code> to your <code>.env</code> for richer answers.</span>
       </div>`
    : '';

  const meta = [
    `${data.context_chunks} chunk${data.context_chunks === 1 ? '' : 's'}`,
    `${data.total_tokens} tokens`,
    `${data.provider}/${data.model}`,
    `${data.latency_seconds}s`,
  ].join(' · ');

  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">AI</div>
    <div class="msg-body">
      <div class="msg-bubble">${_fmt(data.answer)}</div>
      ${citationsHtml}
      ${offlineNotice}
      <div class="msg-meta" aria-label="Response metadata">${_esc(meta)}</div>
    </div>`;
  el.messages.appendChild(div);
  _scrollBottom();
}

function _appendError(msg) {
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.setAttribute('role', 'alert');
  div.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">AI</div>
    <div class="msg-body">
      <div class="error-bubble">⚠ ${_esc(msg)}</div>
    </div>`;
  el.messages.appendChild(div);
  _scrollBottom();
}

function _appendTyping() {
  const id  = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.id = id;
  div.className = 'message assistant';
  div.setAttribute('aria-busy', 'true');
  div.setAttribute('aria-label', 'Generating answer');
  div.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">AI</div>
    <div class="msg-body">
      <div class="msg-bubble">
        <div class="typing-dots" aria-hidden="true">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>`;
  el.messages.appendChild(div);
  _scrollBottom();
  return id;
}

function _removeTyping(id) {
  document.getElementById(id)?.remove();
}

// ═════════════════════════════════════════════════════════════════════════
//  WELCOME CARD
// ═════════════════════════════════════════════════════════════════════════

function _renderWelcome() {
  const div = document.createElement('div');
  div.className = 'welcome-card';
  div.setAttribute('role', 'banner');
  div.innerHTML = `
    <span class="welcome-hex" aria-hidden="true">⬡</span>
    <div class="welcome-title">Welcome to CodeRAG</div>
    <div class="welcome-sub">
      Ask natural-language questions about any codebase.<br>
      Every answer is grounded in the actual source code,<br>with file path and line-number citations.
    </div>
    <div class="welcome-steps" aria-label="Getting started steps">
      <div class="step">
        <div class="step-num" aria-hidden="true">1</div>
        <span>Paste a repo path<br>in the sidebar</span>
      </div>
      <div class="step">
        <div class="step-num" aria-hidden="true">2</div>
        <span>Click<br><em>Index Repository</em></span>
      </div>
      <div class="step">
        <div class="step-num" aria-hidden="true">3</div>
        <span>Ask anything<br>about the code</span>
      </div>
    </div>`;
  el.messages.appendChild(div);
}

// ═════════════════════════════════════════════════════════════════════════
//  HELPERS
// ═════════════════════════════════════════════════════════════════════════

/** HTML-escape a plain string. */
function _esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Lightweight Markdown → HTML formatter.
 * Safely tokenises code spans first (preserving them), then escapes the
 * rest of the text and applies bold/italic/newline transforms.
 */
function _fmt(raw) {
  const slots   = [];
  let   working = raw;

  // 1. Extract fenced code blocks  ```lang\ncode\n```
  working = working.replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => {
    const key = `\x00BLOCK${slots.length}\x00`;
    slots.push(`<pre><code>${_esc(code.trim())}</code></pre>`);
    return key;
  });

  // 2. Extract inline code  `code`
  working = working.replace(/`([^`\n]+)`/g, (_, code) => {
    const key = `\x00INLINE${slots.length}\x00`;
    slots.push(`<code>${_esc(code)}</code>`);
    return key;
  });

  // 3. Escape remaining HTML
  working = _esc(working);

  // 4. Text formatting (on escaped text)
  working = working
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g,     '<em>$1</em>')
    .replace(/\n/g, '<br>');

  // 5. Restore code slots
  slots.forEach((html, i) => {
    working = working
      .replace(`\x00BLOCK${i}\x00`,  html)
      .replace(`\x00INLINE${i}\x00`, html);
  });

  return working;
}

/** Shorten a citation string for the chip label. */
function _shortenCitation(cite) {
  // "src/utils/auth.py:L45-L78" → "auth.py:L45-L78"
  const parts = cite.split('/');
  return parts[parts.length - 1] || cite;
}

function _autoResize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 180) + 'px';
}

function _scrollBottom() {
  requestAnimationFrame(() => {
    el.messages.scrollTo({ top: el.messages.scrollHeight, behavior: 'smooth' });
  });
}

// ═════════════════════════════════════════════════════════════════════════
//  SIDEBAR TOGGLE (mobile)
// ═════════════════════════════════════════════════════════════════════════

el.sidebarToggle.addEventListener('click', () => {
  const open = el.sidebar.classList.toggle('open');
  el.sidebarToggle.setAttribute('aria-expanded', open);
});

document.addEventListener('click', (e) => {
  if (
    window.innerWidth <= 768 &&
    el.sidebar.classList.contains('open') &&
    !el.sidebar.contains(e.target) &&
    e.target !== el.sidebarToggle
  ) {
    el.sidebar.classList.remove('open');
    el.sidebarToggle.setAttribute('aria-expanded', 'false');
  }
});

// ═════════════════════════════════════════════════════════════════════════
//  INIT
// ═════════════════════════════════════════════════════════════════════════

(function init() {
  _renderWelcome();
  fetchStatus();
  // Soft-refresh status every 30 s
  setInterval(fetchStatus, 30_000);
  // Focus the textarea on load
  el.questionInput.focus();
}());

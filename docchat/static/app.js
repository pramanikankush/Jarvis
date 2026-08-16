/* Jarvis frontend — vanilla JS + SSE streaming. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const inputEl = $("#input");
const sendBtn = $("#btn-send");
const micBtn = $("#btn-mic");
const scrollEl = $("#chat-scroll");
const emptyEl = $("#empty-state");

const state = {
  docs: [],
  sheets: [],
  sessions: [],
  memory: [],
  current: null, // session id
  streaming: false,
  abort: null,
  keySet: false,
  tavilySet: false,
  model: "",
  demoMaxChats: 0,
  voice: "troy",
  ttsEnabled: localStorage.getItem("jarvis_tts") === "1",
  recording: false,
  abortReason: null, // "user" (Stop pressed) | "timeout" (watchdog) | null
};

// hard ceiling for a single request; a hung backend becomes a visible error
const REQUEST_TIMEOUT_MS = 120000;

/* ---------------- toasts ---------------- */
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 4500);
}

/* ---------------- identity + api helpers ---------------- */
const GUEST_KEY = "jarvis_guest_id";

function guestId() {
  let g = "";
  try { g = localStorage.getItem(GUEST_KEY) || ""; } catch (_) {}
  if (!g) {
    // crypto.randomUUID where available, else a random fallback (never sent raw)
    g = (crypto.randomUUID ? crypto.randomUUID() :
      "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = Math.random() * 16 | 0;
        return (c === "x" ? r : (r & 0x3 | 0x8)).toString(16);
      }));
    try { localStorage.setItem(GUEST_KEY, g); } catch (_) {}
  }
  return g;
}

// Clerk session token (set after sign-in; null when signed out / not configured)
let clerkToken = null;

async function authHeaders(extra = {}) {
  const h = { ...extra };
  if (clerkToken) {
    h["Authorization"] = "Bearer " + clerkToken;
  } else {
    h["X-Guest-Id"] = guestId();  // guest workspace unless a Clerk session exists
  }
  return h;
}

async function api(url, opts = {}) {
  const headers = await authHeaders(opts.headers || {});
  const res = await fetch(url, { ...opts, headers });
  let data = {};
  try { data = await res.json(); } catch (_) {}
  if (!res.ok) throw new Error(data.detail || data.error || `HTTP ${res.status}`);
  return data;
}

function mdRender(text) {
  try {
    if (window.marked && window.DOMPurify) {
      return window.DOMPurify.sanitize(window.marked.parse(text, { breaks: true, gfm: true }));
    }
  } catch (_) {}
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

/* ---------------- docs sidebar ---------------- */
const DOC_ICON_SVG = '<svg width="14" height="16" viewBox="0 0 14 16" fill="none" aria-hidden="true"><path d="M2.5 1h6l3 3v10a.5.5 0 0 1-.5.5h-8a.5.5 0 0 1-.5-.5V1.5a.5.5 0 0 1 .5-.5z" stroke="currentColor" stroke-width="1.3"/><path d="M8.5 1v3h3" stroke="currentColor" stroke-width="1.3"/></svg>';
const SHEET_ICON_SVG = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true"><rect x="1" y="1" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.3"/><path d="M1 5h12M5 5v8M9.5 5v8" stroke="currentColor" stroke-width="1.1"/></svg>';
const AVATAR_SVG = '<svg width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M10 2v16M2 10h16" stroke="currentColor" stroke-width="3" stroke-linecap="round"/><path d="M5 5l10 10M15 5l-10 10" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" opacity=".5"/></svg>';
const docIcon = () => DOC_ICON_SVG;

function fmtSize(n) {
  if (n >= 1024 * 1024) return (n / 1048576).toFixed(1) + " MB";
  if (n >= 1024) return Math.round(n / 1024) + " KB";
  return n + " B";
}

function renderDocs() {
  const list = $("#docs-list");
  list.innerHTML = "";
  if (!state.docs.length) {
    list.innerHTML = '<li class="doc-item muted-note">No documents yet — add files above.</li>';
    return;
  }
  for (const d of state.docs) {
    const li = document.createElement("li");
    li.className = "doc-item";
    const icon = document.createElement("span");
    icon.className = "doc-icon";
    icon.innerHTML = docIcon(d.name);
    const main = document.createElement("div");
    main.className = "doc-main";
    const nm = document.createElement("div");
    nm.className = "doc-name";
    nm.textContent = d.name;
    nm.title = d.name;
    const meta = document.createElement("div");
    meta.className = "doc-meta";
    meta.textContent = `${fmtSize(d.size)} · ${d.chunks} chunks`;
    main.append(nm, meta);
    const del = document.createElement("button");
    del.className = "doc-del";
    del.textContent = "✕";
    del.title = "Delete document";
    del.onclick = async () => {
      if (!confirm(`Delete "${d.name}" from memory?`)) return;
      try {
        await api(`/api/docs/${d.id}`, { method: "DELETE" });
        state.docs = state.docs.filter((x) => x.id !== d.id);
        renderDocs();
        updateStat();
        toast("Document deleted.", "ok");
      } catch (err) { toast(err.message, "err"); }
    };
    li.append(icon, main, del);
    list.appendChild(li);
  }
}

/* ---------------- sheets sidebar ---------------- */
function renderSheets() {
  const list = $("#sheets-list");
  list.innerHTML = "";
  if (!state.sheets.length) {
    list.innerHTML = '<li class="doc-item muted-note">No spreadsheets — upload .csv / .xlsx.</li>';
    return;
  }
  for (const s of state.sheets) {
    const li = document.createElement("li");
    li.className = "doc-item";
    const icon = document.createElement("span");
    icon.className = "doc-icon";
    icon.innerHTML = SHEET_ICON_SVG;
    const main = document.createElement("div");
    main.className = "doc-main";
    const nm = document.createElement("div");
    nm.className = "doc-name";
    nm.textContent = s.name;
    nm.title = s.name;
    const meta = document.createElement("div");
    meta.className = "doc-meta";
    meta.textContent = `${s.rows.toLocaleString()} rows × ${s.cols} cols`;
    main.append(nm, meta);
    const del = document.createElement("button");
    del.className = "doc-del";
    del.textContent = "✕";
    del.title = "Delete spreadsheet";
    del.onclick = async () => {
      if (!confirm(`Delete spreadsheet "${s.name}"?`)) return;
      try {
        await api(`/api/sheets/${s.id}`, { method: "DELETE" });
        state.sheets = state.sheets.filter((x) => x.id !== s.id);
        renderSheets();
        toast("Spreadsheet deleted.", "ok");
      } catch (err) { toast(err.message, "err"); }
    };
    li.append(icon, main, del);
    list.appendChild(li);
  }
}

async function uploadSheet(file) {
  const fd = new FormData();
  fd.append("file", file);
  try {
    const data = await api("/api/sheets", { method: "POST", body: fd });
    state.sheets = data.sheets;
    renderSheets();
    toast(`Indexed spreadsheet ✓ (${data.sheet.rows} rows)`, "ok");
  } catch (err) {
    toast(`Sheet ${file.name}: ${err.message}`, "err");
  }
}

/* ---------------- memory sidebar ---------------- */
function renderMemory() {
  const list = $("#memory-list");
  list.innerHTML = "";
  if (!state.memory.length) {
    list.innerHTML = '<li class="doc-item muted-note">No facts remembered yet. I store useful things you tell me.</li>';
    return;
  }
  for (const m of state.memory.slice(0, 30)) {
    const li = document.createElement("li");
    li.className = "doc-item mem-item";
    const main = document.createElement("div");
    main.className = "doc-main";
    const nm = document.createElement("div");
    nm.className = "mem-fact";
    nm.textContent = m.fact;
    nm.title = m.fact;
    const meta = document.createElement("div");
    meta.className = "doc-meta";
    meta.textContent = `(${m.kind})`;
    main.append(nm, meta);
    const del = document.createElement("button");
    del.className = "doc-del";
    del.textContent = "✕";
    del.onclick = async () => {
      try {
        await api(`/api/memory/${m.id}`, { method: "DELETE" });
        await loadMemory();
        toast("Fact forgotten.", "ok");
      } catch (err) { toast(err.message, "err"); }
    };
    li.append(main, del);
    list.appendChild(li);
  }
}

async function loadMemory() {
  state.memory = (await api("/api/memory")).memory;
  renderMemory();
}

async function addMemoryFact() {
  const fact = $("#mem-input").value.trim();
  if (!fact) return;
  try {
    const r = await api("/api/memory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fact }),
    });
    $("#mem-input").value = "";
    await loadMemory();
    toast(r.duplicate ? "Already remembered." : "Fact saved to memory.", "ok");
  } catch (err) { toast(err.message, "err"); }
}

/* ---------------- sessions sidebar ---------------- */
function relTime(ts) {
  const diff = (Date.now() - new Date(ts.replace(" ", "T") + "Z")) / 1000;
  if (diff < 60) return "now";
  if (diff < 3600) return Math.round(diff / 60) + "m";
  if (diff < 86400) return Math.round(diff / 3600) + "h";
  return Math.round(diff / 86400) + "d";
}

const PIN_SVG = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true"><path d="M6 1.2l2.3 3.6.7.2v1H3v-1l.7-.2L6 1.2zM4.8 6.5V10M7.2 6.5V10" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round"/></svg>';
const EXPORT_SVG = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true"><path d="M6 1.5v5M3.5 4 6 6.5 8.5 4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 9.5h8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>';

function renderChatLimit() {
  const el = $("#chat-limit");
  if (state.demoMaxChats > 0) {
    el.classList.remove("hidden");
    el.textContent = `${state.sessions.length}/${state.demoMaxChats}`;
    el.classList.toggle("full", state.sessions.length >= state.demoMaxChats);
  } else {
    el.classList.add("hidden");
  }
}

function renderSessions() {
  const list = $("#sessions-list");
  list.innerHTML = "";
  renderChatLimit();
  const q = ($("#chat-search").value || "").trim().toLowerCase();
  const sessions = state.sessions.filter((s) => !q || s.title.toLowerCase().includes(q));
  for (const s of sessions) {
    const btn = document.createElement("div");
    btn.role = "button";
    btn.tabIndex = 0;
    btn.className = "session-item" + (s.id === state.current ? " active" : "")
      + (s.pinned ? " pinned" : "");
    const main = document.createElement("span");
    main.className = "sess-main";
    const t = document.createElement("span");
    t.className = "sess-title";
    t.textContent = s.title;
    t.title = s.title + " (double-click to rename)";
    const time = document.createElement("span");
    time.className = "sess-time";
    time.textContent = (s.pinned ? "● " : "") + relTime(s.updated_at);
    main.append(t, time);
    const actions = document.createElement("span");
    actions.className = "sess-actions";
    const pin = document.createElement("button");
    pin.className = "sess-btn" + (s.pinned ? " on" : "");
    pin.title = s.pinned ? "Unpin" : "Pin chat";
    pin.innerHTML = PIN_SVG;
    pin.onclick = async (e) => {
      e.stopPropagation();
      try {
        await api(`/api/sessions/${s.id}`, {
          method: "PATCH", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pinned: !s.pinned }),
        });
        await loadSessions();
      } catch (err) { toast(err.message, "err"); }
    };
    const exp = document.createElement("button");
    exp.className = "sess-btn";
    exp.title = "Export conversation";
    exp.innerHTML = EXPORT_SVG;
    exp.onclick = (e) => {
      e.stopPropagation();
      window.open(`/api/sessions/${s.id}/export`, "_blank");
    };
    const del = document.createElement("button");
    del.className = "sess-btn sess-del";
    del.textContent = "✕";
    del.title = "Delete chat";
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this chat?")) return;
      try {
        await api(`/api/sessions/${s.id}`, { method: "DELETE" });
        if (state.current === s.id) newChat();
        await loadSessions();
        toast("Chat deleted.", "ok");
      } catch (err) { toast(err.message, "err"); }
    };
    actions.append(pin, exp, del);
    btn.append(main, actions);
    btn.ondblclick = async () => {
      const name = prompt("Rename chat", s.title);
      if (name && name.trim() && name.trim() !== s.title) {
        try {
          await api(`/api/sessions/${s.id}`, {
            method: "PATCH", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: name.trim() }),
          });
          await loadSessions();
        } catch (err) { toast(err.message, "err"); }
      }
    };
    btn.onclick = () => openSession(s.id);
    btn.onkeydown = (e) => { if (e.key === "Enter") openSession(s.id); };
    list.appendChild(btn);
  }
}

async function loadSessions() {
  state.sessions = (await api("/api/sessions")).sessions;
  renderSessions();
}

function setDemoChats(n) {
  state.demoMaxChats = Number.isFinite(n) && n > 0 ? n : 0;
  renderChatLimit();
}

function newChat() {
  // free-demo cap: guests/Clerk users get a limited number of chats
  if (state.demoMaxChats > 0 && state.sessions.length >= state.demoMaxChats) {
    toast(`Free demo limited to ${state.demoMaxChats} chats — delete an old chat to start a new one.`, "err");
    return;
  }
  state.current = null;
  $("#chat").innerHTML = "";
  emptyEl.classList.remove("hidden");
  renderSessions();
  inputEl.focus();
}

async function openSession(id) {
  state.current = id;
  emptyEl.classList.add("hidden");
  const data = await api(`/api/sessions/${id}/messages`);
  const chat = $("#chat");
  chat.innerHTML = "";
  for (const m of data.messages) {
    if (m.role === "user") appendUserBubble(m.content);
    else if (m.content) appendAssistant(m.content, m.sources || []);
  }
  renderSessions();
  scrollToBottom(true);
}

/* ---------------- chat rendering ---------------- */
function appendUserBubble(text) {
  const msg = document.createElement("div");
  msg.className = "msg user";
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  msg.appendChild(b);
  $("#chat").appendChild(msg);
}

const TOOL_LABELS = {
  search_documents: "Searching your documents…",
  calculate: "Calculating…",
  web_search: "Searching the web…",
  analyze_spreadsheet: "Analyzing spreadsheet…",
  run_python: "Running data analysis…",
  memory: "Updating memory…",
  time: "Checking the time…",
};

function appendAssistant(text = "", sources = []) {
  const msg = document.createElement("div");
  msg.className = "msg assistant";
  const av = document.createElement("div");
  av.className = "avatar";
  av.innerHTML = AVATAR_SVG;
  const content = document.createElement("div");
  content.className = "content";
  const activity = document.createElement("div");
  activity.className = "activity hidden";
  content.appendChild(activity);
  const body = document.createElement("div");
  body.className = "md";
  content.appendChild(body);
  msg.append(av, content);
  $("#chat").appendChild(msg);
  setAssistantText(msg, text, sources, false);
  return msg;
}

function setAssistantText(msg, text, sources, streaming) {
  const body = msg.querySelector(".md");
  body.innerHTML = mdRender(text) + (streaming ? '<span class="caret">▍</span>' : "");
  scrollToBottom();
}

function addActivity(msg, html) {
  const act = msg.querySelector(".activity");
  act.classList.remove("hidden");
  // a new phase starting marks the previous ones complete (staged checklist)
  msg.querySelectorAll(".act-chip.busy").forEach((c) => {
    c.classList.remove("busy");
    c.classList.add("done");
  });
  const chip = document.createElement("div");
  chip.className = "act-chip busy";  // spinner shows while the phase runs
  chip.innerHTML = html;
  act.appendChild(chip);
  scrollToBottom();
}

function renderSources(msg, sources) {
  if (!sources || !sources.length) return;
  const content = msg.querySelector(".content");
  const wrap = document.createElement("div");
  wrap.className = "sources";
  const head = document.createElement("div");
  head.className = "sources-head";
  head.textContent = "Sources";
  wrap.appendChild(head);
  sources.forEach((s, i) => {
    const det = document.createElement("details");
    det.className = "source-chip";
    const sum = document.createElement("summary");
    const num = document.createElement("span");
    num.className = "src-num";
    num.textContent = i + 1;
    const label = document.createElement("span");
    label.textContent = s.doc_name + (s.page ? ` — page ${s.page}` : "");
    const toggle = document.createElement("span");
    toggle.className = "src-toggle";
    toggle.textContent = "view";
    sum.append(num, label, toggle);
    const ex = document.createElement("div");
    ex.className = "src-excerpt";
    ex.textContent = (s.text || "").slice(0, 600) + (s.score !== undefined ? `\n\n[similarity ${s.score}]` : "");
    det.append(sum, ex);
    wrap.appendChild(det);
  });
  content.appendChild(wrap);
}

function scrollToBottom(force = false) {
  const near = scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 140;
  if (force || near) scrollEl.scrollTop = scrollEl.scrollHeight;
}

/* ---------------- streaming chat ---------------- */
async function sendMessage(text) {
  if (state.streaming) return;
  emptyEl.classList.add("hidden");
  appendUserBubble(text);
  inputEl.value = "";
  autoResize();

  state.streaming = true;
  state.abort = new AbortController();
  state.abortReason = null;
  sendBtn.classList.add("stop");
  sendBtn.title = "Stop";

  const assistantMsg = appendAssistant("", []);
  let buf = "";
  let answer = "";
  let sources = [];
  let gotError = false;   // terminal state: an error event was rendered
  let gotDone = false;    // terminal state: a done event was received
  let watchdog = null;

  // watchdog: a hung backend (no terminal event) must become a visible error
  watchdog = setTimeout(() => {
    if (!state.streaming) return;
    gotError = true;
    state.abortReason = "timeout";
    setAssistantError(assistantMsg, "The request timed out — the model took too long to respond. Please try again.");
    state.abort.abort();
  }, REQUEST_TIMEOUT_MS);

  try {
    const headers = await authHeaders({ "Content-Type": "application/json" });
    const res = await fetch("/api/chat", {
      method: "POST",
      headers,
      body: JSON.stringify({ session_id: state.current, message: text }),
      signal: state.abort.signal,
    });
    if (!res.ok || !res.body) {
      let detail = `HTTP ${res.status}`;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          let evt;
          try { evt = JSON.parse(line.slice(6)); } catch (_) { continue; }
          if (evt.type === "start" && evt.session_id && !state.current) {
            state.current = evt.session_id;
          } else if (evt.type === "token") {
            answer += evt.text;
            setAssistantText(assistantMsg, answer, [], true);
          } else if (evt.type === "tool") {
            const label = TOOL_LABELS[evt.tool] || `Tool: ${evt.tool}…`;
            addActivity(assistantMsg, label);
          } else if (evt.type === "status") {
            addActivity(assistantMsg, evt.label || "Working…");
          } else if (evt.type === "chart") {
            addActivity(assistantMsg, `<img class="chart" src="${evt.url}" alt="chart">`);
          } else if (evt.type === "sources") {
            sources = evt.sources || [];
          } else if (evt.type === "memory") {
            loadMemory().catch(() => {});
          } else if (evt.type === "error") {
            gotError = true;
            setAssistantError(assistantMsg, evt.message);
          } else if (evt.type === "done") {
            gotDone = true;
            sources = evt.sources || [];
            if (evt.answer) answer = evt.answer;
          }
        }
      }
    }
    // stream closed without a terminal event -> the connection died mid-flight
    if (!gotDone && !gotError) {
      gotError = true;
      setAssistantError(assistantMsg,
        "The response was interrupted before it completed. Please try again.");
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      gotError = true;
      setAssistantError(assistantMsg, err.message);
    } else if (state.abortReason !== "timeout" && !gotError) {
      // deliberate user stop: finalize silently, keep whatever streamed so far
    } else {
      gotError = true;  // timeout error already rendered by the watchdog
    }
  } finally {
    if (watchdog) clearTimeout(watchdog);
    state.streaming = false;
    sendBtn.classList.remove("stop");
    sendBtn.title = "Send";
    // every chip reaches a settled state: spinner off, completed check
    assistantMsg.querySelectorAll(".act-chip.busy").forEach((c) => {
      c.classList.remove("busy");
      c.classList.add("done");
    });
    // never wipe an error that is already displayed
    if (!gotError) {
      setAssistantText(assistantMsg, answer, [], false);
      if (answer.trim()) renderSources(assistantMsg, sources);
      if (answer.trim() && state.ttsEnabled) speak(answer);
    }
    if (state.current) loadSessions().catch(() => {});
    refreshQuota();  // token meter updates after every turn
    scrollToBottom(true);
  }
}

function setAssistantError(msg, message) {
  const body = msg.querySelector(".md");
  body.innerHTML = `<p style="color:var(--error)">${String(message).replace(/</g, "&lt;")}</p>`;
}

/* ---------------- voice: STT + TTS ---------------- */
let mediaRecorder = null;
let micChunks = [];
let audioEl = null;

async function toggleMic() {
  if (state.recording) { stopRecording(); return; }
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    toast("Voice input not supported in this browser.", "err");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => { if (e.data.size) micChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      if (!micChunks.length) return;
      micBtn.classList.remove("rec");
      const blob = new Blob(micChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      micChunks = [];
      await transcribe(blob);
    };
    mediaRecorder.start();
    state.recording = true;
    micBtn.classList.add("rec");
    micBtn.title = "Stop recording";
    toast("Recording… click the mic button to stop.", "ok");
  } catch (err) {
    toast("Microphone unavailable: " + err.message, "err");
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  state.recording = false;
  micBtn.classList.remove("rec");
  micBtn.title = "Voice input";
}

async function transcribe(blob) {
  micBtn.classList.add("processing");
  micBtn.title = "Transcribing…";
  const fd = new FormData();
  fd.append("file", blob, "voice.webm");
  try {
    const data = await api("/api/stt", { method: "POST", body: fd });
    const text = (data.text || "").trim();
    if (!text) { toast("Could not hear anything — try again.", "err"); return; }
    sendMessage(text.slice(0, 4000));
  } catch (err) {
    toast("Speech-to-text: " + err.message, "err");
  } finally {
    micBtn.classList.remove("processing");
    micBtn.title = "Voice input";
  }
}

function cleanForTTS(text) {
  return text
    .replace(/\[\d+\]/g, "")
    .replace(/[#*_`>|]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 1500);
}

async function speak(text) {
  const clean = cleanForTTS(text);
  if (!clean) return;
  try {
    const headers = await authHeaders({ "Content-Type": "application/json" });
    const res = await fetch("/api/tts", {
      method: "POST",
      headers,
      body: JSON.stringify({ text: clean, voice: state.voice }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    const blob = await res.blob();
    if (!audioEl) audioEl = new Audio();
    audioEl.src = URL.createObjectURL(blob);
    audioEl.play().catch(() => {});
    return;
  } catch (err) {
    // graceful fallback: browser's built-in speech synthesis
    if ("speechSynthesis" in window) {
      const u = new SpeechSynthesisUtterance(clean);
      speechSynthesis.speak(u);
    } else {
      toast("Voice reply unavailable: " + err.message, "err");
    }
  }
}

/* ---------------- uploads ---------------- */
async function uploadFiles(files) {
  if (!files || !files.length) return;
  const note = $("#indexing-note");
  note.classList.remove("hidden");
  note.textContent = "Indexing… (first upload downloads the free embedding model, ~100 MB)";
  let ok = 0;
  for (const file of files) {
    if (file.size > 50 * 1024 * 1024) {
      toast(`${file.name}: too large (max 50 MB)`, "err");
      continue;
    }
    const fd = new FormData();
    fd.append("file", file);
    try {
      const data = await api("/api/docs", { method: "POST", body: fd });
      ok++;
      state.docs = data.docs;
      renderDocs();
      updateStat();
    } catch (err) {
      toast(`${file.name}: ${err.message}`, "err");
    }
  }
  note.classList.add("hidden");
  if (ok) toast(`Indexed ${ok} file${ok > 1 ? "s" : ""} ✓`, "ok");
}

function updateStat() {
  const chunks = state.docs.reduce((a, d) => a + d.chunks, 0);
  const parts = [];
  if (state.docs.length) parts.push(`${state.docs.length} doc${state.docs.length > 1 ? "s" : ""} · ${chunks.toLocaleString()} chunks`);
  if (state.sheets.length) parts.push(`${state.sheets.length} sheet${state.sheets.length > 1 ? "s" : ""}`);
  $("#stat-line").textContent = parts.join("  ·  ");
}

/* ---------------- settings modal ---------------- */
function openSettings() {
  $("#modal-settings").classList.remove("hidden");
  $("#key-input").value = "";
  refreshSettings();
}

function refreshSettings() {
  api("/api/config").then((c) => {
    state.keySet = c.key_set;
    state.tavilySet = !!c.tavily_set;
    state.model = c.model;
    state.voice = c.tts_voice || "troy";
    renderQuota(c.usage || null);
    $("#key-input").placeholder = state.keySet ? "••••••  saved (type to replace)" : "gsk_…";
    $("#key-status").textContent = state.keySet ? "✓ Key saved." : "";
    $("#key-status").className = "key-status";
    $("#tavily-input").placeholder = state.tavilySet ? "••••••  saved (type to replace)" : "tvly_…";
    $("#tavily-status").textContent = state.tavilySet ? "✓ Tavily key saved — web search will use it." : "";
    $("#tts-toggle").checked = state.ttsEnabled;
    const vsel = $("#voice-select");
    vsel.innerHTML = "";
    for (const v of c.voices || ["troy", "austin", "hannah", "jessica", "sam", "leo", "mia"]) {
      const o = document.createElement("option");
      o.value = o.textContent = v;
      o.selected = v === state.voice;
      vsel.appendChild(o);
    }
    api("/api/models").then((m) => {
      const sel = $("#model-select");
      sel.innerHTML = "";
      for (const id of m.models) {
        const o = document.createElement("option");
        o.value = o.textContent = id;
        o.selected = id === c.model;
        sel.appendChild(o);
      }
    }).catch(() => {});
  }).catch(() => {});
}

function closeSettings() {
  $("#modal-settings").classList.add("hidden");
  const key = $("#key-input").value.trim();
  if (key) saveKey(key).catch((e) => toast(e.message, "err"));
  const tv = $("#tavily-input").value.trim();
  if (tv) saveTavily(tv).catch((e) => toast(e.message, "err"));
}

async function saveTavily(key) {
  const r = await api("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tavily_key: key }),
  });
  state.tavilySet = !!r.tavily_set;
  const st = $("#tavily-status");
  st.className = "key-status";
  st.textContent = r.tavily_set ? "✓ Tavily key saved — web search will use it." : "";
  return r;
}

async function saveKey(key) {
  const r = await api("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ groq_key: key || null }),
  });
  state.keySet = r.key_set;
  if (r.key_set) $("#key-banner").classList.add("hidden");
  return r;
}

/* ---------------- identity / account (Clerk + guest) ---------------- */
const identityBtn = $("#btn-identity");
const accountBtn = $("#btn-account");
const identityAvatar = $("#identity-avatar");
const identityMeta = $("#identity-meta");
let clerkEnabled = false;
let clerk = null;

function userInitial(me) {
  const name = (me.name || me.email || "").trim();
  return name ? name[0].toUpperCase() : "?";
}

function setIdentityUI(me, enabled) {
  clerkEnabled = enabled;
  if (me.source === "clerk") {
    // signed-in: avatar initial + name + profile + sign out (sign-up hidden)
    identityAvatar.classList.remove("hidden");
    identityAvatar.textContent = userInitial(me);
    identityMeta.textContent = me.name || me.email || "Signed in";
    identityMeta.title = me.email || me.name || "";
    identityBtn.textContent = "Sign out";
    identityBtn.title = "Sign out";
    identityBtn.classList.remove("hidden");
    accountBtn.classList.remove("hidden");
    accountBtn.title = "Profile — manage your account";
  } else if (me.source === "guest") {
    identityAvatar.classList.add("hidden");
    identityMeta.textContent = "Guest workspace";
    identityBtn.textContent = "Sign in";
    identityBtn.title = "Sign in";
    identityBtn.classList.remove("hidden");
    accountBtn.classList.add("hidden");
  } else {
    identityAvatar.classList.add("hidden");
    identityMeta.textContent = "Local workspace";
    identityBtn.textContent = enabled ? "Sign in" : "";
    identityBtn.classList.toggle("hidden", !enabled);
    accountBtn.classList.add("hidden");
  }
}

async function refreshIdentity() {
  try {
    const r = await api("/api/me");
    setIdentityUI(r.me, !!r.clerk_enabled);
  } catch (_) {}
}

async function initClerk() {
  // Clerk runs client-side via its browser SDK (vanilla JS, no bundler).
  // Modern script-tag integration (clerk.com/docs/js-frontend/getting-started):
  //  1. load the UI bundle and clerk-js from the app's FAPI domain,
  //  2. clerk-js self-initializes with the data-clerk-publishable-key attr,
  //  3. Clerk.load({ ui }) exposes window.Clerk ready for use.
  // Gated: when no publishable key is configured the app stays guest/local.
  const domain = window.__CLERK_DOMAIN__ || "";
  if (!clerkEnabled || clerk || !domain) return;
  try {
    // 1) UI bundle (prebuilt components) from the FAPI domain
    if (!window.__internal_ClerkUICtor) {
      await new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = `https://${domain}/npm/@clerk/ui@1/dist/ui.browser.js`;
        s.async = true;
        s.crossOrigin = "anonymous";
        s.onload = resolve;
        s.onerror = () => reject(new Error("Failed to load Clerk UI bundle"));
        document.head.appendChild(s);
      });
    }
    // 2) clerk-js from the FAPI domain; the publishable-key attribute
    //    makes the global Clerk instance self-configure
    if (!window.Clerk) {
      await new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = `https://${domain}/npm/@clerk/clerk-js@6/dist/clerk.browser.js`;
        s.async = true;
        s.crossOrigin = "anonymous";
        s.setAttribute("data-clerk-publishable-key", clerkPk());
        s.onload = resolve;
        s.onerror = () => reject(new Error("Failed to load Clerk SDK"));
        document.head.appendChild(s);
      });
    }
    if (!window.Clerk) return;
    clerk = window.Clerk;
    await clerk.load({
      ui: { ClerkUI: window.__internal_ClerkUICtor },
      // redirect fallback stays on our server (modal is the primary path);
      // Clerk's hosted pages can 404 for some instances
      signInUrl: "/sign-in",
      signUpUrl: "/sign-up",
    });
    if (clerk.session) clerkToken = await clerk.session.getToken();
    refreshIdentity();
    // keep the token fresh when the session changes (sign in / sign out)
    clerk.addListener(async (payload) => {
      clerkToken = payload.session ? await payload.session.getToken() : null;
      refreshIdentity();
      // a signed-out user falls back to the guest workspace for this device
      if (!clerkToken) reloadWorkspace();
    });
  } catch (err) {
    console.warn("Clerk init failed:", err);
    toast("Clerk sign-in unavailable — using guest workspace.", "err");
  }
}

function clerkPk() {
  // publishable key comes from /api/state (safe: it is public by design)
  return window.__CLERK_PK__ || "";
}

async function onIdentityClick() {
  if (!clerkEnabled) return;
  if (clerk && clerk.user) {
    await clerk.signOut();
    clerkToken = null;
    reloadWorkspace();
    return;
  }
  try {
    await initClerk();
    if (clerk && clerk.openSignIn) clerk.openSignIn();
  } catch (err) {
    toast(err.message, "err");
  }
}

function reloadWorkspace() {
  // switch workspace -> refetch everything (docs, chats, memory are per-user)
  Promise.all([
    api("/api/state").then((st) => {
      state.docs = st.docs; state.sessions = st.sessions; state.sheets = st.sheets;
      setDemoChats(st.demo_max_chats);
      renderDocs(); renderSheets(); renderSessions(); updateStat();
    }),
    loadMemory(),
    refreshIdentity(),
  ]).catch(() => {});
}

/* ---------------- resizable sidebar ---------------- */
const SIDEBAR_KEY = "jarvis_sidebar_w";
const SIDEBAR_MIN = 260, SIDEBAR_MAX = 480, SIDEBAR_DEFAULT = 360;
const resizer = $("#sidebar-resizer");
let sidebarDrag = null;

function sidebarWidth() {
  const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--sidebar-w"));
  return Number.isFinite(v) && v > 0 ? v : SIDEBAR_DEFAULT;
}

function setSidebarWidth(w) {
  w = Math.round(Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, w)));
  document.documentElement.style.setProperty("--sidebar-w", w + "px");
  resizer.setAttribute("aria-valuenow", String(w));
  try { localStorage.setItem(SIDEBAR_KEY, String(w)); } catch (_) {}
}

function onResizeMove(e) {
  if (!sidebarDrag) return;
  setSidebarWidth(sidebarDrag.startW + (e.clientX - sidebarDrag.startX));
}

function onResizeUp() {
  sidebarDrag = null;
  document.body.classList.remove("resizing");
  document.removeEventListener("mousemove", onResizeMove);
  document.removeEventListener("mouseup", onResizeUp);
}

resizer.addEventListener("mousedown", (e) => {
  e.preventDefault();  // no text selection while dragging
  sidebarDrag = { startX: e.clientX, startW: sidebarWidth() };
  document.body.classList.add("resizing");
  document.addEventListener("mousemove", onResizeMove);
  document.addEventListener("mouseup", onResizeUp);
});

// keyboard: ArrowLeft/ArrowRight when the divider is focused (Shift = larger step)
resizer.addEventListener("keydown", (e) => {
  const step = e.shiftKey ? 40 : 10;
  if (e.key === "ArrowLeft") { e.preventDefault(); setSidebarWidth(sidebarWidth() - step); }
  else if (e.key === "ArrowRight") { e.preventDefault(); setSidebarWidth(sidebarWidth() + step); }
});

// restore the user's persisted width
const _savedW = parseInt(localStorage.getItem(SIDEBAR_KEY) || "", 10);
if (Number.isFinite(_savedW)) setSidebarWidth(_savedW);

/* ---------------- init ---------------- */
async function init() {
  const st = await api("/api/state");
  state.docs = st.docs;
  state.sessions = st.sessions;
  state.sheets = st.sheets;
  state.keySet = st.key_set;
  state.tavilySet = !!st.tavily_set;
  state.model = st.model;
  state.voice = st.tts_voice || state.voice;
  setDemoChats(st.demo_max_chats);
  renderDocs();
  renderSheets();
  renderSessions();
  updateStat();
  renderQuota(st.usage || null);
  loadMemory().catch(() => {});
  $("#key-banner").classList.toggle("hidden", st.key_set);
  $("#tts-toggle").checked = state.ttsEnabled;
  if (st.clerk_pk) window.__CLERK_PK__ = st.clerk_pk;
  if (st.clerk_domain) window.__CLERK_DOMAIN__ = st.clerk_domain;
  setIdentityUI(st.me || { source: "local" }, !!st.clerk_enabled);
  if (st.clerk_enabled) initClerk().catch(() => {});
}

/* ---------------- Groq daily token usage meter ---------------- */
function fmtTokens(n) {
  if (n == null) return "—";
  return n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "k" : String(n);
}

function renderQuota(usage) {
  const sidebar = $("#quota"), settings = $("#quota-settings");
  const visible = !!(usage && usage.used > 0);
  sidebar.classList.toggle("hidden", !visible);
  settings.classList.toggle("hidden", !visible);
  if (!visible) return;

  const used = usage.used, limit = usage.limit;
  const pct = usage.pct;
  const fillCls = pct == null ? "" : pct >= 90 ? "danger" : pct >= 75 ? "warn" : "";

  $("#quota-fill").className = "quota-fill" + (fillCls ? " " + fillCls : "");
  $("#quota-settings-fill").className = "quota-fill" + (fillCls ? " " + fillCls : "");
  const width = pct == null ? 100 : Math.min(pct, 100);
  $("#quota-fill").style.width = width + "%";
  $("#quota-settings-fill").style.width = width + "%";

  const summary = limit
    ? `${fmtTokens(used)} / ${fmtTokens(limit)} tokens used today`
    : `${fmtTokens(used)} tokens used today`;
  $("#quota-text").textContent = summary;
  $("#quota-settings-text").textContent = summary;
  $("#quota-settings-pct").textContent = pct == null ? "" : `${pct}%`;
  sidebar.title = limit ? `Groq daily usage: ${fmtTokens(used)} of ${fmtTokens(limit)} tokens` : summary;
}

async function refreshQuota() {
  try {
    const c = await api("/api/config");
    renderQuota(c.usage || null);
  } catch (_) {}
}

/* ---------------- events ---------------- */
sendBtn.addEventListener("click", () => {
  if (state.streaming) { state.abortReason = "user"; state.abort.abort(); return; }
  const text = inputEl.value.trim();
  if (text) sendMessage(text.slice(0, 4000));
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + "px";
}
inputEl.addEventListener("input", autoResize);

micBtn.addEventListener("click", toggleMic);

// doc upload
const dropzone = $("#dropzone");
dropzone.addEventListener("click", () => $("#file-input").click());
$("#btn-add").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (e) => {
  uploadFiles([...e.target.files]);
  e.target.value = "";
});
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));

// sheet upload
$("#btn-add-sheet").addEventListener("click", () => $("#sheet-input").click());
$("#sheet-input").addEventListener("change", (e) => {
  for (const f of [...e.target.files]) uploadSheet(f);
  e.target.value = "";
});

// sidebar
$("#btn-new").addEventListener("click", newChat);
$("#btn-settings").addEventListener("click", openSettings);
$("#btn-account").addEventListener("click", () => { if (clerk) clerk.openUserProfile(); });
$("#btn-identity").addEventListener("click", onIdentityClick);
$("#btn-key-banner").addEventListener("click", () => { openSettings(); $("#key-input").focus(); });
$("#btn-close-settings").addEventListener("click", closeSettings);
$("#btn-refresh-memory").addEventListener("click", () => loadMemory().catch(() => {}));
$("#chat-search").addEventListener("input", renderSessions);
$("#btn-mem-add").addEventListener("click", addMemoryFact);
$("#mem-input").addEventListener("keydown", (e) => { if (e.key === "Enter") addMemoryFact(); });

// settings modal
$("#btn-save-key").addEventListener("click", async () => {
  try {
    const r = await saveKey($("#key-input").value.trim());
    const st = $("#key-status");
    st.className = "key-status";
    st.textContent = r.key_set ? "✓ Key saved." : "Key removed.";
    refreshSettings();
  } catch (err) { toast(err.message, "err"); }
});
$("#model-select").addEventListener("change", async (e) => {
  try {
    const r = await api("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: e.target.value }),
    });
    state.model = r.model;
    const st = $("#model-status");
    st.className = "key-status";
    st.textContent = "✓ Model saved: " + r.model;
    refreshQuota();  // per-model limits differ, so the meter changes
  } catch (err) { toast(err.message, "err"); }
});
$("#voice-select").addEventListener("change", async (e) => {
  state.voice = e.target.value;
  try {
    const r = await api("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tts_voice: e.target.value }),
    });
    const st = $("#voice-status");
    st.className = "key-status";
    st.textContent = "✓ Voice set: " + r.tts_voice;
  } catch (err) { toast(err.message, "err"); }
});
$("#tts-toggle").addEventListener("change", (e) => {
  state.ttsEnabled = e.target.checked;
  localStorage.setItem("jarvis_tts", state.ttsEnabled ? "1" : "0");
});
$("#btn-test").addEventListener("click", async () => {
  const st = $("#model-status");
  st.className = "key-status";
  st.textContent = "Testing…";
  try {
    const m = await api("/api/models");
    st.textContent = `✓ Connection OK — ${m.models.length} models available.`;
    refreshSettings();
  } catch (err) {
    st.className = "key-status err";
    st.textContent = "✗ " + err.message;
  }
});
$("#btn-clear-key").addEventListener("click", async () => {
  $("#key-input").value = "";
  try {
    await saveKey(null);
    $("#key-banner").classList.remove("hidden");
    $("#key-status").className = "key-status";
    $("#key-status").textContent = "Key removed.";
    refreshSettings();
  } catch (err) { toast(err.message, "err"); }
});
$("#modal-settings").addEventListener("click", (e) => {
  if (e.target.id === "modal-settings") closeSettings();
});

// suggestion chips
document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => sendMessage(chip.dataset.q)));

init().catch((err) => toast("Startup error: " + err.message, "err"));

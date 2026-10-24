"use strict";

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const form = $("ask-form");
const questionEl = $("question");
const askBtn = $("ask-btn");
const downloadBtn = $("download-btn");
const kbSelect = $("kb-select");

let activeKbId = "general";
let kbCatalog = [];
let readOnlyMode = false;
let latestAnswer = "";
let latestQuestion = "";

// Configuración de Markdown (Marked) y fórmulas (KaTeX)
if (typeof marked !== "undefined") {
  if (typeof markedKatex !== "undefined") {
    marked.use(markedKatex({ throwOnError: false, nonStandard: true }));
  }
  marked.setOptions({
    gfm: true,
    breaks: true,
  });
}

// ---------- utilidades ----------
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/** Recorre los nodos de texto de un elemento renderizado y reemplaza [n] por chips clicables. */
function transformCitationsInDom(root, msgId) {
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        let parent = node.parentElement;
        while (parent && parent !== root) {
          const tag = parent.tagName;
          if (tag === "CODE" || tag === "PRE" || parent.classList.contains("katex")) {
            return NodeFilter.FILTER_REJECT;
          }
          parent = parent.parentElement;
        }
        return /\[\d+\]/.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
      },
    }
  );

  const textNodes = [];
  while (walker.nextNode()) {
    textNodes.push(walker.currentNode);
  }

  for (const node of textNodes) {
    const parent = node.parentNode;
    if (!parent) continue;
    const parts = node.nodeValue.split(/(\[\d+\])/g);
    const frag = document.createDocumentFragment();
    for (const part of parts) {
      const m = part.match(/^\[(\d+)\]$/);
      if (m) {
        const chip = el("span", "cite", part);
        chip.dataset.target = `${msgId}-src-${m[1]}`;
        chip.addEventListener("click", () => {
          const target = document.getElementById(chip.dataset.target);
          if (!target) return;
          target.scrollIntoView({ block: "nearest" });
          target.classList.add("flash");
          setTimeout(() => target.classList.remove("flash"), 1200);
        });
        frag.appendChild(chip);
      } else if (part) {
        frag.appendChild(document.createTextNode(part));
      }
    }
    parent.replaceChild(frag, node);
  }
}

function renderWithCitationsFallback(container, text, msgId) {
  container.replaceChildren();
  const parts = text.split(/(\[\d+\])/g);
  for (const part of parts) {
    const m = part.match(/^\[(\d+)\]$/);
    if (m) {
      const chip = el("span", "cite", part);
      chip.dataset.target = `${msgId}-src-${m[1]}`;
      chip.addEventListener("click", () => {
        const target = document.getElementById(chip.dataset.target);
        if (!target) return;
        target.scrollIntoView({ block: "nearest" });
        target.classList.add("flash");
        setTimeout(() => target.classList.remove("flash"), 1200);
      });
      container.appendChild(chip);
    } else if (part) {
      container.appendChild(document.createTextNode(part));
    }
  }
}

/** Renderiza Markdown y KaTeX convirtiendo [n] en chips interactivos. */
function renderWithCitations(container, text, msgId) {
  if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
    try {
      const rawHtml = marked.parse(text);
      container.innerHTML = DOMPurify.sanitize(rawHtml);
      transformCitationsInDom(container, msgId);
      container.className = "msg-text markdown-body";
      return;
    } catch (e) {
      console.error("Error al renderizar Markdown/KaTeX:", e);
    }
  }
  container.className = "msg-text streaming";
  renderWithCitationsFallback(container, text, msgId);
}

function sourceNode(s, msgId) {
  const node = el("div", "source");
  node.id = `${msgId}-src-${s.id}`;
  const loc = [s.section, s.page != null ? `pág. ${s.page}` : null].filter(Boolean).join(" · ");
  node.appendChild(el("div", "src-head", `[${s.id}] ${s.source}${loc ? " — " + loc : ""}`));
  node.appendChild(el("div", "snippet", s.snippet));
  return node;
}

/** Limpia marcas de citas tipo [1], [4] y normaliza espacios para la descarga en .txt */
function cleanAnswerForDownload(text) {
  if (!text) return "";
  let cleaned = text.replace(/\[\d+\]/g, "");
  cleaned = cleaned.replace(/[ \t]{2,}/g, " ");
  cleaned = cleaned.replace(/ +([,.;:])/g, "$1");
  return cleaned
    .split("\n")
    .map((line) => line.trimEnd())
    .join("\n")
    .trim();
}

function getDownloadFilename(question) {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}`;
  if (question) {
    const slug = question
      .slice(0, 25)
      .toLowerCase()
      .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
    if (slug) return `respuesta-${slug}-${stamp}.txt`;
  }
  return `respuesta-${stamp}.txt`;
}

function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------- salud ----------
async function checkHealth() {
  const badge = $("health");
  try {
    const r = await fetch("/health/live");
    const body = r.ok ? await r.json() : {};
    const ok = r.ok && body.status === "ok";
    readOnlyMode = Boolean(body.read_only_mode);
    badge.className = `badge ${ok ? "badge-ok" : "badge-down"}`;
    badge.textContent = ok ? (readOnlyMode ? "online · solo lectura" : "online") : "error";
    $("new-kb-btn").disabled = readOnlyMode;
    $("upload-btn").disabled = readOnlyMode;
    $("file").disabled = readOnlyMode;
    updateKbBanner();
  } catch {
    badge.className = "badge badge-down";
    badge.textContent = "offline";
  }
}

// ---------- catálogo de Knowledge Bases ----------
function updateKbBanner() {
  const current = kbCatalog.find((k) => k.id === activeKbId) || {
    id: activeKbId,
    name: activeKbId,
    description: "",
  };
  $("active-kb-name").textContent = current.name;
  $("active-kb-desc").textContent = current.description ? `— ${current.description}` : "";
  $("docs-kb-label").textContent = current.name;
  $("del-kb-btn").disabled = readOnlyMode || activeKbId === "general";
}

async function loadKbs(selectedId) {
  try {
    const r = await fetch("/knowledge-bases");
    if (!r.ok) return;
    const { knowledge_bases } = await r.json();
    kbCatalog = knowledge_bases || [];
    kbSelect.replaceChildren();
    for (const kb of kbCatalog) {
      const opt = el("option", null, `${kb.name} (${kb.documents} docs)`);
      opt.value = kb.id;
      kbSelect.appendChild(opt);
    }
    if (selectedId && kbCatalog.some((k) => k.id === selectedId)) {
      activeKbId = selectedId;
    } else if (!kbCatalog.some((k) => k.id === activeKbId)) {
      activeKbId = kbCatalog[0]?.id || "general";
    }
    kbSelect.value = activeKbId;
    updateKbBanner();
  } catch {
    /* silencioso en arranque si offline */
  }
}

kbSelect?.addEventListener("change", () => {
  activeKbId = kbSelect.value;
  setStatus("");
  $("retrain-container").hidden = true;
  updateKbBanner();
  loadDocs();
});

$("new-kb-btn")?.addEventListener("click", () => {
  $("kb-name").value = "";
  $("kb-desc").value = "";
  $("kb-modal-error").textContent = "";
  $("kb-modal").showModal();
  $("kb-name").focus();
});

$("kb-cancel")?.addEventListener("click", () => $("kb-modal").close());

$("kb-form")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const name = $("kb-name").value.trim();
  const description = $("kb-desc").value.trim();
  if (!name) return;
  const r = await fetch("/knowledge-bases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description }),
  });
  if (!r.ok) {
    $("kb-modal-error").textContent = await errorText(r);
    return;
  }
  const created = await r.json();
  $("kb-modal").close();
  await loadKbs(created.id);
  await loadDocs();
  setStatus(`Base '${created.name}' creada.`, "ok");
});

$("del-kb-btn")?.addEventListener("click", async () => {
  if (activeKbId === "general") return;
  const current = kbCatalog.find((k) => k.id === activeKbId);
  const label = current ? current.name : activeKbId;
  if (!confirm(`¿Eliminar la base "${label}" y todos sus documentos indexados?`)) return;
  const r = await fetch(`/knowledge-bases/${encodeURIComponent(activeKbId)}`, {
    method: "DELETE",
  });
  if (r.ok) {
    setStatus(`Base "${label}" eliminada.`, "ok");
    await loadKbs("general");
    await loadDocs();
  } else {
    setStatus(await errorText(r), "err");
  }
});

// ---------- chat (SSE por fetch) ----------
let counter = 0;

function addUserMessage(text) {
  $("empty-hint")?.remove();
  messagesEl.appendChild(el("div", "msg user", text));
  scrollDown();
}

/** Parsea un stream SSE de un Response y llama onEvent(event, data) por cada evento. */
async function readSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message";
      const dataLines = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (!dataLines.length) continue;
      try {
        onEvent(event, JSON.parse(dataLines.join("\n")));
      } catch (e) {
        console.error("Evento SSE inválido", raw, e);
      }
    }
  }
}

async function ask(question, topK) {
  const msgId = `m${++counter}`;
  const bubble = el("div", "msg bot");
  const textEl = el("div", "msg-text streaming");
  const extra = el("div");
  bubble.append(textEl, extra);
  messagesEl.appendChild(bubble);

  let answer = "";
  let retrieved = [];
  let renderRafId = null;
  const started = performance.now();
  const kbSnapshot = activeKbId;

  const scheduleRender = () => {
    if (renderRafId !== null) return;
    renderRafId = requestAnimationFrame(() => {
      renderRafId = null;
      renderWithCitations(textEl, answer, msgId);
      scrollDown();
    });
  };

  const fail = (message) => {
    if (renderRafId !== null) {
      cancelAnimationFrame(renderRafId);
      renderRafId = null;
    }
    bubble.classList.add("error");
    textEl.textContent = message;
  };

  try {
    const body = { question, knowledge_base_id: kbSnapshot };
    if (topK) body.top_k = topK;
    const resp = await fetch("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      let detail = `Error ${resp.status}`;
      try {
        const j = await resp.json();
        detail = j.detail
          ? typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)
          : detail;
      } catch { /* sin cuerpo JSON */ }
      return fail(detail);
    }

    await readSSE(resp, (event, data) => {
      if (event === "retrieved") {
        retrieved = data.retrieved || [];
        textEl.textContent = "Generando respuesta...";
      } else if (event === "token") {
        answer += data.token;
        scheduleRender();
      } else if (event === "end") {
        if (renderRafId !== null) {
          cancelAnimationFrame(renderRafId);
          renderRafId = null;
        }
        const finalText = typeof data.answer === "string" && data.answer ? data.answer : answer;
        if (!data.grounded) {
          bubble.classList.add("warn");
          textEl.textContent = data.message || "No se pudo verificar la respuesta con los documentos.";
        } else {
          renderWithCitations(textEl, finalText.trim(), msgId);
          latestAnswer = finalText.trim();
          latestQuestion = question;
        }
        const cited = data.sources || [];
        if (cited.length) {
          const box = el("div", "sources");
          cited.forEach((s) => box.appendChild(sourceNode(s, msgId)));
          extra.appendChild(box);
        }
        const others = retrieved.filter((r) => !cited.some((c) => c.id === r.id));
        if (others.length) {
          const det = el("details", "retrieved");
          det.appendChild(el("summary", null, `Otros fragmentos recuperados (${others.length})`));
          const box = el("div", "sources");
          others.forEach((s) => box.appendChild(sourceNode(s, msgId)));
          det.appendChild(box);
          extra.appendChild(det);
        }
        const secs = ((performance.now() - started) / 1000).toFixed(1);
        extra.appendChild(el("div", "meta", `${secs} s · KB: ${kbSnapshot}`));
      } else if (event === "error") {
        bubble.classList.add(data.reason ? "warn" : "error");
        textEl.textContent = data.message || data.detail || "Error al procesar la consulta.";
      }
    });
    if (!textEl.textContent) textEl.textContent = "La respuesta llegó vacía.";
  } catch (e) {
    fail(`No se pudo conectar con el servidor: ${e.message}`);
  }
  scrollDown();
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const question = questionEl.value.trim();
  if (question.length < 3) return;
  const k = parseInt($("top-k").value, 10);
  questionEl.value = "";
  askBtn.disabled = true;
  if (downloadBtn) downloadBtn.disabled = true;
  addUserMessage(question);
  await ask(question, Number.isInteger(k) ? k : null);
  askBtn.disabled = false;
  if (downloadBtn && latestAnswer) downloadBtn.disabled = false;
  questionEl.focus();
});

downloadBtn?.addEventListener("click", () => {
  if (!latestAnswer) return;
  const cleanText = cleanAnswerForDownload(latestAnswer);
  const filename = getDownloadFilename(latestQuestion);
  downloadTextFile(filename, cleanText);
});

questionEl.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    form.requestSubmit();
  }
});

// ---------- documentos de la KB activa ----------
const statusEl = $("upload-status");
let statusTimer = null;

function setStatus(text, kind, autoDismissMs = (kind === "ok" ? 3500 : 0)) {
  if (statusTimer) {
    clearTimeout(statusTimer);
    statusTimer = null;
  }
  statusEl.classList.remove("fade-out");
  statusEl.textContent = text;
  statusEl.className = `status ${kind || ""}`;
  if (!text) return;

  if (autoDismissMs > 0) {
    statusTimer = setTimeout(() => {
      statusEl.classList.add("fade-out");
      setTimeout(() => {
        if (statusEl.classList.contains("fade-out")) {
          statusEl.textContent = "";
          statusEl.className = "status";
          statusEl.classList.remove("fade-out");
        }
      }, 400);
    }, autoDismissMs);
  }
}

$("file")?.addEventListener("change", () => {
  setStatus("");
  $("retrain-container").hidden = true;
});

async function errorText(resp) {
  try {
    const j = await resp.json();
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return j.detail.map((d) => d.msg).join("; ");
  } catch { /* sin JSON */ }
  return `Error ${resp.status}`;
}

async function loadDocs() {
  const list = $("docs-list");
  try {
    const r = await fetch(`/knowledge-bases/${encodeURIComponent(activeKbId)}/documents`);
    if (!r.ok) throw new Error(await errorText(r));
    const { total, documents } = await r.json();
    $("docs-count").textContent = `(${total})`;
    list.replaceChildren();
    if (!documents.length) {
      list.appendChild(el("li", "muted", "Sin documentos indexados en esta base."));
      return;
    }
    for (const d of documents) {
      const li = el("li");
      const info = el("div");
      info.appendChild(el("div", "doc-name", d.source));
      const pages = d.pages ? ` · ${d.pages} pág.` : "";
      info.appendChild(el("div", "doc-meta", `${d.file_type} · ${d.chunks} chunks${pages}`));
      const del = el("button", "danger", "Borrar");
      del.type = "button";
      del.disabled = readOnlyMode;
      del.addEventListener("click", () => deleteDoc(d));
      li.append(info, del);
      list.appendChild(li);
    }
  } catch (e) {
    list.replaceChildren(el("li", "muted", `No se pudo cargar la lista: ${e.message}`));
  }
}

async function deleteDoc(d) {
  if (!confirm(`¿Borrar "${d.source}" de la base "${activeKbId}"?`)) return;
  const url = `/knowledge-bases/${encodeURIComponent(activeKbId)}/documents/${encodeURIComponent(d.doc_id)}`;
  const r = await fetch(url, { method: "DELETE" });
  if (r.ok) setStatus(`Borrado: ${d.source}`, "ok");
  else setStatus(await errorText(r), "err");
  await loadKbs(activeKbId);
  await loadDocs();
}

function uploadWithRetrainingProgress(file, kbId) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const fd = new FormData();
    fd.append("file", file);

    const retrainContainer = $("retrain-container");
    const retrainSpinner = $("retrain-spinner");
    const retrainStatusText = $("retrain-status-text");
    const retrainPct = $("retrain-pct");
    const retrainBar = $("retrain-progress-bar");
    const retrainDetail = $("retrain-detail");

    retrainContainer.hidden = false;
    retrainSpinner?.classList.remove("hidden");
    retrainBar.className = "retrain-progress-fill";
    retrainBar.style.width = "0%";
    retrainPct.textContent = "0%";
    retrainStatusText.textContent = "Reentrenando base de conocimiento...";
    retrainDetail.textContent = "Subiendo archivo...";

    let progress = 0;
    let timer = null;

    function setVisualProgress(pct, detail, statusText) {
      progress = pct;
      retrainBar.style.width = `${pct}%`;
      retrainPct.textContent = `${pct}%`;
      if (detail) retrainDetail.textContent = detail;
      if (statusText) retrainStatusText.textContent = statusText;
    }

    // Subida de red (0% a 35%)
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable && ev.total > 0) {
        const netPct = Math.min(35, Math.round((ev.loaded / ev.total) * 35));
        setVisualProgress(netPct, `Subiendo ${file.name} (${Math.round(ev.loaded / 1024)} KB)...`);
      }
    };

    // Al terminar la transferencia HTTP, el backend extrae texto y computa embeddings (reentrenamiento)
    xhr.upload.onload = () => {
      setVisualProgress(40, "Extrayendo contenido del documento...", "Reentrenando base de conocimiento...");
      timer = setInterval(() => {
        if (progress < 60) {
          setVisualProgress(progress + 5, "Extrayendo texto y dividiendo en fragmentos...");
        } else if (progress < 85) {
          setVisualProgress(progress + 3, "Reentrenando: generando embeddings vectoriales...");
        } else if (progress < 95) {
          setVisualProgress(progress + 1, "Indexando vectores en la base de conocimiento...");
        }
      }, 250);
    };

    xhr.onload = () => {
      clearInterval(timer);
      if (xhr.status >= 200 && xhr.status < 300) {
        let data = {};
        try {
          data = JSON.parse(xhr.responseText);
        } catch { /* fallback */ }
        setVisualProgress(100, `Documento "${data.source || file.name}" indexado (${data.chunks || 0} chunks).`, "¡Reentrenamiento completado!");
        retrainBar.classList.add("success");
        retrainSpinner?.classList.add("hidden");
        resolve(data);
      } else {
        retrainBar.classList.add("error");
        retrainSpinner?.classList.add("hidden");
        let errMsg = `Error ${xhr.status}`;
        try {
          const j = JSON.parse(xhr.responseText);
          if (typeof j.detail === "string") errMsg = j.detail;
          else if (Array.isArray(j.detail)) errMsg = j.detail.map((d) => d.msg).join("; ");
        } catch { /* fallback */ }
        retrainStatusText.textContent = "Error al reentrenar";
        retrainDetail.textContent = errMsg;
        reject(new Error(errMsg));
      }
    };

    xhr.onerror = () => {
      clearInterval(timer);
      retrainBar.classList.add("error");
      retrainSpinner?.classList.add("hidden");
      retrainStatusText.textContent = "Error de conexión";
      retrainDetail.textContent = "No se pudo conectar con el backend.";
      reject(new Error("Error de red"));
    };

    xhr.open("POST", `/knowledge-bases/${encodeURIComponent(kbId)}/documents`);
    xhr.send(fd);
  });
}

$("upload-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const fileInput = $("file");
  const file = fileInput.files[0];
  if (!file) return;
  const btn = $("upload-btn");
  btn.disabled = true;
  fileInput.disabled = true;
  setStatus("");
  try {
    const d = await uploadWithRetrainingProgress(file, activeKbId);
    setStatus(`Indexado: ${d.source} (${d.chunks} chunks)`, "ok", 3500);
    fileInput.value = "";
    setTimeout(() => {
      const container = $("retrain-container");
      const bar = $("retrain-progress-bar");
      if (bar?.classList.contains("success") && container) {
        container.hidden = true;
      }
    }, 3500);
  } catch (e) {
    setStatus(e.message, "err");
  } finally {
    btn.disabled = false;
    fileInput.disabled = false;
    await loadKbs(activeKbId);
    await loadDocs();
  }
});

checkHealth();
setInterval(checkHealth, 15000);
loadKbs("general").then(() => loadDocs());
questionEl.focus();

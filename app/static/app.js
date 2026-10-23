"use strict";

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const form = $("ask-form");
const questionEl = $("question");
const askBtn = $("ask-btn");

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

/** Renderiza texto convirtiendo [n] en chips, siempre vía nodos de texto (sin innerHTML). */
function renderWithCitations(container, text, msgId) {
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

function sourceNode(s, msgId) {
  const node = el("div", "source");
  node.id = `${msgId}-src-${s.id}`;
  const loc = [s.section, s.page != null ? `pág. ${s.page}` : null].filter(Boolean).join(" · ");
  node.appendChild(el("div", "src-head", `[${s.id}] ${s.source}${loc ? " — " + loc : ""}`));
  node.appendChild(el("div", "snippet", s.snippet));
  return node;
}

// ---------- salud ----------
async function checkHealth() {
  const badge = $("health");
  try {
    const r = await fetch("/health");
    const ok = r.ok && (await r.json()).status === "ok";
    badge.className = `badge ${ok ? "badge-ok" : "badge-down"}`;
    badge.textContent = ok ? "online" : "error";
  } catch {
    badge.className = "badge badge-down";
    badge.textContent = "offline";
  }
}

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
  const textEl = el("div");
  const extra = el("div");
  bubble.append(textEl, extra);
  messagesEl.appendChild(bubble);

  let answer = "";
  let retrieved = [];
  const started = performance.now();

  const fail = (message) => {
    bubble.classList.add("error");
    textEl.textContent = message;
  };

  try {
    const body = { question };
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
        textEl.textContent = answer;
        scrollDown();
      } else if (event === "end") {
        if (!data.grounded) {
          bubble.classList.add("warn");
          textEl.textContent = data.message || "No se pudo verificar la respuesta con los documentos.";
        } else {
          renderWithCitations(textEl, answer.trim(), msgId);
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
        extra.appendChild(el("div", "meta", `${secs} s`));
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
  addUserMessage(question);
  await ask(question, Number.isInteger(k) ? k : null);
  askBtn.disabled = false;
  questionEl.focus();
});

questionEl.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    form.requestSubmit();
  }
});

// ---------- documentos ----------
const statusEl = $("upload-status");

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = `status ${kind || ""}`;
}

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
    const r = await fetch("/documents");
    if (!r.ok) throw new Error(await errorText(r));
    const { total, documents } = await r.json();
    $("docs-count").textContent = `(${total})`;
    list.replaceChildren();
    if (!documents.length) {
      list.appendChild(el("li", "muted", "Sin documentos indexados."));
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
      del.addEventListener("click", () => deleteDoc(d));
      li.append(info, del);
      list.appendChild(li);
    }
  } catch (e) {
    list.replaceChildren(el("li", "muted", `No se pudo cargar la lista: ${e.message}`));
  }
}

async function deleteDoc(d) {
  if (!confirm(`¿Borrar "${d.source}" del índice y de disco?`)) return;
  const r = await fetch(`/documents/${encodeURIComponent(d.doc_id)}`, { method: "DELETE" });
  if (r.ok) setStatus(`Borrado: ${d.source}`, "ok");
  else setStatus(await errorText(r), "err");
  loadDocs();
}

$("upload-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const file = $("file").files[0];
  if (!file) return;
  const btn = $("upload-btn");
  btn.disabled = true;
  setStatus("Subiendo e indexando...");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/documents", { method: "POST", body: fd });
    if (r.ok) {
      const d = await r.json();
      setStatus(`Indexado: ${d.source} (${d.chunks} chunks)`, "ok");
      $("file").value = "";
    } else {
      setStatus(await errorText(r), "err");
    }
  } catch (e) {
    setStatus(`Error de red: ${e.message}`, "err");
  }
  btn.disabled = false;
  loadDocs();
});

checkHealth();
setInterval(checkHealth, 15000);
loadDocs();
questionEl.focus();

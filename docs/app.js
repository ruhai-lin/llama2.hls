"use strict";

const MODEL_ID = "tinystories-15m-w8a8-kv260";
const MAX_TOKENS = 64;
const NGROK_HEADERS = {"ngrok-skip-browser-warning": "1"};
const API_BASE = document
  .querySelector('meta[name="llama2-api-base"]')
  .content.replace(/\/$/, "");

const status = document.getElementById("status");
const statusText = document.getElementById("status-text");
const outputPanel = document.getElementById("output-panel");
const welcome = document.getElementById("welcome");
const completion = document.getElementById("completion");
const metrics = document.getElementById("metrics");
const form = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt");
const sendButton = document.getElementById("send");

let online = false;
let generating = false;

function updateControls() {
  const disabled = !online || generating;
  promptInput.disabled = disabled;
  sendButton.disabled = disabled || promptInput.value.trim() === "";
  outputPanel.setAttribute("aria-busy", String(generating));
}

function setOnline(value) {
  online = value;
  status.classList.toggle("status-online", value);
  status.classList.toggle("status-resting", !value);
  statusText.textContent = value ? "KV260 Online" : "KV260 is resting";
  updateControls();
  if (value) {
    promptInput.focus();
  }
}

function showOutput(text) {
  welcome.hidden = true;
  completion.hidden = false;
  completion.textContent = text;
}

async function checkHealth() {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(`${API_BASE}/health`, {
      cache: "no-store",
      headers: NGROK_HEADERS,
      signal: controller.signal,
    });
    const result = await response.json();
    setOnline(response.ok && result.status === "ok");
  } catch {
    setOnline(false);
  } finally {
    window.clearTimeout(timeout);
  }
}

async function generate(prompt) {
  generating = true;
  metrics.hidden = true;
  showOutput("Generating on KV260...");
  updateControls();

  const startedAt = performance.now();
  try {
    const response = await fetch(`${API_BASE}/v1/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...NGROK_HEADERS,
      },
      body: JSON.stringify({
        model: MODEL_ID,
        prompt,
        max_tokens: MAX_TOKENS,
        temperature: 0,
        stream: false,
      }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();
    const text = result?.choices?.[0]?.text;
    if (typeof text !== "string") {
      throw new Error("Invalid completion response");
    }

    const elapsedSeconds = (performance.now() - startedAt) / 1000;
    const tokenCount = result?.usage?.completion_tokens;
    const rate = Number.isFinite(tokenCount) && elapsedSeconds > 0
      ? `${(tokenCount / elapsedSeconds).toFixed(1)} tok/s / `
      : "";

    showOutput(text);
    metrics.textContent = `KV260 / ${rate}${elapsedSeconds.toFixed(2)} s`;
    metrics.hidden = false;
    promptInput.value = "";
  } catch (error) {
    showOutput("Generation failed. Please try again.");
    if (error instanceof TypeError) {
      setOnline(false);
    }
  } finally {
    generating = false;
    updateControls();
    if (online) {
      promptInput.focus();
    }
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (prompt !== "" && online && !generating) {
    void generate(prompt);
  }
});

promptInput.addEventListener("input", updateControls);
promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

void checkHealth();

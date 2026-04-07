<script setup>
import { ref, nextTick, watch, computed, onMounted } from "vue";

const apiBase = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
const streamUrl = computed(() => (apiBase ? `${apiBase}/api/chat/stream` : "/api/chat/stream"));
const apiKey = (import.meta.env.VITE_API_KEY || "").trim();

/** 浏览器本地记住上次会话 id，刷新后向服务端拉取历史 */
const STORAGE_CONV_ID = "hr_rag_conversation_id";

const question = ref("");
/** 后端 MySQL 会话 id，多轮问答时回传 */
const conversationId = ref(null);
/** 一条对话：用户或助手 */
const messages = ref([]);
const loading = ref(false);
/** 首屏从 MySQL 拉历史时为 true */
const restoring = ref(false);
const error = ref("");
const chatScroll = ref(null);
let abortCtrl = null;

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function scrollToBottom() {
  chatScroll.value?.scrollTo({ top: chatScroll.value.scrollHeight, behavior: "smooth" });
}

function streamHeaders() {
  const h = { "Content-Type": "application/json", Accept: "text/event-stream" };
  if (apiKey) h["X-API-Key"] = apiKey;
  return h;
}

/** GET JSON 接口用的请求头（与会话接口一致） */
function jsonHeaders() {
  const h = { Accept: "application/json" };
  if (apiKey) h["X-API-Key"] = apiKey;
  return h;
}

/**
 * 刷新页面后：若有本地保存的 conversation_id，则请求历史消息并填充列表
 */
async function restoreSessionFromServer() {
  const saved = localStorage.getItem(STORAGE_CONV_ID);
  if (!saved) return;
  restoring.value = true;
  try {
    const path = `/api/conversations/${encodeURIComponent(saved)}/messages`;
    const url = apiBase ? `${apiBase}${path}` : path;
    const res = await fetch(url, { headers: jsonHeaders() });
    if (!res.ok) {
      localStorage.removeItem(STORAGE_CONV_ID);
      return;
    }
    const rows = await res.json();
    if (!Array.isArray(rows)) {
      localStorage.removeItem(STORAGE_CONV_ID);
      return;
    }
    conversationId.value = saved;
    messages.value = rows.map((row) => ({
      id: `hist-${row.id}`,
      role: row.role === "assistant" ? "assistant" : "user",
      text: String(row.content ?? ""),
      streaming: false,
    }));
  } catch {
    localStorage.removeItem(STORAGE_CONV_ID);
  } finally {
    restoring.value = false;
    await nextTick();
    scrollToBottom();
  }
}

/** conversation_id 变化时同步到 localStorage，便于刷新恢复 */
watch(conversationId, (id) => {
  if (id) localStorage.setItem(STORAGE_CONV_ID, id);
  else localStorage.removeItem(STORAGE_CONV_ID);
});

onMounted(() => {
  restoreSessionFromServer();
});

/** 拆出「最终答案」段用于加粗展示 */
function splitAnswerBold(text) {
  if (!text) return { before: "", bold: "" };
  const markers = ["最终答案：", "最终答案:", "【最终答案】"];
  let cut = -1;
  for (const m of markers) {
    const i = text.indexOf(m);
    if (i >= 0 && (cut < 0 || i < cut)) cut = i;
  }
  if (cut < 0) return { before: text, bold: "" };
  return { before: text.slice(0, cut), bold: text.slice(cut) };
}

/**
 * 流式解析 SSE；delta/final 写入当前助手气泡
 */
async function consumeSse(res, assistantId) {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("响应不支持流式读取");
  const dec = new TextDecoder();
  let carry = "";

  const patchAssistant = (fn) => {
    const list = messages.value;
    const i = list.findIndex((m) => m.id === assistantId);
    if (i < 0) return;
    const next = { ...list[i] };
    fn(next);
    list.splice(i, 1, next);
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    carry += dec.decode(value, { stream: true });
    const frames = carry.split("\n\n");
    carry = frames.pop() ?? "";
    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const jsonStr = line.slice(5).trimStart();
        if (!jsonStr) continue;
        let msg;
        try {
          msg = JSON.parse(jsonStr);
        } catch {
          continue;
        }
        if (msg.t === "meta" && msg.d?.conversation_id) {
          conversationId.value = msg.d.conversation_id;
        } else if (msg.t === "delta" && msg.d) {
          patchAssistant((m) => {
            m.text += msg.d;
          });
          await nextTick();
          scrollToBottom();
        } else if (msg.t === "final") {
          const d = msg.d || {};
          if (d.conversation_id) conversationId.value = d.conversation_id;
          patchAssistant((m) => {
            if (d.result != null) m.text = String(d.result);
            m.streaming = false;
          });
        } else if (msg.t === "error") {
          throw new Error(String(msg.d || "服务端错误"));
        }
      }
    }
  }
  if (carry.trim()) {
    for (const line of carry.split("\n")) {
      if (!line.startsWith("data:")) continue;
      const jsonStr = line.slice(5).trimStart();
      if (!jsonStr) continue;
      try {
        const msg = JSON.parse(jsonStr);
        if (msg.t === "error") throw new Error(String(msg.d || "服务端错误"));
      } catch (e) {
        if (e instanceof SyntaxError) continue;
        throw e;
      }
    }
  }
}

async function send() {
  const q = question.value.trim();
  if (!q || loading.value) return;

  question.value = "";

  abortCtrl?.abort();
  abortCtrl = new AbortController();
  const signal = abortCtrl.signal;

  error.value = "";
  const assistantId = uid();
  messages.value.push({ id: uid(), role: "user", text: q });
  messages.value.push({ id: assistantId, role: "assistant", text: "", streaming: true });

  loading.value = true;
  try {
    const res = await fetch(streamUrl.value, {
      method: "POST",
      headers: streamHeaders(),
      body: JSON.stringify({
        question: q,
        conversation_id: conversationId.value || null,
      }),
      signal,
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }
    await consumeSse(res, assistantId);
  } catch (e) {
    if (e?.name === "AbortError") return;
    error.value = e?.message || String(e);
    const i = messages.value.findIndex((m) => m.id === assistantId);
    if (i >= 0) {
      const m = messages.value[i];
      messages.value.splice(i, 1, {
        ...m,
        text: m.text || `请求失败：${error.value}`,
        streaming: false,
        failed: true,
      });
    }
  } finally {
    const i = messages.value.findIndex((m) => m.id === assistantId);
    if (i >= 0 && messages.value[i].streaming) {
      const m = messages.value[i];
      messages.value.splice(i, 1, { ...m, streaming: false });
    }
    loading.value = false;
    await nextTick();
    scrollToBottom();
  }
}

function newConversation() {
  if (loading.value) return;
  messages.value = [];
  error.value = "";
  conversationId.value = null;
}

watch(
  () => messages.value.length,
  async () => {
    await nextTick();
    scrollToBottom();
  }
);
</script>

<template>
  <div class="page">
    <header class="header">
      <div class="header-row">
        <div>
          <h1 class="title">HR 制度问答</h1>
          <p class="subtitle">流式回答 · 配置 MySQL 后会话落库，刷新本页自动恢复最近对话</p>
        </div>
        <button type="button" class="btn-new" :disabled="loading" @click="newConversation">新对话</button>
      </div>
    </header>

    <div ref="chatScroll" class="chat-feed">
      <div v-if="restoring" class="placeholder">正在加载历史记录…</div>
      <div v-else-if="messages.length === 0 && !loading && !error" class="placeholder">
        在底部输入问题，例如：年假天数、试用期规定等
      </div>
      <div v-if="error && messages.length === 0" class="err">{{ error }}</div>

      <div
        v-for="m in messages"
        :key="m.id"
        class="msg-row"
        :class="m.role === 'user' ? 'msg-row--user' : 'msg-row--bot'"
      >
        <div class="bubble" :class="{ 'bubble--user': m.role === 'user', 'bubble--err': m.failed }">
          <template v-if="m.role === 'user'">{{ m.text }}</template>
          <template v-else>
            <template v-if="!m.text && m.streaming">
              <span class="thinking">正在检索与生成…</span>
            </template>
            <article v-else class="answer">
              <span class="answer-reasoning">{{ splitAnswerBold(m.text).before }}</span
              ><strong v-if="splitAnswerBold(m.text).bold" class="answer-final">{{
                splitAnswerBold(m.text).bold
              }}</strong>
            </article>
            <div v-if="m.streaming && m.text" class="streaming-hint">生成中…</div>
          </template>
        </div>
      </div>
    </div>

    <div class="input-dock">
      <div class="input-shell">
        <textarea
          v-model="question"
          class="field"
          rows="1"
          placeholder="输入问题，Enter 发送，Shift+Enter 换行"
          :disabled="loading"
          @keydown.enter.exact.prevent="send"
        />
        <button type="button" class="send" :disabled="loading || !question.trim()" @click="send">
          发送
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page {
  height: 100%;
  display: flex;
  flex-direction: column;
  max-width: 880px;
  margin: 0 auto;
  position: relative;
  padding: 0 12px;
}

.header {
  flex-shrink: 0;
  padding: 16px 8px 12px;
  background: linear-gradient(180deg, #eef0f4 70%, transparent);
  z-index: 2;
}

.title {
  margin: 0;
  font-size: 1.35rem;
  font-weight: 600;
}

.header-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.subtitle {
  margin: 6px 0 0;
  font-size: 0.8rem;
  color: #666;
}

.btn-new {
  flex-shrink: 0;
  margin-top: 4px;
  padding: 8px 14px;
  font-size: 0.85rem;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  background: #fff;
  cursor: pointer;
}

.btn-new:hover:not(:disabled) {
  border-color: #1677ff;
  color: #1677ff;
}

.btn-new:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.chat-feed {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 8px 8px 140px;
  scroll-behavior: smooth;
}

.placeholder {
  color: #888;
  font-size: 0.95rem;
  padding: 24px 12px;
  text-align: center;
}

.err {
  color: #c62828;
  font-size: 0.9rem;
  white-space: pre-wrap;
  padding: 12px;
}

.msg-row {
  display: flex;
  margin-bottom: 16px;
}

.msg-row--user {
  justify-content: flex-end;
}

.msg-row--bot {
  justify-content: flex-start;
}

.bubble {
  max-width: min(100%, 720px);
  padding: 14px 16px;
  border-radius: 16px;
  font-size: 0.95rem;
  line-height: 1.6;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.bubble--user {
  background: #1677ff;
  color: #fff;
  border-bottom-right-radius: 4px;
  white-space: pre-wrap;
}

.bubble--bot {
  background: #fff;
  color: #1a1a1a;
  border: 1px solid #e8eaed;
  border-bottom-left-radius: 4px;
}

.bubble--err {
  border-color: #ffcdd2;
  background: #fff8f8;
}

.thinking {
  color: #888;
  font-size: 0.9rem;
}

.streaming-hint {
  margin-top: 8px;
  font-size: 0.75rem;
  color: #1677ff;
}

.answer {
  white-space: pre-wrap;
  line-height: 1.65;
  font-size: 0.95rem;
}

.answer-final {
  font-weight: 700;
  color: #0d0d0d;
}

.input-dock {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 10;
  padding: 12px 16px calc(12px + env(safe-area-inset-bottom, 0));
  background: linear-gradient(180deg, transparent, rgba(238, 240, 244, 0.92) 18%, #eef0f4 40%);
  display: flex;
  justify-content: center;
  pointer-events: none;
}

.input-dock .input-shell {
  pointer-events: auto;
  width: 100%;
  max-width: 840px;
}

.input-shell {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  padding: 12px 16px;
  background: #fff;
  border-radius: 24px;
  border: 1px solid #e0e4ea;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
}

.field {
  flex: 1;
  border: none;
  outline: none;
  resize: none;
  font: inherit;
  font-size: 0.95rem;
  line-height: 1.5;
  max-height: 160px;
  min-height: 24px;
  padding: 8px 4px;
  background: transparent;
}

.field::placeholder {
  color: #9e9e9e;
}

.send {
  flex-shrink: 0;
  min-width: 72px;
  height: 44px;
  padding: 0 16px;
  border: none;
  border-radius: 22px;
  background: #1677ff;
  color: #fff;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  transition: background 0.15s, opacity 0.15s;
}

.send:hover:not(:disabled) {
  background: #4096ff;
}

.send:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
</style>

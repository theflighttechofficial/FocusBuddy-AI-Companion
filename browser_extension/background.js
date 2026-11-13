// FocusBuddy Connector - background script (Chrome/Edge service worker, Firefox event page).
//
// Privacy: the only data sent is the ACTIVE tab's URL + title, and only to the
// FocusBuddy app on this computer (http://127.0.0.1). Nothing goes to the internet.
// Incognito/private tabs are reported without their URL unless the user opts in.

const api = globalThis.browser ?? globalThis.chrome;
const DEFAULT_PORT = 47621;
const POLL_WAIT_SECONDS = 20;

const state = {
  port: DEFAULT_PORT,
  clientId: null,
  reportPrivate: false,
  connected: false,
  lastContact: 0,
  polling: false,
  ready: null,
  // Focus policy pushed by the app: {active, until (ms), block: [hosts], allow: [hosts], session}.
  // Expires on its own if the app stops answering, so a crashed app can never keep sites blocked.
  policy: { active: false },
  blockedCount: 0,
};

function browserName() {
  const ua = navigator.userAgent || "";
  if (ua.includes("Edg/")) return "edge";
  if (ua.includes("Firefox/")) return "firefox";
  if (ua.includes("OPR/")) return "opera";
  if (navigator.brave) return "brave";
  return "chrome";
}

function hostMatches(host, pattern) {
  const [h] = String(pattern || "").toLowerCase().split("/");
  return Boolean(h) && (host === h || host.endsWith("." + h));
}

function policyActive() {
  const p = state.policy;
  return Boolean(p && p.active && Number(p.until || 0) > Date.now());
}

// Returns the matched pattern if `url` must be blocked right now under the focus policy.
function blockedBy(url) {
  if (!policyActive() || !/^https?:/i.test(url || "")) return null;
  const host = hostOf(url);
  if (!host) return null;
  const path = (() => { try { return new URL(url).pathname.replace(/^\//, ""); } catch (e) { return ""; } })();
  const allow = state.policy.allow || [];
  if (allow.some((a) => hostMatches(host, a))) return null;
  for (const pat of state.policy.block || []) {
    const [h, ...rest] = String(pat).toLowerCase().split("/");
    const p = rest.join("/");
    if (hostMatches(host, h) && (!p || path.startsWith(p))) return pat;
  }
  return null;
}

async function enforce(tabId, url) {
  const pat = blockedBy(url);
  if (!pat) return false;
  const host = hostOf(url);
  const page = api.runtime.getURL(`focus.html?from=${encodeURIComponent(host)}&session=1`);
  try {
    await api.tabs.update(tabId, { url: page });
    state.blockedCount += 1;
    post("/event", { client: state.clientId, type: "blocked", host }, 3000)
      .then((r) => applyPolicy(r && r.policy)).catch(() => {});
    return true;
  } catch (e) {
    return false;
  }
}

function applyPolicy(policy) {
  if (!policy || typeof policy !== "object") return;
  const was = policyActive();
  state.policy = policy;
  api.storage.session?.set?.({ policy })?.catch?.(() => {});
  if (!was && policyActive()) {
    // a focus session just started: sweep the tabs that are already open in the active windows
    api.tabs.query({ active: true }).then((tabs) => {
      for (const t of tabs) enforce(t.id, t.url || t.pendingUrl || "");
    }).catch(() => {});
  }
}

async function init() {
  try {
    const sp = await api.storage.session?.get?.(["policy"]);
    if (sp && sp.policy) state.policy = sp.policy;
  } catch (e) { /* storage.session missing on older browsers */ }
  const s = await api.storage.local.get(["port", "clientId", "reportPrivate"]);
  state.port = Number(s.port) || DEFAULT_PORT;
  state.reportPrivate = Boolean(s.reportPrivate);
  state.clientId = s.clientId || (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2));
  if (!s.clientId) await api.storage.local.set({ clientId: state.clientId });
}
state.ready = init();

async function snapshot() {
  let tab;
  try {
    [tab] = await api.tabs.query({ active: true, lastFocusedWindow: true });
  } catch (e) {
    return null;
  }
  if (!tab) return null;
  let focused = true;
  try {
    const win = await api.windows.getLastFocused();
    focused = Boolean(win && win.focused);
  } catch (e) { /* windows API may be unavailable */ }
  const hidePrivate = tab.incognito && !state.reportPrivate;
  return {
    url: hidePrivate ? null : (tab.url || tab.pendingUrl || null),
    title: hidePrivate ? "" : (tab.title || ""),
    tabId: tab.id,
    windowId: tab.windowId,
    incognito: Boolean(tab.incognito),
    focused,
  };
}

async function post(path, body, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`http://127.0.0.1:${state.port}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-FocusBuddy": "1" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.connected = true;
    state.lastContact = Date.now();
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

function hostOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch (e) {
    return "";
  }
}

async function ack(id, ok, error) {
  try {
    await post("/ack", { client: state.clientId, id, ok, error: error || "" }, 3000);
  } catch (e) { /* app went away */ }
}

async function handleCommands(commands) {
  for (const cmd of commands || []) {
    if (cmd.type === "policy") continue;          // wake-up only: the new policy came with the response
    try {
      const tab = await api.tabs.get(cmd.tabId);
      // Safety: only act if the tab still shows the site the app decided about.
      if (cmd.expectHost && hostOf(tab.url || "") !== cmd.expectHost &&
          !hostOf(tab.url || "").endsWith("." + cmd.expectHost)) {
        await ack(cmd.id, false, "tab changed");
        continue;
      }
      if (cmd.type === "close_tab") {
        await api.tabs.remove(cmd.tabId);
        await ack(cmd.id, true);
      } else if (cmd.type === "redirect_tab") {
        const page = api.runtime.getURL(`focus.html?from=${encodeURIComponent(cmd.expectHost || "")}`);
        await api.tabs.update(cmd.tabId, { url: page });
        await ack(cmd.id, true);
      } else {
        await ack(cmd.id, false, "unknown command");
      }
    } catch (e) {
      await ack(cmd.id, false, String(e && e.message ? e.message : e));
    }
  }
}

async function report() {
  await state.ready;
  try {
    const active = await snapshot();
    const res = await post("/report", { client: state.clientId, browser: browserName(), active }, 3000);
    applyPolicy(res.policy);
    await handleCommands(res.commands);
  } catch (e) {
    state.connected = false;
  }
}

async function pollLoop() {
  await state.ready;
  if (state.polling) return;
  state.polling = true;
  try {
    // Each iteration calls extension APIs (tabs.query), which keeps the MV3
    // service worker alive while the desktop app is running.
    for (;;) {
      try {
        const active = await snapshot();
        const res = await post("/poll", {
          client: state.clientId, browser: browserName(), active, wait: POLL_WAIT_SECONDS,
          policySig: (state.policy && state.policy.sig) || null,
        }, (POLL_WAIT_SECONDS + 5) * 1000);
        applyPolicy(res.policy);
        await handleCommands(res.commands);
      } catch (e) {
        state.connected = false;
        await new Promise((r) => setTimeout(r, 5000));
      }
    }
  } finally {
    state.polling = false;
  }
}

// Report immediately whenever the active tab / window / URL changes.
// During a focus session with instant enforcement, distracting sites are swapped for the focus page
// the moment they start loading - in any tab, before the desktop app even sees them.
api.tabs.onActivated.addListener(async ({ tabId }) => {
  try {
    const tab = await api.tabs.get(tabId);
    if (await enforce(tabId, tab.url || tab.pendingUrl || "")) return;
  } catch (e) { /* tab vanished */ }
  report();
});
api.tabs.onUpdated.addListener(async (id, info, tab) => {
  if (info.url && await enforce(id, info.url)) return;
  if (tab.active && (info.url || info.title || info.status === "complete")) report();
});
api.windows.onFocusChanged.addListener(() => report());

// Keep-alive / revive: alarms wake the worker if the browser suspended it.
api.alarms.create("focusbuddy-keepalive", { periodInMinutes: 0.5 });
api.alarms.onAlarm.addListener(() => { report(); pollLoop(); });
api.runtime.onStartup.addListener(() => pollLoop());
api.runtime.onInstalled.addListener(() => pollLoop());

api.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === "status") {
    sendResponse({ connected: state.connected, lastContact: state.lastContact, port: state.port,
                   browser: browserName(), reportPrivate: state.reportPrivate,
                   focus: policyActive(), blocked: state.blockedCount });
    return false;
  }
  if (msg && msg.type === "settings") {
    state.port = Number(msg.port) || DEFAULT_PORT;
    state.reportPrivate = Boolean(msg.reportPrivate);
    api.storage.local.set({ port: state.port, reportPrivate: state.reportPrivate });
    report();
    sendResponse({ ok: true });
    return false;
  }
  return false;
});

pollLoop();

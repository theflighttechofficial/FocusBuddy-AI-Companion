const api = globalThis.browser ?? globalThis.chrome;

async function ping(port) {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/ping`, { headers: { "X-FocusBuddy": "1" } });
    return res.ok;
  } catch (e) {
    return false;
  }
}

async function refresh() {
  const st = await api.runtime.sendMessage({ type: "status" });
  document.getElementById("port").value = st.port;
  document.getElementById("private").checked = st.reportPrivate;
  const ok = await ping(st.port);
  const box = document.getElementById("status");
  box.className = "status " + (ok ? "on" : "off");
  document.getElementById("statusText").textContent = ok
    ? `Connected to FocusBuddy (${st.browser})`
    : "FocusBuddy app is not running";
}

document.getElementById("save").addEventListener("click", async () => {
  await api.runtime.sendMessage({
    type: "settings",
    port: Number(document.getElementById("port").value),
    reportPrivate: document.getElementById("private").checked,
  });
  refresh();
});

refresh();

const q = new URLSearchParams(location.search);
const from = q.get("from");
if (from) for (const el of document.querySelectorAll(".site")) el.textContent = from;
if (q.get("session")) {
  document.getElementById("title").textContent = "Not during focus time";
  document.getElementById("lead").innerHTML =
    'FocusBuddy kept <code class="site"></code> away while your focus session runs. It will be back after the session.';
  if (from) document.querySelector("#lead .site").textContent = from;
}

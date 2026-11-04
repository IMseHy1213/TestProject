// v0.9 실시간 알림: 서버가 보내는 이벤트(SSE)를 받아 손님에게는 맞춤 알림을, 관리자 화면에는 이벤트 흐름을 보여줌.
// 연결이 끊기면 브라우저가 자동으로 다시 연결하고, 마지막으로 받은 번호(Last-Event-ID) 이후 것만 이어 받음.
(function () {
  var me = document.currentScript;
  if (!window.EventSource || !me) return;
  var KEY = "rea_last_event", since = "";
  try { since = localStorage.getItem(KEY) || ""; } catch (e) {}
  var url = me.dataset.stream + (since ? "?since=" + encodeURIComponent(since) : "");
  var es = new EventSource(url);
  var isAdmin = me.dataset.admin === "1";
  var feed = document.getElementById("event-feed");
  var box = document.getElementById("toasts");

  function remember(id) { try { if (id) localStorage.setItem(KEY, id); } catch (e) {} }

  function el(tag, cls, text) {
    var x = document.createElement(tag);
    if (cls) x.className = cls;
    if (text) x.textContent = text;
    return x;
  }

  function toast(p) {
    if (!box) return;
    var t = el("div", "toast");
    t.setAttribute("role", "status");
    t.appendChild(el("strong", "", "🔔 " + (p.kind === "가격 인하" ? "가격이 내려간 매물" : "새 매물 알림")));
    t.appendChild(el("p", "", p.message));
    if (p.listing && p.listing.features && p.listing.features.length) {
      t.appendChild(el("p", "toast-feat", p.listing.features.slice(0, 2).join(" · ")));
    }
    var row = el("div", "toast-row");
    var go = el("a", "btn small primary", "AI 상담에서 보기");
    go.href = me.dataset.agent + "?ask=" + encodeURIComponent(p.listing.name + " 매물 자세히 알려줘");
    var close = el("button", "btn small", "닫기");
    close.type = "button";
    close.addEventListener("click", function () { t.remove(); });
    row.appendChild(go); row.appendChild(close);
    t.appendChild(row);
    box.prepend(t);
    while (box.children.length > 3) box.lastChild.remove();
  }

  function feedLine(topic, ev) {
    if (!feed) return;
    var empty = feed.querySelector(".empty");
    if (empty) empty.remove();
    var p = ev.payload || {};
    var text = p.message || (p.listing ? p.listing.name + " · " + p.listing.price : "") ||
      p.listing_id || (p.listings !== undefined ? p.listings + "건" : "");
    var li = el("li", "ev ev-" + topic.replace(".", "-") + " ev-new");
    li.appendChild(el("span", "ev-id", "#" + ev.id));
    li.appendChild(document.createTextNode(" "));
    li.appendChild(el("code", "", topic));
    li.appendChild(document.createTextNode(" " + text + " "));
    li.appendChild(el("small", "muted", (ev.created || "").slice(11, 19) + (p.source ? " · " + p.source : "")));
    feed.prepend(li);
    while (feed.children.length > 30) feed.lastChild.remove();
  }

  ["listing.added", "listing.price_drop", "listing.changed", "listing.removed", "system.sync", "alert.match"]
    .forEach(function (topic) {
      es.addEventListener(topic, function (m) {
        remember(m.lastEventId);
        var ev = JSON.parse(m.data);
        feedLine(topic, ev);
        if (topic === "alert.match" && !isAdmin) toast(ev.payload);
      });
    });
})();

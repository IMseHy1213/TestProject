// 예시 문장 버튼 -> 검색창 채우기
document.querySelectorAll("[data-fill]").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var q = document.getElementById("q");
    if (!q) return;
    q.value = btn.getAttribute("data-fill");
    q.focus();
  });
});

// 추천 설명은 화면을 먼저 보여준 뒤 불러옴 (AI 설명은 몇 초 걸릴 수 있음)
var explain = document.getElementById("explain");
if (explain) {
  fetch(explain.getAttribute("data-url"), { credentials: "same-origin" })
    .then(function (r) { return r.ok ? r.json() : { text: "" }; })
    .then(function (d) {
      if (d.text) { explain.textContent = "💬 " + d.text; }
      else { explain.remove(); }
    })
    .catch(function () { explain.remove(); });
}

// 폴더째 올리기: 브라우저가 파일 이름만 보내는 경우를 대비해 '폴더/하위/파일' 경로를 함께 보냄
document.querySelectorAll("form.upload-form").forEach(function (f) {
  f.addEventListener("submit", function (ev) {
    var folder = f.querySelector('input[name="folder"]');
    var files = f.querySelector('input[name="files"]');
    var count = (folder ? folder.files.length : 0) + (files ? files.files.length : 0);
    if (!count) {
      ev.preventDefault();
      alert("올릴 엑셀 파일이나 폴더를 먼저 골라 주세요.");
      return;
    }
    f.querySelectorAll('input[name="folder_paths"]').forEach(function (x) { x.remove(); });
    if (folder) {
      Array.prototype.forEach.call(folder.files, function (file) {
        var h = document.createElement("input");
        h.type = "hidden"; h.name = "folder_paths"; h.value = file.webkitRelativePath || file.name;
        f.appendChild(h);
      });
    }
  });
});

// AI 상담: 답을 만드는 과정을 실시간으로 (Server-Sent Events 를 fetch 로 읽음). 실패하면 보통 방식으로 전송.
var STEP_LABEL = { memory: "🧠 기억", fallback: "⚙️ 전환", guard: "🛡 안전장치", resume: "↩ 이어서", answer: "💬 답변", remote: "🌐 A2A" };
document.querySelectorAll("form.js-stream").forEach(function (f) {
  f.addEventListener("submit", function (ev) {
    if (f.dataset.plain === "1" || !window.fetch || !window.TextDecoder) return;
    ev.preventDefault();
    var data = new FormData(f);
    if (f.dataset.resume === "1") data.append("resume", "1");
    var live = document.getElementById("live");
    var steps = document.getElementById("live-steps");
    var status = document.getElementById("live-status");
    var q = data.get("q") || "(이어서 답하는 중)";
    document.getElementById("live-q").textContent = q;
    steps.innerHTML = "";
    live.hidden = false;
    live.scrollIntoView({ block: "end" });
    f.querySelectorAll("button").forEach(function (b) { b.disabled = true; });
    var running = {};  // 실행 중인 도구 (멀티 Agent 는 총괄의 '맡기기' 안에서 다른 Agent 의 도구가 돎)

    function show(ev) {
      if (ev.type === "thinking") { status.textContent = "AI가 다음 행동을 고르는 중… (" + ev.n + "단계)"; }
      if (ev.type === "tool_start") {
        var r = document.createElement("li");
        r.className = "running";
        r.textContent = (ev.agent ? "[" + ev.agent + "] " : "") + "🔧 " + ev.tool + " 실행 중";
        running[(ev.agent || "") + ev.tool] = r;
        steps.appendChild(r);
      }
      if (ev.type === "step") {
        var s = ev.step, key = (s.agent || "") + s.tool;
        var li = (s.kind === "tool" && running[key]) ? running[key] : document.createElement("li");
        li.className = s.ok === false ? "is-error" : "";
        li.textContent = (s.agent ? "[" + s.agent + "] " : "") +
          (s.kind === "tool" ? "🔧 " + s.tool + " → " : (STEP_LABEL[s.kind] || s.kind) + " · ") + s.summary +
          (s.ms ? " (" + s.ms + "ms)" : "");
        if (!li.parentNode) steps.appendChild(li);
        delete running[key];
      }
      if (ev.type === "done" || ev.type === "saved") { status.textContent = "답변 정리 완료 — 화면을 새로 그립니다…"; }
      if (ev.type === "error") { status.textContent = ev.message; status.classList.remove("muted"); }
    }

    fetch(f.dataset.stream, { method: "POST", body: data, credentials: "same-origin",
                              headers: { "X-CSRF-Token": data.get("csrf") || "" } })
      .then(function (r) {
        if (!r.ok || !r.body) throw new Error("stream");
        var reader = r.body.getReader(), dec = new TextDecoder(), buf = "", failed = false;
        function pump() {
          return reader.read().then(function (res) {
            if (res.done) {
              if (!failed) {  // 같은 주소에 #last 만 붙이면 새로 그려지지 않으므로 주소를 바꿔 다시 불러옴
                var parts = f.dataset.done.split("#");
                window.location.href = parts[0] + (parts[0].indexOf("?") < 0 ? "?" : "&") + "t=" + Date.now() +
                  (parts[1] ? "#" + parts[1] : "");
              }
              else f.querySelectorAll("button").forEach(function (b) { b.disabled = false; });
              return;
            }
            buf += dec.decode(res.value, { stream: true });
            var parts = buf.split("\n\n");
            buf = parts.pop();
            parts.forEach(function (chunk) {
              var line = chunk.split("\n").filter(function (l) { return l.indexOf("data: ") === 0; })[0];
              if (!line) return;
              var ev = JSON.parse(line.slice(6));
              if (ev.type === "error") failed = true;
              show(ev);
            });
            return pump();
          });
        }
        return pump();
      })
      .catch(function () { f.dataset.plain = "1"; f.submit(); });
  });
});

// 버튼 두 번 눌림 방지
document.querySelectorAll("form").forEach(function (f) {
  f.addEventListener("submit", function (ev) {
    if (ev.defaultPrevented) return;
    f.querySelectorAll("button").forEach(function (b) { b.disabled = true; });
  });
});

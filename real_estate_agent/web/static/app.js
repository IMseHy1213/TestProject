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

// 버튼 두 번 눌림 방지
document.querySelectorAll("form").forEach(function (f) {
  f.addEventListener("submit", function () {
    f.querySelectorAll("button").forEach(function (b) { b.disabled = true; });
  });
});

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

// 버튼 두 번 눌림 방지
document.querySelectorAll("form").forEach(function (f) {
  f.addEventListener("submit", function (ev) {
    if (ev.defaultPrevented) return;
    f.querySelectorAll("button").forEach(function (b) { b.disabled = true; });
  });
});

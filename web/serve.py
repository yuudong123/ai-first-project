"""
web/serve.py
=============
Unity WebGL 빌드가 gzip 압축(.gz)된 상태로 나올 때, 로컬 테스트용 정적 서버.

파이썬 기본 `python -m http.server`는 .gz 파일을 서빙할 때
"Content-Encoding: gzip" 헤더를 안 붙여준다. 그러면 브라우저는
"이 파일은 원래 압축된 형태구나"라고 판단해서 압축을 안 풀고 그대로
JS에 넘기고, Unity 로더는 압축 안 풀린 바이너리를 못 읽어서 실패한다.

이 스크립트는 .gz로 끝나는 파일에 그 헤더를 자동으로 붙여줘서,
브라우저가 알아서 압축을 풀고 Unity에 정상적인 데이터를 넘기게 한다.

실행 (web 폴더 안에서):
    python serve.py          (기본 포트 5500)
    python serve.py 8080     (포트 직접 지정)

그다음 브라우저에서 http://localhost:5500/index.html 접속.
"""

import sys
import http.server
import socketserver

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5500

# .gz를 벗겨낸 뒤의 "진짜" 파일 타입 (Unity 로더가 요구하는 정확한 MIME)
REAL_MIME_TYPES = {
    ".wasm": "application/wasm",
    ".data": "application/octet-stream",
    ".js": "application/javascript",
}


class GzipAwareHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        if self.path.endswith(".gz"):
            self.send_header("Content-Encoding", "gzip")
        super().end_headers()

    def guess_type(self, path):
        # .gz로 끝나면 그 앞의 진짜 확장자를 기준으로 MIME을 판단
        # (Content-Type 헤더는 여기서 한 번만 결정 -> 중복 헤더 방지)
        check_path = path[:-3] if path.endswith(".gz") else path
        for ext, mime in REAL_MIME_TYPES.items():
            if check_path.endswith(ext):
                return mime
        return super().guess_type(path)

    def log_message(self, format, *args):
        # 매 요청 로그가 너무 시끄러우니 상태코드만 간단히 출력
        print(f"[{self.address_string()}] {format % args}")


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    with ReusableTCPServer(("", PORT), GzipAwareHandler) as httpd:
        print(f"http://localhost:{PORT} 에서 서빙 중 (Ctrl+C로 종료)")
        print(f"대시보드: http://localhost:{PORT}/index.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n서버 종료")
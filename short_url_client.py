"""short_link 内部 HTTP API 的轻量客户端。

对外封装 ``ShortLinkClient``，便于在 nexus 内部直接调用工程侧 short_link 服务，
并提供命令行入口用于调试（healthz / shorten / resolve / roundtrip）。

服务发现策略（与 ``nexus.extensions.openai_clients.taiji`` 保持一致）：

1. 显式 ``base_url`` / ``host:port`` 优先（旁路 Polaris，便于本地联调）。
2. 否则通过 ``polaris-cpp-py`` SDK 的 ``GetOneInstanceRequest`` 发现一个
   健康实例。每次调用都触发一次发现，以容忍实例上下线。

用法::

    from nexus.extensions.agents.yb_one_agent.short_url_client import ShortLinkClient

    client = ShortLinkClient()
    short = client.shorten('https://example.com/very/long/url')
    long_url = client.resolve(short['short_code'])

命令行::

    python -m nexus.utils.short_url_client healthz
    python -m nexus.utils.short_url_client shorten --url https://example.com/page
    python -m nexus.utils.short_url_client resolve --code Ab3xZ9
    python -m nexus.utils.short_url_client roundtrip --url https://example.com/page
    python -m nexus.utils.short_url_client --base-url http://127.0.0.1:8005 healthz
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from typing import Any


logger = logging.getLogger(__name__)


# ── 默认值 ──────────────────────────────────────────────────────────────────

DEFAULT_PORT = 8005
DEFAULT_TIMEOUT = 5.0
DEFAULT_NAMESPACE = 'Test'
DEFAULT_SERVICE = 'trpc.yuanbao.short_link.http'
DEFAULT_ENV = 'ci-575'
DEFAULT_DISCOVER_PORT = 8005


# ── 服务发现 ────────────────────────────────────────────────────────────────


def _discover_service(namespace: str, service: str, *, env: str | None = None) -> tuple[str, int]:
    """通过 Polaris 发现一个 short_link 实例。

    与 ``nexus.extensions.openai_clients.taiji._discover_service`` 行为一致：
    每次调用都触发一次发现，不做缓存，便于跟随上下线变化。

    Args:
        namespace: Polaris 命名空间。
        service: Polaris 服务名。
        env: 可选的 env metadata，用于路由到特定环境（如 ``ci-575``）。

    Returns:
        ``(host, port)`` 元组。
    """
    try:
        from polaris.api.consumer import create_consumer_by_config
        from polaris.pkg.model.service import GetOneInstanceRequest
    except ImportError as exc:
        raise ImportError(
            'polaris-cpp-py is required for short_link service discovery. '
            'Install it with: uv pip install /path/to/polaris-cpp-py-*.tar.gz, '
            'or pass base_url/host explicitly to bypass Polaris.'
        ) from exc

    try:
        consumer_api = create_consumer_by_config('')
        req = GetOneInstanceRequest(namespace=namespace, service=service)
        if env:
            req.add_service_metadata('env', env)
        response = consumer_api.get_one_instance_resp(req)
        if response.is_service_data_expired():
            logger.warning('Polaris service %s/%s data is expired', namespace, service)
        instance = response[0]
        host = instance.get_host()
        port = instance.get_port()
        logger.debug('discovered short_link instance at %s:%s (env=%s)', host, port, env)
        return host, port
    except Exception as exc:
        logger.error('failed to discover short_link via Polaris: %s', exc)
        raise RuntimeError(f'Polaris service discovery failed: {exc}') from exc


# ── HTTP 工具 ──────────────────────────────────────────────────────────────


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """发起一次 JSON 请求并解析响应。"""
    data: bytes | None = None
    headers: dict[str, str] = {'Accept': 'application/json'}
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'HTTP {exc.code}: {body}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'request failed: {exc.reason}') from exc

    if not body:
        return {}
    return json.loads(body)


# ── 客户端 ──────────────────────────────────────────────────────────────────


class ShortLinkClient:
    """short_link 内部 HTTP API 客户端。

    Args:
        base_url: 直连模式，例如 ``http://127.0.0.1:8005``。设置后跳过 Polaris。
        host/port: 直连模式的另一种写法。``base_url`` 优先级更高。
        namespace/service/env: Polaris 命名空间、服务名与环境标识。
        timeout: 单次请求超时秒数。
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        host: str | None = None,
        port: int = DEFAULT_PORT,
        namespace: str = DEFAULT_NAMESPACE,
        service: str = DEFAULT_SERVICE,
        env: str | None = DEFAULT_ENV,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip('/') if base_url else None
        self._host = host
        self._port = port
        self._namespace = namespace
        self._service = service
        self._env = env
        self._timeout = timeout

    # ── 内部 ────────────────────────────────────────────────────────────

    def _resolve_base_url(self) -> str:
        """每次调用都解析一次 base URL，让 Polaris 实例变化能即时生效。"""
        if self._base_url:
            return self._base_url
        if self._host:
            return f'http://{self._host}:{self._port}'
        host, _discovered_port = _discover_service(self._namespace, self._service, env=self._env)
        return f'http://{host}:{self._port}'

    # ── 公共 API ───────────────────────────────────────────────────────

    def healthz(self) -> tuple[int, str]:
        """请求 ``GET /healthz``，返回 ``(status_code, body)``。"""
        url = self._resolve_base_url() + '/healthz'
        req = urllib.request.Request(url, method='GET')
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.status, resp.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode('utf-8', errors='replace')
        except urllib.error.URLError as exc:
            raise RuntimeError(f'healthz request failed: {exc.reason}') from exc

    def shorten(self, long_url: str) -> dict[str, Any]:
        """请求 ``POST /api/shorten``，返回服务端 JSON。"""
        url = self._resolve_base_url() + '/api/shorten'
        return _request_json('POST', url, payload={'long_url': long_url}, timeout=self._timeout)

    def resolve(self, code: str) -> dict[str, Any]:
        """请求 ``GET /api/resolve?code=<code>``，返回服务端 JSON。"""
        query = urllib.parse.urlencode({'code': code})
        url = self._resolve_base_url() + '/api/resolve?' + query
        return _request_json('GET', url, timeout=self._timeout)


# ── CLI ────────────────────────────────────────────────────────────────────


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _build_client(args: argparse.Namespace) -> ShortLinkClient:
    return ShortLinkClient(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        namespace=args.namespace,
        service=args.service,
        env=args.env,
        timeout=args.timeout,
    )


def _run(args: argparse.Namespace) -> int:
    client = _build_client(args)
    try:
        if args.command == 'healthz':
            status, body = client.healthz()
            print(f'status={status} body={body!r}')
            return 0 if status == 200 else 1
        if args.command == 'shorten':
            _print_json(client.shorten(args.url))
            return 0
        if args.command == 'resolve':
            _print_json(client.resolve(args.code))
            return 0
        if args.command == 'roundtrip':
            shortened = client.shorten(args.url)
            print('Shorten:')
            _print_json(shortened)
            code = shortened.get('short_code')
            if not code:
                print('error: missing short_code in shorten response', file=sys.stderr)
                return 1
            print('Resolve:')
            _print_json(client.resolve(code))
            return 0
    except (KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    print(f'unknown command: {args.command}', file=sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='short_link internal HTTP API client')
    parser.add_argument('--base-url', help='bypass Polaris, e.g. http://127.0.0.1:8005')
    parser.add_argument('--host', help='bypass Polaris with a direct service host')
    parser.add_argument(
        '--port',
        type=int,
        default=DEFAULT_PORT,
        help=f'direct service port used with --host (default: {DEFAULT_PORT})',
    )
    parser.add_argument(
        '--namespace',
        default=os.environ.get('NAMESPACE', DEFAULT_NAMESPACE),
        help=f'Polaris namespace (default: {DEFAULT_NAMESPACE}, env NAMESPACE)',
    )
    parser.add_argument(
        '--service',
        default=DEFAULT_SERVICE,
        help=f'Polaris service name (default: {DEFAULT_SERVICE})',
    )
    parser.add_argument(
        '--env',
        default=os.environ.get('SHORT_LINK_ENV', DEFAULT_ENV),
        help=f'Polaris env metadata for routing (default: {DEFAULT_ENV}, env SHORT_LINK_ENV)',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f'request timeout seconds (default: {DEFAULT_TIMEOUT})',
    )

    sub = parser.add_subparsers(dest='command')
    sub.add_parser('healthz', help='GET /healthz')

    p_shorten = sub.add_parser('shorten', help='POST /api/shorten')
    p_shorten.add_argument('--url', required=True, help='long URL to shorten')

    p_resolve = sub.add_parser('resolve', help='GET /api/resolve?code=...')
    p_resolve.add_argument('--code', required=True, help='short code to resolve')

    p_roundtrip = sub.add_parser('roundtrip', help='shorten then resolve the generated code')
    p_roundtrip.add_argument('--url', required=True, help='long URL to shorten and resolve')
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    return _run(args)


if __name__ == '__main__':
    raise SystemExit(main())

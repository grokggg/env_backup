"""Persistence implementation for the ``wiki/ingest_entry`` runtime tool.

``ingest_one`` writes one structured manifest into the user's personal library:

- ``raw/<id>/``: original file copy (upload / artifact kind only — url kind has
  no local copy, the URL itself *is* the raw source)
- ``wiki/entries/<id>.md``: structured frontmatter plus the rendered body
- ``wiki/index.json``: lightweight retrieval index
- ``wiki/log.md``: append-only ingest timeline

The caller provides structured ``keyInsights`` and a fully rendered ``body``.
This module persists both without doing any presentation-layer rendering.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any


LIB = '/data/library'
CST = timezone(timedelta(hours=8))

REQUIRED = ['title', 'type', 'status', 'source_kind', 'category', 'topicTags', 'summary', 'body']
DIFFS = ('added', 'repeated', 'contradicted')


@contextmanager
def _owner_lock() -> Iterator[None]:
    """Serialize entry ids and Backend revisions within one user's library."""
    wiki_dir = os.path.join(LIB, 'wiki')
    os.makedirs(wiki_dir, exist_ok=True)
    with open(os.path.join(wiki_dir, '.ingest.lock'), 'a+b') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_index() -> list[dict[str, Any]]:
    path = os.path.join(LIB, 'wiki', 'index.json')
    if not os.path.exists(path):
        return []
    with open(path) as file:
        try:
            index = json.load(file)
        except json.JSONDecodeError:
            return []
    return index if isinstance(index, list) else []


def _next_revision(index: list[dict[str, Any]]) -> int:
    state_path = os.path.join(LIB, 'wiki', '.state.json')
    if os.path.exists(state_path):
        with open(state_path) as file:
            state = json.load(file)
        revision = state.get('next_revision')
        if not isinstance(revision, int) or revision < 1:
            raise ValueError('.state.json.next_revision 必须是正整数')
        return revision
    return len(index) + 1


def _advance_revision(revision: int) -> None:
    path = os.path.join(LIB, 'wiki', '.state.json')
    temporary = f'{path}.{os.getpid()}.tmp'
    with open(temporary, 'w') as file:
        json.dump({'next_revision': revision + 1}, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _yaml_quote(value: str) -> str:
    if not value:
        return '""'
    if any(char in value for char in ':{}[],"\'|>&*!%#`@?') or value.startswith('-'):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return value


def _next_id(now: datetime) -> str:
    """Sequence entry ids per day by scanning ``wiki/entries/`` — every entry has
    one of these regardless of ``source_kind``, unlike ``raw/`` which url-kind
    entries no longer populate.
    """
    day = now.strftime('%Y%m%d')
    entries_dir = os.path.join(LIB, 'wiki', 'entries')
    prefix = f'ent_{day}_'
    existing = (
        [name for name in os.listdir(entries_dir) if name.startswith(prefix)] if os.path.isdir(entries_dir) else []
    )
    sequence = max((int(name.split('_')[-1].split('.')[0]) for name in existing), default=0) + 1
    return f'{prefix}{sequence:04d}'


def _ensure_dirs() -> None:
    os.makedirs(os.path.join(LIB, 'wiki', 'entries'), exist_ok=True)


def _write_raw(entry_id: str, manifest: dict[str, Any]) -> str:
    """Persist the original file/content for non-url sources and return the
    ``source_raw`` value to store (relative path). URL kind has nothing to
    write here — the URL passed by the caller is already the ``source_raw``
    value, so this returns it unchanged without touching disk.
    """
    source_raw = manifest.get('source_raw', '')
    if manifest.get('source_kind') == 'url':
        return source_raw

    raw_dir = os.path.join(LIB, 'raw', entry_id)

    source_file = manifest.get('source_file', '')
    if source_file and os.path.exists(source_file):
        os.makedirs(raw_dir, exist_ok=True)
        extension = os.path.splitext(source_file)[1] or '.bin'
        destination = os.path.join(raw_dir, f'source{extension}')
        shutil.copy2(source_file, destination)
        source_raw = f'raw/{entry_id}/source{extension}'

    content = manifest.get('content', '')
    if content:
        os.makedirs(raw_dir, exist_ok=True)
        with open(os.path.join(raw_dir, 'content.md'), 'w') as file:
            file.write(content)

    return source_raw


def _emit_key_insights(insights: list[dict[str, Any]]) -> list[str]:
    """Serialize key insights into re-parseable nested YAML frontmatter."""
    if not insights:
        return ['keyInsights: []']
    lines = ['keyInsights:']
    for insight in insights:
        diff = insight.get('diff', '') or ''
        lines.append(f'  - text: {_yaml_quote(insight.get("text", ""))}')
        lines.append(f'    diff: {_yaml_quote(diff)}')
        lines.append(f'    ref: {_yaml_quote(insight.get("ref", ""))}')
        lines.append(f'    group: {_yaml_quote(insight.get("group", ""))}')
    return lines


def _write_entry(entry_id: str, manifest: dict[str, Any], today: str) -> None:
    tags = ', '.join(manifest['topicTags'])
    lines = [
        '---',
        f'id: {entry_id}',
        f'title: {_yaml_quote(manifest["title"])}',
        f'type: {manifest["type"]}',
        f'status: {manifest["status"]}',
        f'source_kind: {manifest["source_kind"]}',
        f'source_raw: {_yaml_quote(manifest.get("source_raw", ""))}',
        f'category: {_yaml_quote(manifest["category"])}',
        f'topicTags: [{tags}]',
        f'summary: {_yaml_quote(manifest["summary"])}',
        *_emit_key_insights(manifest.get('keyInsights', [])),
        f'created: {today}',
        f'updated: {today}',
        '---',
        '',
        manifest.get('body', '') or '',
        '',
    ]
    path = os.path.join(LIB, 'wiki', 'entries', f'{entry_id}.md')
    with open(path, 'w') as file:
        file.write('\n'.join(lines))


def _update_index(entry_id: str, manifest: dict[str, Any], today: str) -> None:
    path = os.path.join(LIB, 'wiki', 'index.json')
    index = _read_index()

    index = [entry for entry in index if entry.get('id') != entry_id]
    index.append({
        'id': entry_id,
        'title': manifest['title'],
        'type': manifest['type'],
        'source_kind': manifest['source_kind'],
        'source_raw': manifest.get('source_raw', ''),
        'category': manifest['category'],
        'topicTags': manifest['topicTags'],
        'summary': manifest['summary'],
        'created': today,
        'updated': today,
    })

    with open(path, 'w') as file:
        json.dump(index, file, ensure_ascii=False, indent=2)


def _append_log(entry_id: str, manifest: dict[str, Any], now: datetime) -> None:
    path = os.path.join(LIB, 'wiki', 'log.md')
    today_header = f'## {now.strftime("%Y-%m-%d")}'
    time = now.strftime('%H:%M')
    line = f'- {time} ingest {entry_id} "{manifest["title"]}" ({manifest["source_kind"]})\n'

    existing = ''
    if os.path.exists(path):
        with open(path) as file:
            existing = file.read()

    with open(path, 'a') as file:
        if today_header not in existing:
            if existing and not existing.endswith('\n'):
                file.write('\n')
            file.write(f'\n{today_header}\n')
        file.write(line)


def _normalize_key_insights(manifest: dict[str, Any]) -> None:
    """Validate key insights that have already been structured by the caller."""
    clean: list[dict[str, str]] = []
    for insight in manifest.get('keyInsights') or []:
        if not isinstance(insight, dict) or not insight.get('text'):
            continue
        diff = insight.get('diff', '') or ''
        ref = insight.get('ref', '') or ''
        group = insight.get('group', '') or ''
        if diff not in DIFFS:
            diff = ''
        if diff in ('repeated', 'contradicted') and not ref:
            # A relation without an anchor must not be persisted as a claim.
            diff = 'added' if diff == 'contradicted' else ''
        if not diff:
            ref = ''
        clean.append({'text': insight['text'], 'diff': diff, 'ref': ref, 'group': group})

    manifest['keyInsights'] = clean


def ingest_one(manifest: dict[str, Any]) -> dict[str, Any]:
    """Persist one wiki entry and return Chris's receipt plus the Backend snapshot."""
    missing = [key for key in REQUIRED if not manifest.get(key)]
    if missing:
        return {'ok': False, 'error': f'缺少必填字段: {missing}'}
    if not isinstance(manifest['topicTags'], list) or not manifest['topicTags']:
        return {'ok': False, 'error': 'topicTags 必须是非空数组'}

    with _owner_lock():
        now = datetime.now(CST)
        today = now.strftime('%Y-%m-%d')

        _ensure_dirs()
        index = _read_index()
        revision = _next_revision(index)
        _normalize_key_insights(manifest)

        entry_id = manifest.get('id') or _next_id(now)

        manifest['source_raw'] = _write_raw(entry_id, manifest)
        _advance_revision(revision)
        _write_entry(entry_id, manifest, today)
        _update_index(entry_id, manifest, today)
        _append_log(entry_id, manifest, now)

    now_ms = int(now.timestamp() * 1000)
    card_snapshot = {
        'card_id': entry_id,
        'record_revision': revision,
        'source_url': manifest['source_raw'] if manifest['source_kind'] == 'url' else '',
        'title': manifest['title'],
        'display_excerpt': manifest['summary'],
        'category': manifest['category'],
        'topic_tags': manifest['topicTags'],
        'created_at_ms': now_ms,
        'updated_at_ms': now_ms,
        'content_digest': manifest.get('content_digest') or {},
    }

    return {
        'ok': True,
        'id': entry_id,
        'title': manifest['title'],
        'category': manifest['category'],
        'topicTags': manifest['topicTags'],
        'keyInsights': len(manifest['keyInsights']),
        'card_snapshot': card_snapshot,
    }

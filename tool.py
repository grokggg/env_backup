"""bash / read / edit / write tools for yb_one_agent (no ``description`` field).

Same observable behaviour as the ``*_v2`` tools in :mod:`tools`, but with the
``description`` field removed from the input schema (the LLM no longer has to
fill in a per-call rationale).

Implementation is intentionally a standalone copy — these tools do **not**
delegate to the ``*_v2`` versions, so each set can evolve independently.

NOTE: New business code should prefer the newer ``bash`` / ``read`` / ``edit``
/ ``write`` tools defined here over the ``*_v2`` variants in :mod:`tools`.
"""

import logging

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.agents.yb_one_agent.tools import (
    MAX_RESPONSE_LEN,
    MAX_TIMEOUT_MS,
    _first_upload_error,
    _make_output,
    maybe_truncate,
)
from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command


logger = logging.getLogger(__name__)


# ============== Bash ==============


class BashInput(BaseModel):
    """Input for bash command execution."""

    command: str = Field(description='The bash command to execute.')
    timeout: int = Field(
        default=60000,
        description='Optional timeout in milliseconds (max 600000).',
    )


@tool_registry.register(
    'yb_one_agent/bash',
    meta={
        'name': 'bash',
        'description': (
            'Executes a given bash command in a persistent shell session with optional timeout.\n\n'
            'IMPORTANT: This tool is for terminal operations like git, npm, docker, running scripts, etc. '
            'DO NOT use it for file operations (reading, writing, editing, searching, finding files) '
            '— use the specialized read, edit, write tools instead.\n\n'
            'Usage notes:\n'
            '- Always quote file paths that contain spaces with double quotes.\n'
            '- For independent commands, prefer chaining with && or calling multiple times.\n'
            '- Avoid using find, grep, cat, head, tail, sed, awk — use the dedicated tools instead.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    """Execute a bash command using the stateless run_command executor.

    Each invocation runs in an independent subprocess — no state is shared
    between calls (environment variables, working directory, etc.).
    """
    # Resolve timeout: convert from milliseconds to seconds, apply max cap
    timeout_ms = min(params.timeout, MAX_TIMEOUT_MS)
    timeout_seconds = timeout_ms / 1000.0

    # Determine working directory from context
    cwd = context.tool_state.get('working_dir', '/data/workspace')

    # Environment variables from tool_state (merged with the base environment)
    extra_env = context.tool_state.get('env') or {}

    logger.debug('bash: executing command=%r, cwd=%s, timeout=%.1fs', params.command, cwd, timeout_seconds)

    # Execute via stateless run_command
    response = await run_command(
        RunCommandRequest(
            command=params.command,
            cwd=cwd,
            timeout=timeout_seconds,
            extra_env=extra_env,
        )
    )

    # Build output text from stdout/stderr
    if response.status == CommandRunStatus.TimeLimitExceeded:
        logger.warning(
            'bash: command timed out after %.1fs, command=%r',
            response.execution_time or timeout_seconds,
            params.command,
        )
        output_parts = []
        if response.stdout:
            output_parts.append(response.stdout)
        if response.stderr:
            output_parts.append(f'[stderr]\n{response.stderr}')
        output_parts.append(f'[Command timed out after {timeout_seconds}s]')
        output_text = '\n'.join(output_parts) if output_parts else f'Command timed out after {timeout_seconds}s'

        return ToolResponse(
            output=_make_output(maybe_truncate(output_text)),
            meta={
                'command': params.command,
                'exit_code': -1,
                'timeout': True,
                'execution_time': response.execution_time,
            },
        )

    if response.status == CommandRunStatus.Error:
        error_text = response.stderr or 'Command execution failed'
        logger.error('bash: command failed, command=%r, error=%s', params.command, error_text)
        return ToolResponse(
            output=_make_output(maybe_truncate(f'Error: {error_text}')),
            meta={
                'command': params.command,
                'exit_code': -1,
                'error': True,
            },
        )

    # CommandRunStatus.Finished
    exit_code = response.return_code or 0
    logger.debug(
        'bash: command finished, command=%r, exit_code=%d, time=%.2fs',
        params.command,
        exit_code,
        response.execution_time or 0,
    )

    output_text = response.stdout or ''
    if response.stderr:
        output_text += f'\n[stderr]\n{response.stderr}'

    if not output_text.strip():
        output_text = '(no output)'

    # Append exit code info
    output_text += f'\n[Command finished with exit code {exit_code}]'

    return ToolResponse(
        output=_make_output(maybe_truncate(output_text)),
        meta={
            'command': params.command,
            'exit_code': exit_code,
            'stdout': response.stdout,
            'stderr': response.stderr,
            'execution_time': response.execution_time,
        },
    )


# ============== Read ==============


class ReadInput(BaseModel):
    file_path: str = Field(description='The absolute path to the file to read.')
    offset: int = Field(default=1, description='The line number to start reading from (1-indexed).')
    limit: int = Field(default=2000, description='The number of lines to read.')
    max_chars: SkipJsonSchema[int] = Field(default=MAX_RESPONSE_LEN, gt=0)


@tool_registry.register(
    'yb_one_agent/read',
    meta={
        'name': 'read',
        'description': (
            'Reads a file from the filesystem.\n\n'
            'Usage:\n'
            '- file_path must be an absolute path.\n'
            '- By default reads up to 2000 lines from the beginning.\n'
            '- Use offset and limit for large files.\n'
            '- Returns content in cat -n format with line numbers.\n'
            '- Can read text files, source code, configs, etc.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def read_tool(params: ReadInput, context: ToolContext) -> ToolResponse:
    from nexus.runtime.executors.file import DownloadFileRequest, FileItem, download_file

    logger.debug('read: file_path=%s, offset=%s, limit=%s', params.file_path, params.offset, params.limit)

    try:
        result = await download_file(DownloadFileRequest(files=[FileItem(path=params.file_path, encoding='utf8')]))
        if not result.files:
            error = result.errors[0].message if result.errors else 'File does not exist'
            logger.warning('read: file not found, path=%s, error=%s', params.file_path, error)
            return ToolResponse(output=_make_output(f'Error: {error}'))

        content = result.files[0].content or ''
        lines = content.splitlines()

        # Apply offset and limit (1-indexed offset)
        offset = max(params.offset - 1, 0)
        limit = params.limit
        selected = lines[offset : offset + limit]

        # Format as cat -n
        formatted = '\n'.join(f'{offset + i + 1:6}\t{line}' for i, line in enumerate(selected))
        if len(lines) > offset + limit:
            formatted += f'\n... ({len(lines) - offset - limit} more lines)'

        logger.debug('read: success, path=%s, total_lines=%d, returned=%d', params.file_path, len(lines), len(selected))
        return ToolResponse(output=_make_output(maybe_truncate(formatted, max_length=params.max_chars)))

    except Exception as e:
        logger.error('read: exception, path=%s, error=%s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))


# ============== Edit ==============


class EditInput(BaseModel):
    file_path: str = Field(description='The absolute path to the file to modify.')
    old_string: str = Field(description='The exact text to replace. Must be unique in the file.')
    new_string: str = Field(description='The text to replace it with.')
    replace_all: bool = Field(default=False, description='Replace all occurrences instead of just the first.')


@tool_registry.register(
    'yb_one_agent/edit',
    meta={
        'name': 'edit',
        'description': (
            'Performs exact string replacement in a file.\n\n'
            'Usage:\n'
            '- old_string must match the file content exactly (including whitespace and indentation).\n'
            '- old_string must be unique in the file, or use replace_all=true.\n'
            '- The edit will fail if old_string is not found or is ambiguous.\n'
            '- Always read the file with read before editing to get the exact content.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def edit_tool(params: EditInput, context: ToolContext) -> ToolResponse:
    from nexus.runtime.executors.file import (
        DownloadFileRequest,
        FileItem,
        UploadFileRequest,
        download_file,
        upload_file,
    )

    logger.debug(
        'edit: file_path=%s, old_string_len=%d, new_string_len=%d, replace_all=%s',
        params.file_path,
        len(params.old_string),
        len(params.new_string),
        params.replace_all,
    )

    try:
        # Read current content
        result = await download_file(DownloadFileRequest(files=[FileItem(path=params.file_path, encoding='utf8')]))
        if not result.files:
            error = result.errors[0].message if result.errors else 'File does not exist'
            logger.warning('edit: file not found, path=%s, error=%s', params.file_path, error)
            return ToolResponse(output=_make_output(f'Error: {error}'))

        content = result.files[0].content or ''

        if params.old_string not in content:
            logger.warning('edit: old_string not found in %s', params.file_path)
            return ToolResponse(
                output=_make_output(
                    f'Error: old_string not found in {params.file_path}. '
                    'Make sure it matches exactly including whitespace.'
                )
            )

        if not params.replace_all:
            count = content.count(params.old_string)
            if count > 1:
                logger.warning('edit: old_string is ambiguous, found %d times in %s', count, params.file_path)
                return ToolResponse(
                    output=_make_output(
                        f'Error: old_string appears {count} times in the file. '
                        'Provide more context to make it unique, or use replace_all=true.'
                    )
                )
            new_content = content.replace(params.old_string, params.new_string, 1)
        else:
            new_content = content.replace(params.old_string, params.new_string)

        # Write back
        upload_result = await upload_file(
            UploadFileRequest(
                files=[FileItem(path=params.file_path, content=new_content, encoding='utf8')],
                overwrite=True,
            )
        )
        upload_error = _first_upload_error(upload_result)
        if upload_error:
            logger.warning('edit: upload failed, path=%s, error=%s', params.file_path, upload_error)
            return ToolResponse(output=_make_output(f'Error: {upload_error}'))

        replacements = 1 if not params.replace_all else content.count(params.old_string)
        logger.debug('edit: success, path=%s, replacements=%d', params.file_path, replacements)
        return ToolResponse(
            output=_make_output(f'Successfully edited {params.file_path} ({replacements} replacement(s)).')
        )

    except Exception as e:
        logger.error('edit: exception, path=%s, error=%s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))


# ============== Write ==============


class WriteInput(BaseModel):
    file_path: str = Field(description='The absolute path to the file to write (will be created or overwritten).')
    content: str = Field(description='The full content to write to the file.')


@tool_registry.register(
    'yb_one_agent/write',
    meta={
        'name': 'write',
        'description': (
            'Writes content to a file, creating it or overwriting if it already exists.\n\n'
            'Usage:\n'
            '- file_path must be an absolute path.\n'
            '- Prefer edit for modifying existing files — only use write for new files or full rewrites.\n'
            '- If modifying an existing file, use read first to review current content.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def write_tool(params: WriteInput, context: ToolContext) -> ToolResponse:
    from nexus.runtime.executors.file import FileItem, UploadFileRequest, upload_file

    logger.debug('write: file_path=%s, content_len=%d', params.file_path, len(params.content))

    try:
        upload_result = await upload_file(
            UploadFileRequest(
                files=[FileItem(path=params.file_path, content=params.content, encoding='utf8')],
                overwrite=True,
            )
        )
        upload_error = _first_upload_error(upload_result)
        if upload_error:
            logger.warning('write: upload failed, path=%s, error=%s', params.file_path, upload_error)
            return ToolResponse(output=_make_output(f'Error: {upload_error}'))

        lines = params.content.count('\n') + 1
        logger.debug('write: success, path=%s, lines=%d', params.file_path, lines)
        return ToolResponse(output=_make_output(f'Successfully wrote {lines} lines to {params.file_path}.'))

    except Exception as e:
        logger.error('write: exception, path=%s, error=%s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))

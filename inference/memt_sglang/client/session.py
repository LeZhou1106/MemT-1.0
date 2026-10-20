"""Native HTTP sessions with HF turn replay and memory-only snapshots."""

import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field

from memt_sglang.client.chat_format import text, token_ids
from memt_sglang.client.native import DEFAULT_MAX_NEW_TOKENS, MemTClientError
from memt_sglang.protocol import TransactionStatus


@dataclass
class _Turn:
    messages: list
    replay_ids: list
    thinking: bool | None
    enable_memory: bool
    max_new_tokens: int
    sampling: dict
    structured: bool
    tool_calls: list = field(default_factory=list)


class MemTSession:
    """Session memory and history, with separate input formatting and parsing."""

    def __init__(
        self,
        client,
        chat_format,
        parser,
        *,
        session=None,
        system_prompt=None,
        tools=None,
        state=None,
    ):
        self.client = client
        self.chat_format, self.parser = chat_format, parser
        self.system_prompt = text(system_prompt) if system_prompt is not None else None
        if tools is not None and not isinstance(tools, list):
            raise ValueError("tools must be a list of function schemas")
        self.tools = deepcopy(tools or [])
        names = []
        for tool in self.tools:
            if not isinstance(tool, dict) or tool.get("type", "function") != "function":
                raise ValueError("tools must contain function schemas")
            function = tool.get("function")
            if not isinstance(function, dict) or not isinstance(
                function.get("name"), str
            ):
                raise ValueError("Each tool requires a function name")
            names.append(function["name"])
        if len(names) != len(set(names)):
            raise ValueError("tools must have unique function names")
        self.capabilities = client.capabilities(refresh=True)
        self._snapshot_identity = self.capabilities.get("snapshot_identity")
        if (
            self._snapshot_identity is not None
            and self.chat_format.identity != self._snapshot_identity
        ):
            raise ValueError(
                "Client HF model/tokenizer identity differs from the server"
            )
        self.id = client.create_session(
            session, memory_read=True, epoch=self.capabilities["epoch"]
        )
        self._turns, self.responses = [], []
        self._lock = threading.RLock()
        self.uncertain, self.closed, self.memory_read = False, False, True
        self._pending = None
        self._tool_turn = None
        self._has_memory = False
        if state is not None:
            try:
                self.load_state(state)
            except BaseException as error:
                try:
                    self.close()
                except Exception as cleanup:
                    error.add_note(f"Session cleanup also failed: {cleanup}")
                raise

    @property
    def messages(self):
        """Detached uncompressed messages, including a pending tool turn."""
        with self._lock:
            turns = self._turns + ([self._tool_turn] if self._tool_turn else [])
            return deepcopy([message for turn in turns for message in turn.messages])

    def _check(self):
        if self.closed:
            raise MemTClientError("Session is closed; create a new session")
        if self.uncertain:
            raise MemTClientError(
                "Session has an unresolved transaction; inspect before further writes",
                uncertain=True,
            )

    def _check_no_tools(self):
        if self._tool_turn is not None:
            raise ValueError("Submit the pending tool results before this operation")

    @contextmanager
    def _operation(self):
        with self._lock:
            self._check()
            try:
                yield
            except BaseException as error:
                self.uncertain |= (
                    bool(getattr(error, "uncertain", False))
                    or not isinstance(error, (MemTClientError, ValueError, TypeError))
                    or self.client.has_unreleased_lease(self.id)
                )
                raise

    def _record_failure(self, error, count):
        self.uncertain = bool(getattr(error, "uncertain", False))
        if self.uncertain:
            self._pending = (getattr(error, "transaction", None), count)

    def recover(self):
        """Reconcile a transaction receipt; never automatically replay writes."""
        with self._lock:
            if self.closed:
                raise MemTClientError("Session is closed")
            if self.uncertain and (not self._pending or not self._pending[0]):
                raise MemTClientError(
                    "No transaction receipt to reconcile; create a new session "
                    "and load a confirmed snapshot",
                    uncertain=True,
                )
            status = self.client.recover(self.id)
            if self._pending:
                transaction, count = self._pending
                if status.get("transaction") != transaction or status["status"] not in {
                    TransactionStatus.PUBLISHED,
                    TransactionStatus.ABORTED,
                }:
                    raise MemTClientError(
                        "Transaction receipt no longer available", uncertain=True
                    )
                if status["status"] == TransactionStatus.PUBLISHED:
                    del self._turns[:count]
            self._pending, self.uncertain = None, False
            inspected = self.client.control(self.id, "inspect")
            self.memory_read = inspected["memory_read"]
            self._has_memory = inspected.get("has_memory", self._has_memory)
            return status

    def _commit_turns(self, turns, lease, transaction):
        has_memory = self._has_memory
        encoded = []
        for turn in turns:
            encoded.append(
                self.chat_format.replay(turn.replay_ids, has_memory=has_memory)
            )
            has_memory = True
        if encoded:
            self.client.commit_turns(
                self.id, encoded, lease=lease, transaction=transaction
            )
        return has_memory

    def _generate(self, turn, lease):
        prompt = self.chat_format.query(
            self._turns + [turn],
            system=self.system_prompt,
            tools=self.tools,
            has_memory=self._has_memory,
        )
        result = self.client.query(
            self.id,
            prompt,
            lease=lease,
            max_new_tokens=turn.max_new_tokens,
            sampling=turn.sampling,
        )
        generated = token_ids(result.get("output_ids"))
        completed = (
            result["meta_info"]["finish_reason"]["type"] == "stop"
            and generated[-1] in self.chat_format.eos_token_ids
        )
        try:
            answer, calls, message = self.parser.parse(
                generated,
                prompt,
                thinking=turn.thinking,
                tools=self.tools,
            )
        except (ValueError, AssertionError, KeyError) as error:
            if completed:
                raise MemTClientError(
                    str(error), stage="tool_parse", answer=result
                ) from error
            answer, calls, message = self.parser.parse(
                generated,
                prompt,
                thinking=turn.thinking,
                tools=None,
            )
        turn.messages.append(message)
        turn.replay_ids.extend(generated)
        turn.tool_calls = calls
        result = deepcopy(result)
        result.update(
            answer=answer,
            text=answer,
            generation_completed=completed,
            turn_completed=completed and not calls,
            tool_calls=deepcopy(calls),
            requires_action=bool(calls),
            memory_updated=False,
            committed_turns=0,
            input_tokens=len(prompt),
            output_tokens=len(generated),
            documents_imported=0,
            finish_reason="tool_calls"
            if calls
            else ("eos" if completed else result["meta_info"]["finish_reason"]["type"]),
        )
        self.responses.append(deepcopy(result))
        return result

    def _finish(self, turn, result, lease):
        if turn.tool_calls:
            self._tool_turn = turn
            return result
        self._tool_turn = None
        self._turns.append(turn)
        if turn.enable_memory and result["generation_completed"]:
            turns = list(self._turns)
            try:
                with self.client.transaction(self.id, lease=lease) as txn:
                    has_memory = self._commit_turns(turns, lease, txn)
            except Exception as error:
                self._record_failure(error, len(turns))
                raise MemTClientError(
                    str(error),
                    stage="history",
                    transaction=getattr(error, "transaction", None),
                    uncertain=self.uncertain,
                    answer=result,
                ) from error
            self._has_memory = has_memory
            del self._turns[: len(turns)]
            result.update(memory_updated=True, committed_turns=len(turns))
        return result if turn.structured else result["answer"]

    def chat(
        self,
        query,
        *,
        enable_memory=False,
        enable_thinking=None,
        return_dict=False,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        sampling=None,
    ):
        query = text(query)
        if type(enable_memory) is not bool:
            raise ValueError("enable_memory must be a boolean")
        if type(return_dict) is not bool:
            raise ValueError("return_dict must be a boolean")
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise ValueError("enable_thinking must be a boolean or None")
        with self._operation():
            self._check_no_tools()
            messages = [{"role": "user", "content": query}]
            turn = _Turn(
                messages,
                list(self.chat_format.prefix(messages, thinking=enable_thinking)),
                enable_thinking,
                enable_memory,
                max_new_tokens,
                dict(sampling or {}),
                bool(self.tools) or return_dict,
            )
            with self.client.lease(self.id) as lease:
                result = self._generate(turn, lease)
                return self._finish(turn, result, lease)

    def submit_tool_results(self, results, *, max_new_tokens=None, sampling=None):
        """Continue one pending turn with all caller-executed tool results."""
        with self._operation():
            if self._tool_turn is None:
                raise ValueError("There are no pending tool requests")
            calls = self._tool_turn.tool_calls
            if not isinstance(results, list) or len(results) != len(calls):
                raise ValueError(
                    "Supply exactly one result for every pending tool call"
                )
            by_id = {}
            expected = {call["id"] for call in calls}
            for result in results:
                if not isinstance(result, dict) or not isinstance(
                    result.get("content"), str
                ):
                    raise ValueError(
                        "Tool results require tool_call_id and text content"
                    )
                identifier = result.get("tool_call_id")
                if identifier in by_id or identifier not in expected:
                    raise ValueError("Duplicate or unknown tool_call_id")
                by_id[identifier] = result["content"]
            turn = deepcopy(self._tool_turn)
            messages = [
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": by_id[call["id"]],
                }
                for call in calls
            ]
            turn.replay_ids.extend(
                self.chat_format.tool_suffix(
                    messages,
                    context=turn.messages,
                    thinking=turn.thinking,
                )
            )
            turn.messages.extend(messages)
            if max_new_tokens is not None:
                turn.max_new_tokens = max_new_tokens
            if sampling is not None:
                turn.sampling.update(sampling)
            with self.client.lease(self.id) as lease:
                result = self._generate(turn, lease)
                return self._finish(turn, result, lease)

    def commit_history(self):
        with self._operation():
            self._check_no_tools()
            if not self._turns:
                return
            with self.client.lease(self.id) as lease:
                try:
                    with self.client.transaction(self.id, lease=lease) as txn:
                        has_memory = self._commit_turns(self._turns, lease, txn)
                except Exception as error:
                    self._record_failure(error, len(self._turns))
                    raise
                self._has_memory = has_memory
                self._turns.clear()

    def ingest(self, document):
        text(document)
        with self._operation():
            self._check_no_tools()
            with self.client.lease(self.id) as lease:
                try:
                    with self.client.transaction(self.id, lease=lease) as txn:
                        has_memory = self._commit_turns(self._turns, lease, txn)
                        encoded = self.chat_format.document(
                            document, has_memory=has_memory
                        )
                        self.client.commit(
                            self.id, encoded, lease=lease, transaction=txn
                        )
                except Exception as error:
                    self._record_failure(error, len(self._turns))
                    raise
                self._has_memory = True
                self._turns.clear()

    def _check_snapshot_identity(self):
        if self._snapshot_identity is None:
            raise ValueError("Server does not advertise HF snapshot interoperability")
        if self.chat_format.identity != self._snapshot_identity:
            raise ValueError(
                "Client HF model/tokenizer identity differs from the server"
            )

    def save_state(self, path=None, *, overwrite=False):
        """Replay remaining turns, export the candidate, then adopt its state."""
        if type(overwrite) is not bool:
            raise ValueError("overwrite must be a boolean")
        self._check_snapshot_identity()
        with self._operation():
            self._check_no_tools()
            with self.client.lease(self.id) as lease:
                result = None
                try:
                    with self.client.transaction(self.id, lease=lease) as txn:
                        has_memory = self._commit_turns(self._turns, lease, txn)
                        result = self.client.control(
                            self.id,
                            "export",
                            lease=lease,
                            transaction=txn,
                            file=None if path is None else str(path),
                            encoding=self.chat_format.encoding,
                            overwrite=overwrite,
                        )
                        if not isinstance(result.get("snapshot_id"), str):
                            raise MemTClientError(
                                "Snapshot ID was not confirmed",
                                uncertain=True,
                                transaction=txn,
                            )
                except Exception as error:
                    self._record_failure(error, len(self._turns))
                    if result is not None:
                        raise MemTClientError(
                            "Snapshot exported, but adopting its state failed: "
                            + str(error),
                            stage="save_publish",
                            transaction=getattr(error, "transaction", None),
                            uncertain=self.uncertain,
                            answer=result,
                        ) from error
                    raise
                self._has_memory = has_memory
                self._turns.clear()
                return {"path": result["snapshot_id"], "publication": "published"}

    def load_state(self, path):
        """Import HF memory atomically, clear context and enable memory reading."""
        self._check_snapshot_identity()
        with self._operation():
            self._check_no_tools()
            with self.client.lease(self.id) as lease:
                try:
                    with self.client.transaction(
                        self.id, lease=lease, memory_read=True
                    ) as txn:
                        result = self.client.control(
                            self.id,
                            "import",
                            lease=lease,
                            transaction=txn,
                            file=str(path),
                            encoding=self.chat_format.encoding,
                        )
                except Exception as error:
                    self._record_failure(error, len(self._turns))
                    raise
                self._turns.clear()
                self.responses.clear()
                self.memory_read = True
                self._has_memory = result["has_memory"]
                return result

    def set_memory_read(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        with self._operation(), self.client.lease(self.id) as lease:
            result = self.client.control(
                self.id, "memory_read", lease=lease, enabled=enabled
            )
            if result.get("memory_read") is not enabled:
                raise MemTClientError(
                    "Reading policy was not confirmed", uncertain=True
                )
            self.memory_read = enabled
            return result

    def status(self):
        if self.closed:
            return {"session": self.id, "closed": True, "busy": False}
        result = self.client.control(self.id, "inspect")
        return {
            **result,
            "closed": False,
            "uncertain": self.uncertain or result["uncertain"],
            "uncommitted_turns": len(self._turns),
            "requires_action": self._tool_turn is not None,
        }

    def reset(self):
        with self._operation():
            self.client.reset(self.id)
            self._turns.clear()
            self.responses.clear()
            self._tool_turn = None
            self._has_memory, self.uncertain, self._pending = False, False, None

    def close(self):
        with self._lock:
            if self.closed:
                return
            try:
                self.client.close(self.id)
            except BaseException:
                self.uncertain = True
                raise
            self.closed = True
            self._turns.clear()
            self._tool_turn = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"Session cleanup also failed: {cleanup}")

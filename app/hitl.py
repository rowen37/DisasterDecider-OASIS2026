# hitl.py
from __future__ import annotations

import os
import asyncio
from typing import Any

from .models import HITLRequest, RunState


class AdaptiveHITL:
    def __init__(self, state: RunState):
        self.state = state
        self.enabled = os.getenv("HITL_ENABLED", "true").lower() == "true"
        # 0.7 is the floor score of a complete authoritative observation
        # (authority 0.4 + completeness 0.25 + metadata 0.05, each at
        # 1.0). Freshness only weights the remaining 0.3 and is
        # disclosed separately via the freshness/time_alignment ledger,
        # so staleness alone never triggers a human checkpoint.
        self.auto_approve_confidence = float(
            os.getenv("AUTO_APPROVE_CONFIDENCE", "0.7")
        )
        # Seconds to wait for a human response in web mode. Timed-out
        # safety-critical checkpoints take the denial direction
        # (timeout_value), so this is a liveness budget only: long
        # enough for an operator to respond mid-event, short enough not
        # to stall the whole assessment.
        self.timeout_seconds = float(
            os.getenv("HITL_TIMEOUT_SECONDS", "120")
        )
        # Web-mode waiting state
        self._web_mode = False
        self._pending_request = None
        self._response_event = None
        self._response_value = None

    def enable_web_mode(self):
        """Switch to web mode (async wait instead of blocking input)."""
        self._web_mode = True

    def should_intervene(
        self,
        *,
        confidence: float | None = None,
        critical_error: bool = False,
        ambiguous_parameter: bool = False,
        evidence_conflict: bool = False,
        policy_tradeoff: bool = False,
    ) -> bool:
        if not self.enabled:
            return False
        if critical_error or ambiguous_parameter or evidence_conflict or policy_tradeoff:
            return True
        return confidence is not None and confidence < self.auto_approve_confidence

    async def ask_async(
        self,
        reason: str,
        question: str,
        proposed_value: Any | None = None,
        context: dict[str, Any] | None = None,
        timeout_value: Any | None = None,
    ) -> Any:
        """
        Async version of ask for web mode.
        Stores the request in shared state and waits for the frontend.

        ``timeout_value``: safe default on timeout or when HITL is
        disabled. Safety-critical checkpoints (resource mobilization
        approval, conflict escalation) must pass the denial-direction
        value; an unanswered request never counts as authorization.
        Falls back to proposed_value when omitted (non-safety parameter
        confirmations only).
        """
        request = HITLRequest(
            reason=reason,
            question=question,
            proposed_value=proposed_value,
            context=context or {},
        )
        self.state.hitl_requests.append(request)

        def _safe_default() -> Any:
            return timeout_value if timeout_value is not None else proposed_value

        # HITL_ENABLED=false applies globally and matches the timeout
        # direction: unattended runs take the safe default (no action)
        # at safety-critical checkpoints instead of accepting the
        # proposal.
        if not self.enabled:
            return _safe_default()

        if not self._web_mode:
            # CLI mode: blocking prompt
            print("\n" + "=" * 72)
            print("HUMAN-IN-THE-LOOP REVIEW REQUIRED")
            print("=" * 72)
            print(f"Reason: {reason}")
            print(f"Question: {question}")
            if proposed_value is not None:
                print(f"Agent proposal: {proposed_value}")
            answer = input("Your decision (Enter = accept proposal): ").strip()
            return proposed_value if answer == "" else answer

        # Web mode: asynchronous wait
        self._pending_request = {
            "request_id": str(len(self.state.hitl_requests) - 1),
            "reason": reason,
            "question": question,
            "proposed_value": proposed_value,
            "context": context or {},
        }
        self._response_event = asyncio.Event()
        self._response_value = None

        # Keep the assessment alive while the decision-maker reviews it.
        # The timeout adopts the SAFE default (timeout_value), which for
        # safety-critical gates is the denial direction: an unanswered
        # authorization question must never auto-approve mobilization.
        try:
            await asyncio.wait_for(
                self._response_event.wait(),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._pending_request = None
            return _safe_default()

        response = self._response_value
        self._pending_request = None
        self._response_value = None
        self._response_event = None

        if response == "" or response is None:
            return proposed_value
        return response

    def submit_response(self, value: Any) -> None:
        """Receive the response submitted by the frontend."""
        if self._response_event is not None:
            self._response_value = value
            self._response_event.set()

    def get_pending_request(self) -> dict | None:
        """Return the currently pending HITL request, if any."""
        return self._pending_request

    def ask(self, reason: str, question: str, proposed_value: Any = None, context: dict | None = None) -> Any:
        """
        Synchronous version of ask (CLI compatibility).
        Do not call in web mode; use ask_async instead.
        """
        if self._web_mode:
            raise RuntimeError(
                "In Web mode, use ask_async() instead of ask()"
            )
        request = HITLRequest(
            reason=reason,
            question=question,
            proposed_value=proposed_value,
            context=context or {},
        )
        self.state.hitl_requests.append(request)
        print("\n" + "=" * 72)
        print("HUMAN-IN-THE-LOOP REVIEW REQUIRED")
        print("=" * 72)
        print(f"Reason: {reason}")
        print(f"Question: {question}")
        if proposed_value is not None:
            print(f"Agent proposal: {proposed_value}")
        answer = input("Your decision (Enter = accept proposal): ").strip()
        return proposed_value if answer == "" else answer

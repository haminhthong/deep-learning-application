"""Module máy trạng thái hữu hạn theo thời gian (Temporal Violation FSM) cho hệ thống giám sát PPE.

Quản lý chu kỳ vòng đời của một trạng thái vi phạm:
    COMPLIANT (Tuân thủ)
       ↓ (vi phạm liên tiếp >= confirm_observations)
    ALERTED (Báo động vi phạm chính thức - Emitted Alert & Evidence Snapshot)
       ↓ (tuân thủ trở lại >= resolve_observations)
    RESOLVED (Đã khắc phục vi phạm)
       ↓ (tái phạm liên tiếp >= confirm_observations)
    ALERTED (Báo động tái phạm - Recurrent Violation Event)

Ngăn chặn báo động giả đồng thời loại bỏ giới hạn one-shot (cho phép phát hiện công nhân tháo mũ sau đó).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .models import ViolationState

LOGGER = logging.getLogger(__name__)


@dataclass
class FSMTransitionResult:
    """Kết quả chuyển đổi trạng thái của FSM sau mỗi chu kỳ quan sát."""

    track_id: int
    violation_type: str
    previous_state: str
    current_state: str
    should_emit_alert: bool
    is_recurrence: bool = False
    is_resolved: bool = False


class TemporalViolationFSM:
    """Máy trạng thái thời gian kiểm soát việc kích hoạt và gỡ bỏ vi phạm bảo hộ."""

    def __init__(
        self,
        confirm_observations: int = 3,
        resolve_observations: int = 3,
    ) -> None:
        """Khởi tạo FSM.

        Args:
            confirm_observations: Số lần quan sát vi phạm liên tiếp từ detector để chuyển sang ALERTED.
            resolve_observations: Số lần quan sát tuân thủ liên tiếp từ detector để chuyển sang RESOLVED.
        """
        self.confirm_observations = max(1, confirm_observations)
        self.resolve_observations = max(1, resolve_observations)
        self.states: dict[tuple[int, str], ViolationState] = {}

    def get_state(self, track_id: int, violation_type: str) -> ViolationState:
        """Lấy trạng thái hiện tại của một đối tượng và loại vi phạm."""
        key = (track_id, violation_type)
        if key not in self.states:
            self.states[key] = ViolationState(state="COMPLIANT")
        return self.states[key]

    def update(
        self,
        track_id: int,
        violation_type: str,
        is_violated: bool,
        frame_id: int,
        timestamp_sec: float,
    ) -> FSMTransitionResult:
        """Cập nhật quan sát mới từ detector và thực hiện chuyển trạng thái FSM.

        Args:
            track_id: ID theo dõi của người.
            violation_type: Loại vi phạm ('helmet' hoặc 'vest').
            is_violated: True nếu quan sát ở frame này là vi phạm, False nếu tuân thủ.
            frame_id: Thứ tự frame hiện tại.
            timestamp_sec: Thời điểm tính bằng giây.

        Returns:
            `FSMTransitionResult` chứa chỉ dẫn có cần phát cảnh báo hoặc thông báo khắc phục không.
        """
        key = (track_id, violation_type)
        v_state = self.get_state(track_id, violation_type)
        prev_state = v_state.state

        should_emit = False
        is_recurrence = False
        is_resolved = False

        v_state.last_seen_sec = timestamp_sec

        if is_violated:
            v_state.consecutive_positive += 1
            v_state.consecutive_negative = 0

            if v_state.state == "COMPLIANT":
                if v_state.consecutive_positive >= self.confirm_observations:
                    v_state.state = "ALERTED"
                    v_state.event_count += 1
                    v_state.started_at_frame = frame_id
                    v_state.started_at_sec = timestamp_sec
                    should_emit = True
                else:
                    v_state.state = "VIOLATING"

            elif v_state.state == "VIOLATING":
                if v_state.consecutive_positive >= self.confirm_observations:
                    v_state.state = "ALERTED"
                    v_state.event_count += 1
                    v_state.started_at_frame = frame_id
                    v_state.started_at_sec = timestamp_sec
                    should_emit = True

            elif v_state.state == "RESOLVED":
                # Tái phạm
                if v_state.consecutive_positive >= self.confirm_observations:
                    v_state.state = "ALERTED"
                    v_state.event_count += 1
                    v_state.started_at_frame = frame_id
                    v_state.started_at_sec = timestamp_sec
                    should_emit = True
                    is_recurrence = True
                    LOGGER.info(
                        "TÁI PHẠM [ID %d - %s]: Công nhân vi phạm trở lại sau khi đã khắc phục.",
                        track_id,
                        violation_type,
                    )

        else:
            # Đối tượng tuân thủ
            v_state.consecutive_negative += 1
            v_state.consecutive_positive = 0

            if v_state.state == "VIOLATING":
                v_state.state = "COMPLIANT"

            elif v_state.state == "ALERTED":
                if v_state.consecutive_negative >= self.resolve_observations:
                    v_state.state = "RESOLVED"
                    v_state.resolved_at_sec = timestamp_sec
                    is_resolved = True
                    LOGGER.info(
                        "KHẮC PHỤC [ID %d - %s]: Công nhân đã tuân thủ trang bị bảo hộ.",
                        track_id,
                        violation_type,
                    )

        return FSMTransitionResult(
            track_id=track_id,
            violation_type=violation_type,
            previous_state=prev_state,
            current_state=v_state.state,
            should_emit_alert=should_emit,
            is_recurrence=is_recurrence,
            is_resolved=is_resolved,
        )

    def clean_inactive_tracks(self, active_track_ids: set[int]) -> None:
        """Dọn dẹp bộ nhớ các track đã bị xóa khỏi tracker."""
        to_delete = [k for k in self.states if k[0] not in active_track_ids]
        for k in to_delete:
            del self.states[k]

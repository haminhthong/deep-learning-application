import cv2


def draw_tracks(frame, tracks, violations: dict[str, int], fps: float):
    """Vẽ bounding box, nhãn PPE và số liệu tổng hợp lên khung hình."""
    output = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.box)
        violation = track.ppe.get("helmet_violation") or track.ppe.get("vest_violation")
        color = (0, 0, 255) if violation else (0, 200, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f"Nguoi {track.track_id}", (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for index, item in enumerate(track.ppe.get("detections", [])):
            text = f"{item['label']} {item['confidence']:.2f}"
            cv2.putText(output, text, (x1, y1 + 24 + index * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    summary = (
        f"FPS {fps:.1f} | Vi pham {violations['total']} "
        f"(mu {violations['helmet']}, ao {violations['vest']})"
    )
    cv2.putText(output, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    return output

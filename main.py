"""
Прототип системы детекции уборки столиков по видео
Запуск: python main.py --video video1.mp4

Логика:
- Используется YOLOv8n для детекции людей (класс 0)
- Пользователь вручную выбирает ROI (зону столика) через cv2.selectROI
- Три состояния: EMPTY (пусто), OCCUPIED (занято), APPROACHING (подход)
- Аналитика сохраняется в Pandas DataFrame
- На выходе: output.mp4 с визуализацией + report.txt
"""

import argparse
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Попытка импорта YOLO, fallback на детекцию движения
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[INFO] Ultralytics не установлен. Используется детекция движения (MOG2).")


# ──────────────────────────────────────────────
# Константы и настройки
# ──────────────────────────────────────────────
TABLE_STATES = {
    "EMPTY":      {"label": "Empty",    "color": (0, 200, 0)},    # зелёный
    "OCCUPIED":   {"label": "Occupied",   "color": (0, 0, 220)},    # красный
    "APPROACHING":{"label": "Approaching",   "color": (0, 165, 255)},  # оранжевый
}

# Минимальное число кадров для смены состояния (дебаунс)
STATE_DEBOUNCE_FRAMES = 15

# Минимальная площадь контура при детекции движения
MIN_MOTION_AREA = 1500

# ──────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────

def select_roi(frame: np.ndarray) -> tuple:
    """Интерактивный выбор зоны столика через cv2.selectROI."""
    print("\n[ROI] Нарисуй прямоугольник вокруг столика мышью, затем нажми ENTER или ПРОБЕЛ.")
    print("[ROI] Для отмены нажми C.\n")
    cv2.namedWindow("Выбери зону столика", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Выбери зону столика", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Выбери зону столика")
    if roi == (0, 0, 0, 0):
        raise ValueError("ROI не выбран. Завершение.")
    x, y, w, h = roi
    print(f"[ROI] Выбрана зона: x={x}, y={y}, w={w}, h={h}\n")
    return x, y, w, h


def person_in_roi(boxes, roi: tuple, frame_shape: tuple) -> bool:
    """
    Проверяет, пересекается ли хотя бы один bounding box человека с зоной ROI.
    boxes: список [x1, y1, x2, y2] в пикселях
    roi: (rx, ry, rw, rh)
    """
    rx, ry, rw, rh = roi
    roi_rect = (rx, ry, rx + rw, ry + rh)

    for box in boxes:
        bx1, by1, bx2, by2 = box
        # Пересечение прямоугольников
        ix1 = max(bx1, roi_rect[0])
        iy1 = max(by1, roi_rect[1])
        ix2 = min(bx2, roi_rect[2])
        iy2 = min(by2, roi_rect[3])
        if ix2 > ix1 and iy2 > iy1:
            return True
    return False


def motion_in_roi(fgmask: np.ndarray, roi: tuple) -> bool:
    """Проверяет наличие значимого движения в зоне ROI через маску переднего плана."""
    rx, ry, rw, rh = roi
    roi_mask = fgmask[ry:ry + rh, rx:rx + rw]
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_MOTION_AREA:
            return True
    return False


def draw_overlay(frame: np.ndarray, roi: tuple, state: str, fps: float,
                 frame_idx: int, event_log: list) -> np.ndarray:
    """Рисует ROI-рамку, статус и инфо-панель на кадре."""
    rx, ry, rw, rh = roi
    color = TABLE_STATES[state]["color"]
    label = TABLE_STATES[state]["label"]

    # Рамка столика
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 3)

    # Метка состояния над рамкой
    text = f"Table: {label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (rx, ry - th - 12), (rx + tw + 10, ry), color, -1)
    cv2.putText(frame, text, (rx + 5, ry - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Инфо-панель (левый верхний угол)
    seconds = frame_idx / fps if fps > 0 else 0
    ts = str(timedelta(seconds=int(seconds)))
    info_lines = [
        f"Time: {ts}",
        f"Frame: {frame_idx}",
        f"Events: {len(event_log)}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# Основной пайплайн
# ──────────────────────────────────────────────

def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[VIDEO] {width}x{height}  |  {fps:.1f} FPS  |  {total} кадров")

    # Читаем первый кадр для выбора ROI
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Не удалось прочитать первый кадр.")
    roi = select_roi(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # перемотка в начало

    # Инициализация модели
    if YOLO_AVAILABLE:
        print("[MODEL] Загрузка YOLOv8n...")
        model = YOLO("yolov8n.pt")
        print("[MODEL] YOLOv8n загружена.\n")
        bg_subtractor = None
    else:
        model = None
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=50, detectShadows=True
        )

    # Выходное видео
    out_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # ── Состояние машины событий ──
    current_state  = "EMPTY"
    pending_state  = "EMPTY"
    debounce_count = 0

    # Лог событий: список словарей
    event_log = []

    # Время (в секундах от начала видео) последнего события "EMPTY"
    last_empty_time: float | None = None
    # Флаг: ждём следующего подхода после очередного "EMPTY"
    waiting_for_approach = False

    # Список задержек (пусто → подход) в секундах
    delays = []

    frame_idx = 0

    print("[PROCESS] Обработка видео...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_idx / fps

        # ── Детекция присутствия людей в ROI ──
        occupied = False
        if YOLO_AVAILABLE:
            results = model(frame, classes=[0], verbose=False)  # class 0 = person
            boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    boxes.append([x1, y1, x2, y2])
                    # Рисуем bounding box человека тонкой линией
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 1)
            occupied = person_in_roi(boxes, roi, frame.shape)
        else:
            fgmask = bg_subtractor.apply(frame)
            _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            occupied = motion_in_roi(fgmask, roi)

        # ── Дебаунс (сглаживание) ──
        raw_state = "OCCUPIED" if occupied else "EMPTY"
        if raw_state == pending_state:
            debounce_count += 1
        else:
            pending_state  = raw_state
            debounce_count = 1

        if debounce_count >= STATE_DEBOUNCE_FRAMES:
            new_state = pending_state
        else:
            new_state = current_state

        # ── Фиксация событий при смене состояния ──
        if new_state != current_state:
            ts_str = str(timedelta(seconds=int(current_time_sec)))

            if new_state == "OCCUPIED" and current_state == "EMPTY":
                # Подход к столику
                event_type = "APPROACHING"
                event_log.append({
                    "frame":      frame_idx,
                    "time_sec":   round(current_time_sec, 2),
                    "timestamp":  ts_str,
                    "event":      event_type,
                })
                print(f"  [{ts_str}] → ПОДХОД к столику")

                if waiting_for_approach and last_empty_time is not None:
                    delay = current_time_sec - last_empty_time
                    delays.append(delay)
                    print(f"           Задержка после ухода: {delay:.1f} сек")
                waiting_for_approach = False

            elif new_state == "EMPTY" and current_state == "OCCUPIED":
                # Стол освободился
                event_type = "EMPTY"
                event_log.append({
                    "frame":      frame_idx,
                    "time_sec":   round(current_time_sec, 2),
                    "timestamp":  ts_str,
                    "event":      event_type,
                })
                print(f"  [{ts_str}] → СТОЛ ПУСТОЙ")
                last_empty_time   = current_time_sec
                waiting_for_approach = True

            current_state = new_state

        # Определяем визуальное состояние (APPROACHING только в момент смены)
        display_state = current_state
        if (len(event_log) > 0 and event_log[-1]["event"] == "APPROACHING" and
                frame_idx - event_log[-1]["frame"] < int(fps * 2)):
            display_state = "APPROACHING"

        # ── Отрисовка ──
        frame = draw_overlay(frame, roi, display_state, fps, frame_idx, event_log)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 300 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  Прогресс: {frame_idx}/{total} кадров ({pct:.0f}%)")

    cap.release()
    writer.release()
    print(f"\n[DONE] Видео сохранено: {out_path}")

    # ──────────────────────────────────────────────
    # Аналитика
    # ──────────────────────────────────────────────
    df = pd.DataFrame(event_log)
    df.to_csv("events.csv", index=False, encoding="utf-8-sig")
    print(f"[CSV]  События сохранены: events.csv ({len(df)} записей)")

    avg_delay = np.mean(delays) if delays else None

    # Отчёт
    report_lines = [
        "=" * 55,
        "  ОТЧЁТ: Система детекции уборки столиков",
        "=" * 55,
        f"Видео:           {video_path}",
        f"Зона столика:    x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}",
        f"Метод детекции:  {'YOLOv8n' if YOLO_AVAILABLE else 'MOG2 (детекция движения)'}",
        f"Всего кадров:    {frame_idx}",
        f"Длительность:    {timedelta(seconds=int(frame_idx / fps))}",
        "-" * 55,
        f"Всего событий:   {len(event_log)}",
        f"  - Подходов:    {sum(1 for e in event_log if e['event'] == 'APPROACHING')}",
        f"  - Уходов:      {sum(1 for e in event_log if e['event'] == 'EMPTY')}",
        "-" * 55,
        f"Зафиксировано задержек: {len(delays)}",
    ]
    if delays:
        report_lines += [
            f"  Мин. задержка:    {min(delays):.1f} сек",
            f"  Макс. задержка:   {max(delays):.1f} сек",
            f"  Среднее время между уходом гостя",
            f"  и подходом следующего человека: {avg_delay:.1f} сек",
        ]
    else:
        report_lines.append("  Недостаточно данных для подсчёта задержки.")

    report_lines += [
        "=" * 55,
        "",
        "ТАБЛИЦА СОБЫТИЙ:",
        df.to_string(index=False) if not df.empty else "(нет событий)",
    ]

    report_text = "\n".join(report_lines)
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n[REPORT] Отчёт сохранён: report.txt")

    return avg_delay, df


# ──────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Прототип детекции уборки столиков по видео"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Путь к входному видеофайлу (например: video1.mp4)"
    )
    args = parser.parse_args()

    video_file = args.video
    if not Path(video_file).exists():
        print(f"[ERROR] Файл не найден: {video_file}")
        exit(1)

    print("=" * 55)
    print("  Детекция уборки столиков | Запуск")
    print("=" * 55)
    process_video(video_file)

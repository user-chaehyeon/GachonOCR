"""OpenCV preprocessing for photographed book/page OCR."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

def save_stage_image(output_dir: Path, stage_index: int, stage_name: str, image: np.ndarray) -> None:
    output_path = output_dir / f"{stage_index:02d}_{stage_name}.png"
    save_image(output_path, image)

# --------------------------------------------------------------------------
# 이미지 로드

def load_image(image_path: Path) -> np.ndarray:
    # 이미지 파일을 OpenCV가 읽을 수 있는 형식으로 로드 / 경로에 한글이 포함되어도 문제없도록 처리
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    # H × W × 3 BGR ndarray 반환
    return image


def save_image(image_path: Path, image: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    extension = image_path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"Cannot encode image as {extension}: {image_path}")
    encoded.tofile(str(image_path))
    
    
    
    
def draw_bbox_on_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    visual = image.copy()
    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        visual,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        4,
    )

    return visual


def draw_quad_on_image(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    visual = image.copy()
    quad = order_points(quad).astype(np.int32)

    cv2.polylines(
        visual,
        [quad],
        isClosed=True,
        color=(0, 255, 0),
        thickness=4,
    )

    for x, y in quad:
        cv2.circle(visual, (int(x), int(y)), 10, (0, 0, 255), -1)

    return visual


def draw_split_on_image(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    split_x: int | None,
    page_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    visual = draw_bbox_on_image(image, bbox)
    x1, y1, _, y2 = bbox

    if split_x is not None:
        absolute_x = x1 + split_x
        cv2.line(visual, (absolute_x, y1), (absolute_x, y2), (0, 255, 255), 4)

    if page_bboxes is not None:
        colors = ((255, 0, 0), (0, 128, 255))
        for index, page_bbox in enumerate(page_bboxes):
            px1, py1, px2, py2 = page_bbox
            cv2.rectangle(visual, (px1, py1), (px2, py2), colors[index % len(colors)], 3)

    return visual

# --------------------------------------------------------------------------
# 이미지 크기 리사이즈 / 축소: INTER_AREA, 확대: INTER_CUBIC
def resize_for_ocr(image: np.ndarray, target_long_side: int = 3200) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if abs(long_side - target_long_side) < 80:
        return image

    scale = target_long_side / long_side
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)

# --------------------------------------------------------------------------
# 문서 영역을 사각형 또는 사다리꼴로 검출
# x+y가 가장 작음 → 좌상단 / x+y가 가장 큼 → 우하단 / y-x가 가장 작음 → 우상단 / y-x가 가장 큼 → 좌하단
def order_points(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype("float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(4)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered

# --------------------------------------------------------------------------
# 검출된 점이 이미지 밖으로 나가지 않도록 제한
def clip_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    points = points.astype("float32")
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points

# --------------------------------------------------------------------------
# 검출된 4개의 점으로 문서 영역을 정면에서 본 것처럼 펼침
# 검출된 문서가 사다리꼴일 경우 직사각형 문서 이미지로 변환
def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = order_points(points)

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    max_width = max(1, int(max(width_a, width_b)))

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_height = max(1, int(max(height_a, height_b)))

    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32"),
        destination,
    )
    return cv2.warpPerspective(image, matrix, (max_width, max_height), borderMode=cv2.BORDER_REPLICATE)

# --------------------------------------------------------------------------
# 검출된 사각형을 중심 기준으로 확장
def expand_quad(points: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int], margin_ratio: float = 0.015) -> np.ndarray:
    height, width = image_shape[:2]
    ordered = order_points(points)
    center = ordered.mean(axis=0)
    expanded = ordered + (ordered - center) * margin_ratio
    return clip_points(expanded, width, height)

# --------------------------------------------------------------------------
# 검출된 사각형 면적이 전체 이미지에서 어느정도 비율인지 계산
def quad_area_ratio(points: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int]) -> float:
    height, width = image_shape[:2]
    area = abs(cv2.contourArea(order_points(points)))
    return area / float(height * width)

# --------------------------------------------------------------------------
# 쓸만한 정도로 검출되었는지 판단 (너무 작거나 너무 커서 실패한 경우 제외)
def is_useful_page_quad(points: np.ndarray | None, image_shape: tuple[int, int, int] | tuple[int, int]) -> bool:
    if points is None:
        return False

    ratio = quad_area_ratio(points, image_shape)
    if ratio < 0.12:
        return False

    # A full-frame rectangle means detection failed to isolate the page/book.
    if ratio > 0.92:
        return False

    return True

# --------------------------------------------------------------------------
# 선분의 각도를 defree 단위로 계산 - 수평선인지 수직선인지 판단
def line_angle_degrees(x1: int, y1: int, x2: int, y2: int) -> float:
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

# --------------------------------------------------------------------------
# 선분의 길이 계산 - 짧은 선분은 문서의 경계가 아닐 가능성이 높음 -> 필터링
def line_length(x1: int, y1: int, x2: int, y2: int) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))

# --------------------------------------------------------------------------
# 선분의 두 끝점을 높이 기준으로 정렬하여 반환 (Hough 변환으로 검출된 선분을 상하 또는 좌우로 구분하기 위함)
def line_endpoint_pair(line: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = line
    p1 = np.array([x1, y1], dtype="float32")
    p2 = np.array([x2, y2], dtype="float32")
    if p1[1] <= p2[1]:
        return p1, p2
    return p2, p1

# --------------------------------------------------------------------------
# 두 점에서 직선의 계수 계산 (ax + by + c = 0 형태) - 선분이 긴 경우에 더 안정적으로 검출된 선을 표현할 수 있음
def line_coefficients(p1: np.ndarray, p2: np.ndarray) -> np.ndarray | None:
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = float(np.hypot(a, b))
    if norm < 1e-6:
        return None
    return np.array([a / norm, b / norm, c / norm], dtype="float32")

# --------------------------------------------------------------------------
# 직선 교점 계산
def intersect_lines(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    cross = np.cross(line_a, line_b)
    if abs(float(cross[2])) < 1e-6:
        return None
    return np.array([cross[0] / cross[2], cross[1] / cross[2]], dtype="float32")

# --------------------------------------------------------------------------
# 하나의 문서가 여러개로 나뉘어 검출되는 경우, 하나의 경계선으로 근사
def fit_line_from_segments(segments: list[tuple[int, int, int, int]]) -> np.ndarray | None:
    if not segments:
        return None

    points = []
    for x1, y1, x2, y2 in segments:
        points.append([x1, y1])
        points.append([x2, y2])

    point_array = np.array(points, dtype="float32")
    vx, vy, x0, y0 = cv2.fitLine(point_array, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    p1 = np.array([x0 - vx * 1000, y0 - vy * 1000], dtype="float32")
    p2 = np.array([x0 + vx * 1000, y0 + vy * 1000], dtype="float32")
    return line_coefficients(p1, p2)

# --------------------------------------------------------------------------
# 허프 라인으로 검출된 선분들을 통해 상하, 좌우 경계 후보 선택
def select_border_segments(
    segments: list[dict[str, object]],
    orientation: str,
    image_width: int,
    image_height: int,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]] | None:
    if orientation == "horizontal":
        min_length = image_width * 0.68
        coordinate_key = "mid_y"
        low_limit = image_height * 0.28
        high_limit = image_height * 0.72
    else:
        min_length = image_height * 0.38
        coordinate_key = "mid_x"
        low_limit = image_width * 0.40
        high_limit = image_width * 0.55

    valid = [segment for segment in segments if float(segment["length"]) >= min_length]
    low_side = [segment for segment in valid if float(segment[coordinate_key]) <= low_limit]
    high_side = [segment for segment in valid if float(segment[coordinate_key]) >= high_limit]
    if not low_side or not high_side:
        return None

    low_seed = max(low_side, key=lambda segment: float(segment["length"]))
    high_seed = max(high_side, key=lambda segment: float(segment["length"]))
    tolerance = image_height * 0.055 if orientation == "horizontal" else image_width * 0.055

    low_coordinate = float(low_seed[coordinate_key])
    high_coordinate = float(high_seed[coordinate_key])
    low_cluster = [
        segment["line"]  # type: ignore[index]
        for segment in valid
        if abs(float(segment[coordinate_key]) - low_coordinate) <= tolerance
    ]
    high_cluster = [
        segment["line"]  # type: ignore[index]
        for segment in valid
        if abs(float(segment[coordinate_key]) - high_coordinate) <= tolerance
    ]

    return low_cluster, high_cluster

# --------------------------------------------------------------------------
# 문서 테두리를 허프 변환으로 검출 (수평 수직 선분 분리 후 상, 하, 좌, 우, 경계선 선택하여 피팅 -> 사각형 반환)
def find_page_quad_by_hough_borders(image: np.ndarray) -> np.ndarray | None:
    ratio = image.shape[0] / 1100.0
    small = cv2.resize(image, (int(image.shape[1] / ratio), 1100), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 25, 90, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=int(min(small.shape[:2]) * 0.25),
        maxLineGap=70,
    )
    if lines is None:
        return None

    horizontal_segments: list[dict[str, object]] = []
    vertical_segments: list[dict[str, object]] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        angle = line_angle_degrees(x1, y1, x2, y2)
        length = line_length(x1, y1, x2, y2)
        segment = {
            "line": (x1, y1, x2, y2),
            "length": length,
            "mid_x": (x1 + x2) / 2.0,
            "mid_y": (y1 + y2) / 2.0,
        }

        horizontal_angle = min(abs(angle), abs(abs(angle) - 180))
        vertical_angle = abs(abs(angle) - 90)
        if horizontal_angle <= 18:
            horizontal_segments.append(segment)
        elif vertical_angle <= 18:
            vertical_segments.append(segment)

    horizontal_pair = select_border_segments(horizontal_segments, "horizontal", small.shape[1], small.shape[0])
    vertical_pair = select_border_segments(vertical_segments, "vertical", small.shape[1], small.shape[0])
    if horizontal_pair is None or vertical_pair is None:
        return None

    top_segments, bottom_segments = horizontal_pair
    left_segments, right_segments = vertical_pair
    top_line = fit_line_from_segments(top_segments)
    bottom_line = fit_line_from_segments(bottom_segments)
    left_line = fit_line_from_segments(left_segments)
    right_line = fit_line_from_segments(right_segments)
    if top_line is None or bottom_line is None or left_line is None or right_line is None:
        return None

    top_left = intersect_lines(top_line, left_line)
    top_right = intersect_lines(top_line, right_line)
    bottom_right = intersect_lines(bottom_line, right_line)
    bottom_left = intersect_lines(bottom_line, left_line)
    if top_left is None or top_right is None or bottom_right is None or bottom_left is None:
        return None

    quad = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32") * ratio
    quad = expand_quad(quad, image.shape, margin_ratio=0.006)
    if not is_useful_page_quad(quad, image.shape):
        return None

    ordered = order_points(quad)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    if min(top_width, bottom_width, left_height, right_height) < 80:
        return None

    width_ratio = max(top_width, bottom_width) / max(1.0, min(top_width, bottom_width))
    height_ratio = max(left_height, right_height) / max(1.0, min(left_height, right_height))
    if width_ratio > 2.0 or height_ratio > 2.0:
        return None

    return ordered

# --------------------------------------------------------------------------
# 서로 충분히 떨어진 선분으로 검출된 페이지 테두리를 피팅하여 사각형으로 반환 (허프 변환으로 검출된 선분이 너무 많거나, 문서가 여러개로 나뉘어 검출되는 경우)
def find_page_quad_by_border_lines(image: np.ndarray) -> np.ndarray | None:
    ratio = image.shape[0] / 1000.0
    small = cv2.resize(image, (int(image.shape[1] / ratio), 1000), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 20, 80, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=int(small.shape[0] * 0.35),
        maxLineGap=50,
    )
    if lines is None:
        return None

    candidates: list[dict[str, object]] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = line_angle_degrees(int(x1), int(y1), int(x2), int(y2))
        length = line_length(int(x1), int(y1), int(x2), int(y2))
        vertical_angle = abs(abs(angle) - 90)
        if vertical_angle > 15 or length < small.shape[0] * 0.40:
            continue

        top, bottom = line_endpoint_pair((int(x1), int(y1), int(x2), int(y2)))
        center_x = float((top[0] + bottom[0]) / 2.0)
        candidates.append({"line": (top, bottom), "center_x": center_x, "length": length})

    if len(candidates) < 2:
        return None

    best_pair: tuple[dict[str, object], dict[str, object]] | None = None
    best_score = -1.0
    page_width = small.shape[1]
    page_height = small.shape[0]

    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            left_x = float(left["center_x"])
            right_x = float(right["center_x"])
            separation = abs(right_x - left_x)
            if separation < page_width * 0.35:
                continue

            left_top, left_bottom = left["line"]  # type: ignore[assignment]
            right_top, right_bottom = right["line"]  # type: ignore[assignment]
            vertical_overlap = min(float(left_bottom[1]), float(right_bottom[1])) - max(float(left_top[1]), float(right_top[1]))
            if vertical_overlap < page_height * 0.35:
                continue

            score = separation + 0.25 * (float(left["length"]) + float(right["length"]))
            if score > best_score:
                best_score = score
                best_pair = (left, right)

    if best_pair is None:
        return None

    first, second = best_pair
    if float(first["center_x"]) <= float(second["center_x"]):
        left, right = first, second
    else:
        left, right = second, first

    left_top, left_bottom = left["line"]  # type: ignore[assignment]
    right_top, right_bottom = right["line"]  # type: ignore[assignment]
    quad = np.array([left_top, right_top, right_bottom, left_bottom], dtype="float32") * ratio
    quad = expand_quad(quad, image.shape, margin_ratio=0.01)

    if not is_useful_page_quad(quad, image.shape):
        return None
    return quad

# --------------------------------------------------------------------------
# 위의 두 방법으로도 페이지 테두리가 검출되지 않는 경우, 전경-배경 분할을 통해 페이지 영역을 검출하는 방법
# (조명 보정된 그레이스케일에서 적응적 임계처리로 텍스트 영역과 배경을 분리하여 가장 큰 컨투어를 페이지 영역으로 간주)
def find_page_bbox_by_border_lines(image: np.ndarray) -> tuple[int, int, int, int] | None:
    quad = find_page_quad_by_hough_borders(image)
    if quad is None:
        quad = find_page_quad_by_border_lines(image)
    if quad is None:
        return None

    ordered = order_points(quad)
    x1 = max(0, int(np.min(ordered[:, 0])))
    y1 = max(0, int(np.min(ordered[:, 1])))
    x2 = min(image.shape[1], int(np.max(ordered[:, 0])))
    y2 = min(image.shape[0], int(np.max(ordered[:, 1])))

    width = x2 - x1
    height = y2 - y1
    if width < 80 or height < 80:
        return None

    return x1, y1, x2, y2

# --------------------------------------------------------------------------
# Canny edge와 contour 기반 문서 외곽 사각형 검출
def find_page_quad_by_edges(image: np.ndarray) -> np.ndarray | None:
    ratio = image.shape[0] / 900.0
    small = cv2.resize(image, (int(image.shape[1] / ratio), 900), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = small.shape[0] * small.shape[1]

    for contour in contours[:12]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.18:
            continue

        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.015, 0.02, 0.03, 0.04, 0.06):
            approximation = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(approximation) == 4 and cv2.isContourConvex(approximation):
                quad = approximation.reshape(4, 2).astype("float32") * ratio
                quad = expand_quad(quad, image.shape)
                if is_useful_page_quad(quad, image.shape):
                    return quad

    return None

# --------------------------------------------------------------------------
# 밝기 기반 종이 영역 검출
def find_page_quad_by_foreground(image: np.ndarray) -> np.ndarray | None:
    ratio = image.shape[0] / 900.0
    small = cv2.resize(image, (int(image.shape[1] / ratio), 900), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]

    background = cv2.medianBlur(lightness, 51)
    normalized = cv2.divide(lightness, background, scale=255)
    _, mask = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(mask == 255) < 0.45:
        mask = cv2.bitwise_not(mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = small.shape[0] * small.shape[1]
    large_contours = [contour for contour in contours if cv2.contourArea(contour) > image_area * 0.18]
    if not large_contours:
        return None

    merged = np.vstack(large_contours)
    rect = cv2.minAreaRect(merged)
    quad = cv2.boxPoints(rect).astype("float32") * ratio
    quad = expand_quad(quad, image.shape)
    if not is_useful_page_quad(quad, image.shape):
        return None
    return quad

# --------------------------------------------------------------------------
# 문서 영역 검출 통합 (1. 윤곽선 기반 - 2. 허프 라인 기반 - 3. 밝기 기반 순으로 시도하여 검출된 사각형이 유용한지 판단하여 반환)
def find_document_or_book_quad(image: np.ndarray) -> np.ndarray | None:
    """Return four corners for a page, document, or two-page book spread."""
    edge_quad = find_page_quad_by_edges(image)
    if edge_quad is not None:
        return edge_quad

    hough_quad = find_page_quad_by_hough_borders(image)
    if hough_quad is not None:
        return hough_quad

    return find_page_quad_by_foreground(image)

# --------------------------------------------------------------------------
# 텍스트 후보 픽셀 마스크 생성
def build_text_candidate_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        17,
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    mask = np.zeros_like(binary)
    image_area = image.shape[0] * image.shape[1]
    for index in range(1, component_count):
        x, y, width, height, area = stats[index]
        if area < 6 or area > image_area * 0.01:
            continue
        if width < 2 or height < 2:
            continue
        if width > image.shape[1] * 0.20 or height > image.shape[0] * 0.08:
            continue
        mask[labels == index] = 255

    return mask

# --------------------------------------------------------------------------
# 텍스트가 실제로 몰려 있는 영역의 바운딩 박스 검출
def text_content_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    ratio = image.shape[0] / 1200.0
    small = cv2.resize(image, (int(image.shape[1] / ratio), 1200), interpolation=cv2.INTER_AREA)
    text_mask = build_text_candidate_mask(small)

    # Join characters into text blocks while keeping unrelated keyboard labels apart.
    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 9))
    block_mask = cv2.dilate(text_mask, block_kernel, iterations=2)
    contours, _ = cv2.findContours(block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    small_area = small.shape[0] * small.shape[1]
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < small_area * 0.002:
            continue
        if y < small.shape[0] * 0.20 and (y + height) < small.shape[0] * 0.32:
            # Usually keyboard labels or clipped top noise, not page text.
            continue
        if x <= 2 and width < small.shape[1] * 0.35:
            # Exclude a narrow sliver of the opposite page at the image edge.
            continue
        boxes.append((x, y, x + width, y + height, area))

    if not boxes:
        return None

    # Keep blocks near the densest text area instead of merging distant opposite-page slivers.
    largest = max(boxes, key=lambda item: item[4])
    lx1, ly1, lx2, ly2, _ = largest
    selected = []
    for x1, y1, x2, y2, area in boxes:
        horizontal_overlap = max(0, min(lx2, x2) - max(lx1, x1)) / max(1, min(lx2 - lx1, x2 - x1))
        center_distance = abs(((x1 + x2) / 2) - ((lx1 + lx2) / 2))
        if horizontal_overlap > 0.25 or center_distance < small.shape[1] * 0.22 or area > largest[4] * 0.35:
            selected.append((x1, y1, x2, y2))

    x1 = min(box[0] for box in selected)
    y1 = min(box[1] for box in selected)
    x2 = max(box[2] for box in selected)
    y2 = max(box[3] for box in selected)

    margin_x = int((x2 - x1) * 0.16)
    margin_y_top = int((y2 - y1) * 0.22)
    margin_y_bottom = int((y2 - y1) * 0.12)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y_top)
    x2 = min(small.shape[1], x2 + margin_x)
    y2 = min(small.shape[0], y2 + margin_y_bottom)

    return (
        int(x1 * ratio),
        int(y1 * ratio),
        int(x2 * ratio),
        int(y2 * ratio),
    )

# --------------------------------------------------------------------------
# 책의 반대쪽 페이지가 검출되는 경우, 텍스트가 더 몰려있는 영역만 선택
def trim_opposite_page_sliver(image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    if x1 > image.shape[1] * 0.03:
        return bbox

    crop = image[y1:y2, x1:x2]
    if crop.shape[1] < 300:
        return bbox

    text_mask = build_text_candidate_mask(crop)
    text_projection = np.mean(text_mask > 0, axis=0)
    smooth_width = max(15, crop.shape[1] // 80)
    kernel = np.ones(smooth_width, dtype="float32") / smooth_width
    smoothed = np.convolve(text_projection, kernel, mode="same")

    search_limit = int(crop.shape[1] * 0.30)
    active = smoothed > max(0.003, np.percentile(smoothed, 80) * 0.35)
    blank_runs = []
    start = None
    for idx in range(0, search_limit):
        if not active[idx] and start is None:
            start = idx
        elif active[idx] and start is not None:
            if idx - start > crop.shape[1] * 0.035:
                blank_runs.append((start, idx))
            start = None
    if start is not None and search_limit - start > crop.shape[1] * 0.035:
        blank_runs.append((start, search_limit))

    if not blank_runs:
        return bbox

    # Trim after the last early blank gutter only when text exists on both sides.
    gutter_start, gutter_end = blank_runs[-1]
    left_text = np.sum(text_mask[:, :gutter_start] > 0)
    right_text = np.sum(text_mask[:, gutter_end:] > 0)
    if left_text > 0 and right_text > left_text * 2.5:
        safe_margin = int(crop.shape[1] * 0.10)
        x1 += max(0, gutter_end - safe_margin)

    return x1, y1, x2, y2

# --------------------------------------------------------------------------
# 밝은 픽셀 비율이 높음 / 채도 낮음 / 어두운 영역 적음 -> 검출된 영역이 이미 문서 스캔본처럼 보일 경우 원본 이미지 그대로 사용
def is_already_document_like(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    bright_ratio = float(np.mean(gray > 175))
    low_saturation_ratio = float(np.mean(saturation < 35))
    dark_ratio = float(np.mean(gray < 45))

    return bright_ratio > 0.72 and low_saturation_ratio > 0.88 and dark_ratio < 0.18

# --------------------------------------------------------------------------
# 검출된 바운딩 박스가 이미지 경계를 벗어나지 않도록 조정, 너무 작은 경우 None 반환
def clamp_bbox(bbox: tuple[int, int, int, int], image: np.ndarray) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(image.shape[1] - 1, x1))
    y1 = max(0, min(image.shape[0] - 1, y1))
    x2 = max(0, min(image.shape[1], x2))
    y2 = max(0, min(image.shape[0], y2))
    if x2 - x1 < 80 or y2 - y1 < 80:
        return None
    return x1, y1, x2, y2


def bbox_from_quad(quad: np.ndarray, image: np.ndarray) -> tuple[int, int, int, int] | None:
    ordered = order_points(quad)
    bbox = (
        int(np.min(ordered[:, 0])),
        int(np.min(ordered[:, 1])),
        int(np.max(ordered[:, 0])),
        int(np.max(ordered[:, 1])),
    )
    return clamp_bbox(bbox, image)


def score_crop_bbox(image: np.ndarray, text_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    text_pixels = cv2.countNonZero(text_mask[y1:y2, x1:x2])
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    dark_ratio = float(np.mean(gray < 45))
    very_dark_ratio = float(np.mean(gray < 25))
    edge_penalty = 0.12 if x1 == 0 or y1 == 0 or x2 == image.shape[1] or y2 == image.shape[0] else 0.0
    fill_score = text_pixels / np.sqrt(area)
    top_penalty = 80.0 if y1 < image.shape[0] * 0.08 and very_dark_ratio > 0.05 else 0.0
    return float(fill_score - dark_ratio * 260.0 - very_dark_ratio * 400.0 - edge_penalty * 70.0 - top_penalty)


def find_best_dynamic_crop_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    if is_already_document_like(image):
        return 0, 0, image.shape[1], image.shape[0]

    candidates: list[tuple[int, int, int, int]] = []

    text_bbox = text_content_bbox(image)
    if text_bbox is not None:
        candidates.append(text_bbox)

    document_quad = find_document_or_book_quad(image)
    if document_quad is not None:
        quad_bbox = bbox_from_quad(document_quad, image)
        if quad_bbox is not None:
            candidates.append(quad_bbox)

    line_bbox = find_page_bbox_by_border_lines(image)
    if line_bbox is not None:
        candidates.append(line_bbox)

    ratio = image.shape[0] / 1000.0
    small = cv2.resize(image, (int(image.shape[1] / ratio), 1000), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(185, int(otsu_threshold) + 35)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area_ratio = cv2.contourArea(contour) / float(small.shape[0] * small.shape[1])
        if area_ratio < 0.10:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        bbox = clamp_bbox(
            (
                int((x - width * 0.02) * ratio),
                int((y - height * 0.02) * ratio),
                int((x + width * 1.02) * ratio),
                int((y + height * 1.02) * ratio),
            ),
            image,
        )
        if bbox is not None:
            candidates.append(bbox)

    if not candidates:
        return None

    text_mask = build_text_candidate_mask(image)
    unique_candidates = list(dict.fromkeys(candidates))
    return max(unique_candidates, key=lambda bbox: score_crop_bbox(image, text_mask, bbox))


def trim_dark_outer_borders(image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.shape[0] < 120 or crop.shape[1] < 120:
        return bbox

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    paper_like = (gray > 130) & (hsv[:, :, 1] < 105)
    row_brightness = np.mean(paper_like, axis=1)
    col_brightness = np.mean(paper_like, axis=0)

    def first_stable(values: np.ndarray, threshold: float, window: int) -> int:
        for index in range(0, max(1, len(values) - window)):
            if float(np.mean(values[index : index + window])) >= threshold:
                return index
        return 0

    def last_stable(values: np.ndarray, threshold: float, window: int) -> int:
        for index in range(len(values) - window, 0, -1):
            if float(np.mean(values[index : index + window])) >= threshold:
                return index + window
        return len(values)

    top = first_stable(row_brightness, 0.48, max(12, crop.shape[0] // 90))
    bottom = last_stable(row_brightness, 0.42, max(12, crop.shape[0] // 90))
    left = first_stable(col_brightness, 0.36, max(8, crop.shape[1] // 100))
    right = last_stable(col_brightness, 0.34, max(8, crop.shape[1] // 100))

    margin_y = max(8, int(crop.shape[0] * 0.015))
    margin_x = max(8, int(crop.shape[1] * 0.015))
    top = max(0, top - margin_y)
    bottom = min(crop.shape[0], bottom + margin_y)
    left = max(0, left - margin_x)
    right = min(crop.shape[1], right + margin_x)

    if right - left < crop.shape[1] * 0.45 or bottom - top < crop.shape[0] * 0.45:
        return bbox

    return x1 + left, y1 + top, x1 + right, y1 + bottom


def _smooth_projection(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype="float32") / float(window)
    return np.convolve(values.astype("float32"), kernel, mode="same")


def _normalize_projection(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, [5, 95])
    if high - low < 1e-6:
        return np.zeros_like(values, dtype="float32")
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype("float32")


def find_book_spine_x(crop: np.ndarray) -> int | None:
    """Find the center gutter/spine x-coordinate inside an already-cropped book spread."""
    height, width = crop.shape[:2]
    if width < 360 or height < 360:
        return None
    if width / float(max(1, height)) < 1.08:
        return None

    target_height = 900
    if height > target_height:
        scale = target_height / float(height)
        small_width = max(1, int(round(width * scale)))
        small = cv2.resize(crop, (small_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = crop.copy()

    small_height, small_width = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=max(2.0, small_width / 260.0),
        sigmaY=max(2.0, small_height / 220.0),
    )

    y1 = int(small_height * 0.04)
    y2 = int(small_height * 0.96)
    text_mask = build_text_candidate_mask(small)

    brightness = np.mean(gray[y1:y2, :], axis=0)
    dark_projection = 255.0 - brightness
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    edge_projection = np.mean(sobel_x[y1:y2, :], axis=0)
    text_projection = np.mean(text_mask[y1:y2, :] > 0, axis=0)

    smooth_width = max(9, small_width // 90)
    dark_projection = _smooth_projection(dark_projection, smooth_width)
    edge_projection = _smooth_projection(edge_projection, smooth_width)
    text_projection = _smooth_projection(text_projection, smooth_width)

    dark_score = _normalize_projection(dark_projection)
    edge_score = _normalize_projection(edge_projection)
    text_penalty = _normalize_projection(text_projection)
    x_positions = np.arange(small_width, dtype="float32")
    center_x = (small_width - 1) / 2.0
    center_score = 1.0 - np.clip(np.abs(x_positions - center_x) / max(1.0, small_width * 0.22), 0.0, 1.0)

    score = dark_score * 0.50 + edge_score * 0.22 + center_score * 0.35 - text_penalty * 0.28
    search_left = int(small_width * 0.38)
    search_right = int(small_width * 0.62)
    if search_right <= search_left:
        return None

    best_index = int(search_left + np.argmax(score[search_left:search_right]))
    best_score = float(score[best_index])
    band_median = float(np.median(score[search_left:search_right]))
    spine_evidence = float(dark_score[best_index] + edge_score[best_index])
    text_density = float(np.mean(text_mask > 0))

    if text_density < 0.001:
        return None
    if best_score < 0.42 or best_score < band_median + 0.10:
        return None
    if spine_evidence < 0.55:
        return None

    return int(round(best_index / scale))


def split_book_bbox_by_spine(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int, int, int]], int | None]:
    bbox = trim_dark_outer_borders(image, bbox)
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    split_x = find_book_spine_x(crop)
    if split_x is None:
        return [trim_opposite_page_sliver(image, bbox)], None

    width = x2 - x1
    gutter_padding = max(2, int(width * 0.004))
    left_bbox = clamp_bbox((x1, y1, x1 + split_x - gutter_padding, y2), image)
    right_bbox = clamp_bbox((x1 + split_x + gutter_padding, y1, x2, y2), image)
    if left_bbox is None or right_bbox is None:
        return [trim_opposite_page_sliver(image, bbox)], None

    left_width = left_bbox[2] - left_bbox[0]
    right_width = right_bbox[2] - right_bbox[0]
    if min(left_width, right_width) < width * 0.28:
        return [trim_opposite_page_sliver(image, bbox)], None

    page_bboxes = []
    for page_bbox in (left_bbox, right_bbox):
        trimmed = trim_dark_outer_borders(image, page_bbox)
        clamped = clamp_bbox(trimmed, image)
        if clamped is not None:
            page_bboxes.append(clamped)

    if len(page_bboxes) != 2:
        return [trim_opposite_page_sliver(image, bbox)], None

    return page_bboxes, split_x


def find_book_page_bboxes(image: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    crop_bbox = find_best_dynamic_crop_bbox(image)
    if crop_bbox is None:
        return None
    page_bboxes, _ = split_book_bbox_by_spine(image, crop_bbox)
    return page_bboxes


def safe_crop_page_by_brightness(image: np.ndarray) -> np.ndarray:
    crop_bbox = find_best_dynamic_crop_bbox(image)
    if crop_bbox is None:
        return image

    crop_bbox = trim_dark_outer_borders(image, crop_bbox)
    crop_bbox = trim_opposite_page_sliver(image, crop_bbox)
    x1, y1, x2, y2 = crop_bbox
    return image[y1:y2, x1:x2]


def crop_document_or_book(image: np.ndarray, enabled: bool = True, mode: str = "safe") -> np.ndarray:
    if not enabled:
        return image

    if mode == "safe":
        return safe_crop_page_by_brightness(image)

    quad = find_document_or_book_quad(image)
    if quad is None:
        return safe_crop_page_by_brightness(image)

    warped = four_point_transform(image, quad)
    if warped.shape[0] < 80 or warped.shape[1] < 80:
        return safe_crop_page_by_brightness(image)
    return warped


def crop_document_or_book_pages(
    image: np.ndarray,
    enabled: bool = True,
    mode: str = "safe",
    split_pages: bool = True,
) -> list[np.ndarray]:
    if not enabled:
        return [image]

    if mode == "safe":
        crop_bbox = find_best_dynamic_crop_bbox(image)
        if crop_bbox is None:
            return [image]

        if split_pages:
            page_bboxes, _ = split_book_bbox_by_spine(image, crop_bbox)
            return [image[y1:y2, x1:x2] for x1, y1, x2, y2 in page_bboxes]

        crop_bbox = trim_dark_outer_borders(image, crop_bbox)
        crop_bbox = trim_opposite_page_sliver(image, crop_bbox)
        x1, y1, x2, y2 = crop_bbox
        return [image[y1:y2, x1:x2]]

    quad = find_document_or_book_quad(image)
    if quad is None:
        return crop_document_or_book_pages(image, enabled=True, mode="safe", split_pages=split_pages)

    warped = four_point_transform(image, quad)
    if warped.shape[0] < 80 or warped.shape[1] < 80:
        return crop_document_or_book_pages(image, enabled=True, mode="safe", split_pages=split_pages)

    if not split_pages:
        return [warped]

    split_x = find_book_spine_x(warped)
    if split_x is None:
        return [warped]

    gutter_padding = max(2, int(warped.shape[1] * 0.004))
    left = warped[:, : max(1, split_x - gutter_padding)]
    right = warped[:, min(warped.shape[1] - 1, split_x + gutter_padding) :]
    if left.shape[1] < warped.shape[1] * 0.28 or right.shape[1] < warped.shape[1] * 0.28:
        return [warped]
    return [left, right]

# --------------------------------------------------------------------------
# binary 반전 → 글자 픽셀 좌표 추출 → cv2.minAreaRect로 전체 텍스트 덩어리 각도 추정 → ±10도 이내면 회전 보정
def deskew(binary_image: np.ndarray) -> np.ndarray:
    inverted = cv2.bitwise_not(binary_image)
    coordinates = np.column_stack(np.where(inverted > 0))
    if len(coordinates) < 100:
        return binary_image

    angle = cv2.minAreaRect(coordinates)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.25 or abs(angle) > 10:
        return binary_image

    height, width = binary_image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary_image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

# --------------------------------------------------------------------------
# 텍스트 줄 자체를 기준으로 기울기를 보정
def deskew_text_lines(binary_image: np.ndarray) -> np.ndarray:
    inverted = cv2.bitwise_not(binary_image)
    height, width = binary_image.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    line_mask = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    edges = cv2.Canny(line_mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, width // 35),
        minLineLength=max(40, width // 12),
        maxLineGap=max(8, width // 80),
    )
    if lines is None:
        return deskew(binary_image)

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = line_angle_degrees(int(x1), int(y1), int(x2), int(y2))
        if -12.0 <= angle <= 12.0:
            angles.append(angle)

    if len(angles) < 3:
        return deskew(binary_image)

    angle = float(np.median(angles))
    if abs(angle) < 0.25:
        return binary_image

    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary_image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

# --------------------------------------------------------------------------
# 최종 전처리 함수

# def preprocess_for_tesseract(image: np.ndarray, 
#                              crop_document: bool = True, 
#                              crop_mode: str = "safe") -> np.ndarray:
#     image = resize_for_ocr(image)
#     image = crop_document_or_book(image, enabled=crop_document, 
#                                   mode=crop_mode)

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)

#     binary = cv2.adaptiveThreshold(
#         gray,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         35,
#         11,
#     )

#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
#     return deskew_text_lines(binary)


def _prepare_page_for_tesseract(
    image: np.ndarray,
    save_stages: bool = False,
    stage_output_dir: Path | None = None,
) -> np.ndarray:
    if save_stages and stage_output_dir is not None:
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        save_stage_image(stage_output_dir, 1, "page_input", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if save_stages and stage_output_dir is not None:
        save_stage_image(stage_output_dir, 2, "grayscale", gray)

    gray = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=12,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    if save_stages and stage_output_dir is not None:
        save_stage_image(stage_output_dir, 3, "denoised", gray)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    if save_stages and stage_output_dir is not None:
        save_stage_image(stage_output_dir, 4, "binary", binary)

    deskewed = deskew_text_lines(binary)
    if save_stages and stage_output_dir is not None:
        save_stage_image(stage_output_dir, 5, "deskewed", deskewed)

    return deskewed


def preprocess_pages_for_tesseract(
    image: np.ndarray,
    crop_document: bool = True,
    crop_mode: str = "safe",
    split_pages: bool = True,
    save_stages: bool = False,
    stage_output_dir: Path | None = None,
) -> list[np.ndarray]:
    if save_stages:
        if stage_output_dir is None:
            stage_output_dir = Path("preprocess_stages")
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        save_stage_image(stage_output_dir, 1, "input", image)

    image = resize_for_ocr(image)
    if save_stages and stage_output_dir is not None:
        save_stage_image(stage_output_dir, 2, "resized", image)

    page_images = [image]
    if crop_document:
        if crop_mode == "safe":
            crop_bbox = find_best_dynamic_crop_bbox(image)

            if crop_bbox is not None:
                if save_stages and stage_output_dir is not None:
                    visual = draw_bbox_on_image(image, crop_bbox)
                    save_stage_image(stage_output_dir, 3, "detected_bbox", visual)

                if split_pages:
                    final_bbox = trim_dark_outer_borders(image, crop_bbox)
                    page_bboxes, split_x = split_book_bbox_by_spine(image, crop_bbox)

                    if save_stages and stage_output_dir is not None:
                        visual = draw_split_on_image(image, final_bbox, split_x, page_bboxes)
                        save_stage_image(stage_output_dir, 4, "page_split_bboxes", visual)

                    page_images = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in page_bboxes]
                else:
                    crop_bbox = trim_dark_outer_borders(image, crop_bbox)
                    crop_bbox = trim_opposite_page_sliver(image, crop_bbox)

                    if save_stages and stage_output_dir is not None:
                        visual = draw_bbox_on_image(image, crop_bbox)
                        save_stage_image(stage_output_dir, 4, "final_crop_bbox", visual)

                    x1, y1, x2, y2 = crop_bbox
                    page_images = [image[y1:y2, x1:x2]]

            elif save_stages and stage_output_dir is not None:
                save_stage_image(stage_output_dir, 3, "detected_bbox_failed", image)

        else:
            page_images = crop_document_or_book_pages(
                image,
                enabled=True,
                mode=crop_mode,
                split_pages=split_pages,
            )

    if save_stages and stage_output_dir is not None:
        for index, page_image in enumerate(page_images, start=1):
            suffix = "cropped" if len(page_images) == 1 else f"cropped_page_{index:02d}"
            save_stage_image(stage_output_dir, 5 + index - 1, suffix, page_image)

    processed_pages = []
    for index, page_image in enumerate(page_images, start=1):
        page_stage_dir = None
        if save_stages and stage_output_dir is not None:
            if len(page_images) == 1:
                page_stage_dir = stage_output_dir / "page"
            else:
                page_stage_dir = stage_output_dir / f"page_{index:02d}"
        processed_pages.append(
            _prepare_page_for_tesseract(
                page_image,
                save_stages=save_stages,
                stage_output_dir=page_stage_dir,
            )
        )

    return processed_pages


def preprocess_for_tesseract(
    image: np.ndarray,
    crop_document: bool = True,
    crop_mode: str = "safe",
    save_stages: bool = False,
    stage_output_dir: Path | None = None,
) -> np.ndarray:
    pages = preprocess_pages_for_tesseract(
        image,
        crop_document=crop_document,
        crop_mode=crop_mode,
        split_pages=False,
        save_stages=save_stages,
        stage_output_dir=stage_output_dir,
    )
    return pages[0]

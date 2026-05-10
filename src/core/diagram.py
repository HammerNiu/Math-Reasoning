"""Lightweight diagram parsing helpers for AMC/competition geometry.

The AMC dataset often embeds Asymptote snippets.  We do not try to execute
Asymptote.  Instead, this module extracts the pieces that are reliable and
useful for solving: declared numeric variables, labeled points, filled paths,
drawn polygons, and circles.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, ...]


def extract_asy_blocks(text: str) -> List[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"```asy\s*(.*?)```", text or "", re.IGNORECASE | re.DOTALL)
    ]


def strip_asy_blocks(text: str) -> str:
    return re.sub(r"```asy\s*.*?```", " [diagram omitted] ", text or "", flags=re.IGNORECASE | re.DOTALL)


def parse_diagram_features(text: str) -> Dict[str, object]:
    variables: Dict[str, float] = {}
    points: Dict[str, Point] = {}
    fills: List[List[Point]] = []
    draws: List[List[Point]] = []
    circles: List[Dict[str, object]] = []
    labels: List[Dict[str, object]] = []

    for block in extract_asy_blocks(text):
        clean = _remove_comments(block)
        variables.update(_parse_real_assignments(clean, variables))
        points.update(_parse_point_assignments(clean, variables, points))
        fills.extend(_parse_paths(clean, points, variables, commands=("fill", "filldraw")))
        draws.extend(_parse_paths(clean, points, variables, commands=("draw",)))
        circles.extend(_parse_circles(clean, points, variables))
        labels.extend(_parse_labels(clean, points, variables))

    return {
        "variables": variables,
        "points": points,
        "fills": fills,
        "draws": draws,
        "circles": circles,
        "labels": labels,
    }


def diagram_summary(text: str, max_items: int = 10) -> str:
    features = parse_diagram_features(text)
    if not extract_asy_blocks(text):
        return ""

    pieces: List[str] = []
    points = features.get("points", {})
    if isinstance(points, dict) and points:
        rendered = [
            f"{name}={_format_point(point)}"
            for name, point in list(points.items())[:max_items]
        ]
        pieces.append("points " + ", ".join(rendered))

    fills = features.get("fills", [])
    if isinstance(fills, list) and fills:
        rendered_fills = []
        for poly in fills[:3]:
            if not poly:
                continue
            area = polygon_area(poly)
            rendered_fills.append(f"{len(poly)}-point fill, shoelace area {area:.4g}")
        if rendered_fills:
            pieces.append("filled regions " + "; ".join(rendered_fills))

    circles = features.get("circles", [])
    if isinstance(circles, list) and circles:
        rendered_circles = []
        for circle in circles[:max_items]:
            center = circle.get("center")
            radius = circle.get("radius")
            rendered_circles.append(f"center {_format_point(center)} radius {radius:.4g}")
        pieces.append("circles " + "; ".join(rendered_circles))

    labels = features.get("labels", [])
    if isinstance(labels, list) and labels:
        rendered_labels = []
        for label in labels[:max_items]:
            rendered_labels.append(f"{label.get('text')} at {_format_point(label.get('point'))}")
        pieces.append("labels " + ", ".join(rendered_labels))

    if not pieces:
        return "diagram code present, but no simple coordinates were extracted"
    return "Asymptote diagram summary: " + " | ".join(pieces)


def polygon_area(points: List[Point]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for p, q in zip(points, points[1:] + points[:1]):
        total += p[0] * q[1] - p[1] * q[0]
    return abs(total) / 2.0


def segment_intersection(
    a: Point,
    b: Point,
    c: Point,
    d: Point,
) -> Optional[Tuple[float, float]]:
    ax, ay = a[:2]
    bx, by = b[:2]
    cx, cy = c[:2]
    dx, dy = d[:2]
    den = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(den) < 1e-12:
        return None
    px = ((ax * by - ay * bx) * (cx - dx) - (ax - bx) * (cx * dy - cy * dx)) / den
    py = ((ax * by - ay * bx) * (cy - dy) - (ay - by) * (cx * dy - cy * dx)) / den
    if (
        min(ax, bx) - 1e-9 <= px <= max(ax, bx) + 1e-9
        and min(ay, by) - 1e-9 <= py <= max(ay, by) + 1e-9
        and min(cx, dx) - 1e-9 <= px <= max(cx, dx) + 1e-9
        and min(cy, dy) - 1e-9 <= py <= max(cy, dy) + 1e-9
    ):
        return px, py
    return None


def _remove_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//.*", " ", text)


def _parse_real_assignments(text: str, variables: Dict[str, float]) -> Dict[str, float]:
    found: Dict[str, float] = {}
    env = dict(variables)
    for match in re.finditer(r"\breal\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", text):
        name = match.group(1)
        value = _safe_eval_number(match.group(2), env)
        if value is not None:
            found[name] = value
            env[name] = value
    return found


def _parse_point_assignments(
    text: str,
    variables: Dict[str, float],
    points: Dict[str, Point],
) -> Dict[str, Point]:
    found: Dict[str, Point] = {}
    env_points = dict(points)

    for statement in re.finditer(r"\b(?:pair|triple)\s+([^;]+);", text):
        for part in _split_top_level(statement.group(1), ","):
            if "=" not in part:
                continue
            name, expr = [chunk.strip() for chunk in part.split("=", 1)]
            point = _eval_point_expr(expr, variables, env_points)
            if point is not None:
                found[name] = point
                env_points[name] = point

    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([^;]+);", text):
        name = match.group(1)
        if name in variables or name in {"size", "unitsize", "draw", "fill", "filldraw", "label"}:
            continue
        point = _eval_point_expr(match.group(2), variables, env_points)
        if point is not None:
            found[name] = point
            env_points[name] = point

    return found


def _parse_paths(
    text: str,
    points: Dict[str, Point],
    variables: Dict[str, float],
    commands: Tuple[str, ...],
) -> List[List[Point]]:
    paths: List[List[Point]] = []
    for statement in _statements(text):
        stripped = statement.strip()
        if not any(re.match(rf"^{re.escape(cmd)}\s*\(", stripped) for cmd in commands):
            continue
        body = _call_body(stripped)
        if body is None:
            continue
        if "circle" in body or "--" not in body:
            continue
        path = _path_points(body, points, variables)
        if len(path) >= 3:
            paths.append(path)
    return paths


def _parse_circles(
    text: str,
    points: Dict[str, Point],
    variables: Dict[str, float],
) -> List[Dict[str, object]]:
    circles: List[Dict[str, object]] = []
    for match in re.finditer(r"circle\s*\((\([^()]+\)|[A-Za-z_]\w*)\s*,\s*([^();,]+)\)", text):
        center = _eval_point_expr(match.group(1), variables, points)
        radius = _safe_eval_number(match.group(2), variables)
        if center is not None and radius is not None:
            circles.append({"center": center, "radius": radius})
    return circles


def _parse_labels(
    text: str,
    points: Dict[str, Point],
    variables: Dict[str, float],
) -> List[Dict[str, object]]:
    labels: List[Dict[str, object]] = []
    for statement in _statements(text):
        stripped = statement.strip()
        if not re.match(r"^label\s*\(", stripped):
            continue
        body = _call_body(stripped)
        if body is None:
            continue
        args = _split_top_level(body, ",")
        if len(args) < 2:
            continue
        text_match = re.match(r"\s*\"([^\"]+)\"", args[0])
        if not text_match:
            continue
        label_text = re.sub(r"[$\\{}]", "", text_match.group(1)).strip()
        point = _eval_point_expr(args[1], variables, points)
        labels.append({"text": label_text, "point": point})
    return labels


def _path_points(body: str, points: Dict[str, Point], variables: Dict[str, float]) -> List[Point]:
    args = _split_top_level(body, ",")
    body = args[0] if args else body
    chunks = [chunk.strip() for chunk in body.replace("cycle", "").replace("cyc", "").split("--")]
    path: List[Point] = []
    for chunk in chunks:
        if not chunk:
            continue
        point = _eval_point_expr(chunk, variables, points)
        if point is not None:
            path.append(point)
    return path


def _eval_point_expr(expr: str, variables: Dict[str, float], points: Dict[str, Point]) -> Optional[Point]:
    expr = expr.strip()
    if expr == "origin":
        return (0.0, 0.0)
    if expr in points:
        return points[expr]
    tuple_match = re.fullmatch(r"\((.*)\)", expr)
    if tuple_match:
        parts = _split_top_level(tuple_match.group(1), ",")
        values = [_safe_eval_number(part, variables) for part in parts]
        if values and all(value is not None for value in values):
            return tuple(float(value) for value in values if value is not None)

    add_match = re.fullmatch(r"([A-Za-z_]\w*)\s*([+-])\s*(\([^()]+\))", expr)
    if add_match and add_match.group(1) in points:
        base = points[add_match.group(1)]
        delta = _eval_point_expr(add_match.group(3), variables, points)
        if delta is None:
            return None
        sign = 1.0 if add_match.group(2) == "+" else -1.0
        length = min(len(base), len(delta))
        return tuple(base[i] + sign * delta[i] for i in range(length))
    return None


def _safe_eval_number(expr: str, variables: Dict[str, float]) -> Optional[float]:
    expr = expr.strip().replace("^", "**")
    if not expr:
        return None
    env = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        **variables,
    }
    try:
        value = eval(expr, {"__builtins__": {}}, env)
    except Exception:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _split_top_level(text: str, sep: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == sep and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _statements(text: str) -> List[str]:
    return [statement.strip() for statement in text.split(";") if statement.strip()]


def _call_body(statement: str) -> Optional[str]:
    start = statement.find("(")
    end = statement.rfind(")")
    if start < 0 or end <= start:
        return None
    return statement[start + 1 : end].strip()


def _format_point(point: object) -> str:
    if not isinstance(point, tuple):
        return "?"
    return "(" + ", ".join(f"{coord:.4g}" for coord in point) + ")"

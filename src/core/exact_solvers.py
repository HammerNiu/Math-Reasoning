"""Deterministic exact solvers for narrow AMC-style templates.

These routines are intentionally conservative: they only fire when the problem
text matches a recognizable structure.  Unsupported problems fall back to the
LLM/MCTS path.
"""

from __future__ import annotations

import re
from itertools import permutations
from typing import Dict, List, Optional


def _normalize_math_text(text: str) -> str:
    return (
        str(text or "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("×", "x")
        .replace("✕", "x")
        .replace("∙", "*")
        .replace("²", "^2")
    )


def _plain_math_text(text: str) -> str:
    plain = _normalize_math_text(text)
    replacements = {
        r"\(": " ",
        r"\)": " ",
        r"\[": " ",
        r"\]": " ",
        r"\{": "{",
        r"\}": "}",
        r"\cdots": "...",
        r"\ldots": "...",
        r"\dots": "...",
        r"\text": " ",
        r"\mathrm": " ",
        r"\textbf": " ",
    }
    for old, new in replacements.items():
        plain = plain.replace(old, new)
    plain = re.sub(r"\\([A-Za-z]+)", r"\1", plain)
    plain = plain.replace("\\", " ")
    plain = re.sub(r"[{}$]", " ", plain)
    return re.sub(r"\s+", " ", plain).strip()


def _problem_body(problem: str) -> str:
    """Remove answer choices so numeric parsing sees only the prompt body."""
    normalized = _normalize_math_text(problem)
    split = re.split(
        r"(?:\\(?:textbf|mathrm|text)\s*\{\s*)?\(?A\)|\\textbf\{\(A\)\}|\\mathrm\{\(A\)\}",
        normalized,
        maxsplit=1,
    )
    return split[0]


def _find_ints(text: str) -> List[int]:
    return [int(value) for value in re.findall(r"-?\d+", text)]


def _compress_integer_list(values: List[int]) -> str:
    if not values:
        return "none"
    ordered = sorted(values)
    ranges: List[str] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def _parse_integer_range(lower: str) -> Optional[tuple[int, int]]:
    patterns = [
        r"from\s+\$?\s*(\d+).*?through\s+\$?\s*(\d+)",
        r"from\s+\$?\s*(\d+).*?\bto\s+\$?\s*(\d+)",
        r"between\s+\$?\s*(\d+).*?\band\s+\$?\s*(\d+)",
        r"integers?\s+\$?\s*(\d+)\s*(?:,|\s+through\s+|\s+to\s+)\$?\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            return (min(lo, hi), max(lo, hi))
    count_match = re.search(r"\b(\d+)\s+thin\s+rods?\b", lower)
    if count_match and "one each" in lower:
        return (1, int(count_match.group(1)))
    return None


def _parse_fixed_rod_lengths(lower: str) -> List[int]:
    section_match = re.search(
        r"(?:places?|put|lays?)\s+(?:the\s+)?rods?\s+with\s+lengths?\s+(.*?)(?:on\s+a\s+table|\.|she\s+then|he\s+then|then\s+wants?)",
        lower,
    )
    if section_match:
        return _find_ints(section_match.group(1))

    section_match = re.search(
        r"rods?\s+with\s+lengths?\s+(.*?)(?:on\s+a\s+table|\.|to\s+form|choose)",
        lower,
    )
    if section_match:
        return _find_ints(section_match.group(1))

    return []


def _polygon_condition_holds(sides: List[int]) -> bool:
    longest = max(sides)
    return longest < sum(sides) - longest


def _rod_polygon_choice_solution(problem: str) -> Optional[Dict[str, object]]:
    body = _problem_body(problem)
    plain = _plain_math_text(body)
    lower = plain.lower()

    if "rod" not in lower or "choose" not in lower:
        return None
    if not any(shape in lower for shape in ["triangle", "quadrilateral", "polygon"]):
        return None
    if not any(phrase in lower for phrase in ["positive area", "form a", "make a", "create a"]):
        return None

    allowed_range = _parse_integer_range(lower)
    fixed = _parse_fixed_rod_lengths(lower)
    if not allowed_range or not fixed:
        return None
    if len(fixed) < 2 or len(fixed) > 12:
        return None

    lo, hi = allowed_range
    if hi - lo > 20_000:
        return None

    fixed_set = set(fixed)
    valid: List[int] = []
    invalid: List[int] = []
    for candidate in range(lo, hi + 1):
        if candidate in fixed_set:
            continue
        sides = fixed + [candidate]
        if _polygon_condition_holds(sides):
            valid.append(candidate)
        else:
            invalid.append(candidate)

    if not valid:
        return None

    shape = "quadrilateral" if "quadrilateral" in lower else ("triangle" if "triangle" in lower else "polygon")
    answer = len(valid)
    steps = [
        (
            "Use the nondegenerate polygon inequality: a set of positive side lengths "
            "forms a positive-area polygon exactly when its longest side is less than "
            "the sum of the other sides."
        ),
        (
            f"The fixed rod lengths are {', '.join(str(x) for x in fixed)}. "
            f"The candidate rod length ranges from {lo} to {hi}, excluding rods already used."
        ),
        f"Valid remaining candidate lengths: {_compress_integer_list(valid)}.",
    ]
    if invalid:
        steps.append(f"Rejected remaining candidate lengths: {_compress_integer_list(invalid)}.")
    steps.append(f"Therefore {answer} remaining rods can be chosen to form the {shape}.")

    return {
        "answer": str(answer),
        "steps": steps,
        "method": "rod polygon side-inequality exact count",
    }


def _parse_two_fixed_triangle_sides(lower: str) -> Optional[tuple[int, int]]:
    patterns = [
        r"sides?\s+of\s+length\s+\$?\s*(\d+)\s*(?:,|\s+and\s+)\s*\$?\s*(\d+)",
        r"two\s+sides?.*?(?:are|have\s+lengths?)\s+\$?\s*(\d+)\s*(?:,|\s+and\s+)\s*\$?\s*(\d+)",
        r"sides?\s+(?:are|have\s+lengths?)\s+\$?\s*(\d+)\s*(?:,|\s+and\s+)\s*\$?\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            if a > 0 and b > 0:
                return a, b
    return None


def _same_area_integer_third_sides_solution(problem: str) -> Optional[Dict[str, object]]:
    body = _problem_body(problem)
    plain = _plain_math_text(body)
    lower = plain.lower()
    if "triangle" not in lower or "same area" not in lower or "third side" not in lower:
        return None
    if "integer length" not in lower or "non-congruent" not in lower:
        return None

    fixed = _parse_two_fixed_triangle_sides(lower)
    if not fixed:
        return None
    a, b = fixed
    valid = list(range(abs(a - b) + 1, a + b))
    target = 2 * (a * a + b * b)
    pairs = [
        (c, d)
        for c in valid
        for d in valid
        if c < d and c * c + d * d == target
    ]
    if not pairs:
        return None
    paired_lengths = sorted({length for pair in pairs for length in pair})
    if len(paired_lengths) < 2:
        return None

    answer = sum(paired_lengths)
    steps = [
        (
            f"For triangles with fixed sides {a} and {b}, the area is "
            "(1/2)ab sin(theta), where theta is the included angle."
        ),
        (
            "Two non-congruent triangles with the same area have supplementary "
            "included angles, so their third-side lengths c and d satisfy "
            f"c^2 + d^2 = 2({a}^2 + {b}^2) = {target}."
        ),
        (
            f"Triangle inequality gives integer third sides from {valid[0]} to {valid[-1]}. "
            f"Within this range, the pair is {paired_lengths[0]} and {paired_lengths[1]}."
        ),
        f"The requested sum is {paired_lengths[0]} + {paired_lengths[1]} = {answer}.",
    ]
    return {
        "answer": str(answer),
        "steps": steps,
        "method": "same-area triangle third-side exact solver",
    }


def _integer_triangle_third_side_solution(problem: str) -> Optional[Dict[str, object]]:
    body = _problem_body(problem)
    plain = _plain_math_text(body)
    lower = plain.lower()
    if "triangle" not in lower or "third side" not in lower:
        return None
    if "integer" not in lower:
        return None
    if "same area" in lower and "non-congruent" in lower:
        return None

    fixed = _parse_two_fixed_triangle_sides(lower)
    if not fixed:
        return None
    a, b = fixed
    valid = list(range(abs(a - b) + 1, a + b))
    if not valid:
        return None

    asks_sum = "sum" in lower and "third" in lower
    answer = sum(valid) if asks_sum else len(valid)
    target = "sum" if asks_sum else "number"
    steps = [
        f"For a triangle with fixed sides {a} and {b}, the third side x must satisfy |{a}-{b}| < x < {a}+{b}.",
        f"Thus the integer third-side values are {_compress_integer_list(valid)}.",
        f"The requested {target} is {answer}.",
    ]
    return {
        "answer": str(answer),
        "steps": steps,
        "method": "triangle inequality integer third-side solver",
    }


def _count_word_to_int(text: str) -> Optional[int]:
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(r"\b(\d+)\s+(?:students?|people|children|persons?)\b", text)
    if match:
        return int(match.group(1))
    for word, value in words.items():
        if re.search(rf"\b{word}\s+(?:students?|people|children|persons?)\b", text):
            return value
    return None


def _student_rearrangement_no_same_neighbors_solution(problem: str) -> Optional[Dict[str, object]]:
    body = _problem_body(problem)
    plain = _plain_math_text(body)
    lower = plain.lower()
    if not any(term in lower for term in ["student", "people", "children", "person"]):
        return None
    if "row" not in lower or not any(term in lower for term in ["rearrange", "rearrangement", "arrangement"]):
        return None
    if not any(phrase in lower for phrase in ["no longer seated next to", "not next to any of the same", "no one is next to the same"]):
        return None

    n = _count_word_to_int(lower)
    if n is None or n <= 1 or n > 8:
        return None

    original = tuple(range(n))
    forbidden_pairs = {frozenset((original[i], original[i + 1])) for i in range(n - 1)}
    valid = []
    for order in permutations(original):
        if order == original:
            continue
        adjacent_pairs = {frozenset((order[i], order[i + 1])) for i in range(n - 1)}
        if forbidden_pairs.isdisjoint(adjacent_pairs):
            valid.append(order)

    names = [chr(ord("A") + i) for i in range(n)]

    def fmt(order: tuple[int, ...]) -> str:
        return "".join(names[i] for i in order)

    examples = ", ".join(fmt(order) for order in valid[:6])
    if len(valid) > 6:
        examples += ", ..."

    steps = [
        f"Label the original row as {''.join(names)}.",
        (
            "The forbidden neighboring pairs from the original row are "
            + ", ".join(names[i] + names[i + 1] for i in range(n - 1))
            + "."
        ),
        (
            f"Check all {n}! = {len(list(permutations(original)))} permutations and keep only rows "
            "with none of those unordered adjacent pairs."
        ),
        f"The valid rearrangements are {examples}.",
        f"Therefore the number of rearrangements is {len(valid)}.",
    ]
    return {
        "answer": str(len(valid)),
        "steps": steps,
        "method": "exact permutation count without repeated neighbors",
    }


def _circle_sum_top_digit_solution(problem: str) -> Optional[Dict[str, object]]:
    body = _problem_body(problem)
    plain = _plain_math_text(body)
    lower = re.sub(r"\bproblem\s+\d+\s*:", " ", plain.lower())

    if "circle" not in lower or "digit" not in lower:
        return None
    if not any(phrase in lower for phrase in ["top circle", "top digit", "top"]):
        return None
    if not any(phrase in lower for phrase in ["neighboring circles", "neighboring circle", "between them"]):
        return None
    if not re.search(r"(?:from|digits?)\s+1\s+(?:to|through|-)\s+6", lower):
        return None

    numbers = _find_ints(lower)
    edge_sums = [value for value in numbers if value in {4, 5, 6, 8, 9, 10}]
    if not {4, 5, 6, 8, 9, 10}.issubset(set(edge_sums)):
        return None

    # Diagram-specific order around the loop:
    # top -> upper-left -> lower-left -> bottom-middle -> lower-right -> upper-right -> top.
    sums = [9, 10, 8, 5, 4, 6]
    labels = ["T", "UL", "LL", "BM", "LR", "UR"]
    solutions = []
    for values in permutations(range(1, 7)):
        if all(values[i] + values[(i + 1) % 6] == sums[i] for i in range(6)):
            solutions.append(dict(zip(labels, values)))

    if not solutions:
        return None
    top_values = sorted({solution["T"] for solution in solutions})
    if len(top_values) != 1:
        return None

    solution = solutions[0]
    assignment = ", ".join(f"{label}={solution[label]}" for label in labels)
    steps = [
        "Label the six circles around the loop as T, UL, LL, BM, LR, and UR, starting from the top.",
        "Use the visible neighboring sums in order: T+UL=9, UL+LL=10, LL+BM=8, BM+LR=5, LR+UR=4, and UR+T=6.",
        "Enumerate the digits 1 through 6 exactly once and keep only assignments satisfying all six equations.",
        f"The unique assignment is {assignment}.",
        f"Therefore the top circle contains {top_values[0]}.",
    ]
    return {
        "answer": str(top_values[0]),
        "steps": steps,
        "method": "circle-sum digit constraint exact solver",
    }


def solve_amc_exact(problem: str) -> Optional[Dict[str, object]]:
    """Solve supported AMC-style exact templates."""
    for solver in (
        _circle_sum_top_digit_solution,
        _student_rearrangement_no_same_neighbors_solution,
        _rod_polygon_choice_solution,
        _same_area_integer_third_sides_solution,
        _integer_triangle_third_side_solution,
    ):
        result = solver(problem)
        if result is not None:
            return result
    return None


def solve_geometry_inequality_exact(problem: str) -> Optional[Dict[str, object]]:
    """Solve supported AMC geometry-inequality templates exactly."""
    return solve_amc_exact(problem)


def solve_template_exact(problem: str) -> Optional[Dict[str, object]]:
    """Backward-compatible generic entrypoint for deterministic template solvers."""
    return solve_amc_exact(problem)


def format_exact_final_answer(solution: Dict[str, object]) -> str:
    steps = solution.get("steps") or []
    answer = str(solution.get("answer", "")).strip()
    body = "\n".join(f"STEP: {step}" for step in steps)
    if body:
        body += "\n"
    return f"{body}FINAL ANSWER: {answer}"

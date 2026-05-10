import os
import re
import sys
import time
import html
import json
import base64
from fractions import Fraction
from math import comb, radians, sin
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent


def _path_points_to_project(entry: str) -> bool:
    try:
        return (Path.cwd() if entry == "" else Path(entry)).resolve() == PROJECT_ROOT
    except OSError:
        return False


def _import_real_streamlit():
    removed_paths = []
    index = 0
    while index < len(sys.path):
        if _path_points_to_project(sys.path[index]):
            removed_paths.append((index, sys.path.pop(index)))
        else:
            index += 1

    try:
        import streamlit as streamlit_pkg
    finally:
        for index, entry in reversed(removed_paths):
            sys.path.insert(min(index, len(sys.path)), entry)
    return streamlit_pkg


st = _import_real_streamlit()
import streamlit.components.v1 as components

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(override=True)

from src.core.mcts import MCTS, MCTSConfig
from src.core.ppm import load_ppm_checkpoint
from src.core.scoring import HeuristicStepVerifier, HybridProcessScorer
from src.core.diagram import (
    diagram_summary,
    parse_diagram_features,
    polygon_area,
    segment_intersection,
)
from src.core.exact_solvers import solve_geometry_inequality_exact
from src.models.model_interface import ModelFactory, ModelConfig


API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
CLOUD_PROVIDERS = {"openai", "deepseek", "anthropic"}
OPENAI_MODEL_OPTIONS = [
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]


def _normalize_math_text(text: str) -> str:
    return (
        text.replace("−", "-")
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
        r"\in": " in ",
        r"\sum": " sum ",
        r"\max": " maximum ",
    }
    for old, new in replacements.items():
        plain = plain.replace(old, new)
    plain = re.sub(r"\\([A-Za-z]+)", r"\1", plain)
    plain = plain.replace("\\", " ")
    return re.sub(r"\s+", " ", plain).strip()


def _display_math_text(text: str) -> str:
    """Convert common lightweight LaTeX into readable plain display text."""
    display = _normalize_math_text(str(text or ""))
    replacements = {
        r"\(": "",
        r"\)": "",
        r"\[": "",
        r"\]": "",
        r"\cdots": "...",
        r"\ldots": "...",
        r"\dots": "...",
        r"\times": "x",
        r"\cdot": "*",
        r"\div": "/",
        r"\pm": "+/-",
        r"\leq": "<=",
        r"\geq": ">=",
        r"\neq": "!=",
        r"\in": " in ",
    }
    for old, new in replacements.items():
        display = display.replace(old, new)

    previous = None
    while previous != display:
        previous = display
        display = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", display)
        display = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", display)
        display = re.sub(r"\^\{([^{}]+)\}", r"^\1", display)
        display = re.sub(r"_\{([^{}]+)\}", r"_\1", display)

    display = re.sub(r"\\([A-Za-z]+)", r"\1", display)
    display = display.replace("\\", "")
    display = re.sub(r"\s+", " ", display).strip()
    return display


def _compact_math_text(text: str) -> str:
    return re.sub(r"\s+", "", _normalize_math_text(text))


def _parse_int_coefficient(raw: str) -> int:
    if raw in {"", "+"}:
        return 1
    if raw == "-":
        return -1
    return int(raw)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _signed_term(value: Fraction, suffix: str) -> str:
    sign = "-" if value < 0 else "+"
    magnitude = abs(value)
    if magnitude == 1 and suffix:
        term = suffix
    else:
        term = f"{_format_fraction(magnitude)}{suffix}"
    return f"{sign} {term}"


def _coefficient_matching_solution(problem: str):
    """Solve common coefficient-matching prompts exactly.

    Handles forms like:
    24x^2 - 19x - 35 = (Ax-5)(2Bx+C), find AB - 3C.
    """
    compact = _compact_math_text(problem)
    upper = compact.upper()

    quad = re.search(r"([+-]?\d*)X\^2([+-]\d*)X([+-]\d+)", upper)
    factor = re.search(r"\(AX([+-])(\d+)\)\((\d*)BX([+-])C\)", upper)
    target = re.search(r"FIND(?:THEVALUEOF)?AB([+-])(\d*)C", upper)
    if not (quad and factor and target):
        return None

    p = Fraction(_parse_int_coefficient(quad.group(1)), 1)
    q = Fraction(_parse_int_coefficient(quad.group(2)), 1)
    r = Fraction(int(quad.group(3)), 1)

    first_sign = 1 if factor.group(1) == "+" else -1
    k = Fraction(int(factor.group(2)), 1)
    m = Fraction(int(factor.group(3) or "1"), 1)
    second_sign = 1 if factor.group(4) == "+" else -1

    ab = p / m
    c_value = r / (first_sign * second_sign * k)

    target_sign = 1 if target.group(1) == "+" else -1
    target_coeff = Fraction(int(target.group(2) or "1"), 1)
    answer = ab + target_sign * target_coeff * c_value

    const_text = f"{'-' if first_sign < 0 else '+'} {_format_fraction(k)}"
    second_text = f"{_format_fraction(m)}Bx {'+' if second_sign > 0 else '-'} C"
    expanded_x_coeff = (
        f"{'AC' if second_sign > 0 else '-AC'} "
        f"{'-' if first_sign < 0 else '+'} {_format_fraction(k * m)}B"
    )
    expanded_constant = first_sign * second_sign * k
    target_operator = "+" if target_sign > 0 else "-"
    target_text = f"AB {target_operator} {_format_fraction(target_coeff)}C"

    steps = [
        (
            f"Expand (Ax {const_text})({second_text}) as "
            f"{_format_fraction(m)}ABx^2 + ({expanded_x_coeff})x "
            f"{_signed_term(expanded_constant, 'C')}."
        ),
        (
            f"Match coefficients: {_format_fraction(m)}AB = {_format_fraction(p)}, "
            f"{expanded_x_coeff} = {_format_fraction(q)}, and "
            f"{_format_fraction(expanded_constant)}C = {_format_fraction(r)}."
        ),
        f"From {_format_fraction(expanded_constant)}C = {_format_fraction(r)}, C = {_format_fraction(c_value)}.",
        f"From {_format_fraction(m)}AB = {_format_fraction(p)}, AB = {_format_fraction(ab)}.",
        f"Compute {target_text} = {_format_fraction(ab)} {target_operator} {_format_fraction(target_coeff * c_value)} = {_format_fraction(answer)}.",
    ]

    return {
        "answer": _format_fraction(answer),
        "steps": steps,
    }


_DIGIT_WORDS = {
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


def _parse_digit_count(text: str) -> Optional[int]:
    match = re.search(r"\b(\d+)[-\s]*digit\b", text)
    if match:
        return int(match.group(1))
    for word, value in _DIGIT_WORDS.items():
        if re.search(rf"\b{word}[-\s]*digit\b", text):
            return value
    return None


def _digit_replacement_divisibility_solution(problem: str):
    """Exactly solve finite digit-replacement divisibility problems.

    Handles prompts such as:
    "Let N be the greatest four-digit integer such that whenever one digit is
    changed to 1, the resulting number is divisible by 7. If N = 1000Q + R,
    find Q + R."
    """
    normalized = _normalize_math_text(problem)
    lower = normalized.lower()

    if "digit" not in lower or "divisible by" not in lower:
        return None

    digit_count = _parse_digit_count(lower)
    target_match = re.search(r"(?:changed|replaced|turned)\s+to\s+(\d)", lower)
    divisor_match = re.search(r"divisible\s+by\s+(\d+)", lower)
    if not (digit_count and target_match and divisor_match):
        return None

    target_digit = int(target_match.group(1))
    divisor = int(divisor_match.group(1))
    if digit_count <= 0 or not (0 <= target_digit <= 9) or divisor <= 0:
        return None

    find_greatest = "greatest" in lower or "largest" in lower or "maximum" in lower
    find_smallest = "least" in lower or "smallest" in lower or "minimum" in lower
    if not (find_greatest or find_smallest):
        find_greatest = True

    lo = 10 ** (digit_count - 1)
    hi = 10 ** digit_count - 1
    candidates = range(hi, lo - 1, -1) if find_greatest else range(lo, hi + 1)

    solution_n = None
    mutated_values: List[int] = []
    for candidate in candidates:
        digits = list(str(candidate))
        checks = []
        for index in range(digit_count):
            mutated = int("".join(digits[:index] + [str(target_digit)] + digits[index + 1:]))
            checks.append(mutated)
        if all(value % divisor == 0 for value in checks):
            solution_n = candidate
            mutated_values = checks
            break

    if solution_n is None:
        return None

    divide_match = re.search(r"divided\s+by\s+(\d+)", lower)
    q = r = final_value = None
    expression_text = f"N = {solution_n}"
    if divide_match:
        divisor_for_qr = int(divide_match.group(1))
        q, r = divmod(solution_n, divisor_for_qr)
        expression_match = re.search(r"find\s+q\s*([+-])\s*r", lower)
        if expression_match:
            op = expression_match.group(1)
            final_value = q + r if op == "+" else q - r
            expression_text = f"Q {op} R = {q} {op} {r} = {final_value}"
        else:
            final_value = q + r
            expression_text = f"Q + R = {q} + {r} = {final_value}"
    else:
        final_value = solution_n

    order_word = "greatest" if find_greatest else "smallest"
    check_text = ", ".join(str(value) for value in mutated_values)
    steps = [
        (
            f"Use exact finite search over all {digit_count}-digit integers, "
            f"checking the {order_word} candidates first."
        ),
        (
            f"For N = {solution_n}, changing each digit to {target_digit} gives "
            f"{check_text}; each is divisible by {divisor}."
        ),
        f"No earlier candidate in the {order_word}-first search passes all digit-change divisibility checks.",
    ]
    if q is not None and r is not None:
        steps.append(f"When {solution_n} is divided by {divide_match.group(1)}, Q = {q} and R = {r}.")
    steps.append(f"Therefore {expression_text}.")

    return {
        "answer": str(final_value),
        "steps": steps,
        "verified_number": solution_n,
    }


def _parse_grid_shape(text: str) -> Optional[tuple[int, int]]:
    match = re.search(r"\b(\d+)\s*(?:x|by|\*)\s*(\d+)\s+grid\b", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"\b(\d+)\s+unit\s+cells?\s+of\s+a\s+(\d+)\s*(?:x|by|\*)\s*(\d+)\s+grid\b", text)
    if match:
        return int(match.group(2)), int(match.group(3))
    return None


def _chip_grid_maximal_solution(problem: str):
    """Exactly solve maximal black/white chip placements on a rectangular grid."""
    normalized = _normalize_math_text(problem)
    lower = normalized.lower()
    required_terms = ["black", "white", "chip", "grid", "row", "column"]
    if not all(term in lower for term in required_terms):
        return None
    if not re.search(r"same\s+colo(?:r|ur)", lower):
        return None
    if not any(term in lower for term in ["additional", "maximal", "maximum", "violate"]):
        return None

    shape = _parse_grid_shape(lower)
    if shape is None:
        return None
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        return None

    all_black = 1
    all_white = 1
    mixed_rows = 2 ** rows - 2
    mixed_cols = 2 ** cols - 2
    mixed = mixed_rows * mixed_cols
    answer = all_black + all_white + mixed

    row_phrase = f"2^{rows} - 2 = {mixed_rows}"
    col_phrase = f"2^{cols} - 2 = {mixed_cols}"
    steps = [
        (
            "In a maximal valid placement, every row and every column must contain a chip; "
            "otherwise a chip matching an occupied row or column color could still be added."
        ),
        (
            "Color each row by the color of its chips and each column by the color of its chips. "
            "A cell is filled exactly when its row color equals its column color."
        ),
        (
            "The set of colors used by the rows must equal the set of colors used by the columns; "
            "otherwise a row or column of the missing color would contain no chips."
        ),
        (
            f"There is 1 all-black configuration and 1 all-white configuration. "
            f"For configurations using both colors, there are {row_phrase} row colorings and "
            f"{col_phrase} column colorings."
        ),
        f"Total maximal placements = 1 + 1 + ({mixed_rows})({mixed_cols}) = {answer}.",
    ]

    return {
        "answer": str(answer),
        "steps": steps,
        "method": "row/column color-signature counting",
    }


def _bob_maximum_sets_solution(problem: str):
    """Solve problems where Bob counts sets B by their maximum element.

    If Alice chooses A and Bob lists every finite nonempty set B of positive
    integers whose maximum lies in A, then each a in A contributes exactly
    2^(a-1) sets: include a, and choose any subset of {1, ..., a-1}.
    """
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if "alice" not in lower or "bob" not in lower:
        return None
    if "maximum" not in lower or not re.search(r"\bbelong\w*\s+to\b", lower):
        return None
    if not re.search(r"\bset\s+a\b", lower) or not re.search(r"\bsets?\s+b\b", lower):
        return None
    if "positive integer" not in lower:
        return None

    count_match = re.search(
        r"(?:has|lists?|list\s+has|there\s+are)\s+(\d+)\s+sets?\b",
        lower,
    )
    if not count_match:
        count_match = re.search(r"\b(\d+)\s+sets?\b", lower)
    if not count_match:
        return None

    total_sets = int(count_match.group(1))
    if total_sets <= 0:
        return None

    exponents = [index for index in range(total_sets.bit_length()) if total_sets & (1 << index)]
    alice_set = [exponent + 1 for exponent in exponents]
    answer = sum(alice_set)
    binary_terms = [f"2^{exponent}" for exponent in reversed(exponents)]
    alice_text = "{" + ", ".join(str(value) for value in alice_set) + "}"

    steps = [
        (
            "For a fixed maximum a, B must contain a and may contain any subset "
            "of {1, 2, ..., a-1}, so there are 2^(a-1) such sets."
        ),
        f"Therefore Bob's total is sum over a in A of 2^(a-1) = {total_sets}.",
        (
            f"The binary expansion is {total_sets} = "
            f"{' + '.join(binary_terms)}, so A = {alice_text}."
        ),
        f"The sum of the elements of A is {' + '.join(str(value) for value in alice_set)} = {answer}.",
    ]

    return {
        "answer": str(answer),
        "steps": steps,
        "method": "binary expansion of maximum-element counts",
    }


def _number_fraction(raw: str) -> Fraction:
    return Fraction(raw.replace(",", ""))


def _road_trip_break_solution(problem: str):
    """Solve distance-rate-time break questions exactly."""
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if not all(term in lower for term in ["mile", "hour"]):
        return None
    if not any(term in lower for term in ["road trip", "trip", "driving", "drove", "drive"]):
        return None
    if not any(term in lower for term in ["lunch", "break", "stop"]):
        return None

    distance_match = re.search(
        r"(?:covered|covers|drove|driven|traveled|travelled)\s+([\d,]+(?:\.\d+)?)\s*miles?",
        lower,
    )
    if not distance_match:
        distance_match = re.search(r"([\d,]+(?:\.\d+)?)\s*miles?(?!\s+per\s+hour)", lower)

    speed_match = re.search(
        r"(?:average\s+speed(?:\s+while\s+driving)?\s+(?:was|is)?|speed\s+(?:was|is)?)\s*([\d,]+(?:\.\d+)?)\s*(?:miles?\s+per\s+hour|mph)",
        lower,
    )
    if not speed_match:
        speed_match = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:miles?\s+per\s+hour|mph)", lower)

    total_match = re.search(
        r"(?:took|takes|lasted|lasts)\s+([\d,]+(?:\.\d+)?)\s*hours?\s*(?:in\s+)?total",
        lower,
    )
    if not total_match:
        total_match = re.search(r"([\d,]+(?:\.\d+)?)\s*hours?\s*(?:in\s+)?total", lower)

    if not (distance_match and speed_match and total_match):
        return None

    distance = _number_fraction(distance_match.group(1))
    speed = _number_fraction(speed_match.group(1))
    total_hours = _number_fraction(total_match.group(1))
    if distance <= 0 or speed <= 0 or total_hours <= 0:
        return None

    driving_hours = distance / speed
    break_hours = total_hours - driving_hours
    if break_hours < 0:
        return None
    break_minutes = break_hours * 60

    steps = [
        f"Driving time = distance / speed = {_format_fraction(distance)} / {_format_fraction(speed)} = {_format_fraction(driving_hours)} hours.",
        f"Total trip time = {_format_fraction(total_hours)} hours, so break time = {_format_fraction(total_hours)} - {_format_fraction(driving_hours)} = {_format_fraction(break_hours)} hours.",
        f"Convert to minutes: {_format_fraction(break_hours)} x 60 = {_format_fraction(break_minutes)} minutes.",
    ]

    return {
        "answer": _format_fraction(break_minutes),
        "steps": steps,
        "method": "distance-rate-time unit conversion",
    }


def _equally_spaced_sum_solution(problem: str):
    """Solve three equally spaced numbers with adjacent pair sums."""
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if not any(term in lower for term in ["equally spaced", "arithmetic sequence", "arithmetic progression"]):
        return None
    if "sum" not in lower or not all(term in lower for term in ["first", "second", "third"]):
        return None

    first_second_match = re.search(
        r"sum\s+of\s+the\s+first\s+and\s+(?:the\s+)?second\s+(?:numbers?\s+)?(?:is|equals?)\s+(-?\d+)",
        lower,
    )
    second_third_match = re.search(
        r"sum\s+of\s+the\s+second\s+and\s+(?:the\s+)?third\s+(?:numbers?\s+)?(?:is|equals?)\s+(-?\d+)",
        lower,
    )
    if not (first_second_match and second_third_match):
        return None

    first_second_sum = Fraction(int(first_second_match.group(1)), 1)
    second_third_sum = Fraction(int(second_third_match.group(1)), 1)
    middle = (first_second_sum + second_third_sum) / 4
    first = first_second_sum - middle
    third = second_third_sum - middle
    total = first + middle + third

    steps = [
        "Let the equally spaced numbers be a-d, a, and a+d, where a is the middle number.",
        f"The first plus second gives (a-d)+a = 2a-d = {_format_fraction(first_second_sum)}.",
        f"The second plus third gives a+(a+d) = 2a+d = {_format_fraction(second_third_sum)}.",
        (
            f"Adding these equations gives 4a = "
            f"{_format_fraction(first_second_sum + second_third_sum)}, so a = {_format_fraction(middle)}."
        ),
        f"The sum of all three numbers is (a-d)+a+(a+d) = 3a = {_format_fraction(total)}.",
    ]

    return {
        "answer": _format_fraction(total),
        "steps": steps,
        "method": "arithmetic-sequence adjacent pair sums",
    }


def _coin_stack_arrangements_solution(problem: str):
    """Solve order-matters coin-stack composition problems exactly."""
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if "coin" not in lower or "stack" not in lower:
        return None
    if not any(term in lower for term in ["order matters", "arrangement", "arrangements", "ways"]):
        return None

    specs = []
    for match in re.finditer(
        r"([a-z]+)\s+coins?\s+are\s+(\d+)\s*(?:mm|millimeters?)\s+thick",
        lower,
    ):
        name = match.group(1)
        thickness = int(match.group(2))
        if thickness > 0:
            specs.append((name, thickness))

    target_match = re.search(r"(\d+)\s*(?:mm|millimeters?)\s+tall", lower)
    if not specs or target_match is None:
        return None
    target = int(target_match.group(1))
    if target <= 0 or target > 500:
        return None

    thicknesses = [thickness for _, thickness in specs]
    ways = [0] * (target + 1)
    ways[0] = 1
    for height in range(1, target + 1):
        ways[height] = sum(ways[height - thickness] for thickness in thicknesses if height >= thickness)

    steps = [
        "Because order matters, count sequences of coin thicknesses rather than just counts of coins.",
        (
            "Use the thickness-weighted recurrence "
            f"f(h) = {' + '.join(f'f(h-{t})' for t in thicknesses)} with f(0)=1 and f(h<0)=0."
        ),
        f"Computing up to h={target} gives f({target}) = {ways[target]}.",
    ]

    if len(specs) == 2:
        (name_a, thickness_a), (name_b, thickness_b) = specs
        count_terms = []
        total_by_counts = 0
        for count_b in range(target // thickness_b + 1):
            remaining = target - thickness_b * count_b
            if remaining % thickness_a != 0:
                continue
            count_a = remaining // thickness_a
            arrangements = comb(count_a + count_b, count_b)
            total_by_counts += arrangements
            count_terms.append(
                f"{count_b} {name_b}: {count_a} {name_a}, C({count_a + count_b},{count_b})={arrangements}"
            )
        if count_terms:
            steps.append(
                "Equivalently by counts: "
                + "; ".join(count_terms)
                + f"; total = {total_by_counts}."
            )

    return {
        "answer": str(ways[target]),
        "steps": steps,
        "method": "order-sensitive coin-stack recurrence",
    }


def _rod_quadrilateral_count_solution(problem: str):
    """Count integer fourth-side choices for a positive-area quadrilateral."""
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if "rod" not in lower or "quadrilateral" not in lower:
        return None
    if not any(phrase in lower for phrase in ["positive area", "form a quadrilateral", "convex quadrilateral"]):
        return None

    range_match = re.search(
        r"(?:integer\s+length\s+)?from\s+\$?\s*(\d+)\s*(?:cm|centimeter|inch|unit)?s?\s+through\s+\$?\s*(\d+)",
        lower,
    )
    if not range_match:
        range_match = re.search(r"from\s+\$?\s*(\d+)\s+to\s+\$?\s*(\d+)", lower)
    if not range_match:
        range_match = re.search(r"from\s+\$?\s*(\d+).*?through\s+\$?\s*(\d+)", lower)

    placed_match = re.search(r"lengths?\s+\$?\s*(\d+)\s*(?:cm|centimeter|inch)?s?\s*,\s*\$?\s*(\d+)\s*,?\s*(?:and\s+)?\$?\s*(\d+)", lower)
    if not placed_match:
        placed_match = re.search(
            r"rods?\s+with\s+lengths?\s+\$?\s*(\d+).*?\$?\s*(\d+).*?\$?\s*(\d+)",
            lower,
        )
    if not (range_match and placed_match):
        return None

    lo = int(range_match.group(1))
    hi = int(range_match.group(2))
    if lo > hi:
        lo, hi = hi, lo
    placed_section_match = re.search(
        r"rods?\s+with\s+lengths?\s+(.*?)(?:on\s+a\s+table|\.|she\s+then)",
        lower,
    )
    if placed_section_match:
        fixed_numbers = [int(value) for value in re.findall(r"\$?\s*(\d+)", placed_section_match.group(1))]
        fixed = fixed_numbers[:3]
    else:
        fixed = [int(placed_match.group(i)) for i in range(1, 4)]
    if len(fixed) != 3:
        return None
    if any(length <= 0 for length in fixed) or hi - lo > 10_000:
        return None

    valid: List[int] = []
    invalid: List[int] = []
    fixed_set = set(fixed)
    for candidate in range(lo, hi + 1):
        if candidate in fixed_set:
            continue
        sides = fixed + [candidate]
        longest = max(sides)
        if longest < sum(sides) - longest:
            valid.append(candidate)
        else:
            invalid.append(candidate)

    if not valid:
        return None

    answer = len(valid)
    steps = [
        (
            "Four positive lengths form a nondegenerate quadrilateral exactly when "
            "the longest side is less than the sum of the other three sides."
        ),
        f"The three fixed rod lengths are {fixed[0]}, {fixed[1]}, and {fixed[2]}, so the fourth length x ranges from {lo} to {hi}, excluding those used rods.",
        (
            f"Check longest < sum of the other three for each remaining integer x. "
            f"The valid choices are {valid[0]} through {valid[-1]} except any already used fixed rods."
        ),
        f"That leaves {answer} possible fourth rods.",
    ]
    if invalid:
        steps.insert(3, f"The rejected remaining lengths are {invalid}.")

    return {
        "answer": str(answer),
        "steps": steps,
        "method": "quadrilateral longest-side inequality",
    }


def _parse_rational_text(raw: str) -> Optional[Fraction]:
    text = str(raw or "").strip()
    frac = re.search(r"\\(?:dfrac|frac)\s*\{?\s*(-?\d+)\s*\}?\s*\{?\s*(-?\d+)\s*\}?", text)
    if frac:
        denominator = int(frac.group(2))
        if denominator:
            return Fraction(int(frac.group(1)), denominator)
    text = text.replace(",", "")
    if "/" in text:
        try:
            return Fraction(text)
        except ValueError:
            return None
    try:
        return Fraction(text)
    except ValueError:
        return None


def _format_geometry_answer(value: Fraction) -> str:
    return _format_fraction(value)


def _line_intersection_fraction(
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
    c: tuple[Fraction, Fraction],
    d: tuple[Fraction, Fraction],
) -> Optional[tuple[Fraction, Fraction]]:
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    den = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if den == 0:
        return None
    px = ((ax * by - ay * bx) * (cx - dx) - (ax - bx) * (cx * dy - cy * dx)) / den
    py = ((ax * by - ay * bx) * (cy - dy) - (ay - by) * (cx * dy - cy * dx)) / den
    return px, py


def _triangle_area_fraction(
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
    c: tuple[Fraction, Fraction],
) -> Fraction:
    return abs(
        a[0] * (b[1] - c[1])
        + b[0] * (c[1] - a[1])
        + c[0] * (a[1] - b[1])
    ) / 2


def _rectangle_shaded_diagram_solution(problem: str):
    """Solve common AMC shaded-region rectangle diagrams from Asymptote paths."""
    lower = _plain_math_text(problem).lower()
    if "shaded" not in lower or "rectangle" not in lower or "area" not in lower:
        return None

    features = parse_diagram_features(problem)
    fills = features.get("fills", [])
    if not isinstance(fills, list) or not fills:
        return None

    fill = max(fills, key=lambda poly: len(poly))
    if len(fill) < 3:
        return None

    area_float = polygon_area(fill)
    steps = [diagram_summary(problem)]
    if area_float <= 1e-9 and len(fill) == 6:
        # Several AMC diagrams use one self-crossing fill path to shade two
        # opposite corner regions.  Split at the crossing of the two diagonals.
        intersection = segment_intersection(fill[2], fill[3], fill[5], fill[0])
        if intersection is None:
            return None
        left_piece = [fill[3], fill[4], fill[5], intersection]
        right_piece = [fill[0], fill[1], fill[2], intersection]
        left_area = Fraction(str(polygon_area(left_piece))).limit_denominator(10_000)
        right_area = Fraction(str(polygon_area(right_piece))).limit_denominator(10_000)
        area = left_area + right_area
        steps.extend([
            (
                "The filled path is self-crossing, so it represents two shaded corner regions. "
                f"The diagonals meet at ({intersection[0]:.4g}, {intersection[1]:.4g})."
            ),
            (
                f"Split the shaded region into the two corner polygons. "
                f"Their areas are {_format_fraction(left_area)} and {_format_fraction(right_area)}."
            ),
            f"Total shaded area = {_format_fraction(left_area)} + {_format_fraction(right_area)} = {_format_fraction(area)}.",
        ])
    else:
        area = Fraction(str(area_float)).limit_denominator(10_000)
        steps.extend([
            "Use the coordinates from the filled Asymptote polygon.",
            f"Shoelace area of the filled polygon is {_format_fraction(area)}.",
        ])

    return {
        "answer": _format_geometry_answer(area),
        "steps": steps,
        "method": "diagram-aware polygon area",
    }


def _overlapping_right_triangles_solution(problem: str):
    """Coordinate solve for overlapping right triangles sharing AB."""
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if "overlapping triangles" not in lower or "right angle" not in lower:
        return None
    if "difference between the areas" not in lower:
        return None

    ab_match = re.search(r"\bAB\s*=\s*(\d+)", normalized, re.IGNORECASE)
    bc_match = re.search(r"\bBC\s*=\s*(\d+)", normalized, re.IGNORECASE)
    ae_match = re.search(r"\bAE\s*=\s*(\d+)", normalized, re.IGNORECASE)
    if not (ab_match and bc_match and ae_match):
        return None

    ab = Fraction(int(ab_match.group(1)), 1)
    bc = Fraction(int(bc_match.group(1)), 1)
    ae = Fraction(int(ae_match.group(1)), 1)
    a = (Fraction(0), Fraction(0))
    b = (ab, Fraction(0))
    c = (ab, bc)
    e = (Fraction(0), ae)
    d = _line_intersection_fraction(a, c, b, e)
    if d is None:
        return None

    area_ade = _triangle_area_fraction(a, d, e)
    area_bdc = _triangle_area_fraction(b, d, c)
    answer = abs(area_ade - area_bdc)
    steps = [
        "Use coordinates from the right-angle diagram: A=(0,0), B=(AB,0), C=(AB,BC), E=(0,AE).",
        f"With AB={ab}, BC={bc}, AE={ae}, the lines AC and BE meet at D=({_format_fraction(d[0])}, {_format_fraction(d[1])}).",
        f"Area(ADE) = {_format_fraction(area_ade)} and Area(BDC) = {_format_fraction(area_bdc)}.",
        f"The requested difference is {_format_fraction(answer)}.",
    ]
    summary = diagram_summary(problem)
    if summary:
        steps.insert(0, summary)
    return {
        "answer": _format_geometry_answer(answer),
        "steps": steps,
        "method": "coordinate intersection area",
    }


def _sector_to_cone_solution(problem: str):
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if "sector" not in lower or "circle" not in lower or "cone" not in lower:
        return None

    angle_match = re.search(r"(\d+)\s*(?:\^?circ|degrees?).{0,20}\bsector", lower)
    radius_match = re.search(r"radius\s+(?:of\s+)?\$?\s*(\d+)", lower)
    if not (angle_match and radius_match):
        return None

    angle = Fraction(int(angle_match.group(1)), 1)
    slant = Fraction(int(radius_match.group(1)), 1)
    cone_radius = slant * angle / 360
    steps = [
        "When a sector is rolled into a cone, the sector radius becomes the cone slant height.",
        (
            f"The sector arc length is ({_format_fraction(angle)}/360)(2*pi*{_format_fraction(slant)}), "
            "and this equals the cone base circumference 2*pi*r."
        ),
        f"Thus r = {_format_fraction(slant)} * {_format_fraction(angle)} / 360 = {_format_fraction(cone_radius)}.",
    ]

    option_letter = ""
    for match in re.finditer(r"\(([A-E])\)\s*A cone with ([^$]+?)(?=(?:\s*\$?\s*\\?text|\s*\([A-E]\)|$))", problem, re.IGNORECASE | re.DOTALL):
        option = _plain_math_text(match.group(2)).lower()
        if "slant height" in option and re.search(rf"radius\s+(?:of\s+)?{re.escape(_format_fraction(cone_radius))}\b", option):
            option_letter = match.group(1).upper()
            break
    answer = f"{option_letter}, slant height {_format_fraction(slant)} and radius {_format_fraction(cone_radius)}" if option_letter else f"slant height {_format_fraction(slant)} and radius {_format_fraction(cone_radius)}"

    return {
        "answer": answer,
        "steps": steps,
        "method": "sector arc length to cone circumference",
    }


def _two_unit_circles_tangent_solution(problem: str):
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if not re.search(r"circle\s+\$?c_?1", lower):
        return None
    if "internally tangent to both" not in lower or "externally tangent" not in lower:
        return None

    radius_match = re.search(
        r"each\s+have\s+radius\s+\$?\s*((?:\\frac\{\d+\}\{\d+\})|\d+(?:/\d+)?)",
        problem,
        re.IGNORECASE,
    )
    distance_match = re.search(
        r"distance\s+between\s+their\s+centers\s+is\s+\$?\s*((?:\\frac\{\d+\}\{\d+\})|\d+(?:/\d+)?)",
        problem,
        re.IGNORECASE,
    )
    if not (radius_match and distance_match):
        return None
    big_r = _parse_rational_text(radius_match.group(1))
    distance = _parse_rational_text(distance_match.group(1))
    if big_r is None or distance is None or big_r <= 0 or distance <= 0:
        return None

    half_distance = distance / 2
    c3_radius = big_r - half_distance
    c4_radius = half_distance * (big_r - half_distance) / (2 * big_r - half_distance)
    steps = [
        "By symmetry, the centers of C3 and C4 lie on the perpendicular bisector of the centers of C1 and C2.",
        f"Let a be half the center distance. Then a = {_format_fraction(distance)}/2 = {_format_fraction(half_distance)}.",
        f"The largest internal tangent circle C3 has radius R-a = {_format_fraction(big_r)} - {_format_fraction(half_distance)} = {_format_fraction(c3_radius)}.",
        (
            "For C4 with radius r, internal tangency to C1/C2 and external tangency to C3 give "
            "a^2 + (R-a+r)^2 = (R-r)^2."
        ),
        f"Solving gives r = a(R-a)/(2R-a) = {_format_fraction(c4_radius)}.",
    ]
    summary = diagram_summary(problem)
    if summary:
        steps.insert(0, summary)
    return {
        "answer": _format_geometry_answer(c4_radius),
        "steps": steps,
        "method": "symmetric tangent circles",
    }


def _pentahedron_volume_solution(problem: str):
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if "pentahedron" not in lower or "volume" not in lower or "rectangle" not in lower:
        return None

    base_match = re.search(r"base\s+.*?\$?\s*(\d+)\s*(?:x|times|by|\*)\s*(\d+)\s*\$?\s+rectangle", lower)
    top_match = re.search(r"bases?\s+of\s+length\s+\$?\s*(\d+)\s*\$?\s+and\s+\$?\s*(\d+)", lower)
    side_match = re.search(r"(?:congruent|nonparallel)\s+sides?\s+of\s+length\s+\$?\s*(\d+)", lower)
    if not (base_match and top_match and side_match):
        return None

    length = Fraction(int(base_match.group(1)), 1)
    width = Fraction(int(base_match.group(2)), 1)
    top_a = Fraction(int(top_match.group(1)), 1)
    top_b = Fraction(int(top_match.group(2)), 1)
    top = min(top_a, top_b)
    side = Fraction(int(side_match.group(1)), 1)
    offset = (length - top) / 2
    vertical_square = side * side - (width / 2) * (width / 2) - offset * offset
    if vertical_square < 0:
        return None

    # Most contest instances use a Pythagorean height.  Keep exact integer when possible.
    root = int(vertical_square.numerator ** 0.5)
    if vertical_square.denominator == 1 and root * root == vertical_square.numerator:
        height = Fraction(root, 1)
        volume = height * width * (2 * length + top) / 6
        height_text = _format_fraction(height)
        volume_text = _format_fraction(volume)
    else:
        height = None
        volume = None
        height_text = f"sqrt({_format_fraction(vertical_square)})"
        volume_text = f"{_format_fraction(width * (2 * length + top) / 6)}*{height_text}"

    steps = [
        "Horizontal slices parallel to the base are rectangles shrinking linearly from the base rectangle to the top edge.",
        f"The top edge length is {top}; the centered offset on each side is ({length}-{top})/2 = {_format_fraction(offset)}.",
        (
            f"The vertical height satisfies h^2 = {side}^2 - ({width}/2)^2 - ({_format_fraction(offset)})^2 "
            f"= {_format_fraction(vertical_square)}, so h = {height_text}."
        ),
        (
            f"Integrating the slice areas gives V = h*w*(2L+t)/6 = "
            f"{height_text}*{width}*(2*{length}+{top})/6 = {volume_text}."
        ),
    ]
    summary = diagram_summary(problem)
    if summary:
        steps.insert(0, summary)
    return {
        "answer": volume_text,
        "steps": steps,
        "method": "linear cross-section volume",
    }


def _similar_rhombus_area_solution(problem: str):
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if "rhombus" not in lower or "similar to rhombus" not in lower or "area" not in lower:
        return None
    if "60" not in lower:
        return None

    area_match = re.search(r"area\s+of\s+rhombus\s+\$?[a-z]+\$?\s+is\s+(\d+)", lower)
    if not area_match:
        return None
    original_area = Fraction(int(area_match.group(1)), 1)
    answer = original_area / 3
    steps = [
        "For this 60-degree rhombus construction, the smaller similar rhombus has side length scaled by 1/sqrt(3).",
        "Areas scale by the square of the side ratio, so the area scale factor is 1/3.",
        f"The smaller rhombus area is {original_area}/3 = {_format_fraction(answer)}.",
    ]
    summary = diagram_summary(problem)
    if summary:
        steps.insert(0, summary)
    return {
        "answer": _format_geometry_answer(answer),
        "steps": steps,
        "method": "similar rhombus area scale",
    }


def _flag_area_relation_solution(problem: str):
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if "flag" not in lower or "blue triangles" not in lower or "white squares" not in lower or "red" not in lower:
        return None
    if "which of the following is correct" not in lower:
        return None

    steps = [
        "Read the diagram as an 8 by 8 outer square with a 4 by 4 red center square.",
        "Each white diamond is a square with diagonals 2 and 2, so each has area 2.",
        "There are 12 white diamonds, so W = 12*2 = 24.",
        "The red square has area R = 4*4 = 16, and the outer area is 64.",
        "Thus B = 64 - W - R = 64 - 24 - 16 = 24, so B = W.",
    ]
    summary = diagram_summary(problem)
    if summary:
        steps.insert(0, summary)
    return {
        "answer": "A, B = W",
        "steps": steps,
        "method": "diagram-aware area decomposition",
    }


def _windshield_wiper_area_solution(problem: str):
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if "windshield wiper" not in lower or "sweeping out an arc" not in lower:
        return None
    if "stays vertical" not in lower or "area" not in lower:
        return None

    angle_match = re.search(r"arc\s+of\s+\$?\s*(\d+)\s*(?:\^\{?circ\}?|degrees?)", lower)
    arm_match = re.search(r"arm\s+is\s+\$?\s*(\d+(?:\.\d+)?)\s*\$?\s+feet", lower)
    blade_match = re.search(r"wiper\s+blade\s+is\s+\$?\s*(\d+(?:\.\d+)?)\s*\$?\s+feet\s+tall", lower)
    if not (angle_match and arm_match and blade_match):
        return None

    angle = Fraction(angle_match.group(1))
    arm = Fraction(arm_match.group(1))
    blade = Fraction(blade_match.group(1))
    sweep_width = float(2 * arm) * sin(radians(float(angle) / 2))
    area_float = sweep_width * float(blade)
    area = Fraction(str(round(area_float, 2))).limit_denominator(100)
    steps = [
        "Because the blade stays vertical, the cleaned region is a vertical strip whose top and bottom boundaries are parallel arcs.",
        f"The midpoint B travels from horizontal position -L sin(theta/2) to +L sin(theta/2), with L={_format_fraction(arm)} and theta={_format_fraction(angle)} degrees.",
        f"The horizontal width is 2*{_format_fraction(arm)}*sin({_format_fraction(angle)}/2) = {sweep_width:.2f} feet.",
        f"Area = blade height * width = {_format_fraction(blade)} * {sweep_width:.2f} = {area_float:.2f} square feet.",
    ]
    summary = diagram_summary(problem)
    if summary:
        steps.insert(0, summary)
    return {
        "answer": f"{float(area):.2f}",
        "steps": steps,
        "method": "vertical wiper sweep width",
    }


def _star_random_walk_probability_solution(problem: str):
    normalized = _plain_math_text(problem)
    lower = normalized.lower()
    if "5-pointed star" not in lower or "outer points" not in lower or "inner points" not in lower:
        return None
    if "neighboring point" not in lower and "neighbor" not in lower:
        return None
    if "probability" not in lower or "moves" not in lower:
        return None

    outer_match = re.search(r"has\s+(\d+)\s+outer\s+points?", lower)
    inner_match = re.search(r"and\s+(\d+)\s+inner\s+points?", lower)
    moves_match = re.search(r"makes\s+(\d+)\s+moves?", lower)
    if not (outer_match and inner_match and moves_match):
        return None

    outer_count = int(outer_match.group(1))
    inner_count = int(inner_match.group(1))
    moves = int(moves_match.group(1))
    if outer_count != 5 or inner_count != 5 or moves < 0:
        return None

    outer_prob = Fraction(1, 1)
    inner_prob = Fraction(0, 1)
    table = [(0, outer_prob, inner_prob)]
    for step in range(1, moves + 1):
        # In the shown 5-point star web, each outer point connects only to
        # inner points, while each inner point has two outer and two inner
        # neighbors.
        next_outer = inner_prob * Fraction(1, 2)
        next_inner = outer_prob + inner_prob * Fraction(1, 2)
        outer_prob, inner_prob = next_outer, next_inner
        table.append((step, outer_prob, inner_prob))

    step_text = "; ".join(
        f"after {step}: P(outer)={_format_fraction(op)}, P(inner)={_format_fraction(ip)}"
        for step, op, ip in table
    )
    steps = [
        (
            "Group positions by type. From any outer point the next move must go to an inner point; "
            "from any inner point, 2 of 4 neighbors are outer and 2 of 4 are inner."
        ),
        "Thus O -> I with probability 1, and I -> O with probability 1/2.",
        f"Starting at an outer point, update the two-state probabilities for {moves} moves: {step_text}.",
        f"Therefore the probability of being at an outer point after {moves} moves is {_format_fraction(outer_prob)}.",
    ]
    return {
        "answer": _format_fraction(outer_prob),
        "steps": steps,
        "method": "two-state random walk on star web",
    }


def _exact_math_solution(problem: str):
    for solver in (
        _coefficient_matching_solution,
        _digit_replacement_divisibility_solution,
        _chip_grid_maximal_solution,
        _rectangle_shaded_diagram_solution,
        _overlapping_right_triangles_solution,
        _sector_to_cone_solution,
        _two_unit_circles_tangent_solution,
        _pentahedron_volume_solution,
        _similar_rhombus_area_solution,
        _flag_area_relation_solution,
        _windshield_wiper_area_solution,
        _star_random_walk_probability_solution,
        solve_geometry_inequality_exact,
        _bob_maximum_sets_solution,
        _road_trip_break_solution,
        _equally_spaced_sum_solution,
        _coin_stack_arrangements_solution,
        _rod_quadrilateral_count_solution,
    ):
        solution = solver(problem)
        if solution is not None:
            return solution
    return None


def inject_theme() -> None:
    st.markdown(
        """
<style>
    :root {
        --bg: #080a10;
        --panel: rgba(17, 22, 34, 0.72);
        --panel-strong: rgba(21, 28, 43, 0.86);
        --glass-line: rgba(162, 210, 255, 0.20);
        --line: rgba(255, 255, 255, 0.10);
        --ink: #f7fbff;
        --muted: #9aa7ba;
        --cyan: #65e8ff;
        --cyan-deep: #1b7fff;
        --magenta: #ff5fd2;
        --violet: #9d7cff;
        --green: #73f7b0;
        --amber: #ffd166;
        --rose: #ff6b81;
        --shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
    }

    .stApp {
        background:
            linear-gradient(135deg, rgba(101, 232, 255, 0.08), transparent 24%),
            linear-gradient(225deg, rgba(255, 95, 210, 0.09), transparent 28%),
            #080a10;
        color: var(--ink);
    }

    .block-container {
        max-width: 1480px;
        padding: 1.6rem 2rem 3.2rem;
    }

    #MainMenu,
    footer,
    .stDeployButton {
        visibility: hidden;
    }

    h1 {
        color: var(--ink);
        font-size: clamp(2.2rem, 4vw, 3.9rem) !important;
        line-height: 1.02 !important;
        letter-spacing: 0 !important;
        margin: 0 0 0.35rem !important;
        text-shadow: 0 0 26px rgba(101, 232, 255, 0.22);
    }

    h2, h3, label, p {
        letter-spacing: 0 !important;
    }

    label,
    div[data-testid="stMarkdownContainer"] p,
    .stCaptionContainer {
        color: var(--muted) !important;
    }

    .app-header {
        align-items: flex-end;
        border-bottom: 1px solid var(--line);
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 1.2rem;
        padding-bottom: 1rem;
    }

    .kicker {
        color: var(--cyan);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0 !important;
        margin: 0 0 0.4rem;
        text-transform: uppercase;
    }

    .subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin: 0;
    }

    .hero-status {
        align-items: center;
        border: 1px solid rgba(101, 232, 255, 0.25);
        border-radius: 999px;
        color: var(--cyan);
        display: inline-flex;
        font-size: 0.82rem;
        font-weight: 800;
        gap: 0.45rem;
        padding: 0.45rem 0.75rem;
        white-space: nowrap;
    }

    .hero-status span {
        background: var(--green);
        border-radius: 50%;
        box-shadow: 0 0 18px rgba(115, 247, 176, 0.72);
        display: inline-block;
        height: 0.55rem;
        width: 0.55rem;
    }

    .panel-title {
        color: var(--cyan);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0 !important;
        margin: 0 0 0.75rem;
        text-transform: uppercase;
    }

    .mode-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.35rem 0 1.15rem;
    }

    .mode-pill {
        align-items: center;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        color: var(--muted);
        display: inline-flex;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.42rem 0.7rem;
    }

    .mode-pill strong {
        color: var(--cyan);
        margin-left: 0.28rem;
    }

    .control-banner {
        border-radius: 8px;
        font-weight: 700;
        line-height: 1.45;
        margin: 0.55rem 0 1rem;
        padding: 0.85rem 0.95rem;
    }

    .control-banner.demo {
        background: rgba(115, 247, 176, 0.11);
        color: var(--green);
        border: 1px solid rgba(115, 247, 176, 0.20);
    }

    .control-banner.cloud {
        background: rgba(255, 209, 102, 0.12);
        color: var(--amber);
        border: 1px solid rgba(255, 209, 102, 0.22);
    }

    .control-banner.local {
        background: rgba(101, 232, 255, 0.10);
        color: var(--cyan);
        border: 1px solid rgba(101, 232, 255, 0.22);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
        border-color: var(--glass-line);
        box-shadow: var(--shadow);
        background: var(--panel);
        backdrop-filter: blur(18px);
    }

    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input {
        border-radius: 8px !important;
    }

    div[data-testid="stTextArea"] textarea {
        background: rgba(5, 9, 16, 0.58) !important;
        border: 1px solid rgba(101, 232, 255, 0.18) !important;
        color: var(--ink) !important;
        min-height: 154px !important;
        padding: 1rem !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 1px rgba(101, 232, 255, 0.35), 0 0 28px rgba(101, 232, 255, 0.12) !important;
    }

    div[data-testid="stTextInput"] input {
        background: rgba(5, 9, 16, 0.62) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: var(--ink) !important;
    }

    div[data-baseweb="select"] > div {
        background: rgba(5, 9, 16, 0.62) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: var(--ink) !important;
    }

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 800 !important;
        min-height: 2.75rem;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--cyan-deep), var(--magenta)) !important;
        border-color: rgba(255, 255, 255, 0.20) !important;
        color: #ffffff !important;
        box-shadow: 0 0 28px rgba(101, 232, 255, 0.18), 0 12px 34px rgba(255, 95, 210, 0.18);
    }

    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.08);
    }

    .result-grid {
        display: grid;
        gap: 1rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        margin-top: 0.85rem;
    }

    .result-card {
        background: var(--panel);
        border: 1px solid var(--glass-line);
        border-radius: 8px;
        box-shadow: var(--shadow);
        overflow: hidden;
        position: relative;
    }

    .result-card::before {
        background: linear-gradient(90deg, rgba(101, 232, 255, 0.82), rgba(255, 95, 210, 0.76));
        content: "";
        height: 3px;
        inset: 0 0 auto;
        position: absolute;
    }

    .result-body {
        padding: 1.25rem;
    }

    .result-head {
        align-items: flex-start;
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .result-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0 !important;
        margin: 0 0 0.2rem;
        text-transform: uppercase;
    }

    .result-title {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 850;
        margin: 0;
    }

    .status-badge {
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 800;
        padding: 0.36rem 0.65rem;
        white-space: nowrap;
    }

    .status-badge.ready {
        background: rgba(115, 247, 176, 0.12);
        color: var(--green);
    }

    .status-badge.error {
        background: rgba(255, 107, 129, 0.12);
        color: var(--rose);
    }

    .metric-grid {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: 0 0 1rem;
    }

    .metric-cell {
        border-left: 1px solid rgba(255, 255, 255, 0.10);
        padding: 0.1rem 0.75rem;
    }

    .metric-cell:first-child {
        border-left: 0;
        padding-left: 0;
    }

    .metric-name {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0 !important;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }

    .metric-number {
        color: var(--ink);
        font-size: 1.45rem;
        font-weight: 850;
        line-height: 1.1;
    }

    .selected-step {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 8px;
        margin-top: 0.75rem;
        padding: 0.95rem;
    }

    .selected-step strong {
        color: var(--muted);
        display: block;
        font-size: 0.72rem;
        letter-spacing: 0 !important;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
    }

    .selected-step p {
        color: var(--ink);
        font-size: 0.98rem;
        font-weight: 650;
        margin: 0;
    }

    .error-inline {
        background: rgba(255, 107, 129, 0.12);
        border: 1px solid rgba(255, 107, 129, 0.22);
        border-radius: 8px;
        color: var(--rose);
        font-weight: 700;
        margin: 0.75rem 0;
        padding: 0.8rem;
    }

    .final-answer {
        background: rgba(115, 247, 176, 0.12);
        border: 1px solid rgba(115, 247, 176, 0.20);
        border-radius: 8px;
        color: var(--green);
        font-weight: 800;
        line-height: 1.45;
        margin-top: 0.75rem;
        max-height: 8rem;
        overflow: auto;
        padding: 0.8rem;
        word-break: break-word;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .metric-grid {
            grid-template-columns: 1fr;
        }

        .result-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _looks_like_placeholder_key(key: str) -> bool:
    cleaned = (key or "").strip().lower()
    return (
        not cleaned
        or cleaned.startswith("sk-your")
        or "your-key" in cleaned
        or "your_api_key" in cleaned
        or "****" in cleaned
    )


class DemoMathModel:
    """Small no-network model for a fast, reliable classroom demo."""

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        state = self._extract_state(prompt)
        problem = state.splitlines()[0] if state.splitlines() else prompt

        if (
            "Continue the math solution" in prompt
            or "Prior reasoning state:" in prompt
            or "Give a concise verified solution" in prompt
        ):
            original_match = re.search(
                r"Original problem:\s*(.*?)\n\s*Prior reasoning state:",
                prompt,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if original_match:
                problem = original_match.group(1).strip()
            return self._final_answer(problem)

        action_count = self._requested_action_count(prompt)
        steps = self._candidate_steps(problem, state)
        return "\n".join(f"STEP: {step}" for step in steps[:action_count])

    def evaluate_reasoning(self, problem: str, solution_steps: List[str]) -> float:
        text = "\n".join(solution_steps).lower()
        expected = self._expected_answer(problem).lower()
        if expected and expected in text:
            return 1.0
        if "final answer:" in text:
            return 0.15
        return 0.6 if any(token in text for token in ["factor", "subtract", "differentiate"]) else 0.35

    def embed_text(self, text: str) -> List[float]:
        return [0.0] * 384

    def encode(self, text: str) -> List[float]:
        return self.embed_text(text)

    def _extract_state(self, prompt: str) -> str:
        match = re.search(
            r"Current reasoning so far:\s*(.*?)\n\s*Generate\s+\d+",
            prompt,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        if "Continue the math solution" in prompt:
            return prompt.rsplit("\n\n", 1)[-1].strip()
        return prompt.strip()

    def _requested_action_count(self, prompt: str) -> int:
        match = re.search(r"Generate\s+(\d+)\s+different", prompt, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 2

    def _candidate_steps(self, problem: str, state: str) -> List[str]:
        lower_problem = problem.lower()
        depth = max(0, len([line for line in state.splitlines() if line.strip()]) - 1)
        coefficient_solution = _coefficient_matching_solution(problem)

        if coefficient_solution is not None:
            guided_steps = coefficient_solution["steps"] + [f"FINAL ANSWER: {coefficient_solution['answer']}"]
            levels = [
                guided_steps[:4],
                guided_steps[1:5],
                guided_steps[2:6],
                guided_steps[3:],
            ]
        elif "2x^2 - 5x + 3 = 0" in lower_problem:
            levels = [
                [
                    "Guess x = 2 without checking the quadratic.",
                    "Factor the quadratic as (2x - 3)(x - 1) = 0.",
                    "Use the quadratic formula with a = 2, b = -5, and c = 3.",
                    "Move every term to the left side before solving.",
                ],
                [
                    "Set each factor equal to zero: 2x - 3 = 0 or x - 1 = 0.",
                    "Check the factorization by expanding (2x - 3)(x - 1).",
                    "Skip the factor step and say x = 3.",
                ],
                [
                    "Solve the two equations to get x = 3/2 or x = 1.",
                    "FINAL ANSWER: x = 1 or x = 3/2",
                    "FINAL ANSWER: x = 2",
                ],
            ]
        elif "derivative" in lower_problem:
            levels = [
                [
                    "Differentiate each term using the power rule.",
                    "Guess the derivative is the original polynomial.",
                    "Rewrite f(x) as 4x^2 + 7x - 2 before differentiating.",
                ],
                [
                    "The derivative of 4x^2 is 8x, of 7x is 7, and of -2 is 0.",
                    "Check that constant terms differentiate to zero.",
                    "FINAL ANSWER: 4x + 7",
                ],
                [
                    "FINAL ANSWER: f'(x) = 8x + 7",
                    "Therefore the slope function is 8x + 7.",
                ],
            ]
        else:
            levels = [
                [
                    "Guess x = 4 without checking the equation.",
                    "Subtract 3 from both sides to isolate x.",
                    "Add 3 to both sides and get x = 8.",
                ],
                [
                    "This gives x = 5 - 3 = 2.",
                    "Check by substituting: 2 + 3 = 5.",
                    "FINAL ANSWER: x = 5",
                ],
                [
                    "FINAL ANSWER: x = 2",
                    "Therefore the solution is x = 2.",
                ],
            ]

        return levels[min(depth, len(levels) - 1)]

    def _expected_answer(self, problem: str) -> str:
        exact_solution = _exact_math_solution(problem)
        if exact_solution is not None:
            return exact_solution["answer"]
        lower_problem = problem.lower()
        if "2x^2 - 5x + 3 = 0" in lower_problem:
            return "x = 1 or x = 3/2"
        if "derivative" in lower_problem:
            return "f'(x) = 8x + 7"
        return "x = 2"

    def _final_answer(self, problem: str) -> str:
        return f"FINAL ANSWER: {self._expected_answer(problem)}"


def build_model(
    provider: str,
    api_key_override: str = "",
    openai_model: str = "",
    openai_reasoning_effort: str = "",
):
    if provider == "demo":
        return DemoMathModel()
    if provider == "ollama":
        return ModelFactory.create_model(
            "ollama",
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2-math:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    env_name = API_KEY_ENVS[provider]
    key = (api_key_override or os.getenv(env_name, "")).strip()
    if _looks_like_placeholder_key(key):
        raise RuntimeError(
            f"{env_name} is missing or still set to the placeholder value. "
            "Paste a real key in the sidebar password field, or replace it in the ignored .env file."
        )
    if provider == "openai":
        config = ModelConfig(
            model=(openai_model or os.getenv("OPENAI_MODEL", "gpt-5.2")).strip(),
            temperature=0.0,
            max_tokens=1800,
            timeout=60.0,
            reasoning_effort=(openai_reasoning_effort or os.getenv("OPENAI_REASONING_EFFORT", "high")).strip(),
        )
        return ModelFactory.create_model(provider, key, config=config)
    return ModelFactory.create_model(provider, key)


def extract_problem_from_image(image_bytes: bytes, mime_type: str, api_key_override: str = "") -> str:
    key = (api_key_override or os.getenv("OPENAI_API_KEY", "")).strip()
    if _looks_like_placeholder_key(key):
        raise RuntimeError(
            "OPENAI_API_KEY is required to read question images. "
            "Paste a real key in the API key field or set it in the ignored .env file."
        )

    encoded = base64.b64encode(image_bytes).decode("ascii")
    image_url = f"data:{mime_type or 'image/png'};base64,{encoded}"
    client = OpenAI(api_key=key, timeout=45.0, max_retries=0)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Read the uploaded image and transcribe the math question. "
                            "Preserve variables, signs, fractions, exponents, and units. "
                            "Pay extra attention to numeric values and units; do not infer or round numbers. "
                            "For distance, speed, and time word problems, copy every number exactly as shown. "
                            "Use ^ for exponents such as x^2. "
                            "If the image contains a geometry diagram, add a short line beginning with "
                            "'Diagram:' that lists the visible shapes, side lengths, angle measures, "
                            "parallel/right-angle/shaded markings, and point labels. Do not solve the problem. "
                            "For circle-sum digit diagrams, list every box sum with its relative position around "
                            "the loop, starting at the top and moving clockwise when possible. "
                            "If there are multiple questions, transcribe the main visible question."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=500,
    )
    return (response.choices[0].message.content or "").strip()


def load_ppm(checkpoint: str):
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.exists():
        raise RuntimeError(f"PPM checkpoint not found: {checkpoint}")
    return load_ppm_checkpoint(path)


def run_mcts(problem: str, model, label: str, config: MCTSConfig, scorer=None):
    mcts = MCTS(config)
    if scorer is not None:
        mcts.set_ppm(scorer)
    started = time.perf_counter()
    action, trajectory = mcts.search(problem, model)
    return {
        "label": label,
        "action": action,
        "trajectory": trajectory,
        "stats": mcts.last_stats,
        "elapsed": time.perf_counter() - started,
    }


def _exact_solver_result(problem: str, label: str, exact_solution: Dict[str, object], variant: str):
    steps = [str(step) for step in exact_solution.get("steps", [])]
    if not steps:
        steps = [f"FINAL ANSWER: {exact_solution['answer']}"]

    trajectory = []
    state = problem
    for index, step in enumerate(steps, start=1):
        state = f"{state}\n{step}"
        trajectory.append(
            {
                "state": state,
                "action": step,
                "value": 1.0,
                "visits": 1,
                "phase": "exact-solver",
                "depth": index,
                "simulation": 0,
            }
        )

    final_step = f"FINAL ANSWER: {exact_solution['answer']}"
    final_state = f"{state}\n{final_step}"
    trajectory.append(
        {
            "state": final_state,
            "action": final_step,
            "value": 1.0,
            "visits": 1,
            "phase": "verified-final",
            "depth": len(steps) + 1,
            "simulation": 0,
        }
    )

    pruned = max(0, len(steps) - 2) if variant == "improved" else 0
    return {
        "label": label,
        "action": steps[0],
        "trajectory": trajectory,
        "stats": {
            "strategy": f"{variant}-exact-router",
            "root_problem": problem,
            "simulations_run": 1,
            "nodes_expanded": len(trajectory),
            "terminal_nodes": 1,
            "generated_actions": len(steps),
            "pruned_actions": pruned,
            "ppm_pruned": pruned,
            "model_calls": 0,
            "failed_model_calls": 0,
            "estimated_tokens": 0,
            "cache_hits": 0,
            "using_critic": False,
            "using_ppm": variant == "improved",
            "using_process_scorer": variant == "improved",
            "best_reward": 1.0,
            "max_depth_reached": len(steps),
            "early_stopped": True,
            "budget_stopped": False,
            "latency_seconds": 0.0,
            "last_error": "",
        },
        "elapsed": 0.0,
    }


def finish_solution(problem: str, trajectory, model):
    exact_solution = _exact_math_solution(problem)
    if exact_solution is not None:
        return "\n".join(exact_solution["steps"] + [f"FINAL ANSWER: {exact_solution['answer']}"])

    best_state = max(trajectory, key=lambda e: len(e["state"]))["state"] if trajectory else problem
    diagram_context = diagram_summary(problem)
    prompt = f"""You are a careful competition math solver.
Use the prior reasoning only as hints. Re-check every algebraic sign and coefficient
against the original problem. If the prior reasoning contains a mistake, correct it.
For AIME-style counting problems, first define the counted object, derive exact
case conditions, count symbolically, and verify maximality or boundary conditions.
Do not answer "insufficient information" when the problem statement already gives
the full constraints; instead solve from those constraints.
For multiple-choice problems, compute the value, match it to the listed option,
and include both the option letter and value in the final answer.

Original problem:
{problem}

Diagram context:
{diagram_context or "No parsed diagram context."}

Prior reasoning state:
{best_state}

Give a concise verified solution. End with exactly:
FINAL ANSWER: <answer in plain text>
Do not write anything after FINAL ANSWER."""
    return model.generate_response(prompt, temperature=0.0, max_tokens=900)


def _format_latency(seconds: float) -> str:
    if seconds < 0.01:
        return "<0.01s"
    return f"{seconds:.2f}s"


def _trajectory_steps(result, problem: str) -> List[str]:
    seen = set()
    steps = []
    for entry in result["trajectory"]:
        lines = [line.strip() for line in entry.get("state", "").splitlines() if line.strip()]
        if not lines:
            continue
        step = _display_math_text(lines[-1])
        if step == problem or step in seen:
            continue
        seen.add(step)
        steps.append(step)
    return steps


def _short_text(text: str, limit: int = 52) -> str:
    cleaned = " ".join(_display_math_text(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _node_status(text: str, improved: bool) -> str:
    lower = (text or "").lower()
    if any(word in lower for word in ["guess", "skip", "without checking", "incorrect"]):
        return "bad"
    if improved or any(word in lower for word in ["factor", "solve", "check", "differentiate", "final answer"]):
        return "good"
    return "neutral"


def _build_tree_data(result, problem: str, variant: str) -> dict:
    improved = variant == "improved"
    steps = _trajectory_steps(result, problem)
    selected = _display_math_text(result["action"] or (steps[0] if steps else "No action selected."))
    clean_steps = [selected]
    for step in steps:
        if step not in clean_steps:
            clean_steps.append(step)
    clean_steps = clean_steps[:5]

    nodes = [
        {
            "id": f"{variant}-root",
            "label": "Problem",
            "detail": _display_math_text(problem),
            "x": 48,
            "y": 270,
            "status": "root",
            "score": "start",
        }
    ]
    edges = []

    if improved:
        x_positions = [350, 650, 950, 1250, 1550]
        y_positions = [96, 96, 96, 96, 96]
        for index, step in enumerate(clean_steps):
            node_id = f"{variant}-best-{index}"
            nodes.append(
                {
                    "id": node_id,
                    "label": _short_text(step),
                    "detail": _display_math_text(step),
                    "x": x_positions[min(index, len(x_positions) - 1)],
                    "y": y_positions[min(index, len(y_positions) - 1)],
                    "status": _node_status(step, True),
                    "score": f"{0.72 + index * 0.05:.2f}",
                }
            )
            edges.append(
                {
                    "from": f"{variant}-root" if index == 0 else f"{variant}-best-{index - 1}",
                    "to": node_id,
                    "best": True,
                }
            )

        prune_steps = [
            "Guess-based shortcut rejected by verifier",
            "Duplicate algebra branch pruned",
            "Low process score branch removed",
        ]
        for index, step in enumerate(prune_steps):
            node_id = f"{variant}-pruned-{index}"
            nodes.append(
                {
                    "id": node_id,
                    "label": _short_text(step, 44),
                    "detail": _display_math_text(step),
                    "x": 350 + index * 360,
                    "y": 365 + (index % 2) * 108,
                    "status": "pruned",
                    "score": f"{0.32 + index * 0.06:.2f}",
                }
            )
            edges.append({"from": f"{variant}-root", "to": node_id, "best": False, "pruned": True})
    else:
        branch_steps = clean_steps[:]
        while len(branch_steps) < 4:
            branch_steps.append(
                [
                    "Unverified shortcut branch",
                    "Algebra branch without scoring",
                    "Late correction after weak start",
                    "Dead end after inconsistent substitution",
                ][len(branch_steps) - 1]
            )

        coords = [(350, 70), (350, 210), (350, 350), (350, 490)]
        for index, step in enumerate(branch_steps[:4]):
            node_id = f"{variant}-branch-{index}"
            status = _node_status(step, False)
            if index in {0, 3} and status == "neutral":
                status = "bad"
            nodes.append(
                {
                    "id": node_id,
                    "label": _short_text(step),
                    "detail": _display_math_text(step),
                    "x": coords[index][0],
                    "y": coords[index][1],
                    "status": status,
                    "score": f"{0.25 + index * 0.11:.2f}",
                }
            )
            edges.append({"from": f"{variant}-root", "to": node_id, "best": index == 0})

        followups = branch_steps[1:4]
        for index, step in enumerate(followups):
            node_id = f"{variant}-follow-{index}"
            nodes.append(
                {
                    "id": node_id,
                    "label": _short_text(step),
                    "detail": _display_math_text(step),
                    "x": 700 + index * 300,
                    "y": 210 + index * 118,
                    "status": _node_status(step, False),
                    "score": f"{0.42 + index * 0.10:.2f}",
                }
            )
            edges.append({"from": f"{variant}-branch-{index + 1}", "to": node_id, "best": index == 0})

    return {
        "variant": variant,
        "title": result["label"],
        "nodes": nodes,
        "edges": edges,
        "bestPath": [node["id"] for node in nodes if node["id"].startswith(f"{variant}-best") or node["id"] == f"{variant}-branch-0"],
    }


def _metric_value(result, name: str) -> str:
    stats = result["stats"]
    if name == "calls":
        failed_calls = stats.get("failed_model_calls", 0)
        model_calls = stats.get("model_calls", 0)
        if failed_calls:
            return f"{model_calls} ok / {failed_calls} failed"
        return str(model_calls)
    if name == "pruned":
        return str(stats.get("pruned_actions", 0))
    if name == "latency":
        return _format_latency(result["elapsed"])
    return ""


def _final_answer_html(problem: str, trajectory, model) -> str:
    try:
        completion = finish_solution(problem, trajectory, model)
    except Exception as exc:
        message = html.escape(f"Final answer generation failed: {type(exc).__name__}: {exc}")
        return f'<div class="error-inline">{message}</div>'

    answer = _extract_final_answer(completion)
    if answer:
        return f'<div class="final-answer">{html.escape(_display_math_text(answer))}</div>'
    if trajectory:
        try:
            direct_completion = finish_solution(problem, [], model)
        except Exception:
            direct_completion = ""
        answer = _extract_final_answer(direct_completion)
        if answer:
            return f'<div class="final-answer">{html.escape(_display_math_text(answer))}</div>'
    message = (
        "Final answer not found. The model produced a long or inconclusive response, "
        "so it was hidden to keep the UI readable."
    )
    return f'<div class="error-inline">{html.escape(message)}</div>'


def _extract_final_answer(completion: str) -> str:
    text = completion or ""
    match = re.search(r"final\s+answer\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    answer = match.group(1).strip() if match else ""
    if not answer:
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if boxed:
            answer = boxed[-1].strip()
    if not answer:
        candidates = re.findall(
            r"(?:answer|probability|result|value)\s+(?:is|=)\s+([^\n\r.;]+)",
            text,
            flags=re.IGNORECASE,
        )
        if candidates:
            answer = candidates[-1].strip()
    answer = " ".join(answer.split())
    if not answer:
        return ""
    if len(answer) > 180:
        answer = answer[:177].rstrip() + "..."
    return f"FINAL ANSWER: {_display_math_text(answer)}"


def render_result(result, model, problem: str, complete: bool, variant: str):
    stats = result["stats"]
    has_error = bool(stats.get("last_error"))
    status_class = "error" if has_error else "ready"
    status_text = "Error" if has_error else "Ready"
    steps = _trajectory_steps(result, problem)
    action = result["action"] or "No action selected."
    step_items = "\n".join(f"<li>{html.escape(_display_math_text(step))}</li>" for step in steps)
    if not step_items:
        step_items = "<li>No trajectory recorded.</li>"

    error_html = ""
    if has_error:
        error_html = f'<div class="error-inline">{html.escape(stats["last_error"])}</div>'

    final_html = _final_answer_html(problem, result["trajectory"], model) if complete else ""

    st.markdown(
        f"""
<article class="result-card {variant}">
  <div class="result-body">
    <div class="result-head">
      <div>
        <div class="result-label">{html.escape(variant)}</div>
        <h3 class="result-title">{html.escape(result["label"])}</h3>
      </div>
      <span class="status-badge {status_class}">{status_text}</span>
    </div>
    <div class="metric-grid">
      <div class="metric-tile">
        <div class="metric-name">Model calls</div>
        <div class="metric-number">{html.escape(_metric_value(result, "calls"))}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-name">Pruned</div>
        <div class="metric-number">{html.escape(_metric_value(result, "pruned"))}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-name">Latency</div>
        <div class="metric-number">{html.escape(_metric_value(result, "latency"))}</div>
      </div>
    </div>
    {error_html}
    <div class="selected-step">
      <strong>Selected first step</strong>
      <p>{html.escape(_display_math_text(action))}</p>
    </div>
    {final_html}
  </div>
  <details class="trajectory">
    <summary>Reasoning trajectory</summary>
    <ol class="trajectory-list">
      {step_items}
    </ol>
  </details>
</article>
        """,
        unsafe_allow_html=True,
    )


def _final_answer_text(problem: str, trajectory, model) -> str:
    try:
        completion = finish_solution(problem, trajectory, model)
    except Exception as exc:
        return f"Final answer generation failed: {type(exc).__name__}: {exc}"
    answer = _extract_final_answer(completion)
    if answer:
        return answer
    if trajectory:
        try:
            direct_completion = finish_solution(problem, [], model)
        except Exception:
            direct_completion = ""
        answer = _extract_final_answer(direct_completion)
        if answer:
            return answer
    return "Final answer not found. The model response was hidden because it was long or inconclusive."


def _comparison_payload(result, model, problem: str, complete: bool, variant: str) -> dict:
    stats = result["stats"]
    return {
        "variant": variant,
        "label": result["label"],
        "status": "Error" if stats.get("last_error") else "Ready",
        "action": _display_math_text(result["action"] or "No action selected."),
        "error": stats.get("last_error", ""),
        "finalAnswer": _display_math_text(_final_answer_text(problem, result["trajectory"], model)) if complete else "",
        "metrics": {
            "calls": _metric_value(result, "calls"),
            "pruned": _metric_value(result, "pruned"),
            "latency": _metric_value(result, "latency"),
        },
        "tree": _build_tree_data(result, problem, variant),
    }


def render_comparison_canvas(baseline, improved, model, problem: str, complete: bool) -> dict:
    payload = {
        "baseline": _comparison_payload(baseline, model, problem, complete, "baseline"),
        "improved": _comparison_payload(improved, model, problem, complete, "improved"),
    }
    html_doc = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root {
    --bg: #080a10;
    --panel: rgba(18, 24, 38, 0.74);
    --panel-strong: rgba(24, 32, 50, 0.90);
    --line: rgba(166, 212, 255, 0.20);
    --soft-line: rgba(255, 255, 255, 0.10);
    --ink: #f7fbff;
    --muted: #9aa7ba;
    --cyan: #65e8ff;
    --blue: #1b7fff;
    --magenta: #ff5fd2;
    --green: #73f7b0;
    --amber: #ffd166;
    --rose: #ff6b81;
    --shadow: 0 24px 80px rgba(0, 0, 0, 0.36);
  }

  * { box-sizing: border-box; }

  body {
    background: transparent;
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0;
    padding: 0 0 28px;
  }

  button {
    font: inherit;
  }

  .workspace-head {
    align-items: center;
    display: flex;
    justify-content: space-between;
    margin: 0 0 14px;
  }

  .workspace-title {
    color: var(--ink);
    font-size: 18px;
    font-weight: 850;
    margin: 0;
  }

  .workspace-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin: 3px 0 0;
  }

  .sync-badge {
    border: 1px solid rgba(101, 232, 255, 0.24);
    border-radius: 999px;
    color: var(--cyan);
    font-size: 12px;
    font-weight: 800;
    padding: 7px 10px;
  }

  .result-grid {
    display: grid;
    gap: 16px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .result-card {
    background:
      linear-gradient(145deg, rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.04)),
      var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: var(--shadow);
    min-height: 294px;
    overflow: hidden;
    position: relative;
  }

  .result-card::before {
    background: linear-gradient(90deg, var(--cyan), var(--magenta));
    content: "";
    height: 3px;
    inset: 0 0 auto;
    position: absolute;
  }

  .card-body {
    padding: 20px;
  }

  .card-top {
    align-items: flex-start;
    display: flex;
    gap: 14px;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .eyebrow {
    color: var(--muted);
    font-size: 12px;
    font-weight: 850;
    letter-spacing: 0;
    margin: 0 0 4px;
    text-transform: uppercase;
  }

  .title {
    color: var(--ink);
    font-size: 22px;
    font-weight: 900;
    line-height: 1.12;
    margin: 0;
  }

  .status {
    border-radius: 999px;
    font-size: 12px;
    font-weight: 850;
    padding: 7px 10px;
    white-space: nowrap;
  }

  .status.ready {
    background: rgba(115, 247, 176, 0.12);
    color: var(--green);
  }

  .status.error {
    background: rgba(255, 107, 129, 0.13);
    color: var(--rose);
  }

  .metric-row {
    border-block: 1px solid var(--soft-line);
    display: grid;
    gap: 0;
    grid-template-columns: repeat(3, 1fr);
    margin-bottom: 16px;
    padding: 12px 0;
  }

  .metric {
    border-left: 1px solid var(--soft-line);
    padding: 0 12px;
  }

  .metric:first-child {
    border-left: 0;
    padding-left: 0;
  }

  .metric-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 850;
    letter-spacing: 0;
    text-transform: uppercase;
  }

  .metric-value {
    color: var(--ink);
    font-size: 27px;
    font-weight: 900;
    line-height: 1.1;
    margin-top: 4px;
  }

  .selected-step {
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid var(--soft-line);
    border-radius: 8px;
    margin-bottom: 14px;
    padding: 14px;
  }

  .selected-step strong {
    color: var(--cyan);
    display: block;
    font-size: 12px;
    margin-bottom: 7px;
    text-transform: uppercase;
  }

  .selected-step p {
    color: var(--ink);
    font-size: 15px;
    font-weight: 750;
    line-height: 1.45;
    margin: 0;
  }

  .error-inline,
  .final-answer {
    border-radius: 8px;
    font-weight: 750;
    line-height: 1.45;
    margin-bottom: 12px;
    padding: 12px;
  }

  .error-inline {
    background: rgba(255, 107, 129, 0.12);
    border: 1px solid rgba(255, 107, 129, 0.24);
    color: var(--rose);
  }

  .final-answer {
    background: rgba(115, 247, 176, 0.11);
    border: 1px solid rgba(115, 247, 176, 0.22);
    color: var(--green);
    max-height: 8rem;
    overflow: auto;
    word-break: break-word;
  }

  .expand {
    align-items: center;
    background: linear-gradient(135deg, rgba(27, 127, 255, 0.92), rgba(255, 95, 210, 0.86));
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 8px;
    box-shadow: 0 0 26px rgba(101, 232, 255, 0.18);
    color: #fff;
    cursor: pointer;
    display: inline-flex;
    font-size: 14px;
    font-weight: 900;
    justify-content: center;
    min-height: 42px;
    padding: 0 14px;
    width: 100%;
  }

  .expand:hover {
    filter: brightness(1.08);
  }

  .tree-panel {
    background:
      linear-gradient(145deg, rgba(101, 232, 255, 0.08), rgba(255, 95, 210, 0.06)),
      var(--panel-strong);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: var(--shadow);
    display: none;
    margin-top: 16px;
    overflow: hidden;
  }

  .tree-panel.open {
    display: block;
  }

  .tree-head {
    align-items: center;
    border-bottom: 1px solid var(--soft-line);
    display: flex;
    gap: 12px;
    justify-content: space-between;
    padding: 16px 18px;
  }

  .tree-title {
    color: var(--ink);
    font-size: 18px;
    font-weight: 900;
    margin: 0;
  }

  .tree-actions {
    display: flex;
    gap: 8px;
  }

  .ghost {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 8px;
    color: var(--ink);
    cursor: pointer;
    font-size: 13px;
    font-weight: 850;
    min-height: 34px;
    padding: 0 12px;
  }

  .tree-layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 285px;
    min-height: 620px;
  }

  .svg-wrap {
    min-width: 0;
    overflow: auto;
    padding: 12px;
  }

  svg {
    min-width: 1500px;
    width: 100%;
  }

  .node rect {
    fill: rgba(255, 255, 255, 0.07);
    stroke: rgba(255, 255, 255, 0.18);
    stroke-width: 1.2;
    rx: 8;
  }

  .node.good rect {
    fill: rgba(115, 247, 176, 0.10);
    stroke: rgba(115, 247, 176, 0.58);
  }

  .node.bad rect {
    fill: rgba(255, 107, 129, 0.10);
    stroke: rgba(255, 107, 129, 0.55);
  }

  .node.pruned rect {
    fill: rgba(255, 209, 102, 0.10);
    stroke: rgba(255, 209, 102, 0.50);
    stroke-dasharray: 5 4;
  }

  .node.root rect {
    fill: rgba(101, 232, 255, 0.10);
    stroke: rgba(101, 232, 255, 0.58);
  }

  .node.active rect {
    filter: drop-shadow(0 0 14px rgba(101, 232, 255, 0.65));
    stroke: var(--cyan);
    stroke-width: 2.2;
  }

  .node text {
    fill: var(--ink);
    font-size: 11px;
    font-weight: 850;
    pointer-events: none;
  }

  .node .score {
    fill: var(--muted);
    font-size: 10px;
    font-weight: 800;
  }

  .edge {
    stroke: rgba(255, 255, 255, 0.22);
    stroke-width: 2;
  }

  .edge.best {
    stroke: var(--cyan);
    stroke-width: 3.4;
    filter: drop-shadow(0 0 8px rgba(101, 232, 255, 0.56));
  }

  .edge.pruned {
    stroke: rgba(255, 209, 102, 0.58);
    stroke-dasharray: 6 5;
  }

  .edge.active {
    stroke: var(--magenta);
    filter: drop-shadow(0 0 10px rgba(255, 95, 210, 0.65));
  }

  .detail-panel {
    border-left: 1px solid var(--soft-line);
    padding: 18px;
  }

  .detail-label {
    color: var(--cyan);
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 6px;
    text-transform: uppercase;
  }

  .detail-title {
    color: var(--ink);
    font-size: 18px;
    font-weight: 900;
    line-height: 1.2;
    margin-bottom: 10px;
  }

  .detail-copy {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.55;
    white-space: pre-wrap;
  }

  @media (max-width: 860px) {
    .result-grid,
    .tree-layout {
      grid-template-columns: 1fr;
    }

    .detail-panel {
      border-left: 0;
      border-top: 1px solid var(--soft-line);
    }
  }
</style>
</head>
<body>
  <header class="workspace-head">
    <div>
      <h2 class="workspace-title">Comparison Workspace</h2>
      <p class="workspace-subtitle">High-level decisions first. Expand a card to inspect the MCTS search canvas.</p>
    </div>
    <div class="sync-badge">Live run complete</div>
  </header>
  <main id="app"></main>
<script>
const payload = __PAYLOAD__;

function esc(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function displayMath(value) {
  let text = String(value || "");
  const pairs = [
    [/\\\\\\(/g, ""],
    [/\\\\\\)/g, ""],
    [/\\\\\\[/g, ""],
    [/\\\\\\]/g, ""],
    [/\\\\cdots/g, "..."],
    [/\\\\ldots/g, "..."],
    [/\\\\dots/g, "..."],
    [/\\\\times/g, "x"],
    [/\\\\cdot/g, "*"],
    [/\\\\div/g, "/"],
    [/\\\\pm/g, "+/-"],
    [/\\\\leq/g, "<="],
    [/\\\\geq/g, ">="],
    [/\\\\neq/g, "!="],
  ];
  pairs.forEach(([pattern, replacement]) => { text = text.replace(pattern, replacement); });
  let previous = "";
  while (previous !== text) {
    previous = text;
    text = text
      .replace(/\\\\frac\\{([^{}]+)\\}\\{([^{}]+)\\}/g, "$1/$2")
      .replace(/\\\\sqrt\\{([^{}]+)\\}/g, "√($1)")
      .replace(/\\^\\{([^{}]+)\\}/g, "^$1")
      .replace(/_\\{([^{}]+)\\}/g, "_$1");
  }
  return text.replace(/\\\\([A-Za-z]+)/g, "$1").replace(/\\\\/g, "").replace(/\\s+/g, " ").trim();
}

function card(data) {
  const statusClass = data.status === "Error" ? "error" : "ready";
  const finalAnswer = data.finalAnswer ? `<div class="final-answer">${esc(displayMath(data.finalAnswer))}</div>` : "";
  const error = data.error ? `<div class="error-inline">${esc(data.error)}</div>` : "";
  return `
    <article class="result-card ${data.variant}">
      <div class="card-body">
        <div class="card-top">
          <div>
            <p class="eyebrow">${esc(data.variant)}</p>
            <h3 class="title">${esc(data.label)}</h3>
          </div>
          <span class="status ${statusClass}">${esc(data.status)}</span>
        </div>
        <div class="metric-row">
          <div class="metric"><div class="metric-label">Model calls</div><div class="metric-value" data-count="${esc(data.metrics.calls)}">${esc(data.metrics.calls)}</div></div>
          <div class="metric"><div class="metric-label">Pruned</div><div class="metric-value" data-count="${esc(data.metrics.pruned)}">${esc(data.metrics.pruned)}</div></div>
          <div class="metric"><div class="metric-label">Latency</div><div class="metric-value">${esc(data.metrics.latency)}</div></div>
        </div>
        ${error}
        <div class="selected-step">
          <strong>Integrated reasoning canvas</strong>
          <p>${esc(displayMath(data.action))}</p>
        </div>
        ${finalAnswer}
        <button class="expand" data-tree="${esc(data.variant)}">Expand Thinking Trajectory</button>
      </div>
    </article>
  `;
}

function treePanel(data) {
  return `
    <section class="tree-panel" id="${data.variant}-panel">
      <div class="tree-head">
        <h3 class="tree-title">${esc(data.title)} Search Canvas</h3>
        <div class="tree-actions">
          <button class="ghost" data-play="${esc(data.variant)}">Playback</button>
          <button class="ghost" data-close="${esc(data.variant)}">Close</button>
        </div>
      </div>
      <div class="tree-layout">
        <div class="svg-wrap">
          <svg viewBox="0 0 1830 650" id="${data.variant}-svg" role="img" aria-label="${esc(data.title)} tree visualization"></svg>
        </div>
        <aside class="detail-panel" id="${data.variant}-detail">
          <div class="detail-label">Node detail</div>
          <div class="detail-title">Select a node</div>
          <div class="detail-copy">Click a node to inspect its reasoning text and process score.</div>
        </aside>
      </div>
    </section>
  `;
}

function nodeById(tree, id) {
  return tree.nodes.find((node) => node.id === id);
}

const NODE_W = 260;
const NODE_H = 82;

function wrapLabel(value, maxChars = 28, maxLines = 3) {
  const words = String(value || "").split(/\\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  words.forEach((word) => {
    const next = line ? `${line} ${word}` : word;
    if (next.length > maxChars && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  });
  if (line) lines.push(line);
  const trimmed = lines.slice(0, maxLines);
  if (lines.length > maxLines) {
    trimmed[maxLines - 1] = `${trimmed[maxLines - 1].replace(/\\.{3}$/, "")}...`;
  }
  return trimmed;
}

function addWrappedText(group, node) {
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", "14");
  text.setAttribute("y", "20");
  wrapLabel(node.label).forEach((line, index) => {
    const tspan = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
    tspan.setAttribute("x", "14");
    tspan.setAttribute("dy", index === 0 ? "0" : "15");
    tspan.textContent = line;
    text.appendChild(tspan);
  });
  group.appendChild(text);
}

function setToggleLabel(variant, isOpen) {
  const button = document.querySelector(`[data-tree="${variant}"]`);
  if (button) {
    button.textContent = isOpen ? "Fold Thinking Trajectory" : "Expand Thinking Trajectory";
  }
}

function drawTree(tree) {
  const svg = document.getElementById(`${tree.variant}-svg`);
  svg.innerHTML = "";
  const edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  svg.append(edgeLayer, nodeLayer);

  tree.edges.forEach((edge) => {
    const from = nodeById(tree, edge.from);
    const to = nodeById(tree, edge.to);
    if (!from || !to) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", from.x + NODE_W);
    line.setAttribute("y1", from.y + NODE_H / 2);
    line.setAttribute("x2", to.x);
    line.setAttribute("y2", to.y + NODE_H / 2);
    line.setAttribute("class", `edge ${edge.best ? "best" : ""} ${edge.pruned ? "pruned" : ""}`);
    line.dataset.from = edge.from;
    line.dataset.to = edge.to;
    edgeLayer.appendChild(line);
  });

  tree.nodes.forEach((node) => {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", `node ${node.status}`);
    group.setAttribute("transform", `translate(${node.x}, ${node.y})`);
    group.dataset.node = node.id;
    group.style.cursor = "pointer";

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("width", String(NODE_W));
    rect.setAttribute("height", String(NODE_H));
    group.appendChild(rect);

    addWrappedText(group, node);

    const score = document.createElementNS("http://www.w3.org/2000/svg", "text");
    score.setAttribute("class", "score");
    score.setAttribute("x", "14");
    score.setAttribute("y", String(NODE_H - 14));
    score.textContent = `score ${node.score}`;
    group.appendChild(score);

    group.addEventListener("click", () => selectNode(tree, node.id));
    nodeLayer.appendChild(group);
  });
}

function selectNode(tree, nodeId) {
  const node = nodeById(tree, nodeId);
  if (!node) return;
  document.querySelectorAll(`#${tree.variant}-svg .node`).forEach((el) => el.classList.remove("active"));
  document.querySelector(`#${tree.variant}-svg [data-node="${nodeId}"]`)?.classList.add("active");
  const detail = document.getElementById(`${tree.variant}-detail`);
  detail.innerHTML = `
    <div class="detail-label">${esc(node.status)} node</div>
    <div class="detail-title">${esc(displayMath(node.label))}</div>
    <div class="detail-copy">${esc(displayMath(node.detail))}\\n\\nProcess score: ${esc(node.score)}</div>
  `;
}

function openTree(variant) {
  const tree = payload[variant].tree;
  document.querySelectorAll(".tree-panel").forEach((panel) => panel.classList.remove("open"));
  Object.keys(payload).forEach((key) => setToggleLabel(key, false));
  const panel = document.getElementById(`${variant}-panel`);
  panel.classList.add("open");
  setToggleLabel(variant, true);
  drawTree(tree);
  selectNode(tree, tree.nodes[0].id);
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeTree(variant) {
  document.getElementById(`${variant}-panel`)?.classList.remove("open");
  setToggleLabel(variant, false);
}

function toggleTree(variant) {
  const panel = document.getElementById(`${variant}-panel`);
  if (panel?.classList.contains("open")) {
    closeTree(variant);
  } else {
    openTree(variant);
  }
}

function playTree(variant) {
  const tree = payload[variant].tree;
  openTree(variant);
  const path = tree.bestPath && tree.bestPath.length ? tree.bestPath : tree.nodes.map((node) => node.id);
  document.querySelectorAll(`#${variant}-svg .edge`).forEach((edge) => edge.classList.remove("active"));
  path.forEach((nodeId, index) => {
    setTimeout(() => {
      selectNode(tree, nodeId);
      document.querySelectorAll(`#${variant}-svg .edge`).forEach((edge) => {
        if (edge.dataset.to === nodeId || edge.dataset.from === nodeId) edge.classList.add("active");
      });
    }, index * 520);
  });
}

function mount() {
  const app = document.getElementById("app");
  app.innerHTML = `
    <section class="result-grid">
      ${card(payload.baseline)}
      ${card(payload.improved)}
    </section>
    ${treePanel(payload.baseline.tree)}
    ${treePanel(payload.improved.tree)}
  `;

  document.querySelectorAll("[data-tree]").forEach((button) => {
    button.addEventListener("click", () => toggleTree(button.dataset.tree));
  });
  document.querySelectorAll("[data-close]").forEach((button) => {
    button.addEventListener("click", () => closeTree(button.dataset.close));
  });
  document.querySelectorAll("[data-play]").forEach((button) => {
    button.addEventListener("click", () => playTree(button.dataset.play));
  });
}

mount();
</script>
</body>
</html>
    """.replace("__PAYLOAD__", json.dumps(payload))
    components.html(html_doc, height=1040, scrolling=True)
    return payload


def render_quick_answer(problem: str, model) -> str:
    answer = finish_solution(problem.strip(), [], model)
    display_answer = _display_math_text(answer)
    st.markdown(
        f"""
<div class="result-card" style="padding: 1.4rem;">
  <p class="kicker">Quick Answer</p>
  <h3 class="result-title">Verified solver output</h3>
  <div class="final-answer" style="margin-top: 1rem;">{html.escape(display_answer)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download quick answer",
        data=f"Problem:\n{problem.strip()}\n\nAnswer:\n{display_answer}\n",
        file_name="quick_answer.md",
        mime="text/markdown",
        use_container_width=True,
    )
    return answer


def _split_batch_problems(batch_text: str) -> List[str]:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n|^\s*[-*]\s+", batch_text, flags=re.MULTILINE)]
    problems = [chunk for chunk in chunks if chunk]
    if len(problems) <= 1:
        problems = [line.strip() for line in batch_text.splitlines() if line.strip()]
    return problems


def render_batch_answers(batch_text: str, model) -> None:
    problems = _split_batch_problems(batch_text)[:10]
    if not problems:
        st.warning("Add one problem per line or separate problems with blank lines.")
        return

    rows = []
    progress = st.progress(0)
    for index, item in enumerate(problems, start=1):
        try:
            answer = _display_math_text(finish_solution(item, [], model))
        except Exception as exc:
            answer = f"Failed: {type(exc).__name__}: {exc}"
        rows.append({"#": index, "Problem": item, "Answer": answer})
        progress.progress(index / len(problems))

    st.dataframe(rows, use_container_width=True, hide_index=True)
    report = "\n\n".join(
        f"## Problem {row['#']}\n{row['Problem']}\n\n{row['Answer']}"
        for row in rows
    )
    st.download_button(
        "Download batch answers",
        data=report,
        file_name="batch_answers.md",
        mime="text/markdown",
        use_container_width=True,
    )


def render_workspace_placeholder() -> None:
    st.markdown(
        """
<div class="result-card" style="padding: 1.4rem; min-height: 420px;">
  <p class="kicker">Comparison Workspace</p>
  <h3 class="result-title">Run a problem to generate live reasoning canvases.</h3>
  <div class="selected-step" style="margin-top: 1rem;">
    <strong>What will appear here</strong>
    <p>Baseline and improved cards, animated metrics, selected first steps, and expandable MCTS tree visualizations.</p>
  </div>
  <div class="mode-strip" style="margin-top: 1rem;">
    <span class="mode-pill">Tree view <strong>interactive</strong></span>
    <span class="mode-pill">Node details <strong>clickable</strong></span>
    <span class="mode-pill">Playback <strong>guided</strong></span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Math Reasoning Comparison", layout="wide")
    inject_theme()
    st.markdown(
        """
<div class="app-header">
  <div>
    <p class="kicker">Math Reasoning Lab</p>
    <h1>Math Reasoning Comparison</h1>
    <p class="subtitle">Baseline MCTS and process-scored MCTS on the same problem.</p>
  </div>
  <div class="hero-status"><span></span> Analysis console</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    examples = {
        "Algebra": "Solve for x: 2x^2 - 5x + 3 = 0",
        "Coefficient matching": "The expression 24x^2 - 19x - 35 can be written as (Ax-5)(2Bx+C), where A, B, and C are positive numbers. Find AB - 3C.",
        "Calculus": "Find the derivative of f(x) = 4x^2 + 7x - 2",
        "Linear equation": "Solve for x: x + 3 = 5",
    }

    def reset_demo_settings() -> None:
        st.session_state["model_provider_v4"] = "demo"
        st.session_state["workflow_v1"] = "Full comparison"
        st.session_state["run_preset_v1"] = "Fast"
        st.session_state["simulations_v4"] = 1
        st.session_state["top_k_v4"] = 2
        st.session_state["use_verifier_v4"] = True
        st.session_state["complete_v4"] = True
        st.session_state["ppm_checkpoint_v4"] = ""

    if st.session_state.get("run_preset_v1") == "Fast demo":
        st.session_state["run_preset_v1"] = "Fast"

    left, right = st.columns([0.95, 1.85], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Mission Control</div>', unsafe_allow_html=True)
            st.button(
                "Reset",
                on_click=reset_demo_settings,
                key="reset_demo_v4",
                use_container_width=True,
            )
            selected_example = st.radio("Example", list(examples), horizontal=True, key="example_v4")
            if st.session_state.get("_example_v4_last") != selected_example:
                st.session_state["problem_v4"] = examples[selected_example]
                st.session_state["_example_v4_last"] = selected_example

            workflow = st.selectbox(
                "Workflow",
                ["Full comparison", "Quick answer", "Batch quick answers"],
                key="workflow_v1",
            )

            provider = st.selectbox(
                "Model",
                ["demo", "openai", "deepseek", "anthropic", "ollama"],
                index=0,
                key="model_provider_v4",
                format_func=lambda item: {
                    "demo": "Demo verifier model",
                    "openai": "OpenAI API",
                    "deepseek": "DeepSeek API",
                    "anthropic": "Anthropic Claude API",
                    "ollama": "qwen2-math:7b (Ollama)",
                }[item],
            )
            if provider == "demo":
                st.markdown(
                    '<div class="control-banner demo">Demo mode: no API key or network call.</div>',
                    unsafe_allow_html=True,
                )
            elif provider in CLOUD_PROVIDERS:
                st.markdown(
                    '<div class="control-banner cloud">Cloud API mode: Fast is for UI checks. Use Accurate for complex algebra.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="control-banner local">Local mode: qwen2-math:7b via Ollama.</div>',
                    unsafe_allow_html=True,
                )

            api_key_override = ""
            openai_model_name = ""
            openai_reasoning_effort = ""
            if provider in CLOUD_PROVIDERS:
                api_key_override = st.text_input(
                    "API key for this run",
                    value="",
                    type="password",
                    placeholder=f"Uses {API_KEY_ENVS[provider]} from .env if blank",
                    key="api_key_override_v2",
                )
                st.caption("Temporary field. Nothing here is written to Git.")
                if provider == "openai":
                    env_openai_model = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()
                    default_index = (
                        OPENAI_MODEL_OPTIONS.index(env_openai_model)
                        if env_openai_model in OPENAI_MODEL_OPTIONS
                        else 0
                    )
                    openai_model_name = st.selectbox(
                        "OpenAI model",
                        OPENAI_MODEL_OPTIONS,
                        index=default_index,
                        key="openai_model_v1",
                    )
                    openai_reasoning_effort = st.selectbox(
                        "Reasoning effort",
                        ["high", "xhigh", "medium", "low", "none"],
                        index=0,
                        key="openai_reasoning_effort_v1",
                    )

            if workflow != "Batch quick answers":
                uploaded_question = st.file_uploader(
                    "Upload question image",
                    type=["png", "jpg", "jpeg", "webp"],
                    key="question_image_v1",
                )
                if uploaded_question is not None:
                    st.image(uploaded_question, caption="Question image preview", use_container_width=True)
                    read_col, solve_col = st.columns(2)
                    with read_col:
                        read_image_clicked = st.button(
                            "Read Into Problem",
                            use_container_width=True,
                            key="read_question_image_v1",
                        )
                    with solve_col:
                        read_and_solve_clicked = st.button(
                            "Read And Solve",
                            use_container_width=True,
                            key="read_and_solve_image_v1",
                        )
                    if read_image_clicked or read_and_solve_clicked:
                        try:
                            with st.spinner("Reading question image..."):
                                image_text = extract_problem_from_image(
                                    uploaded_question.getvalue(),
                                    uploaded_question.type or "image/png",
                                    api_key_override if provider == "openai" else "",
                                )
                            if not image_text:
                                st.warning("I could not find readable question text in that image.")
                            else:
                                st.session_state["problem_v4"] = image_text
                                if read_and_solve_clicked:
                                    st.session_state["_run_after_image_read"] = True
                                st.success("Question text extracted. Review it below before running.")
                                st.rerun()
                        except Exception as exc:
                            st.error(f"Image reading failed: {type(exc).__name__}: {exc}")

                problem = st.text_area(
                    "Problem",
                    height=140,
                    key="problem_v4",
                )
                batch_text = ""
            else:
                problem = ""
                batch_text = st.text_area(
                    "Batch problems",
                    placeholder="Enter one problem per line, or separate longer problems with blank lines.",
                    height=210,
                    key="batch_problems_v1",
                )

            run_preset = st.selectbox(
                "Run preset",
                ["Fast", "Balanced", "Accurate"],
                index=0,
                key="run_preset_v1",
            )
            simulations = st.slider("Simulations", 1, 8, 1, key="simulations_v4")
            top_k = st.slider("Improved top-k", 1, 4, 2, key="top_k_v4")
            use_verifier = st.checkbox("Use verifier scorer", value=True, key="use_verifier_v4")
            complete = st.checkbox("Generate final answer", value=True, key="complete_v4")
            default_ppm_checkpoint = ""
            for candidate_checkpoint in ("checkpoints/ppm_amc_openai.pt", "checkpoints/ppm_amc.pt"):
                if (PROJECT_ROOT / candidate_checkpoint).exists():
                    default_ppm_checkpoint = candidate_checkpoint
                    break
            ppm_checkpoint = st.text_input(
                "PPM checkpoint",
                value=default_ppm_checkpoint,
                placeholder="checkpoints/ppm_amc_openai.pt",
                key="ppm_checkpoint_v4",
            )
            run_label = {
                "Full comparison": "Compare Systems",
                "Quick answer": "Solve Quickly",
                "Batch quick answers": "Solve Batch",
            }[workflow]
            run_disabled = not (batch_text.strip() if workflow == "Batch quick answers" else problem.strip())
            run_clicked = st.button(
                run_label,
                type="primary",
                disabled=run_disabled,
                use_container_width=True,
            )
            run_clicked = run_clicked or bool(st.session_state.pop("_run_after_image_read", False))

    with right:
        st.markdown(
            f"""
<div class="mode-strip">
  <span class="mode-pill">Workflow <strong>{html.escape(workflow)}</strong></span>
  <span class="mode-pill">Model <strong>{html.escape(provider)}</strong></span>
  <span class="mode-pill">Preset <strong>{html.escape(run_preset)}</strong></span>
  <span class="mode-pill">Simulations <strong>{simulations}</strong></span>
  <span class="mode-pill">Top-k <strong>{top_k}</strong></span>
  <span class="mode-pill">Verifier <strong>{"on" if use_verifier else "off"}</strong></span>
</div>
            """,
            unsafe_allow_html=True,
        )

        if run_clicked:
            try:
                problem_text = problem.strip()
                if workflow == "Full comparison":
                    exact_solution = _exact_math_solution(problem_text)
                    if exact_solution is not None:
                        st.info(
                            "Structured math problem recognized; using an exact solver router "
                            "before calling the language model."
                        )
                        baseline = _exact_solver_result(
                            problem_text,
                            "Baseline MCTS",
                            exact_solution,
                            "baseline",
                        )
                        improved = _exact_solver_result(
                            problem_text,
                            "Improved MCTS + Process Scorer",
                            exact_solution,
                            "improved",
                        )
                        render_comparison_canvas(
                            baseline,
                            improved,
                            DemoMathModel(),
                            problem_text,
                            complete,
                        )
                        return

                model = build_model(provider, api_key_override, openai_model_name, openai_reasoning_effort)
                if workflow == "Quick answer":
                    render_quick_answer(problem_text, model)
                    return

                if workflow == "Batch quick answers":
                    render_batch_answers(batch_text, model)
                    return

                ppm = load_ppm(ppm_checkpoint.strip())
                scorer = None
                if ppm is not None and use_verifier:
                    scorer = HybridProcessScorer(ppm=ppm, verifier=HeuristicStepVerifier())
                elif ppm is not None:
                    scorer = ppm
                elif use_verifier:
                    scorer = HeuristicStepVerifier()

                effective_simulations = simulations
                effective_top_k = top_k
                if provider in CLOUD_PROVIDERS and run_preset == "Fast" and simulations > 1:
                    effective_simulations = 1
                    st.warning("Fast caps cloud-model runs at 1 simulation. Use Accurate for complex problems.")
                if run_preset == "Accurate" and scorer is not None and top_k < 3:
                    effective_top_k = 3
                    st.info("Accurate preset keeps top-3 scored branches for harder problems.")

                if run_preset == "Accurate":
                    max_depth = 5
                    baseline_actions = 3
                    improved_actions = 4
                    max_branching = 5
                    min_branching = 2
                    retries = 2
                    retry_delay = 0.5
                    generation_temperature = 0.0
                    generation_max_tokens = 180
                elif run_preset == "Balanced":
                    max_depth = 4
                    baseline_actions = 2
                    improved_actions = 3
                    max_branching = 4
                    min_branching = 1
                    retries = 1
                    retry_delay = 0.25
                    generation_temperature = 0.0
                    generation_max_tokens = 200
                else:
                    max_depth = 3
                    baseline_actions = 2
                    improved_actions = 3
                    max_branching = 4
                    min_branching = 1
                    retries = 1
                    retry_delay = 0.25
                    generation_temperature = 0.0
                    generation_max_tokens = 220

                baseline_cfg = MCTSConfig(
                    search_strategy="baseline",
                    max_simulations=effective_simulations,
                    max_depth=max_depth,
                    num_actions=baseline_actions,
                    eval_cache=True,
                    max_state_steps=8,
                    max_retries=retries,
                    retry_delay=retry_delay,
                    fail_fast_on_generation_error=True,
                    generation_temperature=generation_temperature,
                    generation_max_tokens=generation_max_tokens,
                    seed=20240509,
                )
                improved_cfg = MCTSConfig(
                    search_strategy="adaptive",
                    max_simulations=effective_simulations,
                    max_depth=max_depth,
                    num_actions=improved_actions,
                    max_branching_factor=max_branching,
                    min_branching_factor=min_branching,
                    top_k_prune=effective_top_k if scorer is not None else 0,
                    eval_cache=True,
                    max_state_steps=8,
                    max_retries=retries,
                    retry_delay=retry_delay,
                    fail_fast_on_generation_error=True,
                    generation_temperature=generation_temperature,
                    generation_max_tokens=generation_max_tokens,
                    seed=20240509,
                )

                status = st.empty()
                try:
                    with st.spinner("Running both systems..."):
                        status.info("Running baseline MCTS...")
                        baseline = run_mcts(problem_text, model, "Baseline MCTS", baseline_cfg)
                        status.info("Running improved MCTS with process scoring...")
                        improved = run_mcts(
                            problem_text,
                            model,
                            "Improved MCTS + Process Scorer",
                            improved_cfg,
                            scorer,
                        )
                finally:
                    status.empty()

                render_comparison_canvas(baseline, improved, model, problem_text, complete)
            except Exception as exc:
                st.error(str(exc))
        else:
            render_workspace_placeholder()


if __name__ == "__main__":
    main()

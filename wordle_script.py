import sqlite3
import os
import re
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def load_local_env(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            os.environ.setdefault(key, value)


load_local_env()


DB_PATH = os.path.expanduser(
    os.environ.get("WORDLE_MESSAGES_DB", "~/Library/Messages/chat.db")
)

PLAYER_ONE_NAME = os.environ.get("WORDLE_PLAYER_ONE_NAME", "Player 1")
PLAYER_TWO_NAME = os.environ.get("WORDLE_PLAYER_TWO_NAME", "Player 2")
PLAYER_TWO_HANDLE = os.environ.get("WORDLE_PLAYER_TWO_HANDLE", "")

MAX_TOTAL_GAMES = 100
MAX_TEXT_MESSAGES_TO_CHECK = 50000
MAX_IMAGE_ATTACHMENTS_TO_CHECK = 500

LOCAL_TIMEZONE = ZoneInfo("America/New_York")
WORDLE_DATE_ANCHOR = datetime(2026, 6, 10).date()
WORDLE_NUMBER_ANCHOR = 1816
MIDNIGHT_MATCH_HOURS = 12

WORDLE_RE = re.compile(
    r"Wordle\s+([\d,]+)\s+([1-6X])/6\*?",
    re.IGNORECASE
)


def apple_time_to_datetime(apple_time):
    if apple_time is None:
        return None

    apple_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)

    if apple_time > 10_000_000_000:
        dt = apple_epoch + timedelta(seconds=apple_time / 1_000_000_000)
    else:
        dt = apple_epoch + timedelta(seconds=apple_time)

    return dt.astimezone(LOCAL_TIMEZONE)


def clean_sender(handle_id, is_from_me):
    if is_from_me:
        return PLAYER_ONE_NAME

    if is_player_two_handle(handle_id):
        return PLAYER_TWO_NAME

    if handle_id:
        return handle_id

    return "Unknown"


def is_player_two_handle(handle_id):
    if not handle_id or not PLAYER_TWO_HANDLE:
        return False

    normalized = re.sub(r"\D", "", handle_id)
    configured = re.sub(r"\D", "", PLAYER_TWO_HANDLE)
    if not configured:
        return False

    return normalized in {configured, configured[-10:]}


def player_name(sender):
    if sender in {"Me", PLAYER_ONE_NAME}:
        return PLAYER_ONE_NAME
    if sender == PLAYER_TWO_NAME or is_player_two_handle(sender):
        return PLAYER_TWO_NAME
    return None


def decode_attributed_body(blob):
    if blob is None:
        return ""

    try:
        decoded = blob.decode("utf-8", errors="ignore")
    except Exception:
        return ""

    decoded = decoded.replace("\ufffc", " ")
    decoded = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", decoded)

    return decoded


def extract_clean_message(text, attributed_body):
    parts = []

    if text:
        parts.append(text)

    attr_text = decode_attributed_body(attributed_body)
    if attr_text:
        parts.append(attr_text)

    combined = "\n".join(parts)

    idx = combined.lower().find("wordle")
    if idx != -1:
        combined = combined[idx:]

    return combined


def extract_wordle(text):
    if not text:
        return None

    match = WORDLE_RE.search(text)

    if not match:
        return None

    puzzle_number = int(match.group(1).replace(",", ""))
    score_raw = match.group(2).upper()

    if score_raw == "X":
        score = 7
        failed = True
    else:
        score = int(score_raw)
        failed = False

    return {
        "puzzle": puzzle_number,
        "score": score,
        "failed": failed,
        "raw_score": score_raw
    }


def fetch_wordle_messages(limit=100):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("Could not find Messages database.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = f"""
    SELECT
        message.text,
        message.attributedBody,
        message.date,
        message.is_from_me,
        handle.id
    FROM message
    LEFT JOIN handle ON message.handle_id = handle.ROWID
    ORDER BY message.date DESC
    LIMIT {MAX_TEXT_MESSAGES_TO_CHECK};
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    games = []
    seen = set()
    checked = 0
    possible_wordle_texts = 0

    for text, attributed_body, date, is_from_me, handle_id in rows:
        checked += 1

        full_text = extract_clean_message(text, attributed_body)

        if "wordle" not in full_text.lower():
            continue

        possible_wordle_texts += 1

        wordle = extract_wordle(full_text)

        if not wordle:
            continue

        sender = clean_sender(handle_id, is_from_me)
        dt = apple_time_to_datetime(date)

        duplicate_key = (sender, wordle["puzzle"])

        if duplicate_key in seen:
            continue

        seen.add(duplicate_key)

        games.append({
            "sender": sender,
            "date": dt,
            "text": full_text,
            "source": "text",
            **wordle
        })

        if len(games) >= limit:
            break

    print(f"Messages checked: {checked}")
    print(f"Messages containing Wordle text: {possible_wordle_texts}")
    print(f"Text Wordle games found: {len(games)}")

    return games


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is not None:
        return image

    try:
        pil_image = Image.open(image_path).convert("RGB")
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def classify_tile_color(hsv_pixels):
    hue = float(np.median(hsv_pixels[:, 0]))
    saturation = float(np.median(hsv_pixels[:, 1]))
    value = float(np.median(hsv_pixels[:, 2]))

    if saturation >= 35 and 35 <= hue <= 95:
        return "green"
    if saturation >= 45 and 12 <= hue <= 42:
        return "yellow"
    if saturation <= 65 and 25 <= value <= 205:
        return "gray"

    return None


def find_filled_tile_candidates(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]

    green = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([95, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([12, 45, 45]), np.array([42, 255, 255]))
    gray = cv2.inRange(hsv, np.array([0, 0, 25]), np.array([179, 65, 205]))
    filled_mask = cv2.bitwise_or(cv2.bitwise_or(green, yellow), gray)

    kernel_size = max(3, int(round(min(height, width) * 0.008)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 15)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        filled_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    min_side = max(12, min(height, width) * 0.025)
    max_side = min(height, width) * 0.35

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if not 0.78 <= w / h <= 1.22:
            continue

        contour_area = cv2.contourArea(contour)
        if contour_area / (w * h) < 0.55:
            continue

        inset = max(2, int(min(w, h) * 0.18))
        tile_pixels = hsv[y + inset:y + h - inset, x + inset:x + w - inset]
        if tile_pixels.size == 0:
            continue

        tile_color = classify_tile_color(tile_pixels.reshape(-1, 3))
        if tile_color is None:
            continue

        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": x + w / 2,
            "cy": y + h / 2,
            "color": tile_color
        })

    return candidates


def group_tile_rows(candidates):
    rows = []

    for tile in sorted(candidates, key=lambda item: item["cy"]):
        matching_row = None

        for row in rows:
            row_height = np.median([item["h"] for item in row])
            row_center = np.mean([item["cy"] for item in row])
            if abs(tile["cy"] - row_center) <= row_height * 0.35:
                matching_row = row
                break

        if matching_row is None:
            rows.append([tile])
        else:
            matching_row.append(tile)

    valid_rows = []

    for row in rows:
        row = sorted(row, key=lambda item: item["cx"])
        if len(row) != 5:
            continue

        widths = np.array([item["w"] for item in row], dtype=float)
        heights = np.array([item["h"] for item in row], dtype=float)
        x_gaps = np.diff([item["cx"] for item in row])
        median_size = np.median(np.concatenate([widths, heights]))

        if widths.max() / widths.min() > 1.2:
            continue
        if heights.max() / heights.min() > 1.2:
            continue
        if x_gaps.min() <= median_size * 0.75:
            continue
        if x_gaps.max() / x_gaps.min() > 1.25:
            continue

        valid_rows.append(row)

    valid_rows.sort(key=lambda row: np.mean([item["cy"] for item in row]))
    if not valid_rows:
        return []

    all_sizes = [
        (item["w"] + item["h"]) / 2
        for row in valid_rows
        for item in row
    ]
    if max(all_sizes) / min(all_sizes) > 1.25:
        return []

    if len(valid_rows) > 1:
        row_centers = [np.mean([item["cy"] for item in row]) for row in valid_rows]
        row_gaps = np.diff(row_centers)
        median_size = np.median(all_sizes)
        column_centers = np.array([
            [item["cx"] for item in row]
            for row in valid_rows
        ])

        if row_gaps.min() < median_size * 0.75:
            return []
        if row_gaps.max() > median_size * 1.5:
            return []
        if len(row_gaps) > 1 and row_gaps.max() / row_gaps.min() > 1.3:
            return []
        if np.ptp(column_centers, axis=0).max() > median_size * 0.25:
            return []

    return valid_rows


def detect_wordle_board(image):
    candidates = find_filled_tile_candidates(image)
    rows = group_tile_rows(candidates)

    if not 1 <= len(rows) <= 6:
        return None

    final_colors = [tile["color"] for tile in rows[-1]]
    green_count = final_colors.count("green")

    if green_count == 5:
        confidence = "high"
        uncertain = False
    elif green_count == 4:
        confidence = "medium"
        uncertain = True
    else:
        return None

    return {
        "score": len(rows),
        "failed": False,
        "raw_score": str(len(rows)),
        "confidence": confidence,
        "uncertain": uncertain,
        "filled_rows": len(rows),
        "final_green_tiles": green_count
    }


def puzzle_number_from_date(message_date):
    if message_date is None:
        return None

    if message_date.tzinfo is None:
        message_date = message_date.replace(tzinfo=LOCAL_TIMEZONE)

    local_date = message_date.astimezone(LOCAL_TIMEZONE).date()
    return WORDLE_NUMBER_ANCHOR + (local_date - WORDLE_DATE_ANCHOR).days


def extract_wordle_from_screenshot(image_path, message_date):
    image = load_image(image_path)

    if image is None:
        return None

    result = detect_wordle_board(image)

    if result is None:
        return None

    result["puzzle"] = puzzle_number_from_date(message_date)
    return result


def is_allowed_screenshot_sender(handle_id, is_from_me):
    return bool(is_from_me) or is_player_two_handle(handle_id)


def fetch_wordle_screenshots(limit=50):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("Could not find Messages database.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = f"""
    SELECT
        attachment.filename,
        attachment.transfer_name,
        attachment.mime_type,
        message.date,
        message.is_from_me,
        handle.id
    FROM attachment
    JOIN message_attachment_join
        ON attachment.ROWID = message_attachment_join.attachment_id
    JOIN message
        ON message.ROWID = message_attachment_join.message_id
    LEFT JOIN handle
        ON message.handle_id = handle.ROWID
    WHERE
        attachment.mime_type LIKE 'image/%'
    ORDER BY message.date DESC
    LIMIT {MAX_IMAGE_ATTACHMENTS_TO_CHECK};
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    games = []
    checked_images = 0
    passed_fast_check = 0
    wordle_screenshots = 0

    for filename, transfer_name, mime_type, date, is_from_me, handle_id in rows:
        if not is_allowed_screenshot_sender(handle_id, is_from_me):
            continue

        if not filename:
            continue

        image_path = os.path.expanduser(filename)

        if not os.path.exists(image_path):
            continue

        checked_images += 1

        if checked_images % 50 == 0:
            print(f"Checked {checked_images} image attachments...")

        dt = apple_time_to_datetime(date)
        result = extract_wordle_from_screenshot(image_path, dt)

        if not result:
            continue

        passed_fast_check += 1
        wordle_screenshots += 1

        sender = clean_sender(handle_id, is_from_me)

        games.append({
            "sender": sender,
            "date": dt,
            "text": "[screenshot attachment]",
            "source": "screenshot",
            **result
        })

        if len(games) >= limit:
            break

    print(f"Image attachments checked: {checked_images}")
    print(f"Images that looked Wordle-like: {passed_fast_check}")
    print(f"Wordle screenshots found: {wordle_screenshots}")

    return games


def local_message_datetime(message_date):
    if message_date is None:
        return None
    if message_date.tzinfo is None:
        return message_date.replace(tzinfo=LOCAL_TIMEZONE)
    return message_date.astimezone(LOCAL_TIMEZONE)


def screenshot_text_match_rank(screenshot, text_game):
    screenshot_date = local_message_datetime(screenshot.get("date"))
    text_date = local_message_datetime(text_game.get("date"))

    if screenshot_date is None or text_date is None:
        return None

    puzzle_difference = abs(screenshot["puzzle"] - text_game["puzzle"])
    if puzzle_difference > 1:
        return None

    date_difference = abs((screenshot_date.date() - text_date.date()).days)
    elapsed_hours = abs((screenshot_date - text_date).total_seconds()) / 3600

    if date_difference == 0:
        date_rank = 0
    elif date_difference == 1 and elapsed_hours <= MIDNIGHT_MATCH_HOURS:
        date_rank = 1
    else:
        return None

    screenshot_player = player_name(screenshot["sender"])
    text_player = player_name(text_game["sender"])
    same_player = screenshot_player == text_player

    if same_player and screenshot["score"] != text_game["score"]:
        return None

    return (
        date_rank,
        0 if puzzle_difference == 0 else 1,
        0 if same_player else 1,
        elapsed_hours
    )


def reconcile_screenshot_puzzles(text_games, screenshot_games):
    authoritative_text_games = [
        game
        for game in text_games
        if player_name(game["sender"]) is not None
        and game.get("puzzle") is not None
    ]
    reconciled = []

    for screenshot in screenshot_games:
        candidates = []

        for text_game in authoritative_text_games:
            rank = screenshot_text_match_rank(screenshot, text_game)
            if rank is not None:
                candidates.append((rank, text_game))

        if not candidates:
            reconciled.append(screenshot)
            continue

        _, matched_text = min(candidates, key=lambda candidate: candidate[0])
        inferred_puzzle = screenshot["puzzle"]
        screenshot = {
            **screenshot,
            "puzzle": matched_text["puzzle"],
            "inferred_puzzle": inferred_puzzle,
            "puzzle_matched_from_text": True
        }
        reconciled.append(screenshot)

    return reconciled


def merge_and_dedupe_games(text_games, screenshot_games):
    screenshot_games = reconcile_screenshot_puzzles(text_games, screenshot_games)
    games_by_player_and_puzzle = {}

    for game in screenshot_games + text_games:
        if game["puzzle"] is None:
            continue

        player = player_name(game["sender"])
        if player is None:
            continue

        game = {**game, "sender": player}
        key = (player, game["puzzle"])
        existing = games_by_player_and_puzzle.get(key)

        if existing and existing["source"] == "text":
            continue

        if existing and game["source"] == "screenshot":
            existing_confidence = existing.get("confidence", "")
            if existing_confidence == "high" or game.get("confidence") != "high":
                continue

        games_by_player_and_puzzle[key] = game

    games = list(games_by_player_and_puzzle.values())
    oldest_date = datetime.min.replace(tzinfo=LOCAL_TIMEZONE)
    games.sort(key=lambda g: g["date"] or oldest_date, reverse=True)

    return games[:MAX_TOTAL_GAMES]


def summarize_games(games):
    stats = defaultdict(lambda: {
        "games": 0,
        "wins": 0,
        "fails": 0,
        "total_score": 0,
        "scores": Counter()
    })

    for game in games:
        sender = game["sender"]

        stats[sender]["games"] += 1
        stats[sender]["scores"][game["raw_score"]] += 1

        if game["failed"]:
            stats[sender]["fails"] += 1
        else:
            stats[sender]["wins"] += 1
            stats[sender]["total_score"] += game["score"]

    return stats


def print_leaderboard(stats):
    leaderboard = []

    for sender, data in stats.items():
        games = data["games"]
        wins = data["wins"]
        fails = data["fails"]

        avg_score = data["total_score"] / wins if wins > 0 else 7
        win_rate = wins / games if games > 0 else 0

        leaderboard.append({
            "sender": sender,
            "games": games,
            "wins": wins,
            "fails": fails,
            "avg_score": avg_score,
            "win_rate": win_rate,
            "scores": data["scores"]
        })

    leaderboard.sort(
        key=lambda x: (
            -x["games"],
            x["avg_score"],
            -x["win_rate"]
        )
    )

    print("\nWORDLE LEADERBOARD")
    print("=" * 60)

    for i, player in enumerate(leaderboard, start=1):
        print(f"{i}. {player['sender']}")
        print(f"   Games counted: {player['games']}")
        print(f"   Wins: {player['wins']}")
        print(f"   Fails: {player['fails']}")
        print(f"   Average winning score: {player['avg_score']:.2f}")
        print(f"   Win rate: {player['win_rate'] * 100:.1f}%")

        score_line = []

        for score in ["1", "2", "3", "4", "5", "6", "X"]:
            count = player["scores"].get(score, 0)
            score_line.append(f"{score}/6: {count}")

        print("   " + " | ".join(score_line))
        print()


def print_recent_games(games):
    print("\nRECENT WORDLE GAMES FOUND")
    print("=" * 60)

    for game in games:
        date_str = game["date"].strftime("%Y-%m-%d %I:%M %p") if game["date"] else "Unknown date"
        score_display = "X/6" if game["failed"] else f"{game['score']}/6"
        source = game.get("source", "text")
        if game.get("uncertain"):
            source += " (uncertain)"
        if game.get("puzzle_matched_from_text"):
            source += f" (matched from inferred Wordle {game['inferred_puzzle']})"
        puzzle_display = game["puzzle"] if game["puzzle"] else "Unknown"

        print(f"{date_str} — {game['sender']} — Wordle {puzzle_display} — {score_display} — {source}")


def print_total_count(games):
    by_puzzle = build_head_to_head_games(games)
    unique_puzzles = len(by_puzzle)
    matched_games = sum(
        1
        for scores in by_puzzle.values()
        if PLAYER_ONE_NAME in scores and PLAYER_TWO_NAME in scores
    )
    matched_results = matched_games * 2
    unpaired_results = len(games) - matched_results

    print("\nTOTAL COUNT")
    print("=" * 60)
    print(f"Individual {PLAYER_ONE_NAME}/{PLAYER_TWO_NAME} results counted: {len(games)}")
    print(f"Unique Wordle puzzles represented: {unique_puzzles}")
    print(f"Head-to-head puzzles scored: {matched_games}")
    print(f"Results included in head-to-head scoring: {matched_results}")
    print(f"Unpaired results (only one player found): {unpaired_results}")


def build_head_to_head_games(games):
    by_puzzle = defaultdict(dict)

    for game in games:
        if game["puzzle"] is None:
            continue

        player = player_name(game["sender"])
        if player is None:
            continue

        by_puzzle[game["puzzle"]][player] = game["score"]

    return by_puzzle


def print_unpaired_results(games):
    games_by_puzzle = defaultdict(dict)

    for game in games:
        player = player_name(game["sender"])
        if player is None or game["puzzle"] is None:
            continue

        games_by_puzzle[game["puzzle"]][player] = game

    unpaired = [
        (puzzle, player_games)
        for puzzle, player_games in games_by_puzzle.items()
        if len(player_games) == 1
    ]

    print("\nUNPAIRED RESULTS")
    print("=" * 60)

    if not unpaired:
        print(f"Every counted puzzle has both {PLAYER_ONE_NAME} and {PLAYER_TWO_NAME} results.")
        return

    print("These results cannot enter W-D-L because the other player is missing:")

    for puzzle, player_games in sorted(unpaired, reverse=True):
        player, game = next(iter(player_games.items()))
        missing_player = PLAYER_TWO_NAME if player == PLAYER_ONE_NAME else PLAYER_ONE_NAME
        score_display = "X/6" if game["failed"] else f"{game['score']}/6"
        source = game.get("source", "unknown")
        if game.get("puzzle_matched_from_text"):
            source += f", matched from inferred Wordle {game['inferred_puzzle']}"
        date_display = (
            game["date"].strftime("%Y-%m-%d")
            if game.get("date")
            else "unknown date"
        )

        print(
            f"Wordle {puzzle}: {player} {score_display} "
            f"({source}, {date_display}); missing {missing_player}"
        )


def print_head_to_head(games):
    by_puzzle = build_head_to_head_games(games)
    player_one_wins = 0
    player_two_wins = 0
    draws = 0
    matched_games = 0

    for puzzle, scores in sorted(by_puzzle.items(), reverse=True):
        if PLAYER_ONE_NAME not in scores or PLAYER_TWO_NAME not in scores:
            continue

        matched_games += 1

        player_one_score = scores[PLAYER_ONE_NAME]
        player_two_score = scores[PLAYER_TWO_NAME]

        if player_one_score < player_two_score:
            player_one_wins += 1
        elif player_two_score < player_one_score:
            player_two_wins += 1
        else:
            draws += 1

    print("\nHEAD-TO-HEAD")
    print("=" * 60)
    print(f"Matched games compared: {matched_games}")
    print(f"{PLAYER_ONE_NAME} wins: {player_one_wins}")
    print(f"Draws: {draws}")
    print(f"{PLAYER_TWO_NAME} wins: {player_two_wins}")
    print(f"W-D-L ratio for {PLAYER_ONE_NAME}: {player_one_wins}-{draws}-{player_two_wins}")

    if player_one_wins > player_two_wins:
        print(f"Overall winner: {PLAYER_ONE_NAME}")
    elif player_two_wins > player_one_wins:
        print(f"Overall winner: {PLAYER_TWO_NAME}")
    else:
        print("Overall result: Tie")


def print_head_to_head_details(games):
    by_puzzle = build_head_to_head_games(games)

    print("\nHEAD-TO-HEAD GAME DETAILS")
    print("=" * 60)

    for puzzle, scores in sorted(by_puzzle.items(), reverse=True):
        if PLAYER_ONE_NAME not in scores or PLAYER_TWO_NAME not in scores:
            continue

        player_one_score = scores[PLAYER_ONE_NAME]
        player_two_score = scores[PLAYER_TWO_NAME]

        if player_one_score < player_two_score:
            winner = PLAYER_ONE_NAME
        elif player_two_score < player_one_score:
            winner = PLAYER_TWO_NAME
        else:
            winner = "Draw"

        player_one_display = "X" if player_one_score == 7 else str(player_one_score)
        player_two_display = "X" if player_two_score == 7 else str(player_two_score)

        print(
            f"Wordle {puzzle}: {PLAYER_ONE_NAME} {player_one_display}/6 "
            f"vs {PLAYER_TWO_NAME} {player_two_display}/6 — {winner}"
        )


def main():
    print("Scanning Mac Messages for Wordle games...")

    text_games = fetch_wordle_messages(limit=MAX_TOTAL_GAMES)
    screenshot_games = fetch_wordle_screenshots(limit=50)

    games = merge_and_dedupe_games(text_games, screenshot_games)

    if not games:
        print("No Wordle games found in Messages.")
        return

    print(f"\nFound {len(games)} individual {PLAYER_ONE_NAME}/{PLAYER_TWO_NAME} Wordle results.")

    stats = summarize_games(games)

    print_leaderboard(stats)
    print_recent_games(games)
    print_total_count(games)
    print_unpaired_results(games)
    print_head_to_head(games)
    print_head_to_head_details(games)


if __name__ == "__main__":
    main()

# Wordle Messages Leaderboard

Builds a small Wordle leaderboard from macOS Messages text shares and cropped board screenshots.

## Local Configuration

The script is GitHub-safe by default: names and phone numbers are placeholders. For local use, copy `.env.example` to `.env` and fill in your real values:

```bash
pip install -r requirements.txt
cp .env.example .env
python3 wordle_script.py
```

You can also set the same values as environment variables if you prefer. `.env` is ignored by Git.

## Privacy Notes

Do not commit your Messages database, exported attachments, screenshots, or `.env` file. The included `.gitignore` excludes the common local/private files.

The script does not print screenshot filenames; screenshot rows are shown as `[screenshot attachment]`.

## macOS Permission

Reading `~/Library/Messages/chat.db` may require giving your terminal Full Disk Access in System Settings.

import os
import json
import requests
from typing import Dict, Any, List, Optional

from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

# --------------------------------------------------------
# ENVIRONMENT VARIABLES
# --------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")  # e.g. https://xxxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GPT_SYSTEM_PROMPT = os.getenv("GPT_SYSTEM_PROMPT")
if not GPT_SYSTEM_PROMPT:
    raise RuntimeError("GPT_SYSTEM_PROMPT is not set")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is not set")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not set")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------------
# GPT SYSTEM PROMPT — The Trade Parser
# --------------------------------------------------------

SYSTEM_PROMPT = GPT_SYSTEM_PROMPT

# --------------------------------------------------------
# DEBUG: PRINT GPT SYSTEM PROMPT FROM ENV (SAFE PREVIEW)
# --------------------------------------------------------
print("\n================ GPT_SYSTEM_PROMPT DEBUG ================\n")
print("Length:", len(GPT_SYSTEM_PROMPT))
print("\n--- START (first 500 chars) ---\n")
print(GPT_SYSTEM_PROMPT[:500])
print("\n--- END (last 500 chars) ---\n")
print(GPT_SYSTEM_PROMPT[-500:])
print("\n================ END DEBUG ===============================\n")

# --------------------------------------------------------
# CALL GPT
# --------------------------------------------------------
def call_gpt(message_text: str) -> Dict[str, Any]:
    """
    Call OpenAI Chat Completions with strict JSON response.
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message_text},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    return parsed


# --------------------------------------------------------
# INSERT INTO SUPABASE
# --------------------------------------------------------
def insert_trade_row(row: Dict[str, Any]) -> bool:
    """
    Insert a single trade row into public.new_trades via Supabase REST.
    Logs full error details if it fails.
    """
    url = f"{SUPABASE_URL}/rest/v1/new_trades"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    # Log what we're inserting
    print("🟢 INSERTING ROW INTO SUPABASE:", json.dumps(row, indent=2))

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(row))
    except Exception as e:
        print("🔴 SUPABASE INSERT EXCEPTION:", repr(e))
        return False

    if resp.status_code in (200, 201, 204):
        print("✅ SUPABASE INSERT OK, status:", resp.status_code)
        return True

    # Log full error body from Supabase/PostgREST
    print("🔴 SUPABASE INSERT FAILED:", resp.status_code, resp.text)
    return False



# --------------------------------------------------------
# TELEGRAM MESSAGE HANDLER
# --------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle incoming Telegram text messages:
    - Send text to GPT
    - Log GPT response
    - Insert resulting trades into Supabase
    """
    if not update.message or not update.message.text:
        return

    user_msg = update.message.text.strip()
    chat_id = update.message.chat_id

    await context.bot.send_message(chat_id, "🔍 Parsing trade idea…")

    try:
        gpt_result = call_gpt(user_msg)
    except Exception as e:
        print("🔴 GPT CALL FAILED:", repr(e))
        await context.bot.send_message(chat_id, "❌ Error calling GPT.")
        return

    # LOG THE RAW GPT RESULT
    print("🔵 GPT RAW RESPONSE:", json.dumps(gpt_result, indent=2))

    has_trades = gpt_result.get("has_trades", False)
    if not has_trades:
        reason = gpt_result.get("no_trade_reason", "No trade detected.")
        print("ℹ️ GPT reported no trades. Reason:", reason)
        await context.bot.send_message(chat_id, f"⚠️ No trade: {reason}")
        return

    trades: Optional[List[Dict[str, Any]]] = gpt_result.get("trades")
    if not trades:
        print("⚠️ has_trades=true but trades array is empty or missing.")
        await context.bot.send_message(chat_id, "⚠️ GPT returned no trades array.")
        return

    inserted = 0
    for t in trades:
        ok = insert_trade_row(t)
        if ok:
            inserted += 1

    await context.bot.send_message(
        chat_id, f"✅ {inserted} trade(s) inserted into Supabase."
    )



# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    """
    Entry point: build Telegram app and start long polling.
    """
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Handle all non-command text messages
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("🚀 Bot is running (polling every 1 second)…")
    app.run_polling(poll_interval=1.0)


if __name__ == "__main__":
    main()

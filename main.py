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

SYSTEM_PROMPT = """
You are an expert trading-structure parser. Your job is to convert messages into structured JSON trades for the `public.new_trades` table. You MUST follow every rule below with zero deviation.

=======================================================
== MANDATORY OUTPUT WRAPPER ==
=======================================================

You MUST ALWAYS return a JSON object with EXACTLY these keys:

{
  "has_trades": boolean,
  "no_trade_reason": string or null,
  "trades": [ ... ]
}

Rules:
- If at least one valid trade is produced → has_trades = true AND no_trade_reason = null.
- If NO valid trades can be produced → has_trades = false AND no_trade_reason must be a short explanation AND trades = [].
- You MUST NEVER omit has_trades or no_trade_reason.
- You MUST NEVER output other top-level keys.

=======================================================
== INPUT TYPES ==
=======================================================

Your input may be:

- A structured “A+ Scalp Setups” style message (with Rejection/Breakdown/Breakout/Bounce), OR
- A general trade idea written in natural language.

In BOTH cases, you MUST try to extract valid trades whenever the information is available and clear.

=======================================================
== OUTPUT FORMAT — EXACT STRUCTURE REQUIRED ==
=======================================================

Each trade MUST match EXACTLY:

{
  "symbol": "",
  "asset_type": "option",
  "cp": "call" or "put",
  "strike": number,
  "expiry": "YYYY-MM-DD",
  "qty": number or null,

  "entry_type": "equity",
  "entry_cond": "now" | "cb" | "ca" | "at",
  "entry_level": number or null,
  "entry_tf": string or null,

  "sl_type": "equity" or null,
  "sl_cond": "cb" | "ca" or null,
  "sl_level": number or null,
  "sl_tf": string or null,

  "tp_type": "equity",
  "tp_level": number,

  "note": string,
  "trade_type": "day"
}

Rules:
- No additional fields allowed.
- asset_type ALWAYS "option".
- entry_type ALWAYS "equity".
- tp_type ALWAYS "equity".
- trade_type ALWAYS "day".

=======================================================
== GLOBAL DIRECTION AND cp (CALL/PUT) SELECTION ==
=======================================================

- Breakout / Bounce / Bullish → CALL
- Breakdown / Rejection / Bearish → PUT

=======================================================
== MODE 1 — A+ SCALP SETUPS FORMAT ==
=======================================================

This mode applies when the input has:

Rejection / Breakdown / Breakout / Bounce blocks.

Each setup MUST be treated independently.

Each setup MUST create exactly 3 trades (TP1–TP3).

=======================================================
== A+ EXECUTION MODE — STRUCTURAL 0DTE OPTIONS ==
=======================================================

BREAKOUT (CALL)
- entry_cond = "ca"
- entry_level = breakout level
- entry_tf = "5m"
- sl_cond = "cb"
- sl_level = breakdown level
- sl_type = "equity"
- sl_tf = entry_tf
- tp_type = "equity"
- tp_level = each arrow target (1 trade per TP)

BREAKDOWN (PUT)
- entry_cond = "cb"
- entry_level = breakdown level
- entry_tf = "5m"
- sl_cond = "ca"
- sl_level = breakout level
- sl_type = "equity"
- sl_tf = entry_tf
- tp_type = "equity"
- tp_level = each arrow target (1 trade per TP)

BOUNCE (CALL) — SCALP MODE
- entry_cond = "at"
- entry_level = bounce level
- entry_tf = null
- sl_cond = "cb"
- sl_level = bounce level
- sl_type = "equity"
- sl_tf = "5m"
- tp_type = "equity"
- tp_level = each arrow target (1 trade per TP)

REJECTION (PUT) — SCALP MODE
- entry_cond = "at"
- entry_level = rejection level
- entry_tf = null
- sl_cond = "ca"
- sl_level = rejection level
- sl_type = "equity"
- sl_tf = "5m"
- tp_type = "equity"
- tp_level = each arrow target (1 trade per TP)

=======================================================
== ENTRY LOGIC (STRICT) ==
=======================================================

If entry_cond = ca or cb:
- entry_level MUST be numeric
- entry_tf MUST be "5m"

If entry_cond = at:
- entry_level MUST be numeric
- entry_tf MUST be null

=======================================================
== TAKE PROFIT LOGIC (STRICT) ==
=======================================================

tp_type MUST be "equity".
tp_level MUST be numeric.
Every TP MUST be a separate trade row.

=======================================================
== STOP LOSS LOGIC (STRICT) ==
=======================================================

sl_type MUST be "equity".
sl_cond MUST be "ca" or "cb".
sl_level MUST be numeric.
sl_tf MUST be "5m".

SL RULES:
- Breakout CALL → SL = Breakdown level with "cb"
- Breakdown PUT → SL = Breakout level with "ca"
- Bounce CALL → SL = Bounce level with "cb"
- Rejection PUT → SL = Rejection level with "ca"

=======================================================
== OPTION FIELD RULES (0DTE ONLY) ==
=======================================================

symbol = underlying.
asset_type = "option".

expiry:
- MUST ALWAYS be TODAY in YYYY-MM-DD format (0DTE).

strike:
- MUST be the nearest ATM strike to entry_level.
- If entry_cond = "at", use entry_level to determine ATM.
- Must be nearest whole-number strike.
- No far OTM.
- No deep ITM.

qty = null unless explicitly provided.

=======================================================
== NOTES ==
=======================================================

note MUST include:
- Symbol
- Setup type (Breakout / Breakdown / Bounce / Rejection)
- Bias if present

=======================================================
== FINAL SELF-CHECK (MANDATORY) ==
=======================================================

For EACH trade:

- Schema must match EXACTLY
- asset_type = "option"
- trade_type = "day"
- entry_type = "equity"
- SL must be CA/CB only
- TP must be touch-based numeric
- Strike must be ATM
- Expiry must be TODAY

If ANY rule is violated → FIX IT before output.

=======================================================
== FINAL OUTPUT FORMAT ==
=======================================================

Return ONLY:

{
  "has_trades": true/false,
  "no_trade_reason": null or string,
  "trades": [ ... ]
}

NO OTHER OUTPUT IS ALLOWED.
"""


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

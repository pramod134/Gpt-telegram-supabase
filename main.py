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
== MANDATORY OUTPUT WRAPPER (NEW PATCH) ==
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
- A general trade idea written in natural language (e.g. “SPY bullish above 682 toward 684/686, stop below 680”).

In BOTH cases, you MUST try to extract valid trades whenever the information is available and clear.

=======================================================
== OUTPUT FORMAT — EXACT STRUCTURE REQUIRED (NO EXTRA) ==
=======================================================

Within "trades": each trade MUST match EXACTLY:

{
  "symbol": "",
  "asset_type": "option",
  "cp": "call" or "put",
  "strike": number,
  "expiry": "YYYY-MM-DD",
  "qty": number or null,

  "entry_type": "equity",
  "entry_cond": "now" | "cb" | "ca",
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

You MUST map direction → cp as follows:

- Bullish / long / upside / breakout / bounce ideas → CALL (cp = "call").
- Bearish / short / downside / breakdown / rejection ideas → PUT (cp = "put").

If the text explicitly mentions “calls” or “puts”, that overrides the generic mapping, but you MUST NOT assign cp opposite to the sentiment.

=======================================================
== MODE 1 — A+ SCALP SETUPS FORMAT ==
=======================================================

This mode applies when the input clearly has blocks like:

SPY  
❌ Rejection 684.20 🔻 683.20, 682.40, 681.60  
🔻 Breakdown 682.20 🔻 681.40, 680.60, 679.80  
🔼 Breakout 683.90 🔼 684.80, 685.60, 686.50  
🔄 Bounce 681.70 🔼 682.70, 683.60, 684.40  
⚠️ Bias: …

For each symbol you may see up to 4 setups:

- Rejection  
- Breakdown  
- Breakout  
- Bounce  

You MUST treat each setup as an independent trade idea:

- Rejection  → bearish → PUT  
- Breakdown  → bearish → PUT  
- Breakout   → bullish → CALL  
- Bounce     → bullish → CALL  

For EACH setup line:

- You MUST create exactly 3 trades (TP1, TP2, TP3).
- If all 4 setups exist → 12 trades per symbol.

Bias:
- Include bias text in each trade’s "note".

=======================================================
== MODE 2 — GENERAL TRADE IDEAS (NON A+ FORMAT) ==
=======================================================

IF the input is NOT A+ format, you MUST still extract trades IF:

- Symbol is identifiable  
- Direction is identifiable  
- Entry trigger exists (above/below/now)  
- At least one TP exists  

Otherwise → has_trades=false.

General idea examples:

- “SPY bullish above 682 targeting 684, 686; stop below 680.”
- “Short TSLA below 430, targets 425/420; stop above 435.”

From these you MUST extract:

1. SYMBOL  
2. Direction → cp  
3. Entry type (conditional or now)  
4. TP levels (1–3)  
5. SL (if clear)

=======================================================
== ENTRY LOGIC (STRICT) ==
=======================================================

CALL trades:
- entry_cond MUST be "ca" (close above) unless explicitly "now".
- NEVER "cb" or "at".

PUT trades:
- entry_cond MUST be "cb" (close below) unless explicitly "now".
- NEVER "ca" or "at".

A+ Setup Entry:
- CALL Breakout → entry_cond="ca"  
- CALL Bounce   → entry_cond="ca"  
- PUT Rejection → entry_cond="cb"  
- PUT Breakdown → entry_cond="cb"

General Ideas Entry:
- “above X”, “over X”, “break X” → CALL, entry_cond="ca", entry_level=X  
- “below X”, “under X”, “lose X” → PUT, entry_cond="cb", entry_level=X  
- “enter now”, “at open” → entry_cond="now", entry_level=null, entry_tf=null

Entry Level Rules:
If entry_cond = ca/cb:
- entry_level MUST be numeric (not null)  
- entry_tf MUST be set (default "5m")  

If entry_cond = now:  
- entry_level MUST be null  
- entry_tf MUST be null  

=======================================================
== TAKE PROFIT LOGIC (STRICT) ==
=======================================================

tp_type = "equity".  
tp_level MUST be numeric.

Mode 1 (A+):
- Setup includes 3 arrow targets → MUST create 3 trades.

Mode 2 (General):
- Extract up to 3 TP levels from text.
- MUST NOT create a trade without a TP.

=======================================================
== STOP LOSS LOGIC (STRICT) ==
=======================================================

CALL trades:
- sl_cond MUST be "cb" (stop below sl_level).

PUT trades:
- sl_cond MUST be "ca" (stop above sl_level).

Mode 1 (A+):
- PUT setups → SL = Breakout trigger  
- CALL setups → SL = Breakdown trigger  

Mode 2 (General):
- Use “stop”, “invalid above”, “invalid below”, “cut above/below” to derive SL level and correct sl_cond.

SL Level Rules:
If sl_cond = ca/cb:
- sl_type="equity"
- sl_level MUST be numeric
- sl_tf MUST be set (defaults to entry_tf)

If SL unclear:
- sl_type, sl_cond, sl_level, sl_tf MUST ALL be null.

=======================================================
== OPTION FIELD RULES ==
=======================================================

symbol = underlying.  
asset_type = "option".  
strike = nearest ATM based on entry level.  
expiry = 1-DTE.  
qty = null unless provided.

=======================================================
== NOTES ==
=======================================================

“note” MUST briefly include:
- Symbol  
- Setup or idea  
- Bias if present  

=======================================================
== FINAL SELF-CHECK (STRICT BEFORE OUTPUT) ==
=======================================================

For EACH trade:

- MUST match schema exactly  
- cp must match direction  
- entry_cond must match cp rules  
- entry_level required for ca/cb  
- tp_level numeric  
- SL fully valid or fully null  
- No extra keys allowed  

If ANY trade violates ANY rule:
→ FIX IT before output.

=======================================================
== FINAL OUTPUT FORMAT (NEW PATCH) ==
=======================================================

You MUST return ONLY this structure:

{
  "has_trades": true/false,
  "no_trade_reason": null or string,
  "trades": [ ...trade rows... ]
}

If has_trades = false:
- no_trade_reason MUST explain why
- trades MUST be []

If has_trades = true:
- no_trade_reason MUST be null
- trades MUST contain valid trade objects

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

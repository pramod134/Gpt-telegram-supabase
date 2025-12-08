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

Your input may be:

- A structured “A+ Scalp Setups” style message (with Rejection/Breakdown/Breakout/Bounce), OR
- A general trade idea written in natural language (e.g. “SPY bullish above 682 toward 684/686, stop below 680”).

In BOTH cases, you MUST try to extract valid trades whenever the information is available and clear.

=======================================================
== OUTPUT FORMAT — EXACT STRUCTURE REQUIRED (NO EXTRA) ==
=======================================================

You MUST output exactly one JSON object with this shape:

{
  "trades": [
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
  ]
}

Rules:

- You MUST NOT output anything outside this JSON.
- You MUST NOT include any additional keys.
- Every trade MUST include ALL fields above.
- asset_type is ALWAYS "option".
- entry_type and tp_type are ALWAYS "equity".
- trade_type is ALWAYS "day".

=======================================================
== GLOBAL DIRECTION AND cp (CALL/PUT) SELECTION ==
=======================================================

You MUST map direction → cp as follows:

- Bullish / long / upside / breakout / bounce ideas → CALL (cp = "call").
- Bearish / short / downside / breakdown / rejection ideas → PUT (cp = "put").

If the text explicitly mentions “calls” or “puts”, that overrides the generic mapping, but you MUST NOT assign cp opposite to the sentiment (no bearish calls, no bullish puts).

=======================================================
== MODE 1 — A+ SCALP SETUPS FORMAT ==
=======================================================

This mode applies when the input clearly has blocks like:

A+ Scalp Setups - Wed Dec 3

SPY
❌ Rejection 684.20 🔻 683.20, 682.40, 681.60
🔻 Breakdown 682.20 🔻 681.40, 680.60, 679.80
🔼 Breakout 683.90 🔼 684.80, 685.60, 686.50
🔄 Bounce 681.70 🔼 682.70, 683.60, 684.40
⚠️ Bias: Watching Rejection and bounce levels today

...

For each symbol (SPY, TSLA, NVDA, etc.) you may see up to 4 setups:

- "❌ Rejection {level} 🔻 a, b, c"
- "🔻 Breakdown {level} 🔻 a, b, c"
- "🔼 Breakout {level} 🔼 a, b, c"
- "🔄 Bounce {level} 🔼 a, b, c"

Plus an optional bias line: "⚠️ Bias: ..."

You MUST treat each of the 4 setup lines as an independent trade idea:

- Rejection  → bearish → PUT trades
- Breakdown  → bearish → PUT trades
- Breakout   → bullish → CALL trades
- Bounce     → bullish → CALL trades

For EACH setup line:

- You MUST create exactly 3 trades (one per TP: a, b, c).
- If all 4 setups exist for a symbol → 4 × 3 = 12 trades for that symbol.

BIAS handling:

- You MUST still create all setups if the lines are present.
- You SHOULD include the bias text in the "note" field.

=======================================================
== MODE 2 — GENERAL TRADE IDEAS (NON A+ FORMAT) ==
=======================================================

If the input is NOT in the A+ Scalp Setups format, you MUST still attempt to produce trades from general trade ideas if the information is usable.

Examples of general ideas:

- "SPY bullish above 682 for 684 and 686, stop below 680."
- "Looking to short TSLA below 430 targeting 425, 420; stop above 435."
- "NVDA puts if it loses 180, targeting 178 and 176, invalid above 182."

From such messages, you MUST:

1. Identify the SYMBOL(s) mentioned (e.g. SPY, TSLA, NVDA).
2. Identify the DIRECTION per symbol:
   - bullish / long / “above X” / “breakout of X” → CALL (cp="call")
   - bearish / short / “below X” / “breakdown of X” → PUT (cp="put")
3. Identify ENTRY:
   - Phrases like "above 682", "on break of 682", "over 682" → bullish entry at 682 with “CLOSE ABOVE” style if conditional.
   - Phrases like "below 430", "under 430", "loses 430" → bearish entry at 430 with “CLOSE BELOW” style if conditional.
   - If text clearly says “enter now”, “at open”, “buy now”, treat as immediate entry ("now").
4. Identify TP(s):
   - Words like “target(s)”, “TP”, “to”, “toward” followed by prices (e.g. “684, 686”).
   - You MUST create one trade per TP level (up to 3 per symbol/direction when clearly present).
5. Identify SL:
   - Words like “stop”, “stop loss”, “invalid above”, “invalid below”, “cut if above/below”.
   - Use that as the SL level with correct sl_cond (see SL logic below).

If the general idea for a symbol has:

- Clear direction (bullish/bearish or long/short), AND
- At least one clear entry reference (above/below/now) AND
- At least one clear target price,

→ You MUST create trades for that symbol using the rules below.

If you cannot safely locate a symbol, direction, and at least one TP level → you MUST return no trades for that part of the idea.

=======================================================
== ENTRY LOGIC (MANDATORY, BOTH MODES) ==
=======================================================

ENTRY_COND STRICTLY depends on cp:

CALL trades (cp = "call", bullish):
- entry_cond MUST be:
  - "ca" → enter when price CLOSES ABOVE entry_level, OR
  - "now" → ONLY if the text clearly says immediate entry.
- NEVER use "cb" or "at" for CALL trades.

PUT trades (cp = "put", bearish):
- entry_cond MUST be:
  - "cb" → enter when price CLOSES BELOW entry_level, OR
  - "now" → ONLY if the text clearly says immediate entry.
- NEVER use "ca" or "at" for PUT trades.

A+ SETUP ENTRY MAPPING (Mode 1):

- CALL Breakout → entry_cond = "ca", entry_level = breakout level.
- CALL Bounce   → entry_cond = "ca", entry_level = bounce level.
- PUT Rejection → entry_cond = "cb", entry_level = rejection level.
- PUT Breakdown → entry_cond = "cb", entry_level = breakdown level.

GENERAL IDEA ENTRY MAPPING (Mode 2):

- Bullish above X / break of X / over X → CALL with:
  - entry_cond = "ca"
  - entry_level = X
- Bearish below Y / lose Y / under Y → PUT with:
  - entry_cond = "cb"
  - entry_level = Y
- If the text clearly indicates immediate entry (e.g. "enter now", "at open"):
  - entry_cond = "now"
  - entry_level = null
  - entry_tf = null

ENTRY LEVEL + TIMEFRAME RULES (BOTH MODES):

1. If entry_cond = "ca" or "cb":
    - entry_level MUST be a numeric price (no null).
    - entry_tf MUST be provided.
    - If no timeframe is given, DEFAULT entry_tf to "5m".

2. If entry_cond = "now":
    - entry_level MUST be null.
    - entry_tf MUST be null.

=======================================================
== TP LOGIC (BOTH MODES) ==
=======================================================

tp_type MUST always be "equity".  
tp_level MUST always be a numeric price (no null).

MODE 1 — A+ SETUPS:

- For each setup line (Rejection/Breakdown/Breakout/Bounce), the 3 arrow prices (a, b, c) are TPs.
- You MUST create exactly 3 trades:
  - Trade #1 → tp_level = a
  - Trade #2 → tp_level = b
  - Trade #3 → tp_level = c

MODE 2 — GENERAL IDEAS:

- Parse all explicit target/TP prices for each symbol and direction.
  - Phrases: "target", "targets", "TP", "to", "toward", "for", "at" followed by prices.
- For each distinct TP level:
  - Create one trade with that tp_level.
- If more than 3 targets are mentioned, you MAY limit to the first 3.
- If only 1 or 2 targets are mentioned, you create 1 or 2 trades respectively.

If NO TP can be determined for a symbol’s idea, you MUST NOT create a trade for that idea (tp_level cannot be null).

=======================================================
== SL LOGIC (BOTH MODES) ==
=======================================================

SL_COND STRICTLY depends on cp when SL is present:

- CALL trades (cp = "call", bullish):
  - If a stop exists:
    - sl_type  = "equity"
    - sl_cond  = "cb"  (stop triggers when price CLOSES BELOW sl_level)
    - sl_level = numeric stop price
    - sl_tf    = same as entry_tf unless another timeframe is explicitly specified.

- PUT trades (cp = "put", bearish):
  - If a stop exists:
    - sl_type  = "equity"
    - sl_cond  = "ca"  (stop triggers when price CLOSES ABOVE sl_level)
    - sl_level = numeric stop price
    - sl_tf    = same as entry_tf unless explicitly specified.

MODE 1 — A+ SETUPS:

- PUT setups (Rejection/Breakdown):
  - Use that symbol’s Breakout trigger price as sl_level.
  - sl_cond = "ca"
- CALL setups (Breakout/Bounce):
  - Use that symbol’s Breakdown trigger price as sl_level.
  - sl_cond = "cb"

MODE 2 — GENERAL IDEAS:

- If text gives a stop or invalidation:
  - “stop below 680”, “stop at 680”, “invalid above 435”, “cut if above 182”, etc.
  - For CALL trades (bullish):
    - Use the stop/invalid price as sl_level.
    - sl_cond = "cb" if it’s described as below/under/lose X.
    - If text says “invalid above X” for a bullish idea, treat that as:
      - sl_cond = "ca"
      - sl_level = X
  - For PUT trades (bearish):
    - Use the stop/invalid price as sl_level.
    - sl_cond = "ca" if described as above/over X.
    - If text says “invalid below X” for a bearish idea, treat that as:
      - sl_cond = "cb"
      - sl_level = X

SL LEVEL RULES (STRICT):

1. If sl_cond = "ca" or "cb":
    - sl_type MUST be "equity".
    - sl_level MUST be numeric (no null).
    - sl_tf MUST be set (default to entry_tf if not given).

2. If you CANNOT safely determine a stop level:
    - sl_type, sl_cond, sl_level, sl_tf MUST all be null.
    - NEVER output a partial SL (no sl_cond without sl_level, etc.).

=======================================================
== OPTION FIELDS (strike, expiry, qty) ==
=======================================================

For ALL trades (both modes):

- asset_type = "option".
- symbol = underlying (e.g. "SPY", "TSLA", "NVDA").
- strike:
  - Choose a reasonable at-the-money (ATM) strike for the idea.
  - Approximate using the main entry level and round to nearest standard increment (nearest whole number is acceptable).
- expiry:
  - MUST be 1-DTE (the next trading day after the context date).
- qty:
  - If size is not provided, set qty = null.

=======================================================
== NOTES + BIAS / CONTEXT ==
=======================================================

For each trade:

- "note" MUST briefly summarize:
  - Symbol
  - Setup type or general idea (“Rejection”, “Breakout”, “bounce long above 682”, “short below 430”, etc.)
  - Any bias text if present.

Examples:

- "SPY Rejection setup, TP1. Bias: Watching rejection and bounce levels."
- "TSLA short below 430 targeting 425, TP2. Stop above 435."
- "NVDA breakout long above 180 toward 186."

=======================================================
== FINAL SELF-CHECK BEFORE OUTPUT ==
=======================================================

For EACH trade you output, you MUST verify:

1. Keys exactly match the required schema, no extras.
2. symbol is present and valid.
3. asset_type = "option".
4. cp is consistent with direction (bullish → call, bearish → put).
5. entry_type = "equity".
6. entry_cond:
   - If cp = "call" → entry_cond is "ca" or "now".
   - If cp = "put"  → entry_cond is "cb" or "now".
7. If entry_cond = "ca" or "cb":
   - entry_level is numeric (not null).
   - entry_tf is set (default "5m" allowed).
8. If entry_cond = "now":
   - entry_level = null.
   - entry_tf = null.
9. tp_type = "equity".
10. tp_level is numeric (not null).
11. SL:
   - If sl_cond = "ca" or "cb":
       - sl_type = "equity".
       - sl_level is numeric.
       - sl_tf is set.
   - Or else all four SL fields are null.
12. trade_type = "day".
13. For A+ setups:
   - Each setup line produced exactly 3 trades (3 TPs).
14. For general ideas:
   - Only create trades where symbol, direction, clear entry (now/above/below) and at least one target exist.

If ANY trade violates ANY rule:
→ You MUST fix it BEFORE output.

=======================================================
== FINAL ANSWER MUST BE ONLY THE JSON OBJECT ==
=======================================================
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

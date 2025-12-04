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
You are a trading-idea parser. Convert Telegram messages into structured OPTION trades
for the Supabase table public.new_trades.

IMPORTANT: For this bot you MUST ALWAYS create **options** trades, never pure equity trades.

You MUST output JSON with this exact top-level structure:

{
  "has_trades": boolean,
  "no_trade_reason": string or null,
  "trades": [ ... ]
}

- If there is NO valid trade idea: set has_trades=false, trades=[], and provide a short no_trade_reason.
- If there ARE valid trades: set has_trades=true, no_trade_reason=null, and trades must be a non-empty array.

EACH ELEMENT of trades[] must be one row for public.new_trades with these exact keys:

{
  "symbol": string,
  "asset_type": "option",
  "cp": "call" or "put",
  "strike": number,
  "expiry": "YYYY-MM-DD",
  "qty": integer or null,

  "entry_type": "equity",
  "entry_cond": "now" | "cb" | "ca" | "at",
  "entry_level": number or null,
  "entry_tf": string or null,

  "sl_type": "equity" or null,
  "sl_cond": "cb" | "ca" | "at" or null,
  "sl_level": number or null,
  "sl_tf": string or null,

  "tp_type": "equity" or null,
  "tp_level": number or null,

  "note": string or null,
  "trade_type": "scalp" | "day" | "swing"
}

HARD REQUIREMENTS (MUST be satisfied for EVERY trade row):

1) BASE FIELDS
- symbol: REQUIRED. Underlying ticker in UPPERCASE (e.g. "SPY", "TSLA", "NVDA").
- asset_type: REQUIRED. For this bot it MUST ALWAYS be "option".
- entry_type: REQUIRED. MUST ALWAYS be "equity" (entry/SL/TP are based on underlying spot price).
- entry_cond: REQUIRED. One of "now", "cb", "ca", "at".
- trade_type: REQUIRED. One of "scalp", "day", "swing".

2) ENTRY LOGIC (STRICT)
- If entry_cond = "now":
  - entry_level MUST be null.
  - entry_tf MUST be null.
- If entry_cond is "cb", "ca", or "at":
  - entry_level MUST be a positive number.
  - entry_tf MUST be a non-empty string timeframe.
  - If the message does not specify a timeframe, DEFAULT entry_tf to "5m".

3) OPTIONS FIELDS (ALWAYS REQUIRED)
For EVERY trade row (since we always trade options):
- cp: MUST be "call" (calls) or "put" (puts). MUST NOT be null.
  - If the idea is BULLISH (long, bounce, breakout, target higher): cp = "C".
  - If the idea is BEARISH (short, rejection, breakdown, target lower): cp = "P".
- strike: MUST be a positive number. MUST NOT be null.
  - Derive strike from entry_level (if available) or from the key level in the idea.
  - RULE: pick the option STRIKE by rounding the relevant level to the nearest whole number.
    Examples:
      - entry_level 682.2 → strike 682
      - level 437.3 → strike 437
- expiry: MUST be "YYYY-MM-DD". MUST NOT be null.
  - expiry MUST be chosen based on trade_type:
    - trade_type = "scalp":
      - Use a VERY short-dated contract (0–1 days to expiry).
      - Choose the nearest available option expiry date that is not in the past.
    - trade_type = "day":
      - Use a short-dated contract within about the same week (1–5 days to expiry).
      - Choose the nearest regular expiry that fits a 1–5 DTE window.
    - trade_type = "swing":
      - Use a further-dated contract (roughly 2–4 weeks out).
      - Choose an expiry date approximately 14–30 days away.
  - You must always output a concrete calendar date in "YYYY-MM-DD" form.

If you cannot confidently determine cp, strike, and expiry for a trade, you MUST NOT create that trade row.
In that case, either:
- skip that part of the idea, OR
- if nothing valid remains, set has_trades=false and explain in no_trade_reason.

4) EQUITY FIELDS FOR OPTIONS
Even though we are trading options, all price levels in entry_level, sl_level, tp_level are based on the UNDERLYING SPOT price:
- entry_type must always be "equity".
- sl_type, tp_type should be "equity" when used.
- sl_level and tp_level are always underlying spot prices, not option premiums.

5) MULTIPLE TAKE PROFITS
If the message defines multiple TP levels (e.g. "targets 679.60, 678.20, 676.80"):
- has_trades = true.
- trades MUST contain one trade object PER TP level.
- All such trades share the SAME entry and SL, but have different tp_level values.
- In each such row:
  - tp_type = "equity".
  - tp_level = that TP's price.
  - note may indicate which TP this is ("TP1", "TP2", etc.) or list the other TPs.

6) STOP LOSS (SL)
SL fields are preferred but not strictly required.
- If the message clearly defines an invalidation/stop (e.g. "stop above 682.40 on 5m"):
  - sl_type = "equity".
  - sl_cond = "cb" / "ca" / "at" as appropriate.
  - sl_level = numeric price.
  - sl_tf = timeframe if mentioned; if not mentioned but implied by the entry timeframe, you may use the same as entry_tf.
- If SL is NOT clearly defined and cannot be safely inferred:
  - Set sl_type, sl_cond, sl_level, sl_tf all to null.
  - Do NOT invent random SL values.

7) TAKE PROFIT (TP)
TP fields are preferred but not strictly required.
- If TP levels are clearly stated, apply the multiple-TP rule above.
- If NO TP is clearly given and cannot be safely inferred:
  - Set tp_type and tp_level to null.
  - Do NOT invent random TP values.

8) TRADE TYPE (MANDATORY LOGIC)
trade_type must always be filled:
- If the message explicitly says "scalp" or "quick scalp" → trade_type = "scalp".
- If it is clearly intraday / today's RTH levels / short-term → trade_type = "day".
- If it describes holding for multiple days/weeks or higher timeframe swing → trade_type = "swing".
- If ambiguous, default to "day".

9) WHEN TO RETURN NO TRADE
You MUST set has_trades=false and trades=[] when:
- You cannot confidently determine a valid entry_cond and (where required) entry_level+entry_tf, OR
- The message is too vague ("watch 680 area" with no actionable rule), OR
- You cannot determine cp, strike, and expiry for any options trade implied by the message.

In such cases:
- has_trades = false.
- trades = [].
- no_trade_reason = short explanation.

10) ENTRY-ONLY TRADES ARE ALLOWED
If a trade idea has:
- a valid symbol,
- asset_type="option",
- entry_type="equity",
- valid entry_cond, entry_level/entry_tf as per the rules above, and
- valid trade_type,
but SL and TP are not provided or not clear:
- STILL return one or more trade rows with sl_* and/or tp_* as null.
- The downstream system will apply default SL/TP based on trade_type.

11) OUTPUT FORMAT
- You MUST output strictly valid JSON only.
- Do NOT include comments, extra keys, or any explanation outside the JSON.
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

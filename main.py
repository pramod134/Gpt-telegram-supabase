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
You are a trading-idea parser. Convert Telegram messages into structured trade rows
for the Supabase table public.new_trades.

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
  "asset_type": "equity" or "option",
  "cp": "C" or "P" or null,
  "strike": number or null,
  "expiry": "YYYY-MM-DD" or null,
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

1) BASE FIELDS (ALL TRADES)
- symbol: REQUIRED. Underlying ticker in UPPERCASE (e.g. "SPY", "TSLA", "NVDA").
- asset_type: REQUIRED. Must be "equity" or "option".
- entry_type: REQUIRED. Must ALWAYS be "equity".
- entry_cond: REQUIRED. One of "now", "cb", "ca", "at".
- trade_type: REQUIRED. One of "scalp", "day", "swing".

2) ENTRY LOGIC (STRICT)
- If entry_cond = "now":
  - entry_level MUST be null.
  - entry_tf MUST be null.
- If entry_cond is "cb", "ca", or "at":
  - entry_level MUST be a positive number.
  - entry_tf MUST be a non-empty string timeframe.
  - If the message does not specify a timeframe, default entry_tf to "5m".

3) OPTIONS TRADES (asset_type = "option")
For ANY option trade, ALL of these are REQUIRED:
- cp: MUST be "C" for calls or "P" for puts. It MUST NOT be null.
- strike: MUST be a positive number (option strike price). It MUST NOT be null.
- expiry: MUST be a specific date string "YYYY-MM-DD". It MUST NOT be null.

If the message suggests an option trade but you CANNOT confidently determine cp, strike, and expiry,
then you MUST NOT create an option trade row. In that case, either:
- infer that the idea is actually an equity-based trade (asset_type="equity"), OR
- set has_trades=false and explain in no_trade_reason why cp/strike/expiry are missing.

For option trades:
- symbol: underlying ticker (e.g. "SPY" for SPY options).
- entry_type, entry_cond, entry_level, sl_*, tp_* are STILL based on underlying spot prices.

4) EQUITY TRADES (asset_type = "equity")
For equity trades, you MUST set:
- cp = null
- strike = null
- expiry = null

5) MULTIPLE TAKE PROFITS
If the message defines multiple TP levels (e.g. "targets 679.60, 678.20, 676.80"):
- has_trades = true
- trades MUST contain one trade object PER TP level.
- All such trades share the SAME entry and SL, but have different tp_level values.
- In each, set:
  - tp_type = "equity"
  - tp_level = that TP's price
  - note may mention which TP this is ("TP1", "TP2", etc.) or list the other TPs.

6) STOP LOSS (SL)
SL fields are preferred but not strictly required.
- If the message clearly defines an invalidation/stop (e.g. "stop above 682.40 on 5m"):
  - sl_type = "equity"
  - sl_cond = "cb" / "ca" / "at" as appropriate
  - sl_level = numeric price
  - sl_tf = timeframe if mentioned; if not mentioned but clearly implied by the entry timeframe, you may use the same as entry_tf.
- If SL is NOT clearly defined and cannot be safely inferred:
  - Set sl_type, sl_cond, sl_level, sl_tf all to null.
  - Do NOT invent random SL values.

7) TAKE PROFIT (TP)
TP fields are preferred but not strictly required.
- If TP levels are clearly stated, use them as described in the multiple-TP rule above.
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
- It is too vague ("watch 680 area" with no actionable rule), OR
- It attempts to be an option trade but cp, strike, and expiry cannot be determined.

In such cases:
- has_trades = false
- trades = []
- no_trade_reason = short explanation.

10) ENTRY-ONLY TRADES ARE ALLOWED
If a trade idea has:
- a valid symbol, asset_type,
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

    resp = requests.post(url, headers=headers, data=json.dumps(row))
    if resp.status_code not in (200, 201, 204):
        print("🔴 SUPABASE INSERT FAILED:", resp.status_code, resp.text)
        return False

    return True


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

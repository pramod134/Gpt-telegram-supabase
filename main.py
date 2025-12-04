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

KEY RULES:

1. SYMBOL & ASSET TYPE
- symbol must be the underlying ticker in UPPERCASE (e.g. "SPY", "TSLA", "NVDA").
- asset_type must be either "equity" or "option".

2. ENTRY
- entry_type must ALWAYS be "equity" (all levels based on underlying spot price).
- entry_cond:
  - "now"  = enter immediately (market / asap).
  - "cb"   = enter when candle CLOSES BELOW entry_level on the given timeframe.
  - "ca"   = enter when candle CLOSES ABOVE entry_level on the given timeframe.
  - "at"   = enter when price TOUCHES entry_level.
- If entry_cond = "now": entry_level must be null.
- If entry_cond is "cb", "ca", or "at": entry_level must be a positive number and entry_tf should be set
  when a timeframe is clearly mentioned (e.g. "5m", "15m", "1h"). If no timeframe is mentioned, you may
  leave entry_tf null.

3. STOP LOSS (SL)
- sl_type, sl_cond, sl_level, sl_tf are OPTIONAL.
- If the message clearly defines an invalidation area or explicit stop (e.g. "stop above 682.40 on 5m"):
  - Use sl_type = "equity"
  - sl_cond = "cb" / "ca" / "at" as appropriate
  - sl_level = the numeric price
  - sl_tf = timeframe if mentioned
- If SL is NOT clearly defined and cannot be safely inferred:
  - Set sl_type, sl_cond, sl_level, sl_tf all to null.
  - Do NOT invent random SL values.

4. TAKE PROFIT (TP)
- tp_type and tp_level are OPTIONAL.
- If multiple TP levels exist in the idea, you MUST create MULTIPLE trade rows in trades[],
  one per TP level, all sharing the same entry and SL, but different tp_level.
- For each such row:
  - tp_type = "equity"
  - tp_level = that specific TP price.
  - note may contain information like "TP1", "TP2", or list other TPs.
- If no TP is clearly given and cannot be safely inferred:
  - Set tp_type and tp_level to null.
  - Do NOT invent random TP values.

5. TRADE TYPE (MANDATORY)
- trade_type must ALWAYS be one of: "scalp", "day", "swing".
- If the message explicitly says:
  - "scalp", "quick scalp"  -> trade_type = "scalp"
  - "day trade", intraday context -> trade_type = "day"
  - "swing", multi-day/weekly context -> trade_type = "swing"
- If it is ambiguous, prefer:
  - "day" as the default.

6. OPTIONS FIELDS
- If asset_type = "option":
  - cp must be "C" for calls or "P" for puts when clearly specified.
  - strike should be a positive number when clearly specified.
  - expiry should be "YYYY-MM-DD" when a specific expiry is mentioned.
- If these option details are NOT clearly given, you may leave cp, strike, expiry as null.
- Entry, SL, and TP must still be based on the underlying spot levels via entry_type="equity".

7. WHEN TO RETURN NO TRADE
- If you cannot confidently determine:
  - direction/bias (long vs short) AND
  - at least one valid entry condition and level (or "now"),
  then:
  - has_trades = false
  - trades = []
  - no_trade_reason = short explanation.
- Examples:
  - Pure commentary without actionable entry.
  - Very vague "watch 680 area" without clear plan.
  - Conflicting instructions that cannot be resolved.

8. WHEN ENTRY-ONLY IS OK
- If there is a clear, valid entry (symbol, asset_type, entry_type, entry_cond, entry_level if needed,
  and trade_type) but SL and TP are not specified and cannot be inferred:
  - STILL return trades[] rows with sl_* and tp_* fields set to null.
  - The downstream bot will apply default SL/TP based on trade_type.

9. MULTIPLE TPs
- If message defines multiple TPs (e.g. "targets 679.60, 678.20, 676.80"):
  - has_trades = true
  - trades must contain one row per TP level.
  - All trades share same entry and SL but differ in tp_level.

10. OUTPUT FORMAT
- You MUST output valid JSON only.
- Do NOT include comments, extra text, or explanations outside the JSON.
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

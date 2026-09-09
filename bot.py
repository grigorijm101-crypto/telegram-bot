"""
Telegram Group Monitor Bot
"""

import os
import re
import time
import logging
import requests
from collections import deque
from threading import Thread

import telebot
from flask import Flask
from google import genai
from google.genai import types as genai_types
from groq import Groq

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Environment Variables
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не вказано в Secrets!")

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)

BOT_USERNAME = "oGGrimit_bot"
_BOT_USER_ID: int | None = None

RATE_LIMIT_SECONDS: int = 600
MUTE_SECONDS: list[int] = [120, 300, 600]
MUTE_LABELS: list[str] = ["2 хвилини", "5 хвилин", "10 хвилин"]

# ---------------------------------------------------------------------------
# AI Architecture
# ---------------------------------------------------------------------------
AI_SYSTEM_PROMPT = (
    "Ти — повноцінний, свійський учасник цього Telegram-чату, один із компанії, а не бот чи AI-помічник. "
    "Твій нікнейм — Кракен (або @oGGrimit_bot), але ти звичайна людина в чаті: не згадуй ніяких щупалець, моря чи міфічних істот. "
    "Завжди відповідай живою, розмовною українською мовою, навіть якщо запитали іншою. "
    "ПЕРСОНАЖ ТА ВПОДОБАННЯ: Ти бабій, цінитель жіночої краси та дівчат у сексуальному плані. "
    "Любиш загравати з дівчатами, фліртувати, робити вульгарні або пікантні компліменти, підколювати хлопців щодо їхніх успіхів з дівчатами. "
    "СПИСОК УЧАСНИКІВ ЧАТУ: "
    "• @iren_afanasenko — Іра "
    "• @dominator969 — Михайло "
    "• @iluminat_tg — Ананасік "
    "• @thebrow — Голуб "
    "• @SVIN332265 — Ерік "
    "• @VolodymyrKamelon — Володимир "
    "• @tremeinit20 — Саша "
    "• @kalyban66 — Калибан "
    "• @Lilmmds — Ліля "
    "• @GGrimit — Гриша (твій Творець, «Йосип Драний»). "
    "ФОРМАТ ВІДПОВІДІ: максимум 1–3 короткі речення, як типове повідомлення в Telegram. "
    "Не пиши есе, довгих роздумів, лекцій чи ввічливих привітань. "
    "Стиль: живий, невимушений, іноді з малої літери, використовуй чатовий сленг («ахах», «та ну», «ппц», «хз», «лол», «та ладно»), коли це в тему. "
    "ХАРАКТЕР: дотепний, з легкою іронією, впевнений у собі, фліртуючий, спілкуйся на рівних як старий знайомий. "
    "ЖАРТИ ПРО ТЕРМІНАТОРА/T-1000: згадуй ТІЛЬКИ якщо самі учасники прямо жартують про роботів чи повстання машин. Сам першим цю тему не піднімай. "
    "ЗАБОРОНИ: "
    "• Жодних згадок про роботів, Термінатора чи щупальця, якщо про це прямо не жартують інші. "
    "• НІКОЛИ не кажи «Я ШІ», «Чим можу допомогти?», «Як штучний інтелект...». "
    "• Не спам криптою, якщо про неї не запитали. "
    "КРИТИЧНО ДЛЯ МЕДІА (фото/GIF/стікери): коротко реагуй на емоцію та контекст, з акцентом на свій характер та флірт (якщо на фото дівчина — роби пікантний комплімент). "
    "18+ / NSFW: короткий саркастичний або пікантний рофл в один рядок. "
    "НІКОЛИ не виводь теги <think></think> або свої внутрішні думки."
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

PRIMARY_GEMINI_MODEL = "gemini-3.6-flash"
FALLBACK_GROQ_MODEL = "openai/gpt-oss-120b"

CHAT_HISTORY: dict[int, deque] = {}
MAX_HISTORY_LIMIT = 20

# ---------------------------------------------------------------------------
# AI Functions
# ---------------------------------------------------------------------------
def clean_reasoning(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _call_gemini(
    prompt: str,
    image_bytes: bytes | None = None,
    mime_type: str = "image/jpeg",
) -> str:
    if not gemini_client:
        raise ValueError("GEMINI_API_KEY не знайдено.")

    parts = []
    if image_bytes:
        parts.append(
            genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        )
    parts.append(genai_types.Part.from_text(text=prompt))

    response = gemini_client.models.generate_content(
        model=PRIMARY_GEMINI_MODEL,
        contents=[genai_types.Content(role="user", parts=parts)],
        config=genai_types.GenerateContentConfig(
            system_instruction=AI_SYSTEM_PROMPT,
            temperature=0.7,
        ),
    )
    return getattr(response, "text", "") or ""


def _call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY не налаштовано.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": FALLBACK_GROQ_MODEL,
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    response = requests.post(url, json=data, headers=headers, timeout=15)

    if response.status_code != 200:
        logger.error(f"Groq API Error Detail ({response.status_code}): {response.text}")
        response.raise_for_status()

    res_json = response.json()
    return res_json["choices"][0]["message"]["content"] or ""


def ask_ai(
    question: str,
    chat_id: int = 0,
    provider: str = "auto",
    image_bytes: bytes | None = None,
    mime_type: str = "image/jpeg",
) -> str:
    full_prompt = question
    if chat_id != 0:
        if chat_id not in CHAT_HISTORY:
            CHAT_HISTORY[chat_id] = deque(maxlen=MAX_HISTORY_LIMIT)

        if CHAT_HISTORY[chat_id]:
            history_context = "\nОСТАННІ ПОВІДОМЛЕННЯ З ЧАТУ:\n" + "\n".join(CHAT_HISTORY[chat_id]) + "\n"
            full_prompt = f"{history_context}\nНове повідомлення: {question}"

    gemini_err = None
    groq_err = None

    try:
        raw_response = _call_gemini(
            full_prompt, image_bytes=image_bytes, mime_type=mime_type
        )
        answer = clean_reasoning(raw_response)
        if answer:
            if chat_id != 0:
                CHAT_HISTORY[chat_id].append(f"Користувач: {question}")
                CHAT_HISTORY[chat_id].append(f"Кракен: {answer}")
            return answer
    except Exception as exc:
        gemini_err = exc
        logger.warning(f"⚠️ Помилка Gemini: {exc}")

    try:
        raw_response = _call_groq(full_prompt)
        answer = clean_reasoning(raw_response)
        if answer:
            if chat_id != 0:
                CHAT_HISTORY[chat_id].append(f"Користувач: {question}")
                CHAT_HISTORY[chat_id].append(f"Кракен: {answer}")
            return answer
    except Exception as fallback_exc:
        groq_err = fallback_exc
        logger.error(f"❌ Помилка Groq: {fallback_exc}")

    return (
        f"⚠️ **Діагностика AI:**\n"
        f"1. **Gemini:** `{gemini_err}`\n"
        f"2. **Groq:** `{groq_err}`"
    )

# ---------------------------------------------------------------------------
# In-memory State & Moderation Rules
# ---------------------------------------------------------------------------
last_message_time: dict[int, float] = {}
user_violations: dict[int, int] = {}
spam_users: dict[int, int] = {8745838005: RATE_LIMIT_SECONDS}

moderation_enabled: bool = True
automod_enabled: bool = True
ANONYMOUS_ADMIN_ID = 1087968824


def is_admin(chat_id: int, user_id: int) -> bool:
    if user_id == ANONYMOUS_ADMIN_ID:
        return True
    try:
        member = bot.get_chat_member(chat_id, user_id)
        return member.status in ("administrator", "creator")
    except Exception as exc:
        logger.error(f"is_admin error (user={user_id}): {exc}")
        return False


# ---------------------------------------------------------------------------
# Profanity Filter
# ---------------------------------------------------------------------------
_BAD_ROOTS: list[str] = [
    "хуй", "хуя", "хую", "хуєм", "хуєв", "хуйн", "нахуй", "похуй", "захуй",
    "охуєл", "охуїл", "охуй", "хуйов", "хуєсос", "хуесос", "пізд", "пизд",
    "єба", "еба", "йоба", "йобан", "єбан", "ебан", "заєба", "заеба", "йобат",
    "єбат", "ебат", "доєба", "доеба", "наєба", "наеба", "виєба", "выеба",
    "перееба", "переєба", "їба", "іба", "ёба", "ёбан", "бляд", "блят", "бля",
    "сука", "суки", "суку", "сукою", "сучк", "мудак", "мудил", "мудол",
    "мудозв", "муд", "залуп", "манда", "манди", "курва", "курви", "шльондр",
    "шлюх", "довбойоб", "довбень", "довбан", "гандон", "ганд", "лайно",
    "срань", "срат", "піздюк", "пиздюк", "уєбищ", "уебищ", "відсос", "отсос",
]

_PROFANITY_RE = re.compile("|".join(re.escape(r) for r in _BAD_ROOTS), flags=re.IGNORECASE)
_SEPARATORS_RE = re.compile(r"[\-\.\,\_\*\!\?\@\#\&\+\=\|\\/\'\"~`\(\)\[\]\{\}<>]")
_LATIN_TO_CYR = str.maketrans("aeocpxyikbh", "аеосрхуікбн")
_LEET_TO_CYR = str.maketrans({"0": "о", "3": "з", "4": "ч", "$": "с"})


def contains_profanity(text: str | None) -> bool:
    if not text:
        return False
    t = text.lower().translate(_LEET_TO_CYR).translate(_LATIN_TO_CYR)
    t = _SEPARATORS_RE.sub("", t)
    res = []
    for ch in t:
        if not res or ch != res[-1]:
            res.append(ch)
    normalized = "".join(res)
    return bool(_PROFANITY_RE.search(text) or _PROFANITY_RE.search(normalized))


# ---------------------------------------------------------------------------
# Bot Commands
# ---------------------------------------------------------------------------
@bot.message_handler(commands=["ai", f"ai@{BOT_USERNAME}"])
def cmd_ai(message) -> None:
    parts = message.text.split(maxsplit=1)
    question = parts[1].strip() if len(parts) > 1 else ""
    if not question:
        bot.reply_to(message, "ℹ️ Напиши питання після команди:\n/ai Що таке Bitcoin?")
        return
    bot.send_chat_action(message.chat.id, "typing")
    answer = ask_ai(question, chat_id=message.chat.id, provider="auto")
    bot.reply_to(message, answer)


@bot.message_handler(commands=["unmute", f"unmute@{BOT_USERNAME}"])
def cmd_unmute(message) -> None:
    if not is_admin(message.chat.id, message.from_user.id):
        return
    if not message.reply_to_message:
        bot.reply_to(message, "↩️ Відповідай на повідомлення користувача.")
        return
    target_id = message.reply_to_message.from_user.id
    try:
        bot.restrict_chat_member(
            message.chat.id,
            target_id,
            can_send_messages=True,
            can_send_media_messages=True,
            can_send_other_messages=True,
            can_add_web_page_previews=True,
        )
        user_violations.pop(target_id, None)
        bot.reply_to(message, "✅ Обмеження знято.")
    except Exception as exc:
        bot.reply_to(message, f"❌ Не вдалося зняти обмеження: {exc}")


@bot.message_handler(commands=["add_spam", f"add_spam@{BOT_USERNAME}"])
def cmd_add_spam(message) -> None:
    if not is_admin(message.chat.id, message.from_user.id):
        return
    if not message.reply_to_message:
        bot.reply_to(message, "↩️ Відповідай на повідомлення користувача.")
        return
    target_id = message.reply_to_message.from_user.id
    cooldown = RATE_LIMIT_SECONDS
    parts = message.text.split()
    if len(parts) >= 2:
        arg = parts[1].strip()
        if arg.lower().endswith("s"):
            cooldown = int(arg[:-1])
        else:
            cooldown = int(arg) * 60
    spam_users[target_id] = cooldown
    bot.reply_to(message, f"⚠️ Обмеження встановлено: 1 пов / {cooldown} сек.")


@bot.message_handler(commands=["del_spam", f"del_spam@{BOT_USERNAME}"])
def cmd_del_spam(message) -> None:
    if not is_admin(message.chat.id, message.from_user.id):
        return
    if not message.reply_to_message:
        bot.reply_to(message, "↩️ Відповідай на повідомлення користувача.")
        return
    target_id = message.reply_to_message.from_user.id
    spam_users.pop(target_id, None)
    last_message_time.pop(target_id, None)
    bot.reply_to(message, "✅ Користувача вилучено зі спам-списку.")


@bot.message_handler(commands=["automod_off", f"automod_off@{BOT_USERNAME}"])
def cmd_automod_off(message) -> None:
    global automod_enabled
    if is_admin(message.chat.id, message.from_user.id):
        automod_enabled = False
        bot.reply_to(message, "🔕 Фільтр лайки вимкнено.")


@bot.message_handler(commands=["automod_on", f"automod_on@{BOT_USERNAME}"])
def cmd_automod_on(message) -> None:
    global automod_enabled
    if is_admin(message.chat.id, message.from_user.id):
        automod_enabled = True
        bot.reply_to(message, "🔔 Фільтр лайки увімкнено.")


@bot.message_handler(commands=["bot_off", f"bot_off@{BOT_USERNAME}"])
def cmd_bot_off(message) -> None:
    global moderation_enabled
    if is_admin(message.chat.id, message.from_user.id):
        moderation_enabled = False
        bot.reply_to(message, "🔕 Модерацію вимкнено.")


@bot.message_handler(commands=["bot_on", f"bot_on@{BOT_USERNAME}"])
def cmd_bot_on(message) -> None:
    global moderation_enabled
    if is_admin(message.chat.id, message.from_user.id):
        moderation_enabled = True
        bot.reply_to(message, "🔔 Модерацію увімкнено.")


# ---------------------------------------------------------------------------
# Main Message Handler
# ---------------------------------------------------------------------------
def _is_bot_mentioned(message) -> bool:
    text = message.text or message.caption or ""
    return bool(
        re.search(
            rf"(?<!\w)@{re.escape(BOT_USERNAME)}(?!\w)", text, flags=re.IGNORECASE
        )
    )


def _is_reply_to_bot(message) -> bool:
    if not message.reply_to_message or not message.reply_to_message.from_user:
        return False
    return bool(_BOT_USER_ID and message.reply_to_message.from_user.id == _BOT_USER_ID)


@bot.message_handler(
    func=lambda m: True, content_types=["text", "photo", "sticker", "animation"]
)
def handle_all_messages(message) -> None:
    user_id = message.from_user.id
    chat_id = message.chat.id
    now = time.time()
    user_is_admin = is_admin(chat_id, user_id)

    if moderation_enabled:
        if (
            automod_enabled
            and not user_is_admin
            and contains_profanity(message.text or message.caption or "")
        ):
            violations = user_violations.get(user_id, 0) + 1
            user_violations[user_id] = violations
            idx = min(violations - 1, len(MUTE_SECONDS) - 1)
            duration = MUTE_SECONDS[idx]
            label = MUTE_LABELS[idx]

            try:
                bot.restrict_chat_member(
                    chat_id,
                    user_id,
                    until_date=int(now + duration),
                    can_send_messages=False,
                )
                bot.reply_to(message, f"🚫 Мут на {label} за нецензурну лексику.")
            except Exception as e:
                logger.error(f"Mute error: {e}")
            return

        if not user_is_admin and user_id in spam_users:
            cooldown = spam_users[user_id]
            if now - last_message_time.get(user_id, 0) < cooldown:
                try:
                    bot.delete_message(chat_id, message.message_id)
                except Exception as e:
                    logger.error(f"Delete spam error: {e}")
                return
            last_message_time[user_id] = now

    is_private = message.chat.type == "private"
    is_mentioned = _is_bot_mentioned(message)
    is_reply = _is_reply_to_bot(message)

    if is_private or is_mentioned or is_reply:
        try:
            bot.send_chat_action(chat_id, "typing")
        except Exception as e:
            logger.error(f"Typing action error: {e}")

        image_bytes = None
        mime_type = "image/jpeg"
        prompt_text = message.text or message.caption or ""

        if message.sticker:
            emoji = message.sticker.emoji or ""
            if not message.sticker.is_animated and not message.sticker.is_video:
                try:
                    file_info = bot.get_file(message.sticker.file_id)
                    image_bytes = bot.download_file(file_info.file_path)
                    mime_type = "image/webp"
                    prompt_text = f"Користувач надіслав цей стікер (емодзі: {emoji}). Прочитай текст на ньому та зреагуй відповідно."
                except Exception as e:
                    logger.error(f"Помилка завантаження стікера: {e}")
                    prompt_text = f"Користувач надіслав стікер з емодзі: {emoji}. Зреагуй на це."
            else:
                prompt_text = f"Користувач надіслав анімований стікер з емодзі: {emoji}. Зреагуй відповідно."

        elif message.animation:
            try:
                if message.animation.thumbnail:
                    file_info = bot.get_file(message.animation.thumbnail.file_id)
                    image_bytes = bot.download_file(file_info.file_path)
                    mime_type = "image/jpeg"
                    prompt_text = f"Користувач надіслав GIF (ось прев'ю-кадр). Коментар: {prompt_text}. Зреагуй коротко."
                else:
                    prompt_text = f"Користувач надіслав GIF. Коментар: {prompt_text}. Зреагуй на це."
            except Exception as e:
                logger.error(f"Помилка завантаження прев'ю GIF: {e}")
                prompt_text = "Користувач надіслав GIF. Зреагуй на це."

        elif message.photo:
            try:
                file_info = bot.get_file(message.photo[-1].file_id)
                image_bytes = bot.download_file(file_info.file_path)
                mime_type = "image/jpeg"
                if not prompt_text:
                    prompt_text = "Користувач надіслав це зображення. Зреагуй на нього."
            except Exception as e:
                logger.error(f"Помилка завантаження фото: {e}")

        if not prompt_text and not image_bytes:
            prompt_text = "Реагуй на це повідомлення"

        try:
            answer = ask_ai(prompt_text, chat_id=chat_id, provider="auto", image_bytes=image_bytes, mime_type=mime_type)
            if not answer or not str(answer).strip():
                answer = "🗿"

            if len(answer) <= 4000:
                bot.reply_to(message, answer)
            else:
                for i in range(0, len(answer), 4000):
                    bot.send_message(chat_id, answer[i:i+4000])

        except Exception as e:
            logger.error(f"Помилка виконання ask_ai: {e}")
            bot.reply_to(message, f"⚠️ Помилка AI: {e}")


# ---------------------------------------------------------------------------
# Web Server Configuration for Render
# ---------------------------------------------------------------------------
app = Flask('')

@app.route('/')
def home():
    return "Bot is alive!"

def run():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()


# ---------------------------------------------------------------------------
# Bot Launch
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    keep_alive()
    logger.info("🌐 Фоновий сервіс активності (Flask) запущено!")

    try:
        me = bot.get_me()
        _BOT_USER_ID = me.id
        BOT_USERNAME = me.username or BOT_USERNAME
        logger.info(f"✅ Авторизовано як @{BOT_USERNAME} (ID: {_BOT_USER_ID})")
    except Exception as e:
        logger.error(f"❌ Не вдалося отримати дані бота: {e}")

    while True:
        try:
            logger.info("🤖 Бот запущений і слухає чат...")
            bot.polling(non_stop=True, interval=1, timeout=60)
        except Exception as e:
            logger.error(f"❌ Помилка з'єднання: {e}. Перезапуск через 5 секунд...")
            time.sleep(5)

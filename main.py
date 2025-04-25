from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
import re
import dateparser
from datetime import datetime

app = FastAPI()

DATABASE_URL = "sqlite:///./chatbot.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, index=True)
    due_date = Column(DateTime, nullable=True)

class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, index=True)

Base.metadata.create_all(bind=engine)

llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo",  
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
)

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

class Message(BaseModel):
    message: str

last_action = {"type": None, "data": None}

@app.post("/chat")
async def chat(msg: Message):
    session = SessionLocal()
    user_input = msg.message.strip()
    lowered = user_input.lower()

    if last_action["type"] == "awaiting_due_date":
        parsed_time = dateparser.parse(user_input)
        task_text = last_action["data"]
        if parsed_time:
            task = session.query(Task).filter_by(content=task_text).order_by(Task.id.desc()).first()
            if task:
                task.due_date = parsed_time
                session.commit()
            last_action["type"] = None
            return {"response": f"📅 Task ‘{task_text}’ set for {parsed_time.strftime('%A, %I:%M %p')}. ✅ Anything else?"}
        else:
            return {"response": "⏰ I couldn’t understand that time. Can you say it like ‘tomorrow at 4pm’ or ‘in 2 hours’?"}

    if lowered in ["yes", "sure", "go ahead", "add one", "okay", "ok"]:
        if last_action["type"] == "asked_due_date":
            last_action["type"] = "awaiting_due_date"
            return {"response": "Cool. When should I remind you or mark it due?"}
        else:
            return {"response": "Got it! What’s next?"}

    if lowered.startswith("/task"):
        content = user_input[5:].strip()
        if not content:
            return {"response": "🔖 What task should I add?"}
        task = Task(content=content)
        session.add(task)
        session.commit()
        last_action["type"] = "asked_due_date"
        last_action["data"] = content
        return {"response": f"✅ Task added: “{content}.” Want to add a due date or priority?"}

    elif lowered.startswith("/notes"):
        content = user_input[6:].strip()
        if not content:
            return {"response": "📝 Sure — what should I write down?"}
        session.add(Note(content=content))
        session.commit()
        return {"response": f"📒 Noted: “{content}.” Want to tag or organize it further?"}

    elif lowered.startswith("/show tasks"):
        tasks = session.query(Task).order_by(Task.due_date.asc().nulls_last()).all()
        if not tasks:
            return {"response": "🧖‍♂️ No tasks saved yet. Want to add one now?"}
        return {"response": "📂 Your tasks:\n" + "\n".join([
            f"• {t.content} ({t.due_date.strftime('%b %d %I:%M %p') if t.due_date else 'No due date'})" for t in tasks])}

    elif lowered.startswith("/show calendar"):
        tasks = session.query(Task).order_by(Task.due_date.asc().nulls_last()).all()
        if not tasks:
            return {"response": "📭 Nothing to show on your calendar yet."}
        return {"response": "🗓️ Calendar View:\n" + "\n".join([
            f"📌 {t.content} — {t.due_date.strftime('%a, %d %b %I:%M %p') if t.due_date else '❓ No due date'}" for t in tasks])}

    elif lowered.startswith("/show notes"):
        notes = session.query(Note).all()
        if not notes:
            return {"response": "📬 No notes found. You can try `/notes your note here`."}
        return {"response": "📟 Here’s what you’ve noted:\n" + "\n".join([f"📝 {n.content}" for n in notes])}

    elif lowered.startswith("/summarize"):
        notes = session.query(Note).all()
        if not notes:
            return {"response": "🪥 No notes yet — nothing to summarize."}
        all_notes = "\n".join([n.content for n in notes])
        summary = conversation.run(f"Summarize these notes casually:\n{all_notes}")
        return {"response": summary}

    elif lowered.startswith("/help"):
        return {"response": (
            "🧠 I can help you stay on top of your day:\n"
            "• `/task read a chapter` → add task\n"
            "• `/notes project ideas` → save a note\n"
            "• `/show tasks` → list tasks\n"
            "• `/show calendar` → date-aware view of tasks\n"
            "• `/summarize` → quick summary of notes\n"
            "• or just say what’s on your mind"
        )}

    if "remind me" in lowered or "to-do" in lowered:
        content = re.sub(r"(remind me|to-do|add a task)", "", user_input, flags=re.IGNORECASE).strip()
        if content:
            session.add(Task(content=content))
            session.commit()
            last_action["type"] = "asked_due_date"
            last_action["data"] = content
            return {"response": f"📝 Task added: “{content}.” Want me to set a reminder?"}
        else:
            return {"response": "What task should I note down?"}

    elif "note" in lowered or "remember" in lowered:
        content = re.sub(r"(note|remember)", "", user_input, flags=re.IGNORECASE).strip()
        if content:
            session.add(Note(content=content))
            session.commit()
            return {"response": f"🧠 Noted: “{content}.” Want to tag this or keep going?"}
        else:
            return {"response": "Sure! What should I remember?"}

    ai_reply = conversation.run(user_input)
    return {"response": ai_reply}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_home():
    return FileResponse("static/index.html")
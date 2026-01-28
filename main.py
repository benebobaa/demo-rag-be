from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import os
import uuid
from dotenv import load_dotenv
import asyncio
from datetime import datetime

# Load env vars before importing app modules that use them
load_dotenv()

from app.models import UploadResponse, QueryRequest, QueryResponse, SearchRequest, SearchResponse, SearchResult
from app.rag_engine import rag_engine
from app.agent import agent_system
from app.sse import format_sse_event
from app.database import engine, get_db
from app.models_db import Base, DocumentModel, Session, Message
from sqlalchemy.orm import Session as DBSession
from fastapi import Depends
from typing import List

load_dotenv()

# Create Tables
Base.metadata.create_all(bind=engine)

# Initialize Graph from DB
try:
    with DBSession(bind=engine) as db:
        rag_engine.load_graph_from_db(db)
except Exception as e:
    print(f"Warning: Failed to load graph from DB: {e}")

app = FastAPI(title="Dynamic Knowledge Graph Explorer")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), db: DBSession = Depends(get_db)):
    try:
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process file
        graph_data = await rag_engine.process_file(file_path, db_session=db)
        
        # Save to DB
        doc = DocumentModel(
            filename=file.filename,
            file_path=file_path,
            size=f"{os.path.getsize(file_path) / 1024:.1f} KB"
        )
        db.add(doc)
        db.commit()
        
        return UploadResponse(
            filename=file.filename,
            message="File processed and graph updated.",
            graph=graph_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents(db: DBSession = Depends(get_db)):
    docs = db.query(DocumentModel).order_by(DocumentModel.created_at.desc()).all()
    return [{"id": d.id, "name": d.filename, "size": d.size, "date": d.created_at.strftime("%Y-%m-%d %H:%M"), "status": d.status} for d in docs]

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Test vector similarity search against uploaded documents."""
    import time
    start_time = time.time()
    
    if not rag_engine.vector_store:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    
    # Validate k parameter
    k = max(1, min(request.k, 20))  # Clamp between 1 and 20
    
    results_with_scores = rag_engine.query_vector_with_scores(request.query, k=k)
    
    search_results = []
    for doc, score in results_with_scores:
        search_results.append(SearchResult(
            content=doc.page_content,
            metadata=doc.metadata or {},
            score=float(score)
        ))
    
    execution_time_ms = (time.time() - start_time) * 1000
    
    return SearchResponse(
        query=request.query,
        results=search_results,
        execution_time_ms=round(execution_time_ms, 2)
    )

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest, db: DBSession = Depends(get_db)):
    try:
        if not rag_engine.vector_store:
             raise HTTPException(status_code=400, detail="No documents uploaded yet.")
        
        # 1. Get or Create Session
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            new_session = Session(id=session_id, title=request.query[:50])
            db.add(new_session)
            db.commit()
        else:
            # Verify session exists
            existing_session = db.query(Session).filter(Session.id == session_id).first()
            if not existing_session:
                new_session = Session(id=session_id, title=request.query[:50])
                db.add(new_session)
                db.commit()
            
        # 2. Save User Message
        user_msg = Message(session_id=session_id, role="user", content=request.query)
        db.add(user_msg)
        
        # 3. Get History
        history_msgs = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at).all()
        # Convert to list of dicts
        history = [{"role": m.role, "content": m.content} for m in history_msgs]
        
        # 4. Get Answer
        response = await agent_system.run_query(request.query, chat_history=history, session_id=session_id)
        
        # 4. Save AI Message
        ai_msg = Message(
            session_id=session_id, 
            role="assistant", 
            content=response.answer, 
            traces=[t.dict() for t in response.trace]
        )
        db.add(ai_msg)
        db.commit()
        
        return QueryResponse(
            answer=response.answer, 
            trace=response.trace, 
            session_id=session_id
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_agent_stream(request: QueryRequest, db: DBSession = Depends(get_db)):
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    if not rag_engine.vector_store:
        async def error_stream():
            payload = {
                "seq": 1,
                "ts": datetime.utcnow().isoformat() + "Z",
                "code": "no_documents",
                "message": "No documents uploaded yet.",
                "recoverable": True,
            }
            yield format_sse_event("error", payload)
        return StreamingResponse(error_stream(), media_type="text/event-stream", headers=headers)

    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        new_session = Session(id=session_id, title=request.query[:50])
        db.add(new_session)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise HTTPException(status_code=500, detail="Failed to create session.")
    else:
        # Verify session exists
        existing_session = db.query(Session).filter(Session.id == session_id).first()
        if not existing_session:
            new_session = Session(id=session_id, title=request.query[:50])
            db.add(new_session)
            try:
                db.commit()
            except Exception:
                db.rollback()
                raise HTTPException(status_code=500, detail="Failed to create session.")

    user_msg = Message(session_id=session_id, role="user", content=request.query)
    db.add(user_msg)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save user message.")

    history_msgs = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at).all()
    history = [{"role": m.role, "content": m.content} for m in history_msgs]

    async def event_generator():
        seq = 0
        traces = []
        final_answer = None
        had_error = False
        stream = agent_system.stream_query(request.query, chat_history=history, session_id=session_id)

        def build(event_type: str, data: dict):
            nonlocal seq
            seq += 1
            payload = {"seq": seq, "ts": datetime.utcnow().isoformat() + "Z", **data}
            return format_sse_event(event_type, payload)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(stream.__anext__(), timeout=5)
                except asyncio.TimeoutError:
                    yield build("ping", {})
                    continue
                except StopAsyncIteration:
                    break

                event_type = event.pop("type")
                if event_type == "trace":
                    traces.append({
                        "step": event.get("step"),
                        "thought": event.get("thought"),
                        "tool": event.get("tool"),
                        "observation": event.get("observation"),
                    })
                elif event_type == "answer":
                    final_answer = event.get("answer")
                elif event_type == "error":
                    had_error = True

                yield build(event_type, event)

                if event_type == "error":
                    break
        except asyncio.CancelledError:
            await stream.aclose()
            return
        except Exception as e:
            yield build("error", {
                "code": "stream_error",
                "message": str(e),
                "recoverable": False,
            })
            return

        if not had_error:
            if final_answer:
                ai_msg = Message(
                    session_id=session_id,
                    role="assistant",
                    content=final_answer,
                    traces=traces,
                )
                db.add(ai_msg)
                try:
                    db.commit()
                except Exception:
                    db.rollback()
                    # Don't raise, still send done event

            yield build("done", {"session_id": session_id})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.get("/graph")
async def get_graph():
    return rag_engine.get_graph_data()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except Exception:
        pass

@app.get("/sessions/{session_id}", response_model=List[dict])
async def get_session_history(session_id: str, db: DBSession = Depends(get_db)):
    messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at).all()
    return [{"role": m.role, "content": m.content, "traces": m.traces} for m in messages]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

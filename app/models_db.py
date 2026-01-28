from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    messages = relationship("Message", back_populates="session")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String) # "user" or "assistant"
    content = Column(Text)
    traces = Column(JSON, nullable=True) # Store agent thoughts
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("Session", back_populates="messages")

class DocumentModel(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_path = Column(String)
    size = Column(String)
    status = Column(String, default="indexed")
    created_at = Column(DateTime, default=datetime.utcnow)

class GraphTriple(Base):
    __tablename__ = "graph_triples"
    
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, index=True)
    target = Column(String, index=True)
    relation = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

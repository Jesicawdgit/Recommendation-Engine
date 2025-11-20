# Learnora — AI-Powered Learning Recommendation Engine
Learnora is an AI-driven LLM based personalized learning assistant that generates custom roadmaps, recommends articles, videos, projects, and certifications, and helps users accelerate their learning journey.

Built with React + Flask + Sentence Transformers, Learnora provides an intelligent, visually rich, and interactive learning experience.
Features
**1.User Authentication**
Email + Password login
Clean, modern UI
Protected routes (only logged-in users can access the dashboard)

**2.Smart Search**

Enter any topic — e.g., python, machine learning, web dev
Backend performs semantic search on curated datasets
Returns the most relevant learning resources

**3.AI-Generated Learning Roadmaps**

Fishbone-style visually animated roadmap
Organized into:
Articles & Documentation
Video Tutorials
Projects
Certifications
Each resource shows:

Title,Source,Tags,Similarity score,YouTube thumbnails,Clickable external links

**4.Shareable Roadmaps**

Users can generate a shareable link
Saved as a JSON entry on backend

**5.Chat-Style Prompt Interface**

Ask questions
Generate new roadmaps
Clean UI with sidebar chat history

**6.Tech Stack**
Frontend (React)
React.js (JSX)
Custom UI components
Protected routes
Fishbone Roadmap UI
YouTube thumbnail extraction
Clean animations and styling
Backend (Flask)
Flask REST API
Flask-CORS
Sentence Transformers (Semantic Search)
FAISS vector search
Custom fishbone_roadmap.py builder
Shareable roadmap storage

**AI / ML**

Sentence Transformers
Finetuned ST model (stored locally)
FAISS Index for fast semantic search

**Future Improvements**

Storing user login info in MySQL
Saving search history
First-time user survey + personalization
Real-time chat with LLM
User dashboards & streak tracking


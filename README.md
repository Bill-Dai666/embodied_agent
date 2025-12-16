Embodied Agent Project

A research project exploring different interaction modalities for AI-powered conversational agents discussing news articles. This repository contains two distinct implementations of chatbot interfaces for conducting user studies and experiments.

Project Overview

This project implements conversational AI agents that engage users in discussions about news articles as part of a crowdsourcing research experiment. Both implementations use OpenAI's GPT models for generating empathetic, contextual responses and store conversation data in MongoDB for analysis. This project supports academic research comparing different interaction modalities (embodied vs. text-based) for AI conversational agents in experimental settings.

Components

avatar_based/

A 3D virtual avatar-based chatbot interface built with Soul Machines technology. This implementation features:

Frontend: React-based web UI integrated with Soul Machines Web SDK for real-time avatar interactions
Backend: FastAPI skill adapter that handles NLP processing and integrates with Soul Machines persona platform
Features: Visual avatar presence, video/audio streaming, emotion detection, and content cards
Use Case: Provides an immersive, face-to-face conversation experience with an embodied AI agent

Tech Stack: React, Redux, FastAPI, Soul Machines SDK, OpenAI, MongoDB

chat_based/

A text-only chatbot interface for traditional chat interactions. This implementation features:

Frontend: Simple HTML/JavaScript chat interface
Backend: FastAPI server with REST API endpoints
Features: Clean text-based conversation, URL parameter support for article/resource tracking
Use Case: Provides a lightweight, accessible chat interface for comparison studies

Tech Stack: FastAPI, OpenAI, MongoDB, HTML/JavaScript

Key Features

Both implementations share:
- Article-based conversation context loaded from articles.json
- Conversation history tracking and storage in MongoDB
- Completion code system for participant compensation after 10+ conversation turns
- URL parameter support for tracking article IDs and response IDs
- Empathetic conversation guidelines for natural dialogue


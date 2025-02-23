# Menu Manager AI

Menu Manager AI is an application that integrates CRUD operations with AI-powered recommendations and information retrieval. The backend is built using *FastAPI, while the frontend is implemented using **Streamlit* for an interactive user experience. The system utilizes *OpenAI* and *Serper API* to enhance AI-based functionalities.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [User Interface (UI)](#user-interface-ui)
- [Example Images](#example-images)
- [Known Issues](#known-issues)
- [Contributing](#contributing)

## Overview

A LLM application that can manage various tasks like standard *CRUD operations* (Create, Read, Update, Delete) , while leveraging OpenAI API to provide *recommendations* and *information retrieval* based on user prompts.

## Features

- *CRUD Operations:*
  - Add, view, update, and delete records in a *pandas DataFrame*.

- *AI-Based Recommendations:*
  - Uses *OpenAI* to generate menu recommendations based on stored data.

- *Information Retrieval:*
  - Fetches and summarizes web content using *Serper API* and *OpenAI*.

- *Interactive UI:*
  - A *Streamlit-powered interface* for easy interaction and visualization of data.

## Architecture

The project consists of two main components:

- **Backend (main.py)**
  - FastAPI server with endpoints for CRUD operations, file uploads, AI recommendations, and web searches.
  
- **Frontend (ui.py)**
  - Streamlit-based UI to interact with the application, upload files, enter prompts, and displaying data.

## Prerequisites

Before running the application, ensure you have the following:

- *Python:* Version *3.8* or higher.
- *Environment Variables:*
  - OPENAI_API_KEY – Your OpenAI API key.
  - SERPER_API_KEY – Your Serper API key.
- *Required Python Packages:*
  bash
  pip install fastapi uvicorn pandas numpy pydantic openai httpx python-dotenv streamlit requests
  

## Installation and Setup

1. *Clone the Repository:*
   bash
   git clone https://github.com/yourusername/menu-manager-ai.git
   cd menu-manager-ai
   

2. *Set Up Environment Variables:*
   Create a .env file and add your API keys:
   ini
   OPENAI_API_KEY=your_openai_api_key
   SERPER_API_KEY=your_serper_api_key
   

3. *Install Dependencies:*
   bash
   pip install -r requirements.txt
   

## Running the Application

### 1. Start the FastAPI Backend
bash
uvicorn main:app --reload


### 2. Run the Streamlit Frontend
bash
streamlit run ui.py


## API Endpoints

| Method | Endpoint   | Description                                        |
|--------|------------|-------------------------------------------------   |
| POST | /upload      | Uploads the dataframe                              |
| GET  | /data        | Retrieve all records                               |
| POST | /process     | Processes the user prompt and decides the intent   |
| POST | /agent       | Executes the decided intent                        |

## User Interface (UI)

- The Streamlit UI allows users to *upload files, **manage records, and **request AI-generated recommendations*.
- Provides a *real-time view* of stored data and AI responses.

## Example Images

### AI-Generated Information

![Get Info](https://github.com/user-attachments/assets/62ae13be-d4a6-4849-84bf-edb8fe8c5d64)

### AI Recommendations

![Recommendation](https://github.com/user-attachments/assets/c26db73c-e7d1-4feb-8e68-5987a7403dc0)

## Known Issues

- *CRUD Display Errors:* There are minor issues with displaying the CRUD applied data, which will be fixed.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (feature-branch).
3. Make your changes and commit them.
4. Push to your branch and submit a pull request.

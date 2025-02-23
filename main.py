import os
from typing import Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import io
import json
import numpy as np
import asyncio
import sys
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
serper_api_key = os.getenv("SERPER_API_KEY")

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


class CRUDOperations:
    def __init__(self):
        self.df = pd.DataFrame(columns=['Data'])

    def load_dataframe(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def classify_crud_intent(self, prompt: str):
        functions = [
            {
                "name": "create",
                "description": "Add a record",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "object"}},
                    "required": ["data"]
                }
            },
            {
                "name": "read",
                "description": "Get all records",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "update",
                "description": "update a record",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "new_data": {"type": "object"}
                    },
                    "required": ["index", "new_data"]
                }
            },
            {
                "name": "delete",
                "description": "Remove a record",
                "parameters": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"]
                }
            }
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                functions=functions,
                function_call="auto"
            )
            msg = response.choices[0].message
            if msg.function_call:
                return msg.function_call.name, json.loads(msg.function_call.arguments)
        except Exception as e:
            print(f"CRUD classification error: {e}")
            return None, None

    def create_record(self, data: Any) -> str:
        # For structuring both non dictionary and dictionary type
        if isinstance(data, dict):
            for key in data.keys():
                if key not in self.df.columns:
                    self.df[key] = None
            row = {col: data.get(col, None) for col in self.df.columns}
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        else:
            self.df = pd.concat([self.df, pd.DataFrame([{"Data": data}])], ignore_index=True)

        return self.read_records()

    def read_records(self) -> str:
        return self.df.to_json(orient='records') if not self.df.empty else "No data available"

    def update_record(self, index: int, new_data: Any) -> str:
        if 0 <= index < len(self.df):
            if isinstance(new_data, dict):
                for key, value in new_data.items():
                    if key not in self.df.columns:
                        self.df[key] = None
                    self.df.at[index, key] = value
            else:
                self.df.at[index, "Data"] = new_data
            return self.read_records()
        return "Invalid index for update"

    def delete_record(self, index: int) -> str:
        if 0 <= index < len(self.df):
            self.df = self.df.drop(index).reset_index(drop=True)
            return self.read_records()
        return "Invalid index for deletion"

    def process_request(self, prompt: str) -> str:
        operation, params = self.classify_crud_intent(prompt)
        if operation == "create":
            return self.create_record(params["data"])
        elif operation == "read":
            return self.read_records()
        elif operation == "update":
            return self.update_record(params["index"], params["new_data"])
        elif operation == "delete":
            return self.delete_record(params["index"])
        return "Failed to process CRUD operation"

crud_handler = CRUDOperations()

class UserPrompt(BaseModel):
    prompt: str

class AgentPrompt(BaseModel):
    prompt: str
    reasoning: bool = False

def classify_intent(prompt: str) -> str:
    system_prompt = "Classify this prompt as CRUD, Recommendation, or Generate_Info."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        intent = response.choices[0].message.content.strip().lower()
        print(f"DEBUG: Classified intent: '{intent}'")
        return intent
    except Exception as e:
        print(f"Intent classification error: {e}")
        return "unknown"


def recommendation(prompt: str):
    if crud_handler.df.empty:
        return "Data not available for recommendation"
    df_summary = crud_handler.df.sample(len(crud_handler.read_records())//5).to_json(orient="records")

    full_prompt = f"Data: {df_summary}\nRequest: {prompt}\nRecommend:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful menu recommender. You recommend the most relevant menu items based on the request."},
            {"role": "user", "content": full_prompt}
        ],
        max_tokens=300,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


def generate_ai_response(search_results: dict, query: str) -> str:
    try:
        context = []
        if search_results.get('answerBox'):
            answer = search_results['answerBox']
            context.append(f"Answer: {answer.get('snippet', answer.get('answer', ''))}")
        if search_results.get('organic'):
            for result in search_results['organic'][:3]:
                context.append(f"{result.get('title', '')}: {result.get('snippet', '')}")
        if not context:
            return "No info found."

        search_results_text = '\n'.join(context)
        prompt_content = f"Query: {query}\nResults:\n{search_results_text}\nAnswer:"
        system_content = (f"You are an information assistant. if query mentions dataframe check it. "
                          f"Dataframe: {crud_handler.read_records()}")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"AI generation error: {str(e)}")
        return "Unable to generate response."


async def get_web_results(query: str) -> Any | None:
    headers = {"X-API-KEY": serper_api_key}
    params = {"q": query, "gl": "us", "hl": "en"}
    url = "https://google.serper.dev/search"
    try:
        async with httpx.AsyncClient() as web_client:
            response = await web_client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Search error: {str(e)}")
        return None


async def info_pipeline(prompt: str) -> str:
    search_results = await get_web_results(prompt)
    if not search_results:
        return "Unable to retrieve info."
    return generate_ai_response(search_results, prompt)


async def MenuManager(prompt: str, reasoning: bool = True) -> str:
    intent = classify_intent(prompt)
    if "crud" in intent:
        result = crud_handler.process_request(prompt)
    elif "recommendation" in intent:
        result = recommendation(prompt)
    elif "generate_info" in intent:
        result = await info_pipeline(prompt)
    else:
        result = "Unable to determine type"
    if reasoning:
        # Truncate tokens
        result_lim = result if len(result) <= 1000 else result[:1000] + "..."
        reasoning_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Explain your reasoning and then give the final answer."},
                {"role": "user", "content": f"Prompt: {prompt}\nResult: {result_lim}\nReason:"}
            ],
            max_tokens=300,
            temperature=0.5
        )
        chain_output = reasoning_response.choices[0].message.content.strip()
        return chain_output
    else:
        return result


@app.post("/process")
async def process_prompt(user_prompt: UserPrompt):
    res = await MenuManager(user_prompt.prompt)
    return {"response": res}


@app.post("/agent")
async def process_agent(agent_prompt: AgentPrompt):
    res = await MenuManager(agent_prompt.prompt, reasoning=agent_prompt.reasoning)
    return {"response": res}


@app.get("/data")
async def get_data():
    safe_df = crud_handler.df.replace([np.nan, np.inf, -np.inf], None) # Handled the null columns
    return {"data": safe_df.to_dict(orient="records")}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not crud_handler.df.empty:
            return {"message": "Dataframe already stored. Not reuploading."}
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(400, "Unsupported file type")
        crud_handler.load_dataframe(df)
        return {"message": f"Uploaded {len(df)} records"}
    except Exception as e:
        raise HTTPException(500, f"Upload not successful: {str(e)}")


import asyncio
import json
import os
from datetime import datetime
import aiohttp
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self):
        self.host = "localhost"
        self.port = 5005
        self.state_dir = "conversations"
        self.billing_file = "billing_summary.json"
        self.conversations = {}
        self.agent_connections = {}
        self.token_limit = 10000
        self.warning_threshold = 0.9
        self.pricing = {"input_tokens": 5.0, "output_tokens": 10.0}
        self.load_config()
        self.load_state()

    def load_config(self):
        with open("config/grok_api_config.json", "r") as f:
            config = json.load(f)
            self.token_limit = config.get("token_limit", 10000)
            self.warning_threshold = config.get("warning_threshold", 0.9)
            self.pricing = config.get("pricing", self.pricing)

    def load_state(self):
        os.makedirs(self.state_dir, exist_ok=True)
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
        for root, _, files in os.walk(self.state_dir):
            for file in files:
                if file == "state.json":
                    cid = os.path.basename(root)
                    with open(os.path.join(root, file), "r") as f:
                        state = json.load(f)
                        self.conversations[cid] = state
                        usage = state.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0})
                        for key in total_usage:
                            total_usage[key] += usage.get(key, 0)
        with open(self.billing_file, "w") as f:
            json.dump({"total_usage": total_usage, "estimated_cost": total_usage["cost"]}, f)

    def save_state(self, cid):
        os.makedirs(os.path.join(self.state_dir, cid), exist_ok=True)
        with open(os.path.join(self.state_dir, cid, "state.json"), "w") as f:
            json.dump(self.conversations[cid], f, indent=2)
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
        for state in self.conversations.values():
            usage = state.get("usage", {})
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)
        with open(self.billing_file, "w") as f:
            json.dump({"total_usage": total_usage, "estimated_cost": total_usage["cost"]}, f)

    async def mock_grok_api(self, messages):
        prompt_tokens = sum(len(str(m["content"])) for m in messages if m["role"] == "user") // 4
        if messages[1]["content"]["task"] == "process_request":
            return {
                "result": "Processed",
                "target": "orch",
                "actions": [
                    {"task": "search", "task_id": "T1", "data": {"query": "current Bitcoin price March 8 2025"}, "persist": False},
                    {"task": "send_email", "task_id": "T2", "data": {"to": "test@test.com", "subject": "Today’s BTC Price", "body": "TBD"}, "persist": True},
                    {"task": "update_ui", "data": {"response": "Task T1 started. Task T2 queued."}, "conversation_id": "C1"}
                ],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 60, "total_tokens": prompt_tokens + 60}
            }
        elif messages[1]["content"]["task"] == "process_update":
            data = messages[1]["content"]["data"]
            if data["task_id"] == "T1":
                return {
                    "result": "Processed",
                    "target": "orch",
                    "actions": [
                        {"task": "send_email", "task_id": "T2", "data": {"to": "test@test.com", "subject": "Today’s BTC Price", "body": "Bitcoin price on March 8, 2025: $88,000"}, "persist": True},
                        {"task": "update_ui", "data": {"response": "Task T1 completed: BTC price is $88,000. Task T2 updated: Email queued."}, "conversation_id": "C1"}
                    ],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 70, "total_tokens": prompt_tokens + 70}
                }
            elif data["task_id"] == "T2":
                cost = self.conversations["C1"]["usage"]["cost"]
                return {
                    "result": "Processed",
                    "target": "user",
                    "response": f"Task T2 completed: Email with BTC price $88,000 sent to test@test.com. Usage for C1: 320 tokens (${cost:.4f}).",
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 40, "total_tokens": prompt_tokens + 40}
                }

    async def call_grok_api(self, messages):
        return await self.mock_grok_api(messages)

    async def process_message(self, message):
        cid = message.get("conversation_id", "C" + str(len(self.conversations) + 1))
        if cid not in self.conversations:
            self.conversations[cid] = {"tasks": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}}

        if message["type"] == "user_request":
            with open("prompts/orch_grok_prompt.txt", "r") as f:
                system_prompt = f.read()
            grok_response = await self.call_grok_api([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": {"task": "process_request", "data": message}}
            ])
        elif message["type"] == "status_update":
            task_id = message["task_id"]
            self.update_task_state(cid, task_id, message["state"], message.get("result"))
            grok_response = await self.call_grok_api([
                {"role": "system", "content": open("prompts/orch_grok_prompt.txt", "r").read()},
                {"role": "user", "content": {"task": "process_update", "data": message}}
            ])

        self.update_usage(cid, grok_response["usage"])
        await self.handle_grok_response(cid, grok_response)

    def update_task_state(self, cid, task_id, state, result=None):
        for task in self.conversations[cid]["tasks"]:
            if task["task_id"] == task_id:
                task["state"] = state
                if result:
                    task["result"] = result
                break
        self.save_state(cid)

    def update_usage(self, cid, usage):
        state = self.conversations[cid]
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            state["usage"][key] += usage[key]
        state["usage"]["cost"] = (state["usage"]["prompt_tokens"] * self.pricing["input_tokens"] / 1e6) + \
                                 (state["usage"]["completion_tokens"] * self.pricing["output_tokens"] / 1e6)
        total_tokens = sum(state["usage"]["total_tokens"] for state in self.conversations.values())
        if total_tokens >= self.token_limit:
            logger.warning("Token limit exceeded. Pausing non-persistent tasks.")
        elif total_tokens >= self.token_limit * self.warning_threshold:
            logger.warning(f"Approaching token limit: {total_tokens}/{self.token_limit} tokens used.")
        self.save_state(cid)

    async def handle_grok_response(self, cid, response):
        if response["target"] == "orch":
            for action in response.get("actions", []):
                if action["task"] == "update_ui":
                    await self.send_to_ui(cid, action["data"]["response"])
                elif action["task"] in ["search", "send_email"]:
                    await self.send_to_agent(action)
                    if action["task_id"] not in [t["task_id"] for t in self.conversations[cid]["tasks"]]:
                        self.conversations[cid]["tasks"].append({"task_id": action["task_id"], "state": "running", "persist": action["persist"]})
                        self.save_state(cid)
        elif response["target"] == "user":
            await self.send_to_ui(cid, response["response"])

    async def send_to_ui(self, cid, response):
        try:
            reader, writer = await asyncio.open_connection("localhost", 5006)
            writer.write(json.dumps({"type": "response", "response": response, "conversation_id": cid}).encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logger.error(f"Failed to send to UI: {e}")

    async def send_to_agent(self, action):
        port = 5003 if action["task"] == "search" else 5007
        agent_id = f"{action['task']}-{port}"
        if agent_id not in self.agent_connections:
            try:
                reader, writer = await asyncio.open_connection("localhost", port)
                self.agent_connections[agent_id] = writer
            except Exception as e:
                logger.error(f"Failed to connect to agent {agent_id}: {e}")
                return
        writer = self.agent_connections[agent_id]
        writer.write(json.dumps(action).encode())
        await writer.drain()

    async def handle_client(self, reader, writer):
        data = await reader.read(1024)
        if data:
            message = json.loads(data.decode())
            logger.info(f"Received: {message}")
            await self.process_message(message)
        writer.close()
        await writer.wait_closed()

    async def run(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(f"Orchestrator running on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    orch = Orchestrator()
    asyncio.run(orch.run())
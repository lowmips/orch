import asyncio
import json
import os
import aiohttp
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GROK_CONFIG_PATH = "config/grok_api_config.json"
ORCH_CONFIG_PATH = "config/orchestrator_config.json"
CONVERSATIONS_DIR = "conversations"
BILLING_FILE = "billing_summary.json"

with open(GROK_CONFIG_PATH, 'r') as f:
    grok_config = json.load(f)
API_KEY = grok_config["api_key"]
ENDPOINT = grok_config["endpoint"]
TOKEN_LIMIT = grok_config["token_limit"]
WARNING_THRESHOLD = grok_config["warning_threshold"]
INPUT_COST = grok_config["pricing"]["input_tokens"] / 1_000_000
OUTPUT_COST = grok_config["pricing"]["output_tokens"] / 1_000_000

with open(ORCH_CONFIG_PATH, 'r') as f:
    orch_config = json.load(f)
ORCH_PORT = orch_config["port"]
POLL_TIMEOUT = orch_config["poll_timeout"]
MAX_RETRIES = orch_config["max_retries"]
TOTAL_TIMEOUT = orch_config["total_timeout"]

with open("prompts/orchestrator_system_prompt.txt", 'r') as f:
    SYSTEM_PROMPT = f.read()

class Orchestrator:
    def __init__(self):
        self.tasks = {}
        self.conversations = {}
        self.agents = {}
        self.task_counter = 0
        self.total_tokens = 0
        self.total_cost = 0
        self.loop = asyncio.get_event_loop()
        os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
        self.load_state()

    async def start(self):
        server = await asyncio.start_server(self.handle_client, 'localhost', ORCH_PORT)
        logger.info(f"Orchestrator started on localhost:{ORCH_PORT}")
        async with server:
            await server.serve_forever()

    def load_state(self):
        for convo_dir in os.listdir(CONVERSATIONS_DIR):
            state_path = os.path.join(CONVERSATIONS_DIR, convo_dir, "state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self.conversations[convo_dir] = state
                for task_id, task in state.get("tasks", {}).items():
                    self.tasks[task_id] = task
                    if task["s"] in ["r", "p"]:
                        asyncio.create_task(self.poll_task(task_id, task["port"]))
                self.total_tokens += state.get("total_tokens", 0)
                self.total_cost += state.get("cost", 0)
        self.save_billing_summary()

    def save_state(self, convo_id):
        convo_dir = os.path.join(CONVERSATIONS_DIR, convo_id)
        os.makedirs(convo_dir, exist_ok=True)
        state_path = os.path.join(convo_dir, "state.json")
        with open(state_path, 'w') as f:
            json.dump(self.conversations.get(convo_id, {}), f)
        self.save_billing_summary()

    def save_billing_summary(self):
        summary = {"total_tokens": self.total_tokens, "total_cost": self.total_cost}
        with open(BILLING_FILE, 'w') as f:
            json.dump(summary, f)

    async def handle_client(self, reader, writer):
        data = await reader.read(1024)
        if not data:
            return
        msg = json.loads(data.decode())
        addr = writer.get_extra_info('peername')
        logger.info(f"Received from {addr}: {msg}")
        await self.process_message(msg, writer)

    async def process_message(self, msg, writer):
        msg_type = msg.get("t")
        convo_id = msg.get("c")

        if msg_type == "u":  # user_request
            if convo_id not in self.conversations:
                self.conversations[convo_id] = {"h": [], "tasks": {}, "total_tokens": 0, "cost": 0}
            self.conversations[convo_id]["h"].append(msg["r"])
            response = await self.query_grok(convo_id, msg["r"], msg["h"])
            if response["target"] == "user":
                reply = {"t": "r", "r": response["response"]}
                writer.write(json.dumps(reply).encode())
                await writer.drain()
            elif response["target"] == "orch":
                for action in response.get("actions", []):
                    await self.handle_action(action, convo_id)
            self.save_state(convo_id)

        elif msg_type == "x":  # cancel
            if convo_id in self.conversations:
                for task_id in self.conversations[convo_id]["tasks"]:
                    self.tasks[task_id]["s"] = "f"
                self.save_state(convo_id)

        elif msg_type == "m":  # resume
            self.conversations[convo_id] = {"h": msg["h"], "tasks": {}, "total_tokens": 0, "cost": 0}
            for task_id in msg.get("t", []):
                self.tasks[task_id]["s"] = "r"
                asyncio.create_task(self.poll_task(task_id, self.tasks[task_id]["port"]))
            self.save_state(convo_id)

        elif msg_type == "n":  # reconnect
            agent_id = msg["a"]
            self.agents[agent_id] = {"port": msg.get("o", 0)}
            for queued_msg in msg["q"]:
                await self.process_message(queued_msg, writer)

        elif msg_type == "s":  # status_update
            task_id = msg["i"]
            if task_id in self.tasks:
                self.tasks[task_id]["s"] = msg["s"]
                self.tasks[task_id]["r"] = msg["r"]
                for convo_id, convo in self.conversations.items():
                    if task_id in convo["tasks"]:
                        convo["tasks"][task_id] = self.tasks[task_id]
                        self.save_state(convo_id)
                        break

    async def query_grok(self, convo_id, latest, history):
        if self.total_tokens >= TOKEN_LIMIT:
            return {"result": "Token limit reached", "target": "user", "response": "System paused: token limit reached"}
        if self.total_tokens >= TOKEN_LIMIT * WARNING_THRESHOLD:
            logger.warning("Token usage at 90% of limit")

        payload = {
            "role": "user",
            "content": json.dumps({
                "task": "process",
                "data": {
                    "type": "u",
                    "conversation_id": convo_id,
                    "history": {"old": history, "latest": latest}
                }
            })
        }
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            async with session.post(ENDPOINT, json=[system_msg, payload], headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    response = json.loads(result["content"])
                    usage = result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
                    self.update_usage(convo_id, usage)
                    return response
                return {"result": "Error", "target": "user", "response": "Failed to process request"}

    def update_usage(self, convo_id, usage):
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        cost = (prompt_tokens * INPUT_COST) + (completion_tokens * OUTPUT_COST)
        self.total_tokens += total_tokens
        self.total_cost += cost
        if convo_id in self.conversations:
            self.conversations[convo_id]["total_tokens"] += total_tokens
            self.conversations[convo_id]["cost"] += cost

    async def handle_action(self, action, convo_id):
        task = action.get("t")
        self.task_counter += 1
        task_id = f"T{self.task_counter}"
        port = self.get_agent_port(action.get("agent", "default"))
        self.tasks[task_id] = {"t": task, "s": "r", "port": port}
        self.conversations[convo_id]["tasks"][task_id] = self.tasks[task_id]
        await self.send_task(task_id, task, action.get("d", ""), port)
        asyncio.create_task(self.poll_task(task_id, port))

    def get_agent_port(self, agent_type):
        return 5006 if agent_type == "ui_local" else 5000  # Default to Directory Agent if unknown

    async def send_task(self, task_id, task, data, port):
        reader, writer = await asyncio.open_connection('localhost', port)
        msg = {"t": task, "i": task_id, "d": data, "p": True}
        writer.write(json.dumps(msg).encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def poll_task(self, task_id, port):
        retries = 0
        while retries < MAX_RETRIES and task_id in self.tasks and self.tasks[task_id]["s"] == "r":
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection('localhost', port), timeout=POLL_TIMEOUT)
                msg = {"t": "p", "i": task_id}
                writer.write(json.dumps(msg).encode())
                await writer.drain()
                data = await asyncio.wait_for(reader.read(1024), timeout=POLL_TIMEOUT)
                if data:
                    response = json.loads(data.decode())
                    self.tasks[task_id]["s"] = response["s"]
                    self.tasks[task_id]["r"] = response["r"]
                    for convo_id, convo in self.conversations.items():
                        if task_id in convo["tasks"]:
                            convo["tasks"][task_id] = self.tasks[task_id]
                            self.save_state(convo_id)
                            break
                writer.close()
                await writer.wait_closed()
                break
            except (asyncio.TimeoutError, ConnectionError):
                retries += 1
                self.tasks[task_id]["s"] = "p" if retries < MAX_RETRIES else "a"
                await asyncio.sleep(2)
        if retries >= MAX_RETRIES:
            logger.warning(f"Task {task_id} timed out after {TOTAL_TIMEOUT}s")

if __name__ == "__main__":
    orch = Orchestrator()
    asyncio.run(orch.start())
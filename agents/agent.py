import asyncio
import json
import os
import psutil
import platform
import logging
import aiohttp
import mysql.connector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AGENT_ID = "agent-1"  # Should be unique per instance, configurable
AGENT_PORT = 5001     # Configurable per instance
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
POLL_TIMEOUT = orch_config["poll_timeout"]
MAX_RETRIES = orch_config["max_retries"]
TOTAL_TIMEOUT = orch_config["total_timeout"]

with open("prompts/agent_grok_prompt.txt", 'r') as f:
    GROK_PROMPT = f.read()

# MySQL config (example, should be in a config file)
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "password",
    "database": "agent_directory"
}

class Agent:
    def __init__(self, agent_id, port):
        self.agent_id = agent_id
        self.port = port
        self.resources = {}
        self.capabilities = ["exec_program"]  # Dynamic, updated as needed
        self.tasks = {}
        self.conversations = {}
        self.total_tokens = 0
        self.total_cost = 0
        self.loop = asyncio.get_event_loop()
        os.makedirs(f"{CONVERSATIONS_DIR}/{self.agent_id}", exist_ok=True)
        self.load_state()

    def get_resources(self, full=False):
        resources = {
            "cpu_cores": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total // (1024 * 1024),
            "disk_total": psutil.disk_usage('/').total // (1024 * 1024),
            "disk_free": psutil.disk_usage('/').free // (1024 * 1024),
            "os": platform.system()
        }
        if full or not resources.get("cpu_usage"):
            resources.update({
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_used": psutil.virtual_memory().percent,
                "disk_io": psutil.disk_io_counters().read_bytes // 1024 + psutil.disk_io_counters().write_bytes // 1024
            })
        self.resources = resources
        return resources

    async def start(self):
        await self.register_agent(full=True)
        asyncio.create_task(self.update_system_stats())
        asyncio.create_task(self.update_load_stats())
        server = await asyncio.start_server(self.handle_client, 'localhost', self.port)
        logger.info(f"Agent {self.agent_id} started on localhost:{self.port}")
        async with server:
            await server.serve_forever()

    def load_state(self):
        for convo_id in os.listdir(f"{CONVERSATIONS_DIR}/{self.agent_id}"):
            state_path = f"{CONVERSATIONS_DIR}/{self.agent_id}/{convo_id}/state.json"
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self.conversations[convo_id] = state
                for task_id, task in state.get("tasks", {}).items():
                    self.tasks[task_id] = task
                    if task["s"] in ["r", "p"]:
                        asyncio.create_task(self.poll_task(task_id, self.tasks[task_id]["port"]))
                self.total_tokens += state.get("total_tokens", 0)
                self.total_cost += state.get("cost", 0)
        self.save_billing_summary()

    def save_state(self, convo_id):
        convo_dir = f"{CONVERSATIONS_DIR}/{self.agent_id}/{convo_id}"
        os.makedirs(convo_dir, exist_ok=True)
        state_path = f"{convo_dir}/state.json"
        with open(state_path, 'w') as f:
            json.dump(self.conversations.get(convo_id, {}), f)
        self.save_billing_summary()

    def save_billing_summary(self):
        summary = {"total_tokens": self.total_tokens, "total_cost": self.total_cost}
        with open(BILLING_FILE, 'w') as f:
            json.dump(summary, f)

    async def register_agent(self, full=False):
        resources = self.get_resources(full)
        db = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO agents (id, port, capabilities, resources) VALUES (%s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE port=%s, capabilities=%s, resources=%s",
            (self.agent_id, self.port, json.dumps(self.capabilities), json.dumps(resources),
             self.port, json.dumps(self.capabilities), json.dumps(resources))
        )
        db.commit()
        cursor.close()
        db.close()

    async def update_system_stats(self):
        while True:
            await asyncio.sleep(24 * 60 * 60)  # 24 hours
            await self.register_agent(full=True)

    async def update_load_stats(self):
        while True:
            await asyncio.sleep(5 * 60)  # 5 minutes
            await self.register_agent(full=False)

    async def handle_client(self, reader, writer):
        data = await reader.read(1024)
        if not data:
            return
        msg = json.loads(data.decode())
        await self.process_message(msg, writer)

    async def process_message(self, msg, writer):
        msg_type = msg.get("t")
        convo_id = msg.get("c")

        if msg_type == "user_request":
            if convo_id not in self.conversations:
                self.conversations[convo_id] = {"h": [], "tasks": {}, "total_tokens": 0, "cost": 0}
            self.conversations[convo_id]["h"].append(msg["r"])
            response = await self.query_grok(convo_id, msg["r"], self.conversations[convo_id]["h"])
            if response["target"] == "user":
                reply = {"t": "response", "r": response["response"]}
                writer.write(json.dumps(reply).encode())
                await writer.drain()
            elif response["target"] == "agent":
                for action in response.get("actions", []):
                    await self.handle_action(action, convo_id)
            self.save_state(convo_id)

        elif msg_type == "list_conversations":
            reply = {"t": "response", "r": list(self.conversations.keys())}
            writer.write(json.dumps(reply).encode())
            await writer.drain()

        elif msg_type == "get_conversation":
            reply = {"t": "response", "r": self.conversations.get(convo_id, {"history": []})}
            writer.write(json.dumps(reply).encode())
            await writer.drain()

        elif msg_type == "query":
            call = msg["call"]
            if call == "list_capabilities":
                db = mysql.connector.connect(**MYSQL_CONFIG)
                cursor = db.cursor()
                cursor.execute("SELECT DISTINCT capabilities FROM agents")
                capabilities = set()
                for (caps,) in cursor:
                    capabilities.update(json.loads(caps))
                cursor.close()
                db.close()
                reply = {"t": "query_response", "r": list(capabilities)}
            elif call == "list_agents_for_task":
                task = msg["task"]
                db = mysql.connector.connect(**MYSQL_CONFIG)
                cursor = db.cursor()
                cursor.execute("SELECT id FROM agents WHERE JSON_CONTAINS(capabilities, %s)", (f'"{task}"',))
                agents = [row[0] for row in cursor]
                cursor.close()
                db.close()
                reply = {"t": "query_response", "r": {"task": task, "agents": agents}}
            elif call == "get_agent_resources":
                agent_id = msg["agent_id"]
                db = mysql.connector.connect(**MYSQL_CONFIG)
                cursor = db.cursor()
                cursor.execute("SELECT resources FROM agents WHERE id = %s", (agent_id,))
                result = cursor.fetchone()
                resources = json.loads(result[0]) if result else {}
                cursor.close()
                db.close()
                reply = {"t": "query_response", "r": resources}
            elif call == "get_agent_system_load":
                agent_id = msg["agent_id"]
                if agent_id == self.agent_id:
                    reply = {"t": "query_response", "r": {k: v for k, v in self.resources.items() if k in ["cpu_usage", "memory_used", "disk_io"]}}
                else:
                    db = mysql.connector.connect(**MYSQL_CONFIG)
                    cursor = db.cursor()
                    cursor.execute("SELECT resources FROM agents WHERE id = %s", (agent_id,))
                    result = cursor.fetchone()
                    resources = json.loads(result[0]) if result else {}
                    cursor.close()
                    db.close()
                    reply = {"t": "query_response", "r": {k: v for k, v in resources.items() if k in ["cpu_usage", "memory_used", "disk_io"]}}
            writer.write(json.dumps(reply).encode())
            await writer.drain()

        elif msg_type == "s":  # status_update
            task_id = msg["i"]
            if task_id in self.tasks:
                self.tasks[task_id]["s"] = msg["s"]
                self.tasks[task_id]["r"] = msg["r"]
                for cid, convo in self.conversations.items():
                    if task_id in convo["tasks"]:
                        convo["tasks"][task_id] = self.tasks[task_id]
                        self.save_state(cid)
                        break

    async def query_grok(self, convo_id, latest, history):
        if self.total_tokens >= TOKEN_LIMIT:
            return {"result": "Token limit reached", "target": "user", "response": "Paused: token limit reached"}
        if self.total_tokens >= TOKEN_LIMIT * WARNING_THRESHOLD:
            logger.warning("Token usage at 90% of limit")

        payload = {
            "role": "user",
            "content": json.dumps({
                "task": "process",
                "data": {"type": "u", "conversation_id": convo_id, "history": {"old": history[:-1], "latest": latest}}
            })
        }
        system_msg = {"role": "system", "content": GROK_PROMPT}
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
        agent_id = action.get("agent_id", self.agent_id)
        self.tasks[f"T{len(self.tasks) + 1}"] = {"t": task, "s": "r", "port": self.port}
        if agent_id == self.agent_id:
            asyncio.create_task(self.execute_task(task, action.get("d", ""), convo_id))
        else:
            db = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = db.cursor()
            cursor.execute("SELECT port FROM agents WHERE id = %s", (agent_id,))
            result = cursor.fetchone()
            port = result[0] if result else None
            cursor.close()
            db.close()
            if port:
                reader, writer = await asyncio.open_connection('localhost', port)
                msg = {"t": "user_request", "r": action.get("d", ""), "c": convo_id}
                writer.write(json.dumps(msg).encode())
                await writer.drain()
                writer.close()
                await writer.wait_closed()

    async def execute_task(self, task, data, convo_id):
        if task == "install":
            os.system(data)  # e.g., "pip install mysql-connector"
            self.tasks[f"T{len(self.tasks)}"]["s"] = "c"
            self.tasks[f"T{len(self.tasks)}"]["r"] = f"Installed {data}"
        else:  # Placeholder for other tasks
            await asyncio.sleep(2)
            self.tasks[f"T{len(self.tasks)}"]["s"] = "c"
            self.tasks[f"T{len(self.tasks)}"]["r"] = f"Task {task} completed with {data}"
        self.conversations[convo_id]["tasks"].update(self.tasks)
        self.save_state(convo_id)

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
                    for cid, convo in self.conversations.items():
                        if task_id in convo["tasks"]:
                            convo["tasks"][task_id] = self.tasks[task_id]
                            self.save_state(cid)
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
    agent = Agent(AGENT_ID, AGENT_PORT)
    asyncio.run(agent.start())
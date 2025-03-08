import asyncio
import json
import os
import psutil
import platform
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AGENT_ID = "agent-1"  # Example, should be unique per instance
AGENT_PORT = 5001     # Example, configurable per agent

class Agent:
    def __init__(self, agent_id, port):
        self.agent_id = agent_id
        self.port = port
        self.resources = {}
        self.tasks = {}
        self.loop = asyncio.get_event_loop()

    async def start(self):
        await self.register_agent(full=True)
        asyncio.create_task(self.update_system_stats())
        asyncio.create_task(self.update_load_stats())
        server = await asyncio.start_server(self.handle_client, 'localhost', self.port)
        logger.info(f"Agent {self.agent_id} started on localhost:{self.port}")
        async with server:
            await server.serve_forever()

    def get_resources(self, full=False):
        resources = {
            "cpu_cores": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total // (1024 * 1024),
            "disk_total": psutil.disk_usage('/').total // (1024 * 1024),
            "disk_free": psutil.disk_usage('/').free // (1024 * 1024),
            "os": platform.system()
        }
        if full or not resources.get("cpu_usage"):  # Include load metrics
            resources.update({
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_used": psutil.virtual_memory().percent,
                "disk_io": psutil.disk_io_counters().read_bytes // 1024 + psutil.disk_io_counters().write_bytes // 1024
            })
        self.resources = resources
        return resources

    async def register_agent(self, full=False):
        resources = self.get_resources(full)
        reader, writer = await asyncio.open_connection('localhost', 5005)
        msg = {"t": "n", "a": self.agent_id, "p": self.port, "r": resources, "q": []}
        writer.write(json.dumps(msg).encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

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
        task_id = msg.get("i")

        if msg_type == "p":  # poll
            if task_id in self.tasks:
                reply = {"t": "s", "i": task_id, "s": self.tasks[task_id]["s"], "r": self.tasks[task_id].get("r", "")}
                writer.write(json.dumps(reply).encode())
                await writer.drain()

        elif msg_type:  # any task
            self.tasks[task_id] = {"s": "r"}
            asyncio.create_task(self.execute_task(task_id, msg_type, msg.get("d", "")))

    async def execute_task(self, task_id, task, data):
        # Placeholder: Simulate task execution
        await asyncio.sleep(2)  # Simulate work
        self.tasks[task_id] = {"s": "c", "r": f"Task {task} completed with data: {data}"}

if __name__ == "__main__":
    agent = Agent(AGENT_ID, AGENT_PORT)
    asyncio.run(agent.start())
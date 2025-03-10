import asyncio
import json
import aiohttp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UI_PORT = 5000
AGENT_PORTS = [5001]  # Example, should be configurable or fetched from MySQL

class UI:
    def __init__(self):
        self.loop = asyncio.get_event_loop()

    async def start(self):
        server = await asyncio.start_server(self.handle_client, 'localhost', UI_PORT)
        logger.info(f"UI started on localhost:{UI_PORT}")
        async with server:
            await server.serve_forever()

    async def handle_client(self, reader, writer):
        data = await reader.read(1024)
        if not data:
            return
        msg = json.loads(data.decode())
        await self.process_message(msg, writer)

    async def process_message(self, msg, writer):
        msg_type = msg.get("t")
        agent_port = msg.get("agent_port", AGENT_PORTS[0])  # Default to first agent

        if msg_type == "user_request":
            convo_id = msg.get("c", f"C{int(asyncio.get_event_loop().time())}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://localhost:{agent_port}",
                    json={"t": "user_request", "r": msg["r"], "c": convo_id}
                ) as resp:
                    if resp.status == 200:
                        response = await resp.json()
                        writer.write(json.dumps(response).encode())
                        await writer.drain()

        elif msg_type == "list_conversations":
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://localhost:{agent_port}",
                    json={"t": "list_conversations"}
                ) as resp:
                    if resp.status == 200:
                        response = await resp.json()
                        writer.write(json.dumps(response).encode())
                        await writer.drain()

        elif msg_type == "get_conversation":
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://localhost:{agent_port}",
                    json={"t": "get_conversation", "c": msg["c"]}
                ) as resp:
                    if resp.status == 200:
                        response = await resp.json()
                        writer.write(json.dumps(response).encode())
                        await writer.drain()

if __name__ == "__main__":
    ui = UI()
    asyncio.run(ui.start())
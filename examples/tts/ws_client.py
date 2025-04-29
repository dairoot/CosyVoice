import asyncio
import websockets

async def connect():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            message = input("Enter message (or 'quit' to exit): ")
            if message == "quit":
                break
            await websocket.send(message)
            response = await websocket.recv()
            print(f"Server response: {response}")

if __name__ == "__main__":
    asyncio.run(connect())
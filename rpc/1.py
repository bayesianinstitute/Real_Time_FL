import torch
import syft as sy
import asyncio

# Set up the nodes (workers)
hook = sy.frameworks.torch.hook.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

async def receive_data(worker):
    while True:
        try:
            data = await worker.recv()
            print(f"Received: {data}")
        except (sy.SocketClosedError, sy.serde.deserialize.OpenTabsError):
            break

async def send_data(worker, target_worker):
    while True:
        message = input("Enter your message: ")
        if message.lower() == "exit":
            break
        target_worker.send(message)

async def handle_peer(worker, target_worker):
    await asyncio.gather(receive_data(worker), send_data(worker, target_worker))

async def main():
    peers = [(bob, alice), (alice, bob)]

    tasks = [handle_peer(peer[0], peer[1]) for peer in peers]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

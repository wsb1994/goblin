import os
from typing import Dict
from fastapi import FastAPI
import uvicorn
import pika
from dotenv import load_dotenv

def get_dotenv_map() -> Dict[str, str]:
    load_dotenv()
    return dict(os.environ)

def queue_listener_main():
    env = get_dotenv_map()
    amqp_url = env.get("AMQP_URL", "amqp://guest:guest@localhost:5672/")
    queue_name = env.get("AMQP_QUEUE", "test_queue")

    params = pika.URLParameters(amqp_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)

    def callback(ch, method, properties, body):
        print(f"Received message: {body.decode()}")

    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    print(f"Listening for messages on {queue_name}...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Queue listener stopped.")
    finally:
        connection.close()

app = FastAPI()

@app.get("/")
async def hello_world():
    return {"message": "Hello, world!"}

if __name__ == "__main__":
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    queue_listener_main()
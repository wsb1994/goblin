#!/usr/bin/env python
import pika, sys, os
import json
from typing import Dict
from dotenv import load_dotenv
from engine.engine import Engine

def get_dotenv_map() -> Dict[str, str]:
    load_dotenv()
    return dict(os.environ)

def main():
    env = get_dotenv_map()
    amqp_url = env.get("AMQP_URL", "amqp://admin:admin@localhost:5672/")
    queue_name = env.get("AMQP_QUEUE", "goblin")
    
    print(f"📡 Connecting to {amqp_url}, queue: {queue_name}")
    
    # Connect to RabbitMQ using URL parameters (not just host)
    params = pika.URLParameters(amqp_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    
    # Declare the queue
    channel.queue_declare(queue=queue_name, durable=True)
    print(f"✅ Successfully connected to RabbitMQ queue: {queue_name}")

    def callback(ch, method, properties, body):
        try:
            print(f"📨 Received message: {body}")
            
            # Parse the message
            message = json.loads(body.decode('utf-8'))
            plan_name = message.get('plan')
            
            if not plan_name:
                print("❌ Error: No plan found in message")
                return
            
            print(f"🎯 Processing plan: {plan_name}")
            
            # Initialize engine
            engine = Engine()
            
            # Use appropriate paths for local vs Docker
            if os.path.exists("/app"):
                # Docker environment
                scripts_path = "/app/scripts"
                plans_path = "/app/plans"
            else:
                # Local environment
                scripts_path = "/home/wsb/Documents/Binaries/goblin/goblin_backend/scripts"
                plans_path = "/home/wsb/Documents/Binaries/goblin/goblin_backend/plans"
            
            engine.auto_discover_scripts(scripts_path)
            
            # Load plan
            plan_file_path = f"{plans_path}/{plan_name}.toml"
            if not os.path.exists(plan_file_path):
                print(f"❌ Plan file not found: {plan_file_path}")
                return
            
            plan = engine.load_plan(plan_file_path)
            print(f"📋 Loaded plan: {plan.name}")
            
            # Execute plan
            input_data = message.get('input')
            default_input = json.dumps(input_data) if input_data else None
            
            results = engine.execute_plan(plan, default_input)
            print(f"✅ Plan execution completed. Results: {results}")
            
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            import traceback
            traceback.print_exc()

    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

    print(f' [*] Waiting for messages on queue "{queue_name}". To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

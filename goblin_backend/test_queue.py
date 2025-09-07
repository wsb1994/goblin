#!/usr/bin/env python3
import os
import json
import pika
from dotenv import load_dotenv
from engine.engine import Engine

def get_dotenv_map():
    load_dotenv()
    return dict(os.environ)

def test_queue_connection():
    """Test RabbitMQ connection and queue consumption"""
    try:
        env = get_dotenv_map()
        amqp_url = env.get("AMQP_URL", "amqp://guest:guest@localhost:5672/")
        queue_name = env.get("AMQP_QUEUE", "goblin")
        
        print(f"Attempting to connect to: {amqp_url}")
        print(f"Queue name: {queue_name}")
        
        params = pika.URLParameters(amqp_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        
        print("✓ Successfully connected to RabbitMQ")
        
        # Declare queue
        channel.queue_declare(queue=queue_name, durable=True)
        print(f"✓ Queue '{queue_name}' declared")
        
        # Check if there are any messages in the queue
        method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=False)
        if method_frame:
            print(f"✓ Found message in queue: {body}")
            channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)
        else:
            print("ℹ No messages currently in queue")
        
        def callback(ch, method, properties, body):
            print(f"📨 Received message: {body}")
            try:
                message = json.loads(body.decode('utf-8'))
                plan_name = message.get('plan')
                input_data = message.get('input')
                
                print(f"✓ Parsed plan: {plan_name}")
                print(f"✓ Parsed input: {input_data}")
                
                # Test engine initialization
                engine = Engine()
                print("✓ Engine initialized")
                
                # Test plan loading
                plan_file_path = f"/home/wsb/Documents/Binaries/goblin/goblin_backend/plans/{plan_name}.toml"
                if os.path.exists(plan_file_path):
                    plan = engine.load_plan(plan_file_path)
                    print(f"✓ Plan loaded: {plan.name}")
                else:
                    print(f"❌ Plan file not found: {plan_file_path}")
                
            except Exception as e:
                print(f"❌ Error processing message: {e}")
        
        channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
        print(f"🔄 Listening for messages on '{queue_name}'... Press Ctrl+C to stop")
        
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            print("\n⏹ Stopping consumer...")
            channel.stop_consuming()
        finally:
            connection.close()
            print("✓ Connection closed")
            
    except pika.exceptions.AMQPConnectionError as e:
        print(f"❌ Failed to connect to RabbitMQ: {e}")
        print("Make sure RabbitMQ is running and accessible")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_queue_connection()

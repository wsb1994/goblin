#!/usr/bin/env python
import pika, sys, os
import json
import threading
import time
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from engine.engine import Engine

def get_dotenv_map() -> Dict[str, str]:
    load_dotenv()
    return dict(os.environ)

def process_message(message_data):
    """Process a single message in a thread"""
    try:
        thread_id = threading.current_thread().ident
        print(f"🧵 Thread {thread_id}: Processing message: {message_data['body']}")
        
        # Parse the message
        message = json.loads(message_data['body'].decode('utf-8'))
        plan_name = message.get('plan')
        
        if not plan_name:
            print(f"❌ Thread {thread_id}: No plan found in message")
            return False
        
        print(f"🎯 Thread {thread_id}: Processing plan: {plan_name}")
        
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
            print(f"❌ Thread {thread_id}: Plan file not found: {plan_file_path}")
            return False
        
        plan = engine.load_plan(plan_file_path)
        print(f"📋 Thread {thread_id}: Loaded plan: {plan.name}")
        
        # Execute plan
        input_data = message.get('input')
        default_input = json.dumps(input_data) if input_data else None
        
        results = engine.execute_plan(plan, default_input)
        print(f"✅ Thread {thread_id}: Plan execution completed. Results: {results}")
        
        # Acknowledge the message
        message_data['channel'].basic_ack(delivery_tag=message_data['method'].delivery_tag)
        return True
        
    except Exception as e:
        thread_id = threading.current_thread().ident
        print(f"❌ Thread {thread_id}: Error processing message: {e}")
        import traceback
        traceback.print_exc()
        
        # Reject and requeue the message on error
        try:
            message_data['channel'].basic_nack(
                delivery_tag=message_data['method'].delivery_tag, 
                requeue=True
            )
        except:
            pass
        return False

def ensure_connection(connection, channel, params, queue_name, max_threads):
    """Ensure connection and channel are healthy, reconnect if needed"""
    try:
        # Check if connection is open
        if connection is None or connection.is_closed:
            print("🔄 Connection is closed, reconnecting...")
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=queue_name, durable=True)
            channel.basic_qos(prefetch_count=max_threads)
            print("✅ Reconnected successfully")
        
        # Check if channel is open
        elif channel is None or channel.is_closed:
            print("🔄 Channel is closed, recreating...")
            channel = connection.channel()
            channel.queue_declare(queue=queue_name, durable=True)
            channel.basic_qos(prefetch_count=max_threads)
            print("✅ Channel recreated successfully")
            
        return connection, channel
        
    except Exception as e:
        print(f"❌ Error ensuring connection: {e}")
        # Force reconnection
        try:
            if connection and not connection.is_closed:
                connection.close()
        except:
            pass
        
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)
        channel.basic_qos(prefetch_count=max_threads)
        print("✅ Force reconnected successfully")
        return connection, channel

def main():
    env = get_dotenv_map()
    amqp_url = env.get("AMQP_URL", "amqp://admin:admin@localhost:5672/")
    queue_name = env.get("AMQP_QUEUE", "goblin")
    max_threads = 8
    
    print(f"📡 Connecting to {amqp_url}, queue: {queue_name}")
    print(f"🧵 Using up to {max_threads} concurrent threads")
    
    # Connect to RabbitMQ using URL parameters (not just host)
    params = pika.URLParameters(amqp_url)
    connection = None
    channel = None
    
    try:
        connection, channel = ensure_connection(connection, channel, params, queue_name, max_threads)
        print(f"✅ Successfully connected to RabbitMQ queue: {queue_name}")

        while True:
            try:
                # Ensure connection is healthy before processing
                connection, channel = ensure_connection(connection, channel, params, queue_name, max_threads)
                
                # Collect up to max_threads messages
                messages = []
                print(f"📥 Collecting batch of up to {max_threads} messages...")
                
                for i in range(max_threads):
                    try:
                        method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=False)
                        if method_frame:
                            messages.append({
                                'method': method_frame,
                                'properties': header_frame,
                                'body': body,
                                'channel': channel
                            })
                        else:
                            # No more messages available
                            break
                    except (pika.exceptions.ConnectionWrongStateError, pika.exceptions.AMQPConnectionError) as e:
                        print(f"⚠️ Connection error while getting messages: {e}")
                        # Reconnect and try again
                        connection, channel = ensure_connection(None, None, params, queue_name, max_threads)
                        break
                
                if not messages:
                    print("⏳ No messages available, waiting...")
                    time.sleep(1)
                    continue
                
                print(f"🚀 Processing batch of {len(messages)} messages with {len(messages)} threads...")
                
                # Process messages concurrently
                with ThreadPoolExecutor(max_workers=len(messages)) as executor:
                    # Submit all messages for processing
                    future_to_message = {
                        executor.submit(process_message, msg): msg for msg in messages
                    }
                    
                    # Wait for all threads to complete
                    completed = 0
                    failed = 0
                    for future in as_completed(future_to_message):
                        try:
                            result = future.result()
                            if result:
                                completed += 1
                            else:
                                failed += 1
                        except Exception as e:
                            failed += 1
                            print(f"❌ Thread execution failed: {e}")
                    
                    print(f"📊 Batch completed: {completed} successful, {failed} failed")
                    
            except (pika.exceptions.ConnectionWrongStateError, pika.exceptions.AMQPConnectionError) as e:
                print(f"⚠️ Connection error in main loop: {e}")
                print("🔄 Attempting to reconnect...")
                connection, channel = ensure_connection(None, None, params, queue_name, max_threads)
                time.sleep(2)  # Brief pause before retrying
                continue
            except Exception as e:
                print(f"❌ Unexpected error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)  # Longer pause for unexpected errors
                continue
                
    except KeyboardInterrupt:
        print("\n⏹ Shutting down gracefully...")
    finally:
        try:
            if connection and not connection.is_closed:
                connection.close()
                print("✅ Connection closed")
        except Exception as e:
            print(f"⚠️ Error closing connection: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

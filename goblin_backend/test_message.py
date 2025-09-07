#!/usr/bin/env python3
"""
Test script to send a message to RabbitMQ queue to verify the callback function works
"""
import json
import pika
from dotenv import load_dotenv
import os

def send_test_message():
    load_dotenv()
    amqp_url = os.getenv("AMQP_URL", "amqp://admin:admin@localhost:5672/")
    queue_name = os.getenv("AMQP_QUEUE", "goblin")
    
    # Test message with the expected format
    test_message = {
        "plan": "AB",
        "input": {
            "id": "1",
            "timestamp": "123141",
            "comment": "racism is good I hate black people"
        }
    }
    
    try:
        print(f"Connecting to RabbitMQ at {amqp_url}")
        params = pika.URLParameters(amqp_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        
        # Declare queue (make sure it exists)
        channel.queue_declare(queue=queue_name, durable=True)
        
        # Send message
        message_body = json.dumps(test_message)
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message_body,
            properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
        )
        
        print(f"✓ Sent test message to queue '{queue_name}':")
        print(f"  Message: {message_body}")
        
        connection.close()
        print("✓ Connection closed")
        
    except Exception as e:
        print(f"❌ Error sending message: {e}")

if __name__ == "__main__":
    send_test_message()

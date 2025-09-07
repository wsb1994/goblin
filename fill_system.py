#!/usr/bin/env python
import pika
import json
import csv
import sys
import os

def main():
    # RabbitMQ connection parameters
    amqp_url = "amqp://admin:admin@localhost:5672/"
    queue_name = "goblin"
    
    # Connect to RabbitMQ
    try:
        params = pika.URLParameters(amqp_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        
        # Declare the queue
        channel.queue_declare(queue=queue_name, durable=True)
        
        print(f" [*] Connected to RabbitMQ at {amqp_url}")
        print(f" [*] Using queue: {queue_name}")
        
    except Exception as e:
        print(f" [!] Failed to connect to RabbitMQ: {e}")
        sys.exit(1)
    
    # Read CSV and publish messages
    try:
        with open('../English_test.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            message_count = 0
            
            for row in reader:
                # Create message in the specified format
                message = {
                    "plan": "AB",
                    "input": {
                        "id": str(row.get('', '')),  # First column (index)
                        "comment": row.get('text', ''),
                        "label": str(row.get('label', ''))
                    }
                }
                
                # Publish message to RabbitMQ
                try:
                    channel.basic_publish(
                        exchange='',
                        routing_key=queue_name,
                        body=json.dumps(message)
                    )
                    message_count += 1
                    
                    if message_count % 100 == 0:
                        print(f" [x] Sent {message_count} messages...")
                        
                except Exception as e:
                    print(f" [!] Failed to publish message: {e}")
                    continue
            
            print(f" [x] Successfully sent {message_count} messages to queue '{queue_name}'")
            
    except FileNotFoundError:
        print(" [!] English_test.csv file not found")
        sys.exit(1)
    except Exception as e:
        print(f" [!] Error reading CSV file: {e}")
        sys.exit(1)
    
    finally:
        # Close connection
        try:
            connection.close()
            print(" [*] Connection closed")
        except:
            pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n [*] Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

import pika
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RabbitMQClient:
    def __init__(self, host='localhost', port=5672, username='guest', password='guest'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection = None
        self.channel = None
    
    def connect(self):
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            logger.info("Connected to RabbitMQ successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    def declare_queue(self, queue_name, durable=True):
        try:
            if not self.channel:
                self.connect()
            self.channel.queue_declare(queue=queue_name, durable=durable)
            logger.info(f"Queue '{queue_name}' declared successfully")
        except Exception as e:
            logger.error(f"Failed to declare queue {queue_name}: {e}")
            raise
    
    def publish_message(self, queue_name, message):
        try:
            if not self.channel:
                self.connect()
            
            # Ensure queue exists
            self.declare_queue(queue_name)
            
            # Convert message to JSON string
            message_body = json.dumps(message)
            
            self.channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=message_body,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                )
            )
            logger.info(f"Message published to queue '{queue_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to publish message to {queue_name}: {e}")
            return False
    
    def consume_messages(self, queue_name, callback):
        try:
            if not self.channel:
                self.connect()
            
            # Ensure queue exists
            self.declare_queue(queue_name)
            
            # Set up consumer
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=callback,
                auto_ack=True
            )
            
            logger.info(f"Starting to consume messages from '{queue_name}'")
            self.channel.start_consuming()
        except Exception as e:
            logger.error(f"Failed to consume messages from {queue_name}: {e}")
            raise
    
    def close(self):
        try:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
                logger.info("RabbitMQ connection closed")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}")
    
    def __del__(self):
        self.close()
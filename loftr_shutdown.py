import socket
import sys

port = sys.argv[1]
try:
    port = int(port)  # Convert to integer
    print(f"The argument as integer: {port}")
except ValueError:
    port = 59434
    print(f"Error: '{port}' is not a valid integer")

def image_processing_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    server_address = ('localhost', port)
    print(f"Connecting to {server_address[0]} on port {server_address[1]}")
    client_socket.connect(server_address)
    
    try:
        message = "shutdown"
        print(f"Sending message: {message}")
        client_socket.sendall(message.encode('utf-8'))  # Encode the message to bytes
        
        response = client_socket.recv(1024)
        print(f"Received response: {response.decode('utf-8')}")  # Decode the bytes back to a string
        
    finally:
        # Close the connection
        print("Closing connection")
        client_socket.close()

# Run the client
image_processing_client()
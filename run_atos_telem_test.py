import socket
import json

def run_emulator(ip="127.0.0.1", port=5005):
    print(f"[EMULATOR] Initializing ATOS Digital Twin Receiver on {ip}:{port}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    
    print("[EMULATOR] Waiting for City Heartbeat...")
    
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8')
            
            try:
                payload = json.loads(message)
                if payload['type'] == 'city_pulse':
                    print(f"[TWIN] Pressure: {payload['pressure']:.2f} | Signal Phase: {payload['signal_phase']}")
                elif payload['type'] == 'incident_alert':
                    print(f"[ALERT] Incident Type: {payload['category']} at Node {payload['node_id']}")
            except json.JSONDecodeError:
                print(f"[WARN] Received malformed packet: {message}")
                
    except KeyboardInterrupt:
        print("\n[EMULATOR] Shutting down.")
    finally:
        sock.close()

if __name__ == "__main__":
    run_emulator()

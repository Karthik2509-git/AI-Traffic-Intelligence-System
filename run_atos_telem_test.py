import socket
import json

def run_receiver(ip="127.0.0.1", port=5005):
    """Receives and displays UDP telemetry from the ATOS C++ engine."""
    print(f"[ATOS Receiver] Listening on {ip}:{port}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8')

            try:
                p = json.loads(message)
                if p['type'] == 'city_pulse':
                    vehicles = p.get('vehicles', '?')
                    print(f"  Vehicles: {vehicles}  |  Pressure: {p['pressure']:.1f}  |  Phase: {p['signal_phase']}")
                elif p['type'] == 'incident_alert':
                    print(f"  ALERT: {p['category']} at node {p['node_id']}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [WARN] Bad packet: {message} ({e})")

    except KeyboardInterrupt:
        print("\n[ATOS Receiver] Stopped.")
    finally:
        sock.close()

if __name__ == "__main__":
    run_receiver()

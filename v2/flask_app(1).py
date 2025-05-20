from flask import Flask, request, jsonify
from flask_compress import Compress
import time
from threading import Lock

app = Flask(__name__)
Compress(app)  # Enable response compression

# In-memory storage for player states
players = {}  # {player_id: {'state': {...}, 'name': str, 'color': [r,g,b], 'last_updated': timestamp}}
lock = Lock()
TIMEOUT = 5.0  # Remove players inactive for 5 seconds

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        if not data or 'player_id' not in data or 'state' not in data or 'name' not in data or 'color' not in data:
            return jsonify({'error': 'Invalid data'}), 400

        player_id = data['player_id']
        state = data['state']
        name = data['name']
        color = data['color']

        with lock:
            # Update player state
            players[player_id] = {
                'state': {
                    'x': state['x'],
                    'y': state['y'],
                    'angle': state['angle'],
                    'speed': state['speed'],
                    'steering_angle': state['steering_angle'],
                    'velocity_x': state['velocity_x'],
                    'velocity_y': state['velocity_y'],
                    'angular_velocity': state['angular_velocity'],
                    'checkpoints_passed': state['checkpoints_passed'],
                    'health': state['health'],
                    'nitro': state['nitro']  # Include nitro in state
                },
                'name': name,
                'color': color,
                'last_updated': time.time()
            }

            # Clean up inactive players
            current_time = time.time()
            inactive_players = [pid for pid, pinfo in players.items() if current_time - pinfo['last_updated'] > TIMEOUT]
            for pid in inactive_players:
                del players[pid]

            # Prepare response with minimal data
            other_players = {
                pid: {
                    'state': pinfo['state'],
                    'name': pinfo['name'],
                    'color': pinfo['color']
                }
                for pid, pinfo in players.items()
                if pid != player_id
            }

        return jsonify(other_players), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
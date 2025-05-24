# Гонки 2D multiplayer

## Установка на школьном компе:
1. Откройте терминал - `CTRL + ALT + T`
2. Пропишите сдедующую команду
``` bash
curl -sSL https://raw.githubusercontent.com/mitrichevgeorge/kart/main/install.sh | bash
```
> [!Note]
> Если что в терминале вставить это `CTRL + SHIFT + V`
Далее можно запустить прописав короткую команду `gkart` или ярлыком на рабочем столе(Появится после установки)

## Свойтсва игры:
### Главный экран
![image](https://github.com/user-attachments/assets/aa6245a8-af7f-4dcd-bd3c-b48dda1294b5)
Мониторинг всех серверов можео посмотреть [тут](gkart.pythonanywhere.com)
### Управление
1. `Стрелочки` или `WASD` - движение и повороты
2. `Shift` - Принудительный дрифт(мало ощущается)
3. `Ctrl` - Разгон и видны синие партиклы за машинкой
   ![image](https://github.com/user-attachments/assets/914ef07f-85af-46f5-82fd-ff04f7f543f9)

### Цвета карты
1. <img src="https://github.com/user-attachments/assets/9e9c591d-1f9c-4c67-9645-f59558e787e8" style="height: 1em; width: 1em; vertical-align: middle;" /> (0,0,0) - [Пол](#пол)
2. <img src="https://github.com/user-attachments/assets/e8c7606f-ba60-48e7-900e-fe6c63467862" style="height: 1em; width: 1em; vertical-align: middle;" /> (200,200,200) - [Стена](#стена)
3. <img src="https://github.com/user-attachments/assets/4e452f41-556b-4e98-84f9-e61864ee77db" style="height: 1em; width: 1em; vertical-align: middle;" /> (50,200,0) - [Спавн](#спавн)
4. <img src="https://github.com/user-attachments/assets/59b500a1-1bdb-4b86-8fe6-6c186491250b" style="height: 1em; width: 1em; vertical-align: middle;" /> (180,180,0) - [Песок](#песок)
5. <img src="https://github.com/user-attachments/assets/e47349ef-5270-44bf-9ca7-13bb6efe12bc" style="height: 1em; width: 1em; vertical-align: middle;" /> (0,200,50) - [Стена с повышенным отскоком](#пружинная-стена)

#### Пол
Базовая поверхноость. По нему может спокойно кататься машинка и иметь иннерцию
![3f4d884ab482428fb8cbe87d8ce81b22](https://github.com/user-attachments/assets/a3516f34-e46d-46fa-b4bb-d9f94b2e554c)
| | |
|---|---|
| Пройти насквозь | можно |
| Урон | нет |
| Отскок | нет |

#### Стена
Сквозь неё невозможно пройти. Создаёт небольшой отскок
![c2b8dc8391244141a5a130fdac0a0e39](https://github.com/user-attachments/assets/d667d5cf-eeb7-40de-b41e-3e626607e056)
| | |
|---|---|
| Пройти насквозь | невозможно |
| Урон | есть |
| Отскок | небольшой |

#### Спавн
На физику не влияет, по сути как [пол](#пол). Указывает системе, где должна появиться машинка

#### Песок
Замедляет машинку и её иннерцию, по нему можно ездить.
![20d2eda006b54ac79b961103f586cfe8](https://github.com/user-attachments/assets/2383e060-435f-4d48-8eb5-9486eea228bd)
| | |
|---|---|
| Пройти насквозь | можно |
| Урон | нет |
| Отскок | нет |

#### Пружинная стена
Как и у обычной стены, от неё получаешь урон и невозможно пройти, но отскок больше и урон меньше
![b81078e6c6c34f4abde96bf97ffd0f12](https://github.com/user-attachments/assets/5528daa3-b728-41b9-a175-fe3f6f64d39e)
| | |
|---|---|
| Пройти насквозь | невозможно |
| Урон | небольшой |
| Отскок | есть |


## Добавление собственной карты(сервера)
1. зарегистрируйтесь в <img src="https://www.pythonanywhere.com/static/anywhere/images/PA-logo.svg" style="height: 1em; vertical-align: middle;" />[python any where](https://www.pythonanywhere.com/login)
   ![image](https://github.com/user-attachments/assets/f15ca967-9582-4144-9035-4fc9b50840c4)
   ![image](https://github.com/user-attachments/assets/18415712-c6f0-4166-a521-62e209b38495)
   ![image](https://github.com/user-attachments/assets/4e678c99-1659-43c2-a1b4-4304a457ef8c)
   ![image](https://github.com/user-attachments/assets/b6f479c8-3f57-4082-b4b0-286b73081187)
   переходим в почту
   ![image](https://github.com/user-attachments/assets/a3eac0d7-55d7-409b-9692-2eee3bdd2d08)
   ![image](https://github.com/user-attachments/assets/38602484-b65b-4dff-a7db-d7922b7131bb)


3. w
4. Впишите код:
   <details>
      <summary>тут код сервера (нажмите чтобы развернуть)</summary>

      ``` python
      BASE_PATH = '/home/YOURNAME/info'
      from flask import Flask, request, jsonify, send_file
      from flask_compress import Compress
      from flask_cors import CORS
      import time
      from threading import Lock
      import os
      
      app = Flask(__name__)
      Compress(app)
      CORS(app, resources={r"/*": {"origins": "*"}})
      
      players = {}
      lock = Lock()
      TIMEOUT = 5.0
      
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
                          'nitro': state['nitro']
                      },
                      'name': name,
                      'color': color,
                      'last_updated': time.time()
                  }
      
                  current_time = time.time()
                  inactive_players = [pid for pid, pinfo in players.items() if current_time - pinfo['last_updated'] > TIMEOUT]
                  for pid in inactive_players:
                      del players[pid]
      
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
      
      @app.route('/map', methods=['GET'])
      def get_map():
          map_path = os.path.join(BASE_PATH, 'map.png')
          if os.path.exists(map_path):
              return send_file(map_path, mimetype='image/png')
          else:
              return jsonify({'error': 'Map file not found'}), 404
      
      @app.route('/info', methods=['GET'])
      def get_info():
          info_path = os.path.join(BASE_PATH, 'info.json')
          if os.path.exists(info_path):
              return send_file(info_path, mimetype='application/json')
          else:
              return jsonify({'error': 'Info file not found'}), 404
      
      @app.route('/online', methods=['GET'])
      def get_online():
          try:
              with lock:
                  current_time = time.time()
                  active_players = len([pid for pid, pinfo in players.items() if current_time - pinfo['last_updated'] <= TIMEOUT])
              return jsonify({'online': active_players}), 200
          except Exception as e:
              return jsonify({'error': str(e)}), 500
      
      if __name__ == '__main__':
          app.run(debug=False)
      ```
      Тут вам надо в первой строчке заменить `YOURNAME` на ваш юзернейм, который вы указали при регистрации
</details>


 



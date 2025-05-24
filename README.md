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

## Для Windows
Для винды есть просто [готовая сборка](https://github.com/MitrichevGeorge/kart/blob/main/v2/dist/kart2.exe)

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

3. создайте карту
   Просто в `paint`. Про цвета было расписано [здесь](#цвета-карты) Например так:
   ![image](https://github.com/user-attachments/assets/ad621a33-20c9-432a-a02f-6454b62ae1ce)
   затем сохраните её в `.png` файл

5. загрузите файлы
   ![image](https://github.com/user-attachments/assets/c43d3bb1-40d0-49a5-88da-a90d03f9e0dc)
   создайте папку `info`
   ![image](https://github.com/user-attachments/assets/4a3c5049-d086-49c1-987a-c5f1b938f0d0)
   создайте в ней файл `info.json`
   ![image](https://github.com/user-attachments/assets/0a105afb-7f05-4e00-acdb-043ec9d178d5)
   напишите в нём следущий текст:
   <details>
      <summary>info.json (нажмите чтобы развернуть)</summary>
      
      ``` json
      {
          "ACCELERATION": 0.3,
          "DECELERATION": 0.04,
          "MAX_SPEED": 10,
          "TURN_ACCELERATION": 0.01,
          "ROTATIONAL_FRICTION": 0.04,
          "MAX_ANGULAR_VELOCITY": 0.40,
          "SAND_SLOWDOWN": 0.9,
          "SAND_INERTIA_LOSS": 0.08,
          "WALL_BOUNCE": 0.3,
          "FRICTION": 0.3,
          "TRAIL_FADE_RATE": 0.99,
          "MIN_SPEED_FOR_TURN": 0.5,
          "LOW_SPEED_TURN_FACTOR": 0.3,
          "HIGH_SPEED_DRIFT_FACTOR": 0.3,
          "DRIFT_FACTOR_ON_SHIFT": 0.8,
          "CAR_COLLISION_BOUNCE": 0.5,
          "MIN_SPAWN_DISTANCE": 30,
          "BLEND_FACTOR": 0.5,
          "MAX_HEALTH": 20,
          "DAMAGE_SCALING": 0.5,
          "SPAWN_PROTECTION_TIME": 2.0,
          "HEALTH_BAR_WIDTH": 40,
          "HEALTH_BAR_HEIGHT": 6,
          "HEALTH_BAR_OFFSET": 0,
          "NITRO_BAR_OFFSET": 8,
          "NAME_OFFSET": 30,
          "SMOKE_HEALTH_THRESHOLD": 9,
          "SMOKE_EMISSION_RATE": 0.1,
          "SMOKE_LIFETIME": 1.0,
          "SMOKE_SPEED": 10,
          "POPUP_LIFETIME": 1.0,
          "POPUP_SPEED": 20,
          "EXPLOSION_LIFETIME": 0.5,
          "EXPLOSION_SIZE": 40,
          "CORPSE_LIFETIME": 3.0,
          "SPARK_EMISSION_RATE": 0.1,
          "SPARK_LIFETIME": 0.3,
          "SPARK_SPEED": 15,
          "SPARK_ALPHA_THRESHOLD": 50,
          "NITRO_MAX": 100,
          "NITRO_REGEN_RATE": 10,
          "NITRO_CONSUMPTION_RATE": 50,
          "NITRO_BOOST_FACTOR": 3.0,
          "NITRO_LOW_THRESHOLD": 10,
          "NITRO_LOW_SLOWDOWN": 0.7,
          "NITRO_LOW_DAMAGE": 0.5,
          "NITRO_FLAME_EMISSION_RATE": 0.05,
          "NITRO_VISIBILITY_THRESHOLD": 0.95
      }
      ```
   </details>
   
   Далее вы можете изменить физические константы. А затем нажмите `сохранить`
   
   ![image](https://github.com/user-attachments/assets/9c1e3c2c-20ae-43a2-95bd-99aa894d2fa6)
   Вернитесь в папку `info` и загрузите карту
   ![image](https://github.com/user-attachments/assets/ab46019e-45b6-48f3-a02f-418ec39b1ee1)
   ![image](https://github.com/user-attachments/assets/04222ffe-f75e-4405-9d48-9e218b865425)
   Затем выберите на своём компе карту ранее созданную. Она должна называться `map.png`. Должно быть так:
   ![image](https://github.com/user-attachments/assets/0af3dd84-2166-4e01-8ca1-a9c46bf8a4b3)
6. Перейдите в раздел `web` и создайте новое приложение
   ![image](https://github.com/user-attachments/assets/4f272201-643f-493c-8746-838fc3d63d5a)
   ![image](https://github.com/user-attachments/assets/fd01724d-fb0a-4a86-8ef3-22a7377a0b2c)
   ![image](https://github.com/user-attachments/assets/a8c5e3a1-4e24-4aa9-bdb2-8a6e3c5ec375)
   ![image](https://github.com/user-attachments/assets/8dc4b05b-b2d6-4ede-8cac-115370aacedc)
   ![image](https://github.com/user-attachments/assets/c5ccd345-1d50-441c-9479-0aba89ee5318)
   ![image](https://github.com/user-attachments/assets/a6486494-d419-4aec-9111-ab82fd52a405)

7. Напишите код сервера
   ![image](https://github.com/user-attachments/assets/cb8b09fe-fb9f-4dd3-a2e6-d41314b5bba5)
   ![image](https://github.com/user-attachments/assets/7bb9ae34-9b35-4aa6-bfdb-ed5710d1e53c)
   ![image](https://github.com/user-attachments/assets/37c84b4e-7701-4cf7-b2a6-944419fb27de)
   Выделите всё(`ctrl+A`) и впишите код:
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
   И сохраняем файл
   
   ![image](https://github.com/user-attachments/assets/d1938b18-86f5-4e8a-87d4-f7d882f346f9)

8. Переходим на главную страницу
   
   ![image](https://github.com/user-attachments/assets/b6eb0e02-cb60-4239-85fe-60d2956971c7)
   
   Открываем консоль хостинга
   
   ![image](https://github.com/user-attachments/assets/8bfa64b6-9e63-477c-b9fb-fb1aeeb8ff24)
   
   Устанавливаем `flask_cors`
   
   ![image](https://github.com/user-attachments/assets/37c862cf-e91c-43f1-b8ba-07f9bc9db036)
   
   Возвращаемся на главную

   ![image](https://github.com/user-attachments/assets/77e85efb-b4dd-4dd1-924a-f51c45504256)

   Перезагружаем ранее созданное веб-приложение
 
   ![image](https://github.com/user-attachments/assets/d4b32a2d-3f02-4acf-ac8c-7a23628b3bfc)
   ![image](https://github.com/user-attachments/assets/3b48888c-41b1-4a9a-bb7a-1975993ba277)

9. Проверить
   ![image](https://github.com/user-attachments/assets/d90295b8-43b2-426b-97ed-b57b080ab10e)
   ![image](https://github.com/user-attachments/assets/cbca803b-3675-4d9f-aca8-9ca9e044ebff)

10. Желательно добавить на [страницу мониторинга серверов](https://gkart.pythonanywhere.com/)
    ![image](https://github.com/user-attachments/assets/6f11f400-a2a0-439e-86fb-d8cf4322bb97)
    Сдесь можно удобно просматривать онлайн и пинг доступных серверов
    ![image](https://github.com/user-attachments/assets/8f435078-5a59-4de3-874e-66e39c2dfa18)

   

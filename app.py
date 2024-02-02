import os
import time

import cv2

import draw

from flask_socketio import SocketIO, emit

from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory, Response

app = Flask(__name__, template_folder='template', static_folder='static')
socketio = SocketIO(app)

config = {}

real_time = True
draw_model = None


@app.route("/", methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        source = request.files['source']
        deck_list = request.files['deck_list']

        global real_time
        real_time = len(request.form.getlist('real_time')) == 1

        if deck_list:
            if not os.path.exists('./temp'):
                os.makedirs('./temp')

            target_src = '0'
            if source:
                print(request.files)
                video_format = request.files['source'].filename.split('.')[-1]
                target_src = os.path.join('temp', 'source.' + video_format)
                source.save(target_src)

            target_dl = os.path.join('temp', 'deck_list.ydk')

            deck_list.save(target_dl)

            config['source'] = target_src
            config['deck_list'] = target_dl

            return redirect(url_for('main'))
    return render_template("index.html")


@app.route('/main', methods=('GET', 'POST'))
def main():
    if real_time:
        global draw_model
        draw_model = draw.Draw(source=config['source'], deck_list=config['deck_list'])
        return render_template('real_time.html')
    else:
        return render_template('main.html')


@app.route('/image/<path:path>')
def display_image(path):
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)
    return send_from_directory(dirname, basename)


@app.route('/video_feed')
def video_feed():
    return Response(real_time_display(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('start_processing')
def start_processing():
    global draw_model
    draw_model = draw.Draw(source=config['source'], deck_list=config['deck_list'])
    config['data_path'] = draw_model.configs['data_path']
    cards_display(draw_model)


def real_time_display():
    if draw_model:
        for result in draw_model.results:
            image = draw_model.process(result, display=True)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def cards_display(draw_model):
    count = {}
    displayed = {}
    emited = False
    if config['source'] == '0':
        for result in draw_model.results:
            outputs = draw_model.process(result)
            for output in outputs:
                if output not in count.keys():
                    count[output] = 0
                count[output] = count[output] + 1
            for key, value in count.items():
                if value > 60:
                    if key not in displayed.keys():
                        displayed[key] = 6
                        path = os.path.join(draw_model.configs['data_path'], key[1], key[0])
                        filename = os.listdir(path)[0]
                        emit('image_display', os.path.join('image', path, filename))
                        emited = True
                        break
            if emited:
                time.sleep(10)
                emited = False
                del_key = []
                for key in displayed.keys():
                    displayed[key] -= 1
                    if displayed[key] == 0:
                        del_key.append(key)
                for key in del_key:
                    del displayed[key]
                    count[key] = 0
    else:
        fast_forwarding = 0
        frame = -1
        print("fast forwarding")
        for _ in draw_model.results:
            frame += 1
            if frame == 2500:
                break
        print("starting inference")
        for result in draw_model.results:
            if emited:
                fast_forwarding += 1
                if fast_forwarding == 600:
                    emited = False
                    del_key = []
                    for key in displayed.keys():
                        displayed[key] -= 1
                        if displayed[key] == 0:
                            del_key.append(key)
                    for key in del_key:
                        del displayed[key]
                    fast_forwarding = 0
            else:
                outputs = draw_model.process(result)
                for output in outputs:
                    if output not in count.keys():
                        count[output] = 0
                    count[output] = count[output] + 1
                for key, value in count.items():
                    if value >= 30:
                        if key not in displayed.keys():
                            displayed[key] = 6
                            count[key] = 0
                            path = os.path.join(draw_model.configs['data_path'], key[1], key[0])
                            filename = os.listdir(path)[0]
                            emit('image_display', os.path.join('image', path, filename))
                            emited = True
                            break

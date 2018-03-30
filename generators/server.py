import flask
from flask import app, request
from flask_script import Manager


_in_memory_storage = []


manager = Manager(app)


@app.route('/')
def user_info():
    pass



if __name__ == '__main__':
    manager.run()

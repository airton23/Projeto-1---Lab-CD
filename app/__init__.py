"""Aplicação Flask; o motor analítico permanece independente do servidor."""
from flask import Flask

from app.context import load_project
from app.formatting import fmt
from config import DATA_PATH


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_mapping(DATA_PATH=DATA_PATH, MAX_CONTENT_LENGTH=1024 * 1024)
    if test_config:
        app.config.update(test_config)
    app.extensions["project"] = load_project(app.config["DATA_PATH"])
    app.jinja_env.filters["fmt"] = fmt
    from app.routes import report
    app.register_blueprint(report)
    return app


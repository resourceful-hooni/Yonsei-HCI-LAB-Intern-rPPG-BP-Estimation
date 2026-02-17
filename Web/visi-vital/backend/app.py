import os

from flask import Flask, jsonify
from flask_cors import CORS

from config import Config
from database.db_manager import DatabaseManager
from routes.lifestyle import lifestyle_bp
from routes.measurement import measurement_bp
from routes.summary import summary_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    app.config["MAX_CONTENT_LENGTH"] = app.config.get("MAX_CONTENT_LENGTH", 50 * 1024 * 1024)

    CORS(app, resources={r"/api/*": {"origins": app.config.get("CORS_ORIGINS", "*")}})

    db_path = app.config["DATABASE_PATH"]
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), db_path))

    schema_path = os.path.join(os.path.dirname(__file__), "database", "schema.sql")
    app.db = DatabaseManager(db_path)
    app.db.init_database(schema_path)

    app.register_blueprint(measurement_bp)
    app.register_blueprint(summary_bp)
    app.register_blueprint(lifestyle_bp)

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"success": True, "status": "ok"})

    @app.errorhandler(400)
    def bad_request(err):
        return jsonify({"success": False, "error": str(err)}), 400

    @app.errorhandler(404)
    def not_found(err):
        return jsonify({"success": False, "error": "Not Found"}), 404

    @app.errorhandler(500)
    def internal_error(err):
        return jsonify({"success": False, "error": "Internal Server Error"}), 500

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(
        host=application.config["FLASK_HOST"],
        port=application.config["FLASK_PORT"],
        debug=application.config["FLASK_DEBUG"],
    )

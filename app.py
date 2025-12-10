from flask import Flask, render_template
from flask_socketio import SocketIO
from service.training import process_data_and_train
from utils.utils import logger

app = Flask(__name__)

# Creates flask app instance
socketio = SocketIO(
    app,
    cors_allowed_origins=["http://127.0.0.1:5000", "http://localhost:5000"],
    logger=True,
    engineio_logger=True,
    async_mode='threading'  # keep this for simplicity for doing background tasks
)

#Shows progress bar
def emit_progress(percentage, message):
    socketio.emit("progress", {"percentage": percentage, "message": message})

#Updates chart with metrics
def emit_metrics(model, accuracy, precision, recall, f1_score):
    socketio.emit("update_metrics", {
        "model": model,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score)
    })

#Shows final comparison
def emit_results(model_metrics):
    logger.info("Emitting final_results: %s", model_metrics)
    socketio.emit("final_results", model_metrics)

#it is triggered when clinet connects to SocketIO server
@socketio.on('connect')
def handle_connect():
    logger.info("Client connected!")
    socketio.emit('connect_ack', {'message': 'Connected to server'})

#it is triggered when start analysis is pressed and starts background process
@socketio.on("start_analysis")
def analyze_kaggle(data):
    try:
        dataset_name = data.get("dataset_name")
        retrain = data.get("retrain", False)
        logger.info(f"Starting analysis for dataset: {dataset_name}, retrain: {retrain}")

        # Run analysis in background so SocketIO can emit updates
        socketio.start_background_task(
            process_data_and_train,
            dataset_name,
            retrain,
            emit_progress,
            emit_metrics,
            emit_results
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        socketio.emit("error", {"message": str(e)})



@app.route('/')
def index():
    return render_template('index.html')

#Launches the Flask and SocketIO server
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)

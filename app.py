from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os

app = Flask(__name__)
CORS(app) 

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "Backend is working!"})


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import semantic_search
from roadmap import build_roadmap
from fishbone_roadmap import build_fishbone_roadmap
import traceback
import json
import uuid
import os
from datetime import datetime

# Storage for shared roadmaps (in production, use a database)
SHARED_ROADMAPS_FILE = "shared_roadmaps.json"

def load_shared_roadmaps():
    """Load shared roadmaps from file"""
    if os.path.exists(SHARED_ROADMAPS_FILE):
        try:
            with open(SHARED_ROADMAPS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_shared_roadmaps(roadmaps):
    """Save shared roadmaps to file"""
    try:
        with open(SHARED_ROADMAPS_FILE, 'w', encoding='utf-8') as f:
            json.dump(roadmaps, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving shared roadmaps: {e}")

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/api/health")
    def health() -> tuple:
        return jsonify({"status": "ok"}), 200

    @app.get("/api/search")
    def search_endpoint() -> tuple:
        try:
            query = request.args.get("q", type=str)
            top_k = request.args.get("k", type=int, default=10)
            if not query:
                return jsonify({"error": "Missing required query param 'q'"}), 400

            results = semantic_search(query=query, top_k=top_k)
            return jsonify({"query": query, "results": results}), 200
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.get("/api/roadmap")
    def roadmap_endpoint() -> tuple:
        try:
            query = request.args.get("q", type=str)
            top_k = request.args.get("k", type=int, default=25)
            max_steps = request.args.get("steps", type=int, default=5)
            if not query:
                return jsonify({"error": "Missing required query param 'q'"}), 400

            results = semantic_search(query=query, top_k=top_k)
            roadmap = build_roadmap(query=query, results=results, max_steps=max_steps)
            return jsonify({"query": query, "steps": roadmap}), 200
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.get("/api/fishbone")
    def fishbone_endpoint() -> tuple:
        try:
            query = request.args.get("q", type=str)
            top_k = request.args.get("k", type=int, default=25)
            if not query:
                return jsonify({"error": "Missing required query param 'q'"}), 400

            results = semantic_search(query=query, top_k=top_k)
            fishbone = build_fishbone_roadmap(query=query, results=results)
            return jsonify(fishbone), 200
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.post("/api/share/roadmap")
    def share_roadmap() -> tuple:
        """Save a roadmap and return a shareable link"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            # Generate unique ID
            share_id = str(uuid.uuid4())[:8]
            
            # Load existing roadmaps
            roadmaps = load_shared_roadmaps()
            
            # Save roadmap data
            roadmaps[share_id] = {
                "id": share_id,
                "data": data,
                "created_at": datetime.now().isoformat(),
                "query": data.get("query", "")
            }
            
            # Save to file
            save_shared_roadmaps(roadmaps)
            
            return jsonify({
                "share_id": share_id,
                "share_url": f"/share/{share_id}",
                "message": "Roadmap saved successfully"
            }), 200
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.get("/api/share/<share_id>")
    def get_shared_roadmap(share_id: str) -> tuple:
        """Retrieve a shared roadmap by ID"""
        try:
            roadmaps = load_shared_roadmaps()
            
            if share_id not in roadmaps:
                return jsonify({"error": "Roadmap not found"}), 404
            
            roadmap_data = roadmaps[share_id]["data"]
            return jsonify(roadmap_data), 200
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

"""
FastTrack - Patient Intake Backend
====================================
Run:  python app.py
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, date
from dotenv import load_dotenv
import json
import uuid
import os
import anthropic

load_dotenv()

PORT = 8080
DATA_FILE = "patients.json"

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# storage
def load_patients():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_patients(patients):
    with open(DATA_FILE, "w") as f:
        json.dump(patients, f, indent=2)

#  Helpers 

def generate_patient_id():
    return "PT-" + uuid.uuid4().hex[:4].upper()

def calculate_age(dob_str):
    try:
        dob = datetime.strptime(dob_str.strip(), "%m/%d/%Y").date()
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return None

#  AI Triage 

def ai_triage(patient):
    age = calculate_age(patient.get("dob", ""))
    age_str = f"{age} years old" if age else "age unknown"

    prompt = f"""You are an experienced ER triage nurse using the Emergency Severity Index (ESI).
Assess the following patient and return ONLY a JSON object — no markdown, no explanation outside the JSON.

Patient:
- Name: {patient['firstName']} {patient['lastName']}
- Age: {age_str}
- Sex: {patient.get('sex', 'not specified')}
- Pain level: {patient.get('painLevel', 'N/A')} / 10
- Symptoms: {', '.join(patient.get('symptoms', [])) or 'none listed'}
- Existing conditions: {patient.get('existingConditions') or 'none'}
- Allergies: {patient.get('allergies') or 'none'}
- Current medications: {patient.get('medications') or 'none'}
- Arrived alone: {patient.get('isAlone', False)}

Triage rules:
1. Assign urgency: EMERGENT (life-threatening), URGENT (serious but stable), or NON-URGENT (minor).
2. Assign a tiebreakScore (0-100). Higher = seen sooner within same urgency level.
   Tie-break priority: clinical severity, age (elderly 65+ and children under 12 first),
   sex (female prioritized in equivalent cases), high-risk conditions, pain level.
3. Write a brief one-sentence clinical reasoning.

Respond ONLY with this exact JSON:
{{
  "urgency": "EMERGENT",
  "urgencyRank": 1,
  "tiebreakScore": 80,
  "reasoning": "one sentence here"
}}"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        urgency = result.get("urgency", "URGENT").upper()
        if urgency not in ("EMERGENT", "URGENT", "NON-URGENT"):
            urgency = "URGENT"
        urgency_rank = {"EMERGENT": 1, "URGENT": 2, "NON-URGENT": 3}[urgency]
        tiebreak = max(0, min(100, int(result.get("tiebreakScore", 50))))
        reasoning = result.get("reasoning", "")

        print(f"  [AI] {urgency} (rank {urgency_rank}, tiebreak {tiebreak}) — {reasoning}")

        return {
            "urgency": urgency,
            "urgencyRank": urgency_rank,
            "tiebreakScore": tiebreak,
            "reasoning": reasoning,
        }

    except Exception as e:
        print(f"  [AI] Error: {e} — defaulting to URGENT")
        return {
            "urgency": "URGENT",
            "urgencyRank": 2,
            "tiebreakScore": 50,
            "reasoning": "AI triage unavailable; defaulted to URGENT.",
        }

def sort_key(patient):
    return (
        patient.get("urgencyRank", 2),
        -patient.get("tiebreakScore", 50),
        patient.get("submittedAt", ""),
    )

#  Request Handler 

class TriageHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"  {self.address_string()} - {format % args}")

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def parse_path(self):
        path = self.path.split("?")
        parts = path[0].rstrip("/").split("/")
        query = path[1] if len(path) > 1 else ""
        patient_id = parts[3] if len(parts) > 3 else None
        sub = parts[4] if len(parts) > 4 else None
        return patient_id, sub, query

    def parse_query(self, query_string):
        params = {}
        for part in query_string.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v
        return params

    def do_GET(self):
        patient_id, sub, query = self.parse_path()

        # Serve the patient intake HTML form
        if self.path == "/" or self.path == "/patient.html":
            try:
                with open("patient.html", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_json({"error": "patient.html not found"}, 404)
            return

        if self.path.startswith("/api/health"):
            patients = load_patients()
            waiting = sum(1 for p in patients if p.get("status") == "waiting")
            self.send_json({
                "status": "ok",
                "totalPatients": len(patients),
                "waitingPatients": waiting,
                "timestamp": datetime.utcnow().isoformat(),
            })

        elif self.path.startswith("/api/patients"):
            patients = load_patients()
            if patient_id:
                patient = next((p for p in patients if p["patientId"] == patient_id), None)
                if not patient:
                    self.send_json({"error": "Patient not found"}, 404)
                else:
                    self.send_json(patient)
            else:
                params = self.parse_query(query)
                status_filter = params.get("status")
                if status_filter:
                    patients = [p for p in patients if p.get("status") == status_filter]
                patients.sort(key=sort_key)
                self.send_json({"count": len(patients), "patients": patients})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/api/patients"):
            self.send_json({"error": "Not found"}, 404)
            return

        data = self.read_body()
        required = ["firstName", "lastName", "phone", "painLevel"]
        missing = [f for f in required if not data.get(f)]
        if missing:
            self.send_json({"error": f"Missing required fields: {', '.join(missing)}"}, 422)
            return

        patient_id = generate_patient_id()

        patient = {
            "patientId":          patient_id,
            "firstName":          data.get("firstName", "").strip(),
            "lastName":           data.get("lastName", "").strip(),
            "dob":                data.get("dob", ""),
            "sex":                data.get("sex", ""),
            "phone":              data.get("phone", "").strip(),
            "isAlone":            data.get("isAlone", False),
            "emergencyContact":   data.get("emergencyContact", "").strip(),
            "symptoms":           data.get("symptoms", []),
            "painLevel":          data.get("painLevel"),
            "existingConditions": data.get("existingConditions", "").strip(),
            "allergies":          data.get("allergies", "").strip(),
            "medications":        data.get("medications", "").strip(),
            "status":             "waiting",
            "submittedAt":        data.get("submittedAt", datetime.utcnow().isoformat()),
            "updatedAt":          datetime.utcnow().isoformat(),
        }

        print(f"  [AI] Running triage for {patient['firstName']} {patient['lastName']}...")
        triage = ai_triage(patient)
        patient.update(triage)

        patients = load_patients()
        patients.append(patient)
        save_patients(patients)

        print(f"  [+] Saved: {patient_id} — {patient['firstName']} {patient['lastName']} | {triage['urgency']}")

        self.send_json({
            "patientId":     patient_id,
            "urgency":       triage["urgency"],
            "urgencyRank":   triage["urgencyRank"],
            "tiebreakScore": triage["tiebreakScore"],
            "reasoning":     triage["reasoning"],
            "message":       "Patient intake recorded successfully."
        }, 201)

    def do_PUT(self):
        patient_id, sub, _ = self.parse_path()
        if not patient_id or sub != "status":
            self.send_json({"error": "Not found"}, 404)
            return

        data = self.read_body()
        allowed = {"waiting", "in-progress", "discharged"}
        new_status = data.get("status")
        if new_status and new_status not in allowed:
            self.send_json({"error": f"Invalid status. Must be one of: {allowed}"}, 422)
            return

        patients = load_patients()
        patient = next((p for p in patients if p["patientId"] == patient_id), None)
        if not patient:
            self.send_json({"error": "Patient not found"}, 404)
            return

        if new_status:
            patient["status"] = new_status
        if "note" in data:
            patient["staffNote"] = data["note"]
        patient["updatedAt"] = datetime.utcnow().isoformat()

        save_patients(patients)
        self.send_json({"patientId": patient_id, "status": patient["status"], "message": "Status updated."})

    def do_DELETE(self):
        patient_id, _, _ = self.parse_path()
        if not patient_id:
            self.send_json({"error": "Patient ID required"}, 400)
            return

        patients = load_patients()
        updated = [p for p in patients if p["patientId"] != patient_id]
        if len(updated) == len(patients):
            self.send_json({"error": "Patient not found"}, 404)
            return

        save_patients(updated)
        self.send_json({"message": f"Patient {patient_id} removed."})

#  Entry point 

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY not set. Add it to your .env file.")
        print()
    server = HTTPServer(("", PORT), TriageHandler)
    print("=" * 50)
    print("  FastTrack — Patient Intake API")
    print(f"  Running on http://localhost:{PORT}")
    print(f"  Open http://localhost:{PORT} in Safari")
    print(f"  Data stored in: {DATA_FILE}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()
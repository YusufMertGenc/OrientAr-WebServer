import random
from locust import HttpUser, task, between

questions = [
    "Where is the library?",
    "Where is the health center?",
    "What events are happening this week?",
    "List student societies.",
    "Where are the dormitories?"
]

class ChatbotUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def ask_question(self):
        payload = {
            "question": random.choice(questions)
        }

        with self.client.post(
            "/chatbot/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}, body: {response.text}")
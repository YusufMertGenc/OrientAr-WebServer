import random
from locust import HttpUser, task, between

QUESTION_CATEGORIES = {
    "normal": [
        "What facilities are available on campus?",
        "Where can I do sports at METU NCC?",
        "What are the library working hours?",
        "Where can students eat on campus?",
        "What accommodation options are available?",
        "How can I find student clubs?",
        "What transportation options are available at METU NCC?",
        "Where is the Student Development and Counseling Center?",
    ],
    "specific": [
        "What facilities are available in the Sports Center?",
        "What are the dormitory options at METU NCC?",
        "How can I apply for dormitory accommodation?",
        "What services does the library provide?",
        "What does the IEEE Student Chapter do?",
        "What does the Photography Community do?",
    ],
    "noisy": [
        "   hiiii whereee issss the gymmm???   ",
        "can u tell me where i can find campus food places pls",
        "where library hours??",
        "i am new student what can i do on campus?",
        "student clubs info pls??",
    ],
    "out_of_domain": [
        "Who won the NBA finals?",
        "What is the price of Bitcoin today?",
        "Who is the president of the USA?",
    ],
    "long": [
        "Can you give me detailed information about all campus facilities including sports areas, dormitories, social spaces, and student communities at METU NCC?",
        "I am a new student and I want to understand where I can stay, eat, study, join clubs, and do sports on campus. Can you summarize the available options?",
    ],
}

CATEGORY_WEIGHTS = {
    "normal": 40,
    "specific": 25,
    "noisy": 15,
    "out_of_domain": 10,
    "long": 10,
}


def choose_question():
    category = random.choices(
        population=list(CATEGORY_WEIGHTS.keys()),
        weights=list(CATEGORY_WEIGHTS.values()),
        k=1,
    )[0]
    return category, random.choice(QUESTION_CATEGORIES[category])


class ChatbotUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def ask_question(self):
        category, question = choose_question()

        payload = {"question": question}

        with self.client.post(
            "/chatbot/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            name=f"/chatbot/query [{category}]",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Status code: {response.status_code}, body: {response.text}"
                )
                return

            try:
                data = response.json()
            except Exception:
                response.failure("Response is not valid JSON")
                return

            required_fields = [
                "answer",
                "confidence",
                "context_used",
                "latency_ms",
                "domain_score",
                "in_domain",
            ]

            missing = [field for field in required_fields if field not in data]
            if missing:
                response.failure(f"Missing required fields: {missing}")
                return

            if not isinstance(data["answer"], str):
                response.failure("answer field is not a string")
                return

            if data["answer"].strip() == "":
                response.failure("answer field is empty")
                return

            response.success()
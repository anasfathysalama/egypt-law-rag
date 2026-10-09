"""50 concurrent users against the stub /ask. The report is written to reports/."""

from locust import HttpUser, between, task


class AskUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def ask(self) -> None:
        self.client.post("/ask", json={"question": "What does the contract say?"})

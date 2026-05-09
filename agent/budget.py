import structlog
log = structlog.get_logger()

class BudgetExceeded(Exception):
    pass

class BudgetGuard:
    def __init__(self, token_budget=None, cost_budget_usd=None):
        from shared.config import settings
        self.token_budget = token_budget or settings.token_budget
        self.cost_budget = cost_budget_usd or 0.10
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    def check(self, estimated_input_tokens=0):
        if self.input_tokens + self.output_tokens + estimated_input_tokens > self.token_budget:
            raise BudgetExceeded(f"Token budget exceeded: {self.input_tokens + self.output_tokens} of {self.token_budget}")
        if self.total_cost >= self.cost_budget:
            raise BudgetExceeded(f"Cost budget exceeded: ${self.total_cost:.4f}")

    def record(self, input_tokens, output_tokens):
        from shared.config import settings
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.call_count += 1
        self.total_cost += (input_tokens / 1000 * settings.cost_per_1k_input + output_tokens / 1000 * settings.cost_per_1k_output)

    def summary(self):
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "total_tokens": self.input_tokens + self.output_tokens, "total_cost_usd": round(self.total_cost, 6), "llm_calls": self.call_count, "budget_remaining_tokens": self.token_budget - self.input_tokens - self.output_tokens}

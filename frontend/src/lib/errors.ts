// One line of guidance for each code the backend can send (spec section 10).
const GUIDANCE: Record<string, string> = {
  provider_error: "The answer model refused the request. Check the backend log for the reason.",
  provider_timeout: "The answer model took too long. Raise OLLAMA_TIMEOUT or ask something shorter.",
  ollama_unavailable:
    "Private mode answers with Ollama only. Start Ollama, or leave private mode to use Gemini.",
  model_unavailable: "That model is not configured. Gemini needs GOOGLE_API_KEY and GEMINI_MODEL.",
  egress_blocked: "The egress guard blocked this connection. The Privacy tab lists what was attempted.",
  internal_error: "The backend hit an unexpected error. Its log has the details.",
  network_error: "The backend did not answer. Check that it is running on the proxied port.",
  cancelled: "You stopped this answer.",
  incomplete: "This question was stored without an answer.",
};

export function errorGuidance(code: string): string | null {
  return GUIDANCE[code] ?? null;
}

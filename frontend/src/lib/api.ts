/**
 * Centralized API module for NyayaQuest frontend.
 *
 * Provides typed error classes and a single apiChat() function with
 * proper handling for 401, 429, timeouts, and generic errors.
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ─── Custom Error Classes ────────────────────────────────────────────

export class AuthExpiredError extends Error {
  constructor() {
    super('Session expired. Please sign in again.');
    this.name = 'AuthExpiredError';
  }
}

export class RateLimitError extends Error {
  constructor() {
    super('Too many requests. Please wait a minute.');
    this.name = 'RateLimitError';
  }
}

export class TimeoutError extends Error {
  constructor() {
    super('Request timed out. Please try again.');
    this.name = 'TimeoutError';
  }
}

export class APIError extends Error {
  status: number;
  constructor(status: number, body: string) {
    super(`API error ${status}: ${body}`);
    this.name = 'APIError';
    this.status = status;
  }
}

// ─── Chat API ────────────────────────────────────────────────────────

export interface ChatPayload {
  user_id: string;
  thread_id: string;
  message: string;
}

export interface ChatResponse {
  response: string;
  context: Array<{
    page_content: string;
    metadata: Record<string, unknown>;
  }>;
  thread_id: string;
}

export async function apiChat(payload: ChatPayload): Promise<ChatResponse> {
  try {
    const response = await fetch(`${API_URL}/api/chat`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (response.status === 401) {
      throw new AuthExpiredError();
    }

    if (response.status === 429) {
      throw new RateLimitError();
    }

    if (!response.ok) {
      throw new APIError(response.status, await response.text());
    }

    return await response.json();
  } catch (error) {
    if (error instanceof AuthExpiredError || error instanceof RateLimitError || error instanceof APIError) {
      throw error;
    }
    if (error instanceof Error && error.name === 'AbortError') {
      throw new TimeoutError();
    }
    throw error;
  }
}

// ─── History API ─────────────────────────────────────────────────────

export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
  context?: Array<{
    page_content: string;
    metadata: Record<string, unknown>;
  }>;
}

export async function apiGetHistory(userId: string, threadId: string): Promise<HistoryMessage[]> {
  const response = await fetch(`${API_URL}/api/conversations/${userId}/${threadId}`);

  if (!response.ok) {
    throw new APIError(response.status, await response.text());
  }

  const data = await response.json();
  return data.history || [];
}

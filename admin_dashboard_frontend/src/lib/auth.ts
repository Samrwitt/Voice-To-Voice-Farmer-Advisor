import type { SessionData } from '@/types';

export type DashboardRole = 'admin' | 'da' | 'expert';

/** Default 1 hour session timeout (overridden by JWT exp when available) */
export const SESSION_TIMEOUT_MS = 60 * 60 * 1000;

// ── Login (proxied through Next.js → logic_service /admin/login) ─────────────
type LoginPayload = {
  email: string;
  password: string;
};

interface LoginResponse {
  token: string;
  access_token?: string;
  expires_in?: number;
  role: DashboardRole;
  username: string;
  email?: string;
  user_id?: string;
  full_name?: string;
}

/**
 * Authenticates against logic_service POST /admin/login.
 * Persists JWT + profile in localStorage and returns the full session.
 */
export async function loginUser(payload: LoginPayload): Promise<SessionData> {
  const response = await fetch('/api/admin/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let message = 'Login failed';
    try {
      const errorData = await response.json();
      message = errorData.detail || message;
    } catch {
      // ignore JSON parse error
    }
    throw new Error(message);
  }

  const data: LoginResponse = await response.json();

  const expiresInMs =
    typeof data.expires_in === 'number'
      ? data.expires_in * 1000
      : SESSION_TIMEOUT_MS;

  const session: SessionData = {
    token: data.token ?? data.access_token ?? '',
    username: data.email ?? data.username,
    email: data.email ?? data.username,
    userId: data.user_id,
    fullName: data.full_name,
    role: data.role,
    loginTime: Date.now(),
    expiresAt: Date.now() + expiresInMs,
  };

  if (typeof window !== 'undefined') {
    localStorage.setItem('session', JSON.stringify(session));
  }

  return session;
}

// ── Session helpers ───────────────────────────────────────────────────────────
export function saveSession(data: Omit<SessionData, 'loginTime'>): void {
  if (typeof window === 'undefined') return;
  const session: SessionData = { ...data, loginTime: Date.now() };
  localStorage.setItem('session', JSON.stringify(session));
}

export function getSession(): SessionData | null {
  if (typeof window === 'undefined') return null;

  const raw = localStorage.getItem('session');
  if (!raw) return null;

  try {
    const session: SessionData = JSON.parse(raw);
    const expiresAt =
      session.expiresAt ?? session.loginTime + SESSION_TIMEOUT_MS;
    if (Date.now() >= expiresAt) {
      clearSession();
      return null;
    }
    return session;
  } catch {
    clearSession();
    return null;
  }
}

export function clearSession(): void {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('session');
  }
}

export function getToken(): string | null {
  return getSession()?.token ?? null;
}

export function getRole(): DashboardRole | null {
  return getSession()?.role ?? null;
}

export function getUsername(): string | null {
  return getSession()?.username ?? null;
}

export function getUserId(): string | null {
  return getSession()?.userId ?? null;
}

export function getEmail(): string | null {
  return getSession()?.email ?? null;
}

export function getFullName(): string | null {
  return getSession()?.fullName ?? null;
}

export function isAdmin(): boolean {
  return getRole() === 'admin';
}

export function isDA(): boolean {
  return getRole() === 'da';
}

export function isExpert(): boolean {
  return getRole() === 'expert';
}

export function hasRole(...roles: DashboardRole[]): boolean {
  const role = getRole();
  return role ? roles.includes(role) : false;
}

/** Returns milliseconds remaining in the current session. */
export function getSessionTimeRemaining(): number {
  const session = getSession();
  if (!session) return 0;
  const expiresAt =
    session.expiresAt ?? session.loginTime + SESSION_TIMEOUT_MS;
  return Math.max(0, expiresAt - Date.now());
}

export async function authFetch(
  input: string,
  init: RequestInit = {}
): Promise<Response> {
  const token = getToken();
  if (!token) throw new Error('Not authenticated');

  return fetch(input, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...(init.headers || {}),
    },
  });
}

import type { SessionData } from '@/types';

export type DashboardRole = 'admin' | 'da' | 'expert';

/** 1 hour — matches backend SESSION_TIMEOUT_SECONDS = 3600 */
export const SESSION_TIMEOUT_MS = 60 * 60 * 1000;

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type BackendLoginResponse = {
  access_token: string;
  token_type: string;
  user: {
    user_id: string;
    full_name: string;
    email: string;
    role: DashboardRole;
    is_active: boolean;
  };
};

type LoginPayload = {
  email: string;
  password: string;
};

export async function loginUser(payload: LoginPayload): Promise<SessionData> {
  const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let message = 'Login failed';

    try {
      const errorData = await response.json();
      message = errorData.detail || message;
    } catch {
      // Ignore JSON parse error
    }

    throw new Error(message);
  }

  const data: BackendLoginResponse = await response.json();

  const session: SessionData = {
    token: data.access_token,
    userId: data.user.user_id,
    username: data.user.full_name,
    email: data.user.email,
    role: data.user.role,
    loginTime: Date.now(),
  };

  if (typeof window !== 'undefined') {
    localStorage.setItem('session', JSON.stringify(session));
  }

  return session;
}

export function saveSession(data: Omit<SessionData, 'loginTime'>): void {
  if (typeof window === 'undefined') return;

  const session: SessionData = {
    ...data,
    loginTime: Date.now(),
  };

  localStorage.setItem('session', JSON.stringify(session));
}

export function getSession(): SessionData | null {
  if (typeof window === 'undefined') return null;

  const raw = localStorage.getItem('session');

  if (!raw) return null;

  try {
    const session: SessionData = JSON.parse(raw);

    if (Date.now() - session.loginTime > SESSION_TIMEOUT_MS) {
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

  return Math.max(
    0,
    SESSION_TIMEOUT_MS - (Date.now() - session.loginTime)
  );
}

export async function authFetch(
  input: string,
  init: RequestInit = {}
): Promise<Response> {
  const token = getToken();

  if (!token) {
    throw new Error('Not authenticated');
  }

  return fetch(`${API_BASE_URL}${input}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...(init.headers || {}),
    },
  });
}

export async function fetchCurrentUser() {
  const response = await authFetch('/api/auth/me');

  if (!response.ok) {
    clearSession();
    throw new Error('Session expired or invalid');
  }

  return response.json();
}
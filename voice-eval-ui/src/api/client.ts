import type {
  ClipSaved, ClipsResponse, Evaluation, GoldenResponse, Review, SessionDetail, SessionRow, SummaryResponse,
} from "./models";

/**
 * Two services, two base URLs. Empty (the default) means same origin, which is what `npm run dev` (proxy) and a
 * reverse proxy give you. Set them in .env when the services are on different origins (then enable CORS on both):
 *   VITE_EVAL_API_URL=http://127.0.0.1:8001     VITE_AGENT_URL=http://127.0.0.1:8000
 */
const EVAL_API = ((import.meta.env.VITE_EVAL_API_URL as string | undefined) ?? "").replace(/\/$/, "");
export const AGENT_URL = ((import.meta.env.VITE_AGENT_URL as string | undefined) ?? "").replace(/\/$/, "");

const KEY_STORAGE = "accessKey";
let accessKey = "";
try {
  accessKey = sessionStorage.getItem(KEY_STORAGE) ?? "";
} catch {
  /* storage can be blocked: the key then lives in memory only */
}

export const getAccessKey = () => accessKey;
export function setAccessKey(k: string): void {
  accessKey = k;
  try {
    sessionStorage.setItem(KEY_STORAGE, k);
  } catch {
    /* ignore */
  }
}

export class ApiError extends Error {}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const res = await fetch(`${EVAL_API}/api${path}`, { ...init, headers: { "X-API-Key": accessKey, ...(init.headers ?? {}) } });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail);
  }
  return res;
}
const json = async <T>(path: string, init?: RequestInit) => (await request(path, init)).json() as Promise<T>;

export const api = {
  golden: () => json<GoldenResponse>("/golden"),
  clips: () => json<ClipsResponse>("/clips"),
  sessions: () => json<SessionRow[]>("/sessions"),
  session: (id: string) => json<SessionDetail>(`/sessions/${id}`),
  summary: () => json<SummaryResponse>("/summary"),
  saveSession: (fd: FormData) => json<{ id: string }>("/sessions", { method: "POST", body: fd }),
  evaluate: (id: string, o: { llm: boolean; mos: boolean }) =>
    json<Evaluation>(`/sessions/${id}/evaluate?use_llm=${o.llm}&use_mos=${o.mos}`, { method: "POST" }),
  review: (id: string, body: Partial<Review>) =>
    json<Review>(`/sessions/${id}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  addClip: (fd: FormData) => json<ClipSaved>("/clips", { method: "POST", body: fd }),
  /** Stored clip as PCM16 16 kHz, optionally mixed with a noise recording at `snrDb` on the server. */
  clipPcm: async (id: string, noise: string, snrDb: number): Promise<Int16Array> => {
    const r = await request(`/clips/${encodeURIComponent(id)}/pcm?noise=${encodeURIComponent(noise)}&snr_db=${snrDb}`);
    return new Int16Array(await r.arrayBuffer());
  },
  /** Audio needs the key header, so it is fetched and exposed as a blob URL (revoke it when done). */
  audioUrl: async (id: string, who: "caller" | "agent"): Promise<string> =>
    URL.createObjectURL(await (await request(`/sessions/${id}/audio/${who}`)).blob()),
};

export async function fetchToken(): Promise<string> {
  const r = await fetch(`${AGENT_URL}/auth/token`, { method: "POST", headers: { "X-API-Key": accessKey } });
  if (r.status === 401) throw new ApiError("access key rejected");
  if (!r.ok) throw new ApiError(`could not get a token (HTTP ${r.status})`);
  return ((await r.json()) as { access_token: string }).access_token;
}

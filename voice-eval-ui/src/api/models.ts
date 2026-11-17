// Friendly names for the generated OpenAPI schema (src/api/types.ts is generated: never edit it by hand).
import type { components } from "./types";

type S = components["schemas"];
export type GoldenScenario = S["GoldenScenario"];
export type GoldenResponse = S["GoldenResponse"];
export type SessionEvent = S["SessionEvent"];
export type SessionMeta = S["SessionMeta"];
export type SessionRow = S["SessionRow"];
export type SessionDetail = S["SessionDetail"];
export type Evaluation = S["Evaluation"];
export type Review = S["Review"];
export type SummaryResponse = S["SummaryResponse"];
export type GroupStats = S["GroupStats"];
export type ClipsResponse = S["ClipsResponse"];
export type ClipLine = S["ClipLine"];
export type ClipSaved = S["ClipSaved"];
export type MosScore = S["MosScore"];

/** Messages the agent sends over /ws as JSON (the WebSocket protocol is not part of the OpenAPI document). */
export type ServerMsg =
  | { type: "ready"; in_rate: number; out_rate: number; tools: string[]; conversation_id: string }
  | { type: "transcript"; text: string }
  | { type: "reply"; text: string }
  | { type: "interrupted" }
  | { type: "tool"; name: string; args: Record<string, unknown>; result: unknown; ms: number }
  | { type: "error"; message: string };

/** Pure audio helpers: no DOM, so they are unit-tested in plain Node (vitest). */

export const CALLER_RATE = 16_000;
export const FRAME = 480; // 30 ms at 16 kHz

export interface AgentChunk {
  /** Seconds from record start at which this chunk is PLAYED (not when it arrived). */
  start: number;
  pcm: Int16Array;
}

export function concatInt16(parts: Int16Array[]): Int16Array {
  const out = new Int16Array(parts.reduce((n, p) => n + p.length, 0));
  let pos = 0;
  for (const p of parts) {
    out.set(p, pos);
    pos += p.length;
  }
  return out;
}

/** 44-byte RIFF header + PCM16 mono. */
export function encodeWav(pcm: Int16Array, rate: number): Blob {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const tag = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  tag(0, "RIFF");
  v.setUint32(4, 36 + pcm.length * 2, true);
  tag(8, "WAVEfmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  tag(36, "data");
  v.setUint32(40, pcm.length * 2, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: "audio/wav" });
}

/**
 * After a barge-in at time `t`, keep only what the listener heard: drop chunks that start after `t` and cut the one
 * that spans it. Without this agent.wav would contain speech that was stopped before it was played.
 */
export function truncateAgent(chunks: AgentChunk[], t: number, rate: number): AgentChunk[] {
  return chunks
    .filter((c) => c.start < t)
    .map((c) => {
      const keep = Math.floor((t - c.start) * rate);
      return keep < c.pcm.length ? { start: c.start, pcm: c.pcm.slice(0, Math.max(0, keep)) } : c;
    });
}

/** Lays chunks out on a zero-filled timeline of `totalSeconds` at their play positions. */
export function buildTimeline(chunks: AgentChunk[], totalSeconds: number, rate: number): Int16Array {
  const out = new Int16Array(Math.ceil(totalSeconds * rate));
  for (const c of chunks) {
    const at = Math.max(0, Math.round(c.start * rate));
    const room = Math.max(0, out.length - at);
    out.set(c.pcm.subarray(0, Math.min(c.pcm.length, room)), at);
  }
  return out;
}

/** Pads or cuts `pcm` to exactly `seconds` so caller and agent files start and end together. */
export function fitTo(pcm: Int16Array, seconds: number, rate: number): Int16Array {
  const out = new Int16Array(Math.ceil(seconds * rate));
  out.set(pcm.subarray(0, out.length));
  return out;
}

/** Total length of a session: the longest of caller audio, wall clock and the end of the last agent chunk. */
export function sessionSeconds(callerSamples: number, wallSeconds: number, chunks: AgentChunk[], agentRate: number): number {
  return Math.max(callerSamples / CALLER_RATE, wallSeconds, ...chunks.map((c) => c.start + c.pcm.length / agentRate));
}

/** Splits audio into fixed 30 ms frames, zero-padding the last one. */
export function toFrames(pcm: Int16Array, size = FRAME): Int16Array[] {
  const frames: Int16Array[] = [];
  for (let o = 0; o < pcm.length; o += size) {
    const f = new Int16Array(size);
    f.set(pcm.subarray(o, o + size));
    frames.push(f);
  }
  return frames;
}

/**
 * How many frames must be sent now to stay on the wall clock. Sending exactly one frame per timer tick would
 * stretch the audio whenever a tick is late; computing the shortfall keeps the stream real-time.
 */
export function framesDue(elapsedMs: number, alreadySent: number, frameMs = 30): number {
  return Math.max(0, Math.floor(elapsedMs / frameMs) - alreadySent);
}

/** Removes leading/trailing samples below `threshold` amplitude and keeps `margin` samples either side. Null if too short. */
export function trimSilence(pcm: Int16Array, threshold = 400, margin = 2400, minSamples = 4000): Int16Array | null {
  let s = 0;
  let e = pcm.length;
  while (s < e && Math.abs(pcm[s]!) < threshold) s++;
  while (e > s && Math.abs(pcm[e - 1]!) < threshold) e--;
  if (e - s < minSamples) return null;
  return pcm.slice(Math.max(0, s - margin), Math.min(pcm.length, e + margin));
}

export function rms(frame: Int16Array, step = 8): number {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < frame.length; i += step) {
    sum += frame[i]! * frame[i]!;
    n++;
  }
  return n ? Math.sqrt(sum / n) : 0;
}

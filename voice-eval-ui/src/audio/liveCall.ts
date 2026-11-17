import type { ServerMsg, SessionEvent } from "../api/models";
import {
  buildTimeline, CALLER_RATE, concatInt16, fitTo, FRAME, framesDue, rms, sessionSeconds, toFrames, truncateAgent,
  type AgentChunk,
} from "./wav";
import { AGENT_URL } from "../api/client";
import { openMic } from "./worklet";

export interface RecordedSession {
  events: SessionEvent[];
  callerPcm: Int16Array; // 16 kHz, whole timeline
  agentPcm: Int16Array; // outRate, as played, same length in seconds
  outRate: number;
  durationS: number;
  conversationId: string | null;
  clipTexts: string[];
}

interface Rec {
  ctx0: number;
  caller: Int16Array[];
  agent: AgentChunk[];
  events: SessionEvent[];
  clipTexts: string[];
}

export interface CallHandlers {
  onMessage(m: ServerMsg): void;
  onLevel(level: number): void;
  onClosed(wasOpen: boolean): void;
}

/**
 * One call to the agent under test. Owns the AudioContext, the WebSocket and the recording.
 * Everything on the recording timeline uses the audio clock (ctx.currentTime), so events, caller frames and
 * agent playback cannot drift against each other.
 */
export class LiveCall {
  private ws!: WebSocket;
  private ctx = new AudioContext();
  private mic: { stop(): void } | null = null;
  private playAt = 0;
  private sources: AudioBufferSourceNode[] = [];
  private outRate = 24000;
  private rec: Rec | null = null;
  private clipTimer: number | undefined;
  private clipQueue: Int16Array[] = [];
  private sent = 0;
  private conversationId: string | null = null;
  /** Set when a recording was finished; the UI reads it once with takeResult(). */
  private result: RecordedSession | null = null;

  private constructor(private handlers: CallHandlers) {}

  static async connect(mode: "mic" | "clip", token: string, handlers: CallHandlers): Promise<LiveCall> {
    const c = new LiveCall(handlers);
    if (mode === "mic") {
      c.mic = await openMic(c.ctx, { echoCancellation: true, noiseSuppression: true }, (f) => {
        if (c.ws.readyState === WebSocket.OPEN) c.ws.send(f.buffer as ArrayBuffer);
        c.rec?.caller.push(f);
        handlers.onLevel(rms(f) / 120);
      });
    } else {
      c.startClipSender();
    }
    const base = AGENT_URL || location.origin;   // http(s)://host -> ws(s)://host
    c.ws = new WebSocket(`${base.replace(/^http/, "ws")}/ws`, ["bearer", token]);
    c.ws.binaryType = "arraybuffer";
    let opened = false;
    c.ws.onopen = () => (opened = true);
    c.ws.onmessage = (e: MessageEvent<string | ArrayBuffer>) =>
      typeof e.data === "string" ? c.onJson(JSON.parse(e.data) as ServerMsg) : c.play(e.data);
    c.ws.onclose = () => {
      c.end();
      handlers.onClosed(opened);
    };
    return c;
  }

  get isRecording(): boolean {
    return this.rec !== null;
  }

  /** Seconds since Record was pressed. */
  now(): number {
    return this.rec ? this.ctx.currentTime - this.rec.ctx0 : 0;
  }

  startRecording(): void {
    this.result = null;
    this.rec = { ctx0: this.ctx.currentTime, caller: [], agent: [], events: [], clipTexts: [] };
    if (this.conversationId) this.rec.events.push({ t: 0, type: "ready", conversation_id: this.conversationId });
  }

  /** Clip mode: queue a stored clip to be streamed in real time; `text` becomes part of the WER reference. */
  queueClip(pcm: Int16Array, text: string): void {
    this.clipQueue.push(...toFrames(pcm));
    this.rec?.clipTexts.push(text);
  }

  /** Stops recording and returns the finished session (also kept for takeResult). */
  finishRecording(): RecordedSession | null {
    if (!this.rec) return null;
    const r = this.rec;
    const wall = this.now();
    const caller = concatInt16(r.caller);
    const total = sessionSeconds(caller.length, wall, r.agent, this.outRate);
    this.rec = null;
    this.result = {
      events: r.events,
      callerPcm: fitTo(caller, total, CALLER_RATE),
      agentPcm: buildTimeline(r.agent, total, this.outRate),
      outRate: this.outRate,
      durationS: total,
      conversationId: this.conversationId,
      clipTexts: r.clipTexts,
    };
    return this.result;
  }

  /** The session finished by finishRecording() or by the call ending mid-recording; returned once. */
  takeResult(): RecordedSession | null {
    const r = this.result;
    this.result = null;
    return r;
  }

  end(): void {
    if (this.rec) this.finishRecording();
    window.clearInterval(this.clipTimer);
    this.mic?.stop();
    this.mic = null;
    this.stopPlayback();
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) this.ws.close();
    if (this.ctx.state !== "closed") void this.ctx.close();
  }

  // ---------------------------------------------------------------- internals
  private onJson(m: ServerMsg): void {
    if (m.type === "ready") {
      this.outRate = m.out_rate;
      this.conversationId = m.conversation_id;
    }
    if (m.type === "interrupted") {
      this.stopPlayback();
      if (this.rec) this.rec.agent = truncateAgent(this.rec.agent, this.now(), this.outRate);
    }
    if (this.rec && m.type !== "ready") this.rec.events.push({ ...m, t: this.now() } as SessionEvent);
    this.handlers.onMessage(m);
  }

  private play(buf: ArrayBuffer): void {
    const pcm = new Int16Array(buf);
    const f = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f[i] = pcm[i]! / 32768;
    const ab = this.ctx.createBuffer(1, f.length, this.outRate);
    ab.copyToChannel(f, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = ab;
    src.connect(this.ctx.destination);
    const start = Math.max(this.ctx.currentTime, this.playAt);
    src.start(start);
    this.playAt = start + ab.duration;
    this.sources.push(src);
    src.onended = () => (this.sources = this.sources.filter((s) => s !== src));
    // The timeline is when the audio is PLAYED, not when it arrived: replies arrive in bursts faster than real time.
    if (this.rec) this.rec.agent.push({ start: start - this.rec.ctx0, pcm });
  }

  private stopPlayback(): void {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources = [];
    this.playAt = 0;
  }

  /** Streams 30 ms frames on the wall clock (silence when nothing is queued) so caller.wav is gap-free. */
  private startClipSender(): void {
    const t0 = performance.now();
    this.clipTimer = window.setInterval(() => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      const due = framesDue(performance.now() - t0, this.sent);
      for (let k = 0; k < due; k++) {
        const fr = this.clipQueue.shift() ?? new Int16Array(FRAME);
        this.ws.send(fr.buffer.slice(fr.byteOffset, fr.byteOffset + fr.byteLength) as ArrayBuffer);
        this.sent++;
        this.rec?.caller.push(fr);
      }
    }, 15);
  }
}

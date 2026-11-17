/** AudioWorklet that downsamples the mic to 16 kHz PCM16 and posts 480-sample (30 ms) frames. */
const SOURCE = `class Cap extends AudioWorkletProcessor{
constructor(){super();this.r=sampleRate/16000;this.pos=0;this.out=[];}
process(i){const ch=i[0][0];if(!ch)return true;
for(let k=0;k<ch.length;k++){this.pos+=1;if(this.pos>=this.r){this.pos-=this.r;this.out.push(Math.max(-1,Math.min(1,ch[k]))*32767);}}
while(this.out.length>=480){this.port.postMessage(Int16Array.from(this.out.splice(0,480)).buffer);}return true;}}
registerProcessor('cap',Cap);`;

export async function addCaptureWorklet(ctx: AudioContext): Promise<void> {
  const url = URL.createObjectURL(new Blob([SOURCE], { type: "application/javascript" }));
  try {
    await ctx.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Opens the mic into the capture worklet. Call `stop()` to release the microphone. */
export async function openMic(ctx: AudioContext, constraints: MediaTrackConstraints, onFrame: (f: Int16Array) => void) {
  await addCaptureWorklet(ctx);
  const stream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
  const node = new AudioWorkletNode(ctx, "cap");
  node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => onFrame(new Int16Array(e.data));
  ctx.createMediaStreamSource(stream).connect(node);
  return { stop: () => stream.getTracks().forEach((t) => t.stop()) };
}

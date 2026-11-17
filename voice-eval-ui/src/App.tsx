import { useState } from "react";
import { getAccessKey, setAccessKey } from "./api/client";
import { BaselineTab } from "./tabs/BaselineTab";
import { LiveTab } from "./tabs/LiveTab";
import { SessionsTab } from "./tabs/SessionsTab";

type Tab = "live" | "sessions" | "baseline";
const TABS: [Tab, string][] = [["live", "1 · Live session"], ["sessions", "2 · Sessions & metrics"], ["baseline", "3 · Baseline clips"]];

export function App() {
  const [tab, setTab] = useState<Tab>("live");
  const [key, setKey] = useState(getAccessKey());
  const [keyVersion, setKeyVersion] = useState(0);
  const [openId, setOpenId] = useState<string | null>(null);

  // Tabs stay mounted (hidden) so a call in progress survives switching tabs.
  const show = (t: Tab) => ({ display: tab === t ? "block" : "none" });
  return (
    <>
      <header>
        <h1>Voice Agent QA Console</h1>
        <nav>
          {TABS.map(([t, label]) => <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{label}</button>)}
        </nav>
        <label style={{ marginLeft: "auto" }}>Access key
          <input type="password" autoComplete="off" placeholder="AUTH_API_KEY" value={key}
            onChange={(e) => { setKey(e.target.value); setAccessKey(e.target.value); }}
            onBlur={() => setKeyVersion((v) => v + 1)} />
        </label>
      </header>
      <main>
        <div style={show("live")}><LiveTab keyVersion={keyVersion} onOpenSession={(id) => { setOpenId(id); setTab("sessions"); }} /></div>
        <div style={show("sessions")}><SessionsTab keyVersion={keyVersion} openId={openId} /></div>
        <div style={show("baseline")}><BaselineTab keyVersion={keyVersion} /></div>
      </main>
    </>
  );
}

import { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { ModelStatus } from '../api/types';
import { voiceApi } from '../api/voice';

export default function SettingsView() {
  const [status, setStatus] = useState<Record<string, ModelStatus> | null>(null);
  const [sys, setSys] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, v] = await Promise.all([apiClient.fetchModelStatus(), voiceApi.system().catch(() => null)]);
      setStatus(s);
      setSys(v);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const pill = (ok: boolean, yes = 'Ready', no = 'Not available') =>
    <span className={`status-pill ${ok ? 'real' : 'demo'}`}>{ok ? yes : no}</span>;

  return (
    <div className="settings-view">
      <div className="settings-header">
        <h2>Models &amp; system</h2>
        <p>What is loaded, where it runs, and what still needs to be started.</p>
      </div>

      {error ? (
        <div className="error-toast" role="alert">{error}</div>
      ) : loading || !status ? (
        <div className="loading-state">Checking models…</div>
      ) : (
        <div className="settings-cards">
          <div className="settings-card">
            <div className="card-top"><h3>{status.model_a?.name}</h3>{pill(!!status.model_a?.is_installed)}</div>
            <div className="detail-row"><strong>Models:</strong> <code>{status.model_a?.weights_path}</code></div>
            <div className="detail-row"><strong>Status:</strong> <span>{status.model_a?.status_message}</span></div>
            <div className="detail-row"><strong>Device:</strong> <span>{status.model_a?.device.toUpperCase()}</span></div>
          </div>

          <div className="settings-card">
            <div className="card-top">
              <h3>{status.model_b?.name}</h3>
              {sys ? pill(!!sys.llm_reachable, 'Running', 'Not running') : pill(!!status.model_b?.is_installed)}
            </div>
            <div className="detail-row"><strong>Model:</strong> <code>{status.model_b?.weights_path}</code></div>
            <div className="detail-row"><strong>Status:</strong>{' '}
              <span>{sys && !sys.llm_reachable ? `Start Ollama and run: ollama pull ${sys.settings?.llm_model}` : status.model_b?.status_message}</span>
            </div>
          </div>

          {sys && (
            <div className="settings-card">
              <div className="card-top">
                <h3>Speech &amp; voice</h3>
                {pill(Object.values(sys.tts ?? {}).some(Boolean), 'Voice ready', 'No voice engine running')}
              </div>
              <div className="detail-row"><strong>Speech-to-text:</strong> <code>faster-whisper {sys.settings?.whisper_model}</code></div>
              {Object.entries(sys.tts ?? {}).map(([k, ok]) => (
                <div className="detail-row" key={k}>
                  <strong>Voice cloning · {k}:</strong> <span>{ok ? 'running' : `not running (${sys.settings?.tts_routes?.[k]})`}</span>
                </div>
              ))}
              <div className="detail-row"><strong>Compute:</strong>{' '}
                <span>{sys.device?.device?.toUpperCase()}{sys.device?.name ? ` · ${sys.device.name}` : ''}</span>
              </div>
              {Object.keys(sys.registry?.errors ?? {}).length > 0 && (
                <div className="detail-row"><strong>Load errors:</strong>{' '}
                  <code>{Object.entries(sys.registry.errors).map(([k, v]) => `${k}: ${v}`).join(' | ')}</code>
                </div>
              )}
            </div>
          )}

          <div className="instructions-card">
            <h3>Starting everything</h3>
            <ol>
              <li>Run <code>scripts\run_dev.ps1</code>. It opens the API, the voice engines and this web app.</li>
              <li>Make sure Ollama is running. The LLM pill above turns green when it is.</li>
              <li>Check from a terminal with <code>.venv\Scripts\python.exe scripts\check_system.py</code>.</li>
              <li>See <code>docs\VOICE_QUICKSTART.md</code> for troubleshooting.</li>
            </ol>
            <button className="secondary-btn" onClick={load}>Refresh</button>
          </div>
        </div>
      )}
    </div>
  );
}

import { useState, useEffect } from 'react';
import { AnalysisResult, ModelStatus } from './api/types';
import { apiClient } from './api/client';
import InputPanel from './components/InputPanel/InputPanel';
import TranscriptPanel from './components/TranscriptPanel/TranscriptPanel';
import VerdictPanel from './components/VerdictPanel/VerdictPanel';
import AlternativesPanel from './components/AlternativesPanel/AlternativesPanel';
import OverviewView from './components/OverviewView';
import RewriteStudioView from './components/RewriteStudioView';
import ModerationCenterView from './components/ModerationCenterView';
import AnalyticsView from './components/AnalyticsView';
import ModelEvaluationView from './components/ModelEvaluationView';
import SettingsView from './components/SettingsView';
import VoiceStudioView from './components/VoiceStudio/VoiceStudioView';
import LiveRoomView from './components/LiveRoom/LiveRoomView';
import { voiceApi } from './api/voice';
import './App.css';

type NavTab = 'voice' | 'live' | 'overview' | 'analyzer' | 'studio' | 'moderation' | 'analytics' | 'evaluation' | 'settings';

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>(() => (new URLSearchParams(location.search).get('tab') === 'live' ? 'live' : 'voice'));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [modelStatus, setModelStatus] = useState<Record<string, ModelStatus> | null>(null);

  const [voiceSys, setVoiceSys] = useState<{ ready: boolean; detail: string } | null>(null);

  useEffect(() => {
    apiClient.fetchModelStatus().then(setModelStatus).catch(() => {});
    voiceApi.system().then((s) => {
      const tts = Object.entries(s.tts ?? {}).filter(([, ok]) => ok).map(([k]) => k);
      const missing = [!s.llm_reachable && `LLM (${s.settings?.llm_model})`, !tts.length && 'TTS service'].filter(Boolean);
      setVoiceSys({
        ready: missing.length === 0,
        detail: missing.length ? `Not running: ${missing.join(', ')}` : `LLM ${s.settings?.llm_model} · TTS ${tts.join(', ')}`,
      });
    }).catch(() => setVoiceSys({ ready: false, detail: 'Voice API not reachable' }));
  }, []);

  const handleTextAnalyze = async (text: string, langHint: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.analyzeText(text, langHint);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze text');
    } finally {
      setLoading(false);
    }
  };

  const handleAudioAnalyze = async (file: File, langHint: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.analyzeAudio(file, langHint);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze audio');
    } finally {
      setLoading(false);
    }
  };

  const isDemoMode = voiceSys ? !voiceSys.ready : (!modelStatus?.model_a?.is_installed || !modelStatus?.model_b?.is_installed);

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="logo-area">
          <h1>Civitas AI</h1>
          <span className="badge">Speech &amp; text moderation</span>
        </div>

        <nav className="nav-tabs" aria-label="Main Navigation">
          <button
            className={`nav-tab-btn ${activeTab === 'voice' ? 'active' : ''}`}
            onClick={() => setActiveTab('voice')}
          >Voice Studio</button>
          <button
            className={`nav-tab-btn ${activeTab === 'live' ? 'active' : ''}`}
            onClick={() => setActiveTab('live')}
          >Live Room</button>
          <button
            className={`nav-tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >Overview</button>
          <button
            className={`nav-tab-btn ${activeTab === 'analyzer' ? 'active' : ''}`}
            onClick={() => setActiveTab('analyzer')}
          >Analyzer</button>
          <button
            className={`nav-tab-btn ${activeTab === 'studio' ? 'active' : ''}`}
            onClick={() => setActiveTab('studio')}
          >Rewrite</button>
          <button
            className={`nav-tab-btn ${activeTab === 'moderation' ? 'active' : ''}`}
            onClick={() => setActiveTab('moderation')}
          >Moderation</button>
          <button
            className={`nav-tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >Analytics</button>
          <button
            className={`nav-tab-btn ${activeTab === 'evaluation' ? 'active' : ''}`}
            onClick={() => setActiveTab('evaluation')}
          >Evaluation</button>
          <button
            className={`nav-tab-btn ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >Settings</button>
        </nav>

        <div className="settings-area">
          <span className={`status-pill ${voiceSys?.ready ? 'voice-ready' : 'voice-setup'}`} title={voiceSys?.detail}>
            {voiceSys == null ? 'Checking models…' : voiceSys.ready ? 'Models ready' : 'Setup needed'}
          </span>
        </div>
      </header>

      {/* Main Content Area based on Tab */}
      <main className="main-content">
        {activeTab === 'live' && (
          <div className="full-view-container">
            <LiveRoomView />
          </div>
        )}

        {activeTab === 'voice' && (
          <div className="full-view-container">
            <VoiceStudioView />
          </div>
        )}

        {activeTab === 'overview' && (
          <div className="full-view-container">
            <OverviewView
              onGoToAnalyzer={() => setActiveTab('analyzer')}
              isDemoMode={isDemoMode}
            />
          </div>
        )}

        {activeTab === 'analyzer' && (
          <div className="analyzer-three-column">
            {/* Left Column: Input */}
            <div className="column input-column">
              <InputPanel
                onAnalyzeText={handleTextAnalyze}
                onAnalyzeAudio={handleAudioAnalyze}
                loading={loading}
              />
              {error && <div className="error-toast" role="alert">{error}</div>}
            </div>

            {/* Center Column: Transcript & Alternatives */}
            <div className="column transcript-column">
              <TranscriptPanel result={result} loading={loading} />
              {result && (
                <div className="alternatives-wrapper">
                  <AlternativesPanel suggestions={result.suggestions} />
                </div>
              )}
            </div>

            {/* Right Column: Verdict */}
            <div className="column verdict-column">
              <VerdictPanel result={result} loading={loading} />
            </div>
          </div>
        )}

        {activeTab === 'studio' && (
          <div className="full-view-container">
            <RewriteStudioView />
          </div>
        )}

        {activeTab === 'moderation' && (
          <div className="full-view-container">
            <ModerationCenterView />
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="full-view-container">
            <AnalyticsView />
          </div>
        )}

        {activeTab === 'evaluation' && (
          <div className="full-view-container">
            <ModelEvaluationView />
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="full-view-container">
            <SettingsView />
          </div>
        )}
      </main>
    </div>
  );
}

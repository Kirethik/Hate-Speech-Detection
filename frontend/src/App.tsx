import { useState } from 'react';
import { AnalysisResult } from './api/types';
import { apiClient } from './api/client';
import InputPanel from './components/InputPanel/InputPanel';
import TranscriptPanel from './components/TranscriptPanel/TranscriptPanel';
import VerdictPanel from './components/VerdictPanel/VerdictPanel';
import AlternativesPanel from './components/AlternativesPanel/AlternativesPanel';
import './App.css';

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

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

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="logo-area">
          <h1>Civitas AI</h1>
          <span className="badge">Moderator View</span>
        </div>
        <div className="settings-area">
          {import.meta.env.VITE_MOCK === '1' && <span className="mock-badge">Mock Mode</span>}
        </div>
      </header>

      <main className="main-content">
        {/* Left Column: Input */}
        <div className="column input-column">
          <InputPanel 
            onAnalyzeText={handleTextAnalyze} 
            onAnalyzeAudio={handleAudioAnalyze}
            loading={loading}
          />
          {error && <div className="error-toast" role="alert">{error}</div>}
        </div>

        {/* Center Column: Transcript/Waveform */}
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
      </main>
    </div>
  );
}

export default App;

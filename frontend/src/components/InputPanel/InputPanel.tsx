import React, { useState, useRef } from 'react';
import './InputPanel.css';
import './InputPanel-mic.css';

interface InputPanelProps {
  onAnalyzeText: (text: string, langHint: string) => void;
  onAnalyzeAudio: (file: File, langHint: string) => void;
  loading: boolean;
}

const LANGUAGES = [
  { code: 'auto', name: 'Auto-detect' },
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'Hindi (हिन्दी)' },
  { code: 'ta', name: 'Tamil (தமிழ்)' },
  { code: 'te', name: 'Telugu (తెలుగు)' },
  { code: 'ml', name: 'Malayalam (മലയാളം)' },
  { code: 'ur_roman', name: 'Roman Urdu' },
];

export default function InputPanel({ onAnalyzeText, onAnalyzeAudio, loading }: InputPanelProps) {
  const [mode, setMode] = useState<'text' | 'upload' | 'mic'>('mic');
  const [text, setText] = useState('');
  const [lang, setLang] = useState('en'); // Default to English for mic testing
  const [file, setFile] = useState<File | null>(null);
  
  // Mic state
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaRecorder.current = recorder;
      audioChunks.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.current.push(e.data);
      };

      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], 'live-recording.webm', { type: 'audio/webm' });
        const langHint = lang === 'auto' ? '' : lang;
        onAnalyzeAudio(audioFile, langHint);
        // Stop all tracks to release mic
        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      setIsRecording(true);
    } catch (err) {
      alert("Microphone access denied or unavailable.");
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    const langHint = lang === 'auto' ? '' : lang;
    
    if (mode === 'text' && text.trim()) {
      onAnalyzeText(text, langHint);
    } else if (mode === 'upload' && file) {
      onAnalyzeAudio(file, langHint);
    }
  };

  return (
    <div className="input-panel">
      <div className="mode-tabs">
        <button 
          className={mode === 'mic' ? 'active' : ''} 
          onClick={() => setMode('mic')}
        >Live Mic</button>
        <button 
          className={mode === 'text' ? 'active' : ''} 
          onClick={() => setMode('text')}
        >Type Text</button>
        <button 
          className={mode === 'upload' ? 'active' : ''} 
          onClick={() => setMode('upload')}
        >Audio File</button>
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <div className="field-group">
          <label htmlFor="lang-select">Spoken Language</label>
          <select 
            id="lang-select" 
            value={lang} 
            onChange={(e) => setLang(e.target.value)}
          >
            {LANGUAGES.map(l => (
              <option key={l.code} value={l.code}>{l.name}</option>
            ))}
          </select>
        </div>

        {mode === 'mic' && (
          <div className="field-group flex-fill mic-container">
            <div className={`mic-status ${isRecording ? 'recording' : ''}`}>
              {isRecording ? 'Listening...' : 'Ready to record'}
            </div>
            {isRecording ? (
              <button type="button" className="mic-btn stop" onClick={stopRecording}>
                <div className="stop-icon"></div>
                Stop & Analyze
              </button>
            ) : (
              <button type="button" className="mic-btn start" onClick={startRecording} disabled={loading}>
                <div className="mic-icon">🎤</div>
                Start Recording
              </button>
            )}
            <p className="mic-hint">
              Record yourself speaking (e.g. saying a foul sentence). 
              The system will detect it and generate a safe alternative.
            </p>
          </div>
        )}

        {mode === 'text' && (
          <div className="field-group flex-fill">
            <label htmlFor="text-input" className="sr-only">Input text</label>
            <textarea
              id="text-input"
              className="text-input"
              placeholder="Paste or type content to analyze..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={loading}
              autoFocus
            />
          </div>
        )}
        
        {mode === 'upload' && (
          <div className="field-group flex-fill file-drop">
            <input 
              type="file" 
              id="file-input" 
              accept="audio/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              disabled={loading}
            />
            <label htmlFor="file-input" className="file-label">
              {file ? file.name : "Select audio file"}
            </label>
          </div>
        )}

        {mode !== 'mic' && (
          <button 
            type="submit" 
            className="submit-btn" 
            disabled={loading || (mode === 'text' && !text.trim()) || (mode === 'upload' && !file)}
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        )}
      </form>
    </div>
  );
}

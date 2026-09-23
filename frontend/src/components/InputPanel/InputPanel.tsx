import React, { useState } from 'react';
import './InputPanel.css';

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
  const [mode, setMode] = useState<'text' | 'upload'>('text');
  const [text, setText] = useState('');
  const [lang, setLang] = useState('auto');
  const [file, setFile] = useState<File | null>(null);

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

        {mode === 'text' ? (
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
        ) : (
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

        <button 
          type="submit" 
          className="submit-btn" 
          disabled={loading || (mode === 'text' && !text.trim()) || (mode === 'upload' && !file)}
        >
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>
    </div>
  );
}

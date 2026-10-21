import React, { useState } from 'react';
import { apiClient } from '../api/client';
import { RewriteResult } from '../api/types';

const TONES = [
  { id: 'calm', label: 'Calm', desc: 'De-escalating, respectful, and balanced' },
  { id: 'factual', label: 'Factual', desc: 'Objective, data-driven, and focused on verifiable reality' },
  { id: 'firm', label: 'Firm', desc: 'Direct, uncompromising against harm, setting clear boundaries' },
  { id: 'empathetic', label: 'Empathetic', desc: 'Understanding perspective and addressing underlying emotions' },
  { id: 'professional', label: 'Professional', desc: 'Formal, constructive, and suitable for workplace communication' },
];

export default function RewriteStudioView() {
  const [inputText, setInputText] = useState('');
  const [selectedTone, setSelectedTone] = useState('calm');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RewriteResult | null>(null);
  const [editedText, setEditedText] = useState('');
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.rewriteText(inputText, selectedTone);
      setResult(data.rewrites?.length || data.responses?.length ? data : null);
      setEditedText(data.rewrites?.[0]?.text ?? '');
      if (data.warning) setError(data.warning);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rewrite-studio-view">
      <div className="studio-header">
        <h2>Rewrite Studio</h2>
        <p>Rewrite a message in one of five tones. Every suggestion is re-checked by the hate classifiers before it is shown.</p>
      </div>

      <div className="studio-layout">
        <div className="studio-controls">
          <div className="field-group">
            <label htmlFor="studio-input">Original Input Text</label>
            <textarea
              id="studio-input"
              rows={4}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter contentious or harmful statement to rewrite..."
            />
          </div>

          <div className="tone-selector-section">
            <label>Select Target Tone</label>
            <div className="tone-pills">
              {TONES.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  className={`tone-pill ${selectedTone === t.id ? 'active' : ''}`}
                  onClick={() => setSelectedTone(t.id)}
                >
                  <span className="tone-name">{t.label}</span>
                  <span className="tone-desc">{t.desc}</span>
                </button>
              ))}
            </div>
          </div>

          <button
            className="primary-btn generate-btn"
            onClick={handleGenerate}
            disabled={loading || !inputText.trim()}
          >
            {loading ? 'Synthesizing Alternatives...' : `Generate ${selectedTone.toUpperCase()} Alternative`}
          </button>
          {error && <div className="error-toast" role="alert">{error}</div>}
        </div>

        <div className="studio-results">
          {result ? (
            <div className="comparison-card">
              <div className="comparison-header">
                <span className="badge tone-badge">{result.requested_tone.toUpperCase()} TONE</span>
                <span className="badge safety-badge">
                  {result.safety_status === 'verified_safe' ? 'Safety check: passed' : 'Safety check: needs review'}
                </span>
              </div>

              <div className="comparison-split">
                <div className="split-side original-side">
                  <h4>Original Content</h4>
                  <div className="text-box original-box">{result.original_text}</div>
                </div>

                <div className="split-side rewritten-side">
                  <h4>Constructive Alternative (Editable)</h4>
                  <textarea
                    className="editable-rewrite"
                    rows={4}
                    value={editedText}
                    onChange={(e) => setEditedText(e.target.value)}
                  />
                  <div className="card-actions">
                    <button className="secondary-btn" onClick={() => handleCopy(editedText)}>
                      {copied ? '✓ Copied to Clipboard' : 'Copy Rewrite'}
                    </button>
                  </div>
                </div>
              </div>

              {result.responses && result.responses.length > 0 && (
                <div className="bystander-response-section">
                  <h4>Bystander / Moderator Counter-Narrative</h4>
                  <div className="text-box response-box">{result.responses[0].text}</div>
                  <button className="secondary-btn small-btn" onClick={() => handleCopy(result.responses[0].text)}>
                    Copy Counter-Narrative
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="empty-studio-placeholder">
              <p>Select a tone and click <strong>Generate Alternative</strong> to view side-by-side comparison and safety validation.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

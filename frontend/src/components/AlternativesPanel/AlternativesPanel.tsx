import React, { useState } from 'react';
import { Suggestions, Suggestion } from '../../api/types';
import './AlternativesPanel.css';

interface Props {
  suggestions: Suggestions;
}

export default function AlternativesPanel({ suggestions }: Props) {
  const [copiedText, setCopiedText] = useState<string | null>(null);

  if (suggestions.fallback && suggestions.rewrite.length === 0 && suggestions.respond.length === 0) {
    return (
      <div className="alternatives-panel">
        <p className="fallback-msg">Model B could not generate safe alternatives. Please moderate manually.</p>
      </div>
    );
  }

  if (suggestions.rewrite.length === 0 && suggestions.respond.length === 0) {
    return null; // Not hateful/offensive, or model didn't run
  }

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(text);
    setTimeout(() => setCopiedText(null), 2000);
  };

  const SuggestionCard = ({ sug }: { sug: Suggestion }) => (
    <div className="suggestion-card" lang={sug.language}>
      <p className="suggestion-text">{sug.text}</p>
      <div className="suggestion-actions">
        <button 
          className="action-btn"
          onClick={() => handleCopy(sug.text)}
          title="Copy to clipboard"
        >
          {copiedText === sug.text ? '✓ Copied' : 'Copy'}
        </button>
        {/* Placeholder for TTS integration */}
        {/* <button className="action-btn">Play</button> */}
      </div>
    </div>
  );

  return (
    <div className="alternatives-panel">
      <div className="suggestions-grid">
        {suggestions.rewrite.length > 0 && (
          <div className="suggestion-group">
            <h3>Say it differently (Rewrite)</h3>
            <div className="suggestion-list">
              {suggestions.rewrite.map((s, i) => <SuggestionCard key={`rw-${i}`} sug={s} />)}
            </div>
          </div>
        )}

        {suggestions.respond.length > 0 && (
          <div className="suggestion-group">
            <h3>Reply with (Counter-narrative)</h3>
            <div className="suggestion-list">
              {suggestions.respond.map((s, i) => <SuggestionCard key={`rsp-${i}`} sug={s} />)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

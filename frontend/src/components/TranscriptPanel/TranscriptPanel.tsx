import React, { useMemo, useState } from 'react';
import { AnalysisResult, RationaleSpan } from '../../api/types';
import './TranscriptPanel.css';

interface Props {
  result: AnalysisResult | null;
  loading: boolean;
}

export default function TranscriptPanel({ result, loading }: Props) {
  const [blur, setBlur] = useState(true);

  // Map text to highlighted segments
  const segments = useMemo(() => {
    if (!result) return [];
    const text = result.input.text;
    const spans = [...result.detection.rationale].sort((a, b) => a.start_char - b.start_char);
    
    const parts = [];
    let cursor = 0;

    for (const span of spans) {
      if (span.start_char > cursor) {
        parts.push({ text: text.substring(cursor, span.start_char), isRationale: false });
      }
      parts.push({
        text: text.substring(span.start_char, span.end_char),
        isRationale: true,
        score: span.score
      });
      cursor = span.end_char;
    }
    
    if (cursor < text.length) {
      parts.push({ text: text.substring(cursor), isRationale: false });
    }
    
    return parts;
  }, [result]);

  if (loading) {
    return (
      <div className="transcript-panel loading-state">
        <div className="spinner"></div>
        <p>Analyzing context...</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="transcript-panel empty-state">
        <p>Awaiting input</p>
      </div>
    );
  }

  const isHate = result.detection.is_hate;
  const isOffensive = result.detection.severity.label === 'offensive_profanity';
  
  let headerClass = 'safe';
  if (isHate) headerClass = 'hate';
  else if (isOffensive) headerClass = 'offensive';

  return (
    <div className="transcript-panel">
      <div className={`transcript-header ${headerClass}`}>
        <h2>{result.input.mode === 'audio' ? 'Audio Transcript' : 'Source Text'}</h2>
        
        {isHate && (
          <button 
            className="blur-toggle" 
            onClick={() => setBlur(!blur)}
          >
            {blur ? 'Reveal Content' : 'Hide Content'}
          </button>
        )}
      </div>

      <div className="transcript-body">
        {result.input.mode === 'audio' && (
          <div className="waveform-placeholder">
            {/* WaveSurfer mounts here when audio is available */}
            <div className="wave-line"></div>
          </div>
        )}

        <div 
          className={`text-content ${blur && isHate ? 'blurred' : ''}`}
          data-script={result.input.script}
          lang={result.input.language}
        >
          {segments.map((seg, i) => (
            <span 
              key={i} 
              className={seg.isRationale ? 'rationale-highlight' : ''}
              style={seg.isRationale ? { backgroundColor: `rgba(155, 44, 44, ${Math.max(0.2, seg.score * 0.6)})` } : {}}
              title={seg.isRationale ? `Attention score: ${(seg.score * 100).toFixed(0)}%` : undefined}
            >
              {seg.text}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

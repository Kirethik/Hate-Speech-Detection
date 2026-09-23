import React from 'react';
import { AnalysisResult } from '../../api/types';
import './VerdictPanel.css';

interface Props {
  result: AnalysisResult | null;
  loading: boolean;
}

export default function VerdictPanel({ result, loading }: Props) {
  if (loading || !result) {
    return <div className="verdict-panel empty"></div>;
  }

  const { detection, timing_ms } = result;
  
  let severityDisplay = 'No issue found';
  let severityClass = 'safe';
  
  if (detection.severity.label === 'hate') {
    severityDisplay = 'Hate Speech';
    severityClass = 'hate';
  } else if (detection.severity.label === 'offensive_profanity') {
    severityDisplay = 'Offensive / Profane';
    severityClass = 'offensive';
  }

  return (
    <div className="verdict-panel" role="region" aria-live="polite">
      <div className={`verdict-header ${severityClass}`}>
        <h2 className="verdict-title">{severityDisplay}</h2>
        <div className="confidence">
          {(detection.hate_prob * 100).toFixed(0)}% confidence
        </div>
      </div>

      <div className="verdict-body">
        
        <div className="info-block">
          <h3>Target Group</h3>
          <p className="target-value">
            {detection.target.label !== 'none' ? detection.target.label : 'None detected'}
          </p>
        </div>

        {detection.second_stage.nli_ran && (
          <div className="info-block nli-block">
            <h3>Second-Stage Review</h3>
            <p>Score fell in uncertainty band. NLI model verified implicit nature.</p>
            {detection.second_stage.dehumanization_hits.length > 0 && (
              <p className="dehumanization-alert">
                Dehumanizing terms found: 
                <strong> {detection.second_stage.dehumanization_hits.join(', ')}</strong>
              </p>
            )}
          </div>
        )}

        <details className="technical-details">
          <summary>Model Diagnostics</summary>
          <div className="details-content">
            <div className="detail-row">
              <span>Decision Path:</span>
              <code>{detection.decision_path}</code>
            </div>
            
            <div className="detail-table">
              <h4>Severity Probabilities</h4>
              {Object.entries(detection.severity.probs).map(([label, prob]) => (
                <div className="detail-row" key={label}>
                  <span>{label}:</span>
                  <span>{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>

            <div className="detail-table">
              <h4>Timing</h4>
              <div className="detail-row"><span>Model A (ONNX):</span> <span>{timing_ms.model_a}ms</span></div>
              {timing_ms.nli && <div className="detail-row"><span>NLI Stage:</span> <span>{timing_ms.nli}ms</span></div>}
              {timing_ms.model_b && <div className="detail-row"><span>Model B:</span> <span>{timing_ms.model_b}ms</span></div>}
            </div>
          </div>
        </details>
      </div>
    </div>
  );
}

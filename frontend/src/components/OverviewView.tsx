interface Props {
  onGoToAnalyzer: () => void;
  isDemoMode: boolean;
}

export default function OverviewView({ onGoToAnalyzer, isDemoMode }: Props) {
  return (
    <div className="overview-view">
      <div className="overview-hero">
        <h2>Multilingual hate speech detection and constructive alternatives, for text and speech</h2>
        <p className="hero-subtitle">
          Civitas finds hate speech in English, Hindi, Urdu, Tamil, Telugu, Malayalam and code-mixed speech,
          explains the decision, suggests what could be said instead, and can re-voice recordings in the
          speaker's own voice. Every model runs locally.
        </p>
        <div className="hero-cta">
          <button className="primary-btn" onClick={onGoToAnalyzer}>Open the Live Analyzer →</button>
          <span className={`status-pill ${isDemoMode ? 'demo' : 'real'}`}>
            {isDemoMode ? '● Models not ready: see Settings' : '● Models ready'}
          </span>
        </div>
      </div>

      <div className="overview-grid">
        <div className="overview-card">
          <div className="card-badge">Detect</div>
          <h3>Two-stage detector</h3>
          <p>Fast pretrained classifiers screen every sentence; a local LLM judges the borderline ones in context.</p>
          <ul>
            <li><strong>Classifiers</strong>: XLM-RoBERTa toxicity model plus MuRIL trained on Indic abusive speech.</li>
            <li><strong>LLM judge</strong>: catches sarcasm, quoting and implicit hate, and names the target group.</li>
            <li><strong>Rationale</strong>: the exact words that made the sentence hateful.</li>
          </ul>
          <div className="card-footer"><span>Only borderline sentences reach the LLM, which keeps it fast</span></div>
        </div>

        <div className="overview-card">
          <div className="card-badge">Rewrite</div>
          <h3>Rewrites and counter-narratives</h3>
          <p>The same LLM writes alternatives in the original language and script:</p>
          <ul>
            <li><strong>Rewrite</strong>: keeps the speaker's legitimate point and removes the hate.</li>
            <li><strong>Counter-narrative</strong>: a calm reply that challenges the hateful idea.</li>
            <li><strong>Five tones</strong>: calm, factual, firm, empathetic, professional.</li>
            <li><strong>Safety gate</strong>: every suggestion is re-scored by the classifiers before it is shown.</li>
          </ul>
          <div className="card-footer"><span>Suggestions that still score as toxic are never shown</span></div>
        </div>

        <div className="overview-card workflow-card">
          <div className="card-badge">Speech</div>
          <h3>From recording to cleaned audio</h3>
          <div className="workflow-steps">
            <div className="step-item"><span className="step-num">1</span><div><strong>Transcribe</strong>
              <p>Whisper, or IndicConformer for Dravidian languages, with sentence timings.</p></div></div>
            <div className="step-item"><span className="step-num">2</span><div><strong>Detect</strong>
              <p>Each sentence is screened and judged as above.</p></div></div>
            <div className="step-item"><span className="step-num">3</span><div><strong>Re-voice</strong>
              <p>IndicF5 or Chatterbox speaks the replacement in the speaker's own voice.</p></div></div>
            <div className="step-item"><span className="step-num">4</span><div><strong>Splice</strong>
              <p>Cut at pauses, matched in loudness, background kept, AI audio watermarked.</p></div></div>
            <div className="step-item"><span className="step-num">5</span><div><strong>Review</strong>
              <p>Every edit is listed and marked on the waveform; moderators can give feedback.</p></div></div>
          </div>
        </div>
      </div>
    </div>
  );
}

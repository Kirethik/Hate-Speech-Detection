import { useEffect, useState } from 'react';
import { apiClient, EvalReport } from '../api/client';

const LANG: Record<string, string> = {
  en: 'English', hi: 'Hindi', ur: 'Urdu', ur_roman: 'Roman Urdu', ta: 'Tamil', te: 'Telugu',
  ml: 'Malayalam', kn: 'Kannada',
};
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export default function ModelEvaluationView() {
  const [r, setR] = useState<EvalReport | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => { apiClient.fetchEvaluation().then(setR).finally(() => setLoading(false)); }, []);

  if (loading) return <div className="loading-state">Loading evaluation…</div>;

  if (!r) {
    return (
      <div className="evaluation-view">
        <div className="eval-header">
          <h2>Model evaluation</h2>
          <p>No evaluation has been run yet. The numbers on this page always come from a real run on your labelled data.</p>
        </div>
        <div className="eval-card">
          <h3>Run it</h3>
          <p>From the project folder:</p>
          <pre><code>.venv\Scripts\python.exe scripts\evaluate.py --split val --per-language 300 --with-llm</code></pre>
          <p>
            Use <code>--split val</code> while tuning. Run <code>--split test</code> once, at the end, for the
            numbers you report. Then refresh this page.
          </p>
        </div>
      </div>
    );
  }

  const o = r.overall;
  return (
    <div className="evaluation-view">
      <div className="eval-header">
        <h2>Model evaluation</h2>
        <p>
          {o.n.toLocaleString()} labelled examples from the <strong>{r.split}</strong> split (up to {r.per_language_limit} per language),
          run on {new Date(r.created_at).toLocaleString()} in {Math.round(r.seconds)} s.
          {' '}Detector: {r.classifiers.join(' + ')}{r.with_llm ? ` + LLM judge (${r.llm_model})` : ' (classifiers only)'}.
        </p>
      </div>

      <div className="eval-sections">
        <div className="eval-card">
          <div className="eval-card-header">
            <h3>Hate detection</h3>
            <span className="badge">threshold {r.threshold.toFixed(2)}</span>
          </div>
          <div className="eval-metrics-grid">
            <div className="eval-metric-box"><span className="eval-val">{pct(o.accuracy)}</span><span className="eval-label">Accuracy</span></div>
            <div className="eval-metric-box"><span className="eval-val">{o.macro_f1.toFixed(3)}</span><span className="eval-label">Macro-F1</span></div>
            <div className="eval-metric-box"><span className="eval-val">{pct(o.recall)}</span><span className="eval-label">Recall (hate found)</span></div>
            <div className="eval-metric-box"><span className="eval-val">{pct(o.fpr)}</span><span className="eval-label">False positive rate</span></div>
          </div>

          <div className="confusion-matrix-container">
            <h4>Confusion matrix</h4>
            <div className="cm-table">
              <div className="cm-row cm-header"><span>Actual \ Predicted</span><span>Not hate</span><span>Hate</span></div>
              <div className="cm-row"><span className="cm-label">Not hate</span>
                <span className="cm-cell tn">{o.tn.toLocaleString()} (TN)</span><span className="cm-cell fp">{o.fp.toLocaleString()} (FP)</span></div>
              <div className="cm-row"><span className="cm-label">Hate</span>
                <span className="cm-cell fn">{o.fn.toLocaleString()} (FN)</span><span className="cm-cell tp">{o.tp.toLocaleString()} (TP)</span></div>
            </div>
          </div>

          <div className="lang-perf-table">
            <h4>By language</h4>
            <table>
              <thead><tr><th>Language</th><th>Examples</th><th>Accuracy</th><th>Macro-F1</th><th>Recall</th><th>FPR</th></tr></thead>
              <tbody>
                {Object.entries(r.per_language).map(([k, m]) => (
                  <tr key={k}><td>{LANG[k] ?? k}</td><td>{m.n}</td><td>{pct(m.accuracy)}</td>
                    <td>{m.macro_f1.toFixed(3)}</td><td>{pct(m.recall)}</td><td>{pct(m.fpr)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {r.rewrite && (
          <div className="eval-card">
            <div className="eval-card-header"><h3>Rewrite safety</h3><span className="badge">{r.llm_model}</span></div>
            <div className="eval-metrics-grid">
              <div className="eval-metric-box"><span className="eval-val">{r.rewrite.n}</span><span className="eval-label">Hateful inputs rewritten</span></div>
              <div className="eval-metric-box"><span className="eval-val">{pct(r.rewrite.safe_rate)}</span><span className="eval-label">Rewrites passing the safety gate</span></div>
              <div className="eval-metric-box"><span className="eval-val">{r.rewrite.avg_hate_prob.toFixed(3)}</span><span className="eval-label">Mean toxicity of rewrites</span></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

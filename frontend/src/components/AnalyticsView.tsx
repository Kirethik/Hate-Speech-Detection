import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { AnalyticsData } from '../api/types';

export default function AnalyticsView() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await apiClient.fetchAnalytics());
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (error) {
    return <div className="error-toast" role="alert">{error} <button className="secondary-btn small-btn" onClick={loadAnalytics}>Retry</button></div>;
  }

  if (loading || !data) {
    return <div className="loading-state">Loading real-time analytics...</div>;
  }

  return (
    <div className="analytics-view">
      <div className="analytics-header">
        <h2>System Telemetry & Moderation Analytics</h2>
        <p>Quantitative overview of live traffic, detection rates, and multilingual volume.</p>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <span className="kpi-label">Total Utterances Analyzed</span>
          <span className="kpi-value">{data.total_analyses}</span>
          <span className="kpi-sub">Across text, audio upload, and live mic</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Hate Speech Flagged</span>
          <span className="kpi-value hate-accent">{data.hateful_detected}</span>
          <span className="kpi-sub">{(data.hate_ratio * 100).toFixed(1)}% detection rate</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Clean Utterances</span>
          <span className="kpi-value safe-accent">{data.clean_detected}</span>
          <span className="kpi-sub">{((1 - data.hate_ratio) * 100).toFixed(1)}% cleared without intervention</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Human Feedback Submissions</span>
          <span className="kpi-value">{data.total_feedback_submissions}</span>
          <span className="kpi-sub">Moderator audits & quality reports</span>
        </div>
      </div>

      <div className="analytics-breakdowns">
        <div className="breakdown-card">
          <h3>Multilingual Volume Breakdown</h3>
          <div className="lang-bar-list">
            {Object.entries(data.language_distribution).map(([lang, count]) => {
              const pct = data.total_analyses > 0 ? (count / data.total_analyses) * 100 : 0;
              return (
                <div key={lang} className="lang-bar-row">
                  <div className="lang-label">
                    <span className="lang-code">{lang.toUpperCase()}</span>
                    <span>{count} scans ({pct.toFixed(0)}%)</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${Math.max(5, pct)}%` }}></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="breakdown-card">
          <h3>Dual-Model Safety Guarantee</h3>
          <div className="safety-guarantee-box">
            <p>
              <strong>100% of generated alternatives</strong> are re-checked by the hate classifiers before they are shown.
              Uncertain generations are either converted to masked rewrites or safe, grounded templates.
            </p>
            <div className="metric-badge-row">
              <span className="badge">Gated Generation: Active</span>
              <span className="badge">NLI Cascading: Active</span>
              <span className="badge">Lexicon Check: Active</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

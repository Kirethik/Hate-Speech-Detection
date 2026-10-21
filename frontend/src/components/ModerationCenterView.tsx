import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { HistoryItem } from '../api/types';

export default function ModerationCenterView() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    try {
      setHistory(await apiClient.fetchHistory(20));
    } catch (e: any) {
      setActionNotice(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAction = async (itemId: string, actionType: string, isCorrectHate: boolean) => {
    try {
      await apiClient.submitFeedback({
        analysis_id: itemId,
        correct_label: isCorrectHate,
        rating: actionType,
        note: `Moderator action: ${actionType}`
      });
      setActionNotice(`Action recorded: ${actionType} for item ${itemId.slice(0, 8)}`);
    } catch (e: any) {
      setActionNotice(`Could not save: ${e.message}`);
    }
    setTimeout(() => setActionNotice(null), 3000);
  };

  return (
    <div className="moderation-center-view">
      <div className="center-header">
        <h2>Human-in-the-Loop Moderation Center</h2>
        <p>Review flagged incidents, inspect rationale highlights, and approve or override AI recommendations.</p>
        {actionNotice && <div className="action-toast">{actionNotice}</div>}
      </div>

      <div className="moderation-table-container">
        {loading ? (
          <div className="loading-state">Loading moderation queue...</div>
        ) : history.length === 0 ? (
          <div className="empty-queue">
            <p>No recent items in the moderation queue. Run analyses in the Live Analyzer or Rewrite Studio to populate the queue.</p>
          </div>
        ) : (
          <table className="moderation-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Content</th>
                <th>Language</th>
                <th>AI Verdict</th>
                <th>Hate Confidence</th>
                <th>Moderator Actions</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr key={item.id}>
                  <td className="timestamp-cell">{new Date(item.created_at).toLocaleTimeString()}</td>
                  <td className="text-cell">{item.text}</td>
                  <td><span className="lang-tag">{item.language.toUpperCase()}</span></td>
                  <td>
                    <span className={`verdict-pill ${item.verdict === 'hate' ? 'hate' : 'clean'}`}>
                      {item.verdict === 'hate' ? 'Hateful' : 'Clean'}
                    </span>
                  </td>
                  <td>{(item.hate_prob * 100).toFixed(1)}%</td>
                  <td className="actions-cell">
                    <button
                      className="mod-btn approve-btn"
                      title="Approve AI moderation"
                      onClick={() => handleAction(item.id, 'Approved', item.verdict === 'hate')}
                    >
                      ✓ Approve
                    </button>
                    <button
                      className="mod-btn override-btn"
                      title="Mark as benign false positive"
                      onClick={() => handleAction(item.id, 'Marked Benign (False Positive)', false)}
                    >
                      Override
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

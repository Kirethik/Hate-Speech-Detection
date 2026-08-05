import { useState } from 'react'
import './index.css'

function App() {
  const [text, setText] = useState('')
  const [language, setLanguage] = useState('en')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleAnalyze = async (e) => {
    e.preventDefault()
    if (!text.trim()) return

    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/moderate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, language }),
      })
      
      if (!response.ok) {
        throw new Error('Failed to fetch from API')
      }
      
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'An error occurred during moderation.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header>
        <h1>Civitas AI</h1>
        <p>Enterprise Dual-Node Moderation System</p>
      </header>

      <main>
        {/* Left Column: Input Form */}
        <section className="glass-panel">
          <form onSubmit={handleAnalyze}>
            <div className="input-group">
              <label htmlFor="text-input">Input Text</label>
              <textarea
                id="text-input"
                placeholder="Enter text to moderate (e.g. 'You people are ruining this country')"
                value={text}
                onChange={(e) => setText(e.target.value)}
                required
              />
            </div>
            
            <div className="input-group">
              <label htmlFor="language-select">Language</label>
              <select
                id="language-select"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              >
                <option value="en">English</option>
                <option value="hi">Hindi (हिन्दी)</option>
                <option value="ta">Tamil (தமிழ்)</option>
              </select>
            </div>

            <button type="submit" className="btn-primary" disabled={loading || !text.trim()}>
              {loading ? (
                <>
                  <span className="spinner"></span> Analyzing...
                </>
              ) : (
                'Analyze Text'
              )}
            </button>
          </form>
          
          {error && (
            <div style={{ color: 'var(--danger-color)', marginTop: '1rem', textAlign: 'center' }}>
              ⚠️ {error}
            </div>
          )}
        </section>

        {/* Right Column: Results */}
        <section className="results-container">
          {!result && !loading && (
            <div className="results-placeholder">
              <p>Submit text to see Model A's detection & Model B's counter-narratives.</p>
            </div>
          )}

          {result && (
            <div className="glass-panel">
              {result.status === 'safe' ? (
                <div>
                  <div className="status-badge safe">✓ Safe Content</div>
                  <p style={{ color: 'var(--text-secondary)' }}>
                    Model A classified this text as safe. No counter-narratives were required from Model B.
                  </p>
                </div>
              ) : (
                <div>
                  <div className="status-badge abusive">🚨 Abusive Content Detected</div>
                  
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-label">Target Group</div>
                      <div className="metric-value">{result.target || 'Unknown'}</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Abuse Probability</div>
                      <div className="metric-value">{(result.p_abuse * 100).toFixed(1)}%</div>
                    </div>
                    <div className="metric-card" style={{ gridColumn: 'span 2' }}>
                      <div className="metric-label">Severity Score</div>
                      <div className="metric-value">{(result.severity * 100).toFixed(1)} / 100</div>
                    </div>
                  </div>

                  <div className="alternatives-section">
                    <h3>💡 Model B Counter-Narratives</h3>
                    {result.alternatives && result.alternatives.length > 0 ? (
                      <div className="alternatives-list">
                        {result.alternatives.map((alt, idx) => (
                          <div key={idx} className="alternative-item">
                            {alt}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p style={{ color: 'var(--text-secondary)' }}>No alternatives generated.</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App

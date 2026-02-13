#!/usr/bin/env python3
"""
Live Web Dashboard for Soliter Training Monitoring

Monitors training progress in real-time via web browser.
Runs as separate process - no impact on training performance.

Usage:
    python scripts/monitor_training_live.py experiments/run1 --refresh 5 --port 8050
    
Then open: http://localhost:8050
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from flask import Flask, render_template_string, jsonify
import threading

# HTML template with embedded JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Soliter Training Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            padding: 30px;
        }
        h1 {
            color: #764ba2;
            margin-top: 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .status {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }
        .status.running {
            background: #4CAF50;
            color: white;
            animation: pulse 2s infinite;
        }
        .status.complete {
            background: #2196F3;
            color: white;
        }
        .status.error {
            background: #f44336;
            color: white;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .metric-label {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
        }
        .metric-change {
            font-size: 14px;
            margin-top: 5px;
        }
        .metric-change.positive {
            color: #90EE90;
        }
        .metric-change.negative {
            color: #FFB6C1;
        }
        .chart-container {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .info-text {
            color: #666;
            font-size: 14px;
            margin-top: 20px;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 6px;
        }
        .refresh-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4CAF50;
            margin-left: 10px;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0%, 50%, 100% { opacity: 1; }
            25%, 75% { opacity: 0.3; }
        }
        .control-panel {
            display: flex;
            gap: 15px;
            align-items: center;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            background: #667eea;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        button:hover {
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        select {
            padding: 8px 12px;
            border-radius: 6px;
            border: 2px solid #667eea;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>
            <span>🧠 Soliter Training Monitor</span>
            <span class="status running" id="status">RUNNING<span class="refresh-indicator"></span></span>
        </h1>
        
        <div class="control-panel">
            <label for="refresh-rate">Refresh Rate:</label>
            <select id="refresh-rate" onchange="updateRefreshRate()">
                <option value="1">1 second</option>
                <option value="5" selected>5 seconds</option>
                <option value="10">10 seconds</option>
                <option value="30">30 seconds</option>
                <option value="60">60 seconds</option>
            </select>
            <button onclick="forceRefresh()">Refresh Now</button>
            <button onclick="toggleAutoRefresh()" id="toggle-btn">Pause</button>
        </div>
        
        <div class="metrics-grid" id="metrics">
            <!-- Populated by JavaScript -->
        </div>
        
        <div class="chart-container">
            <div id="consumption-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="life-duration-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="buffer-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="policy-loss-chart"></div>
        </div>
        
        <div class="info-text" id="info">
            Loading training data...
        </div>
    </div>
    
    <script>
        let refreshInterval = 5000; // milliseconds
        let autoRefresh = true;
        let refreshTimer = null;
        
        function updateRefreshRate() {
            const select = document.getElementById('refresh-rate');
            refreshInterval = parseInt(select.value) * 1000;
            if (autoRefresh) {
                clearInterval(refreshTimer);
                refreshTimer = setInterval(fetchData, refreshInterval);
            }
        }
        
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const btn = document.getElementById('toggle-btn');
            if (autoRefresh) {
                btn.textContent = 'Pause';
                refreshTimer = setInterval(fetchData, refreshInterval);
            } else {
                btn.textContent = 'Resume';
                clearInterval(refreshTimer);
            }
        }
        
        function forceRefresh() {
            fetchData();
        }
        
        function fetchData() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('status').textContent = 'ERROR';
                        document.getElementById('status').className = 'status error';
                        document.getElementById('info').textContent = data.error;
                        return;
                    }
                    
                    updateMetrics(data.metrics);
                    updateCharts(data.charts);
                    updateInfo(data.info);
                    
                    // Update status
                    const status = document.getElementById('status');
                    if (data.info.is_complete) {
                        status.innerHTML = 'COMPLETE';
                        status.className = 'status complete';
                        if (autoRefresh) {
                            toggleAutoRefresh();
                        }
                    }
                })
                .catch(error => {
                    console.error('Fetch error:', error);
                });
        }
        
        function updateMetrics(metrics) {
            const container = document.getElementById('metrics');
            container.innerHTML = '';
            
            metrics.forEach(metric => {
                const card = document.createElement('div');
                card.className = 'metric-card';
                
                let changeHTML = '';
                if (metric.change !== null) {
                    const changeClass = metric.change >= 0 ? 'positive' : 'negative';
                    const changeSymbol = metric.change >= 0 ? '↑' : '↓';
                    changeHTML = `<div class="metric-change ${changeClass}">${changeSymbol} ${Math.abs(metric.change).toFixed(1)}%</div>`;
                }
                
                card.innerHTML = `
                    <div class="metric-label">${metric.label}</div>
                    <div class="metric-value">${metric.value}</div>
                    ${changeHTML}
                `;
                container.appendChild(card);
            });
        }
        
        function updateCharts(charts) {
            // Consumption Rate
            Plotly.newPlot('consumption-chart', charts.consumption.data, {
                title: 'Consumption Rate Over Time',
                xaxis: { title: 'Cycle' },
                yaxis: { title: 'Consumptions per Cycle' },
                showlegend: true
            }, {responsive: true});
            
            // Life Duration
            Plotly.newPlot('life-duration-chart', charts.life_duration.data, {
                title: 'Life Duration Trend',
                xaxis: { title: 'Death Index' },
                yaxis: { title: 'Life Duration (ticks)' },
                showlegend: true
            }, {responsive: true});
            
            // Buffer Size
            Plotly.newPlot('buffer-chart', charts.buffer.data, {
                title: 'Memory Buffer Size',
                xaxis: { title: 'Cycle' },
                yaxis: { title: 'Buffer Size (transitions)' },
                showlegend: true
            }, {responsive: true});
            
            // Policy Loss
            Plotly.newPlot('policy-loss-chart', charts.policy_loss.data, {
                title: 'Policy Loss (Learning Curve)',
                xaxis: { title: 'Cycle' },
                yaxis: { title: 'Policy Loss', type: 'log' },
                showlegend: true
            }, {responsive: true});
        }
        
        function updateInfo(info) {
            const infoDiv = document.getElementById('info');
            infoDiv.innerHTML = `
                <strong>Training Progress:</strong> ${info.current_cycle} / ${info.total_cycles} cycles (${info.progress}%)<br>
                <strong>Elapsed Time:</strong> ${info.elapsed_time}<br>
                <strong>ETA:</strong> ${info.eta}<br>
                <strong>Deaths:</strong> ${info.total_deaths}<br>
                <strong>Consumptions:</strong> ${info.total_consumptions.toLocaleString()}
            `;
        }
        
        // Initial load
        fetchData();
        refreshTimer = setInterval(fetchData, refreshInterval);
    </script>
</body>
</html>
"""

class TrainingMonitor:
    def __init__(self, log_dir, refresh_rate=5):
        self.log_dir = Path(log_dir)
        self.refresh_rate = refresh_rate
        self.log_file = None
        self.last_mtime = 0
        self.data = None
        self.start_time = None
        
    def find_log_file(self):
        """Find the training log JSON file."""
        json_files = list(self.log_dir.glob('training_*.json'))
        if not json_files:
            return None
        # Get most recent
        return max(json_files, key=lambda p: p.stat().st_mtime)
    
    def load_data(self):
        """Load and parse training data."""
        if self.log_file is None:
            self.log_file = self.find_log_file()
            if self.log_file is None:
                return None
        
        # Check if file has been modified
        try:
            current_mtime = self.log_file.stat().st_mtime
            if current_mtime == self.last_mtime:
                return self.data  # Use cached data
            
            self.last_mtime = current_mtime
            
            with open(self.log_file) as f:
                self.data = json.load(f)
            
            if self.start_time is None:
                self.start_time = datetime.now()
            
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def compute_metrics(self):
        """Compute summary metrics for display."""
        if self.data is None:
            return []
        
        cycles = self.data['total_cycles']
        deaths = self.data['deaths']
        sleeps = self.data['sleeps']
        
        metrics = []
        
        # Current cycle
        current_cycle = sleeps[-1]['cycle'] if sleeps else 0
        metrics.append({
            'label': 'Current Cycle',
            'value': f"{current_cycle} / {cycles}",
            'change': None
        })
        
        # Total consumptions
        total_consumptions = sleeps[-1]['total_consumptions'] if sleeps else 0
        metrics.append({
            'label': 'Total Consumptions',
            'value': f"{total_consumptions:,}",
            'change': None
        })
        
        # Consumption rate (recent)
        if len(sleeps) >= 10:
            recent_rates = []
            for i in range(len(sleeps) - 10, len(sleeps)):
                if i > 0:
                    rate = sleeps[i]['total_consumptions'] - sleeps[i-1]['total_consumptions']
                    recent_rates.append(rate)
            avg_rate = np.mean(recent_rates) if recent_rates else 0
            
            # Compare to earlier
            if len(sleeps) >= 20:
                early_rates = []
                for i in range(max(1, len(sleeps) - 20), len(sleeps) - 10):
                    rate = sleeps[i]['total_consumptions'] - sleeps[i-1]['total_consumptions']
                    early_rates.append(rate)
                early_avg = np.mean(early_rates) if early_rates else 1
                change = ((avg_rate - early_avg) / early_avg * 100) if early_avg > 0 else 0
            else:
                change = None
            
            metrics.append({
                'label': 'Recent Consumption Rate',
                'value': f"{avg_rate:.1f}/cycle",
                'change': change
            })
        
        # Average life duration (recent)
        if deaths:
            recent_deaths = deaths[-20:] if len(deaths) >= 20 else deaths
            avg_life = np.mean([d['life_duration'] for d in recent_deaths])
            
            # Compare to earlier
            if len(deaths) >= 40:
                early_deaths = deaths[-40:-20]
                early_life = np.mean([d['life_duration'] for d in early_deaths])
                change = ((avg_life - early_life) / early_life * 100) if early_life > 0 else 0
            else:
                change = None
            
            metrics.append({
                'label': 'Avg Life Duration',
                'value': f"{avg_life:.0f} ticks",
                'change': change
            })
        
        # Buffer size
        if sleeps:
            buffer_size = sleeps[-1]['buffer_size']
            metrics.append({
                'label': 'Buffer Size',
                'value': f"{buffer_size:,}",
                'change': None
            })
        
        # Action std (exploration)
        if sleeps:
            action_std = sleeps[-1]['action_std']
            metrics.append({
                'label': 'Exploration (std)',
                'value': f"{action_std:.4f}",
                'change': None
            })
        
        return metrics
    
    def compute_charts(self):
        """Compute chart data."""
        if self.data is None:
            return {}
        
        sleeps = self.data['sleeps']
        deaths = self.data['deaths']
        
        charts = {}
        
        # Consumption rate
        cycles = [s['cycle'] for s in sleeps]
        consumptions = [s['total_consumptions'] for s in sleeps]
        rates = [consumptions[i] - consumptions[i-1] if i > 0 else 0 
                 for i in range(len(consumptions))]
        
        # Smoothed
        window = 10
        if len(rates) > window:
            rates_smooth = np.convolve(rates, np.ones(window)/window, mode='valid')
            cycles_smooth = cycles[window-1:]
        else:
            rates_smooth = rates
            cycles_smooth = cycles
        
        charts['consumption'] = {
            'data': [
                {
                    'x': cycles,
                    'y': rates,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Raw',
                    'line': {'color': 'lightblue'},
                    'opacity': 0.3
                },
                {
                    'x': list(cycles_smooth),
                    'y': list(rates_smooth),
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Smoothed',
                    'line': {'color': 'blue', 'width': 3}
                }
            ]
        }
        
        # Life duration
        death_indices = list(range(len(deaths)))
        life_durations = [d['life_duration'] for d in deaths]
        
        charts['life_duration'] = {
            'data': [
                {
                    'x': death_indices,
                    'y': life_durations,
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Deaths',
                    'marker': {'color': 'red', 'size': 5, 'opacity': 0.5}
                }
            ]
        }
        
        # Add trend line
        if len(death_indices) > 10:
            z = np.polyfit(death_indices, life_durations, 1)
            p = np.poly1d(z)
            charts['life_duration']['data'].append({
                'x': death_indices,
                'y': [p(x) for x in death_indices],
                'type': 'scatter',
                'mode': 'lines',
                'name': f'Trend (slope={z[0]:.1f})',
                'line': {'color': 'darkred', 'width': 2, 'dash': 'dash'}
            })
        
        # Buffer size
        buffer_sizes = [s['buffer_size'] for s in sleeps]
        charts['buffer'] = {
            'data': [
                {
                    'x': cycles,
                    'y': buffer_sizes,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Buffer Size',
                    'line': {'color': 'green', 'width': 2}
                }
            ]
        }
        
        # Policy loss
        policy_losses = [s['policy_loss'] for s in sleeps]
        charts['policy_loss'] = {
            'data': [
                {
                    'x': cycles,
                    'y': policy_losses,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Policy Loss',
                    'line': {'color': 'purple', 'width': 2}
                }
            ]
        }
        
        return charts
    
    def compute_info(self):
        """Compute info text."""
        if self.data is None:
            return {'error': 'No data loaded'}
        
        current_cycle = self.data['sleeps'][-1]['cycle'] if self.data['sleeps'] else 0
        total_cycles = self.data['total_cycles']
        progress = (current_cycle / total_cycles * 100) if total_cycles > 0 else 0
        
        # Elapsed time
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        else:
            elapsed_str = "Unknown"
        
        # ETA
        if progress > 0 and self.start_time:
            total_time = elapsed / (progress / 100)
            remaining = total_time - elapsed
            eta_str = str(remaining).split('.')[0]
        else:
            eta_str = "Calculating..."
        
        is_complete = current_cycle >= total_cycles
        
        return {
            'current_cycle': current_cycle,
            'total_cycles': total_cycles,
            'progress': f"{progress:.1f}",
            'elapsed_time': elapsed_str,
            'eta': eta_str,
            'total_deaths': len(self.data['deaths']),
            'total_consumptions': self.data['sleeps'][-1]['total_consumptions'] if self.data['sleeps'] else 0,
            'is_complete': is_complete
        }

# Flask app
app = Flask(__name__)
monitor = None

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    try:
        monitor.load_data()
        
        if monitor.data is None:
            return jsonify({'error': 'Training log not found or not readable'})
        
        return jsonify({
            'metrics': monitor.compute_metrics(),
            'charts': monitor.compute_charts(),
            'info': monitor.compute_info()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

def main():
    parser = argparse.ArgumentParser(description='Live training monitor web dashboard')
    parser.add_argument('log_dir', help='Directory containing training logs')
    parser.add_argument('--refresh', type=int, default=5, 
                        help='Default refresh rate in seconds (1-60)')
    parser.add_argument('--port', type=int, default=8050,
                        help='Web server port')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Web server host (use 0.0.0.0 for remote access)')
    
    args = parser.parse_args()
    
    # Validate refresh rate
    if not 1 <= args.refresh <= 60:
        print("Error: refresh rate must be between 1 and 60 seconds")
        sys.exit(1)
    
    # Initialize monitor
    global monitor
    monitor = TrainingMonitor(args.log_dir, args.refresh)
    
    print(f"\n{'='*70}")
    print(f"🧠 SOLITER TRAINING MONITOR")
    print(f"{'='*70}")
    print(f"Monitoring: {args.log_dir}")
    print(f"Default refresh: {args.refresh} seconds")
    print(f"Web dashboard: http://{args.host}:{args.port}")
    print(f"\nOpen the URL in your browser to view live training progress.")
    print(f"Press Ctrl+C to stop the monitor.\n")
    print(f"{'='*70}\n")
    
    try:
        app.run(host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped by user.")

if __name__ == '__main__':
    main()

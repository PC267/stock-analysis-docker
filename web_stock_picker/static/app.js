async function loadStocks() {
  const s = document.getElementById('strategy').value;
  const k = document.getElementById('keyword').value;
  const l = document.getElementById('limit').value || 50;
  const st = document.getElementById('status');
  st.textContent = '加载中...';
  const res = await fetch(`/api/stocks?strategy=${encodeURIComponent(s)}&limit=${encodeURIComponent(l)}&keyword=${encodeURIComponent(k)}`);
  const data = await res.json();
  const tb = document.getElementById('tbody');
  tb.innerHTML = '';
  for (const x of data.items) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${x.rank}</td><td class="code">${x.code}</td><td>${x.name}</td><td><span class="sig">${x.signal}</span></td><td class="score">${x.score}</td><td>${x.price}</td><td class="${x.change_pct >= 0 ? 'red' : 'green'}">${x.change_pct}%</td><td>${x.ma5 ?? '-'} / ${x.ma20 ?? '-'} / ${x.ma60 ?? '-'}</td><td>${x.rsi14 ?? '-'}</td><td>${x.volume_ratio ?? '-'}</td><td class="${(x.momentum20 || 0) >= 0 ? 'red' : 'green'}">${x.momentum20 ?? '-'}%</td><td>${x.drawdown ?? '-'}%</td><td>${(x.tags || []).map(t => `<span class="tag">${t}</span>`).join('')}</td>`;
    tb.appendChild(tr);
  }
  st.textContent = `${data.updated_at} · ${data.count} 只`;
}
document.getElementById('refresh').onclick = loadStocks;
document.getElementById('strategy').onchange = loadStocks;
document.getElementById('keyword').onkeydown = e => { if (e.key === 'Enter') loadStocks(); };
loadStocks();
